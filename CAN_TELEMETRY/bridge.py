"""
CANBridge — Hardware CAN bus to AeroTwin EngineTelemetry bridge.

Connects to a real CAN interface (socketcan on Linux, PEAK/Kvaser on Windows)
using python-can, decodes incoming frames via CANDecoder, assembles complete
EngineTelemetry objects, and exposes them as a blocking generator stream.

For environments without a physical CAN interface, the bridge automatically
falls back to CANSimulator, making the code testable on any developer machine.

Architecture:
    CAN Hardware / OS Driver
         |  (python-can bus object)
         v
    CANBridge._rx_loop()          <- background thread
         |  (raw CAN frames)
         v
    CANDecoder.decode_frame()     <- per-frame decoding
         |  (signal dicts, multi-frame assembly)
         v
    _FrameAssembler               <- waits for all 7 frame IDs
         |  (complete telemetry dict)
         v
    EngineTelemetry(**dict)       <- yielded to caller
"""

import threading
import time
import logging
import queue
from typing import Generator, Optional

from .decoder import CANDecoder
from .dbc import DBC_ENGINE_SIGNALS

log = logging.getLogger(__name__)

# All frame IDs that must be received before assembling one telemetry frame
_REQUIRED_FRAME_IDS = set(DBC_ENGINE_SIGNALS.keys())

# Maximum age (seconds) of a partial telemetry frame before it is discarded
_FRAME_TTL_S = 0.5


class _FrameAssembler:
    """
    Accumulates decoded signal dicts from multiple CAN frame IDs and
    emits a complete merged dict when all required frames have arrived
    within TTL_S of the first frame.
    """

    def __init__(self, required_ids=_REQUIRED_FRAME_IDS, ttl=_FRAME_TTL_S):
        self._required = required_ids
        self._ttl = ttl
        self._pending: dict = {}
        self._seen_ids: set = set()
        self._first_ts: float = 0.0
        self._lock = threading.Lock()

    def feed(self, frame_id: int, signals: dict, timestamp: float):
        """
        Feed decoded signals from one frame. Returns a complete telemetry
        dict if all required frames have been received, else None.
        """
        with self._lock:
            now = timestamp

            # Start a new assembly window
            if not self._seen_ids:
                self._first_ts = now

            # Discard stale partial frames
            if now - self._first_ts > self._ttl:
                log.debug("Assembler TTL expired — discarding partial frame")
                self._pending.clear()
                self._seen_ids.clear()
                self._first_ts = now

            self._pending.update(signals)
            self._seen_ids.add(frame_id)

            if self._required.issubset(self._seen_ids):
                complete = dict(self._pending)
                self._pending.clear()
                self._seen_ids.clear()
                return complete

            return None


class CANBridge:
    """
    High-level CAN bus to EngineTelemetry bridge.

    Parameters
    ----------
    interface : str
        python-can interface name (e.g. "socketcan", "pcan", "kvaser",
        "vector", "virtual").
    channel : str
        Interface channel (e.g. "can0", "PCAN_USBBUS1").
    bitrate : int
        CAN bus bitrate in bits/second (default 500 kbps — SAE J1939 std).
    engine_id : str
        Engine identifier embedded into assembled EngineTelemetry objects.
    fallback_to_sim : bool
        If True and python-can is unavailable or hardware fails to open,
        transparently fall back to CANSimulator (useful for dev/CI).
    """

    def __init__(
        self,
        interface: str = "socketcan",
        channel: str = "can0",
        bitrate: int = 500_000,
        engine_id: str = "AEROTWIN-4-001",
        fallback_to_sim: bool = True,
    ):
        self.interface = interface
        self.channel = channel
        self.bitrate = bitrate
        self.engine_id = engine_id
        self.fallback_to_sim = fallback_to_sim

        self._decoder = CANDecoder(strict=False)
        self._assembler = _FrameAssembler()
        self._queue: queue.Queue = queue.Queue(maxsize=512)
        self._bus = None
        self._rx_thread: Optional[threading.Thread] = None
        self._running = False
        self._using_sim = False

    # ------------------------------------------------------------------ #
    # Lifecycle                                                            #
    # ------------------------------------------------------------------ #

    def start(self):
        """Open the CAN bus and begin receiving frames in a background thread."""
        if self._running:
            return

        try:
            import can
            self._bus = can.Bus(
                interface=self.interface,
                channel=self.channel,
                bitrate=self.bitrate,
            )
            log.info("CAN bridge opened: %s / %s @ %d bps",
                     self.interface, self.channel, self.bitrate)
        except Exception as exc:
            if self.fallback_to_sim:
                log.warning(
                    "python-can unavailable (%s). Falling back to CANSimulator.", exc
                )
                self._using_sim = True
            else:
                raise

        self._running = True
        target = self._sim_loop if self._using_sim else self._rx_loop
        self._rx_thread = threading.Thread(target=target, daemon=True,
                                           name="can-rx")
        self._rx_thread.start()

    def stop(self):
        """Shut down the background receiver and close the bus."""
        self._running = False
        if self._rx_thread:
            self._rx_thread.join(timeout=2.0)
        if self._bus:
            self._bus.shutdown()
            self._bus = None
        log.info("CAN bridge stopped.")

    # ------------------------------------------------------------------ #
    # Consumer API                                                         #
    # ------------------------------------------------------------------ #

    def stream(self, timeout: float = 5.0) -> Generator:
        """
        Blocking generator that yields assembled EngineTelemetry dicts.
        Raises StopIteration when the bridge is stopped.

        Parameters
        ----------
        timeout : float
            Maximum seconds to wait for the next complete telemetry frame.
        """
        while self._running:
            try:
                frame = self._queue.get(timeout=timeout)
                yield frame
            except queue.Empty:
                log.warning("CAN stream timeout — no frames received in %.1fs", timeout)

    def latest(self) -> Optional[dict]:
        """Return the most recently assembled telemetry dict, or None."""
        try:
            return self._queue.get_nowait()
        except queue.Empty:
            return None

    @property
    def stats(self) -> dict:
        return {
            "decoder": self._decoder.stats,
            "queue_depth": self._queue.qsize(),
            "using_simulator": self._using_sim,
        }

    # ------------------------------------------------------------------ #
    # Internal receiver loops                                              #
    # ------------------------------------------------------------------ #

    def _rx_loop(self):
        """Hardware CAN receive loop — runs in background thread."""
        import can
        while self._running:
            try:
                msg = self._bus.recv(timeout=0.1)
                if msg is None:
                    continue
                self._process_msg(msg.arbitration_id, bytes(msg.data),
                                  msg.timestamp)
            except Exception as exc:
                log.error("CAN rx error: %s", exc)

    def _sim_loop(self):
        """Fallback loop using CANSimulator when hardware is absent."""
        from .simulator import CANSimulator
        sim = CANSimulator(engine_id=self.engine_id, dt=0.01)
        for fid, payload, ts in sim.generate():
            if not self._running:
                break
            self._process_msg(fid, payload, ts)

    def _process_msg(self, frame_id: int, payload: bytes, timestamp: float):
        """Decode a single CAN message and attempt to assemble a telemetry frame."""
        signals = self._decoder.decode_frame(frame_id, payload)
        if not signals:
            return
        complete = self._assembler.feed(frame_id, signals, timestamp)
        if complete is not None:
            complete.setdefault("timestamp", timestamp)
            complete.setdefault("engine_id", self.engine_id)
            complete.setdefault("simulation_time", timestamp)
            complete.setdefault("operating_mode", "CAN_LIVE")
            # Push into consumer queue (drop oldest if full)
            if self._queue.full():
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    pass
            self._queue.put_nowait(complete)
