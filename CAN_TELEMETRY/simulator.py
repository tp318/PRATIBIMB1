"""
CANSimulator — Software CAN bus simulator for PRATIBIMB.

Generates synthetic CAN frames from EngineTelemetry values produced by the
AeroTwin physics engine, encodes them using the DBC signal table, and emits
(frame_id, payload, timestamp) tuples at the correct inter-frame gap timing.

This lets the full CAN telemetry pipeline be exercised end-to-end on any
development machine or CI runner without physical CAN hardware.

Frame emission order follows priority:
    0x601 (RPM/throttle) -> 0x602 (thermal) -> 0x603 (fluids)
    -> 0x604 (torques) -> 0x605/606 (cylinders) -> 0x607 (derived)

Each frame is emitted 1 ms apart to mimic real bus scheduling.
"""

import time
import math
import logging
from typing import Generator, Tuple

from .dbc import DBC_ENGINE_SIGNALS
from .decoder import CANDecoder

log = logging.getLogger(__name__)

# Ordered list of frame IDs to emit per physics step
_EMIT_ORDER = [0x601, 0x602, 0x603, 0x604, 0x605, 0x606, 0x607]

# Inter-frame delay on real 500 kbps CAN: 8-byte frame ≈ 0.128 ms.
# We simulate 1 ms between frames to stay realistic without burning CPU.
_INTER_FRAME_DELAY_S = 0.001


class CANSimulator:
    """
    Software CAN bus simulator.

    Internally runs the AeroTwin MVEM physics engine and encodes its
    EngineTelemetry output into CAN frames using the DBC signal table.

    Parameters
    ----------
    engine_id : str
        Identifier embedded in every telemetry packet.
    dt : float
        Physics step size in seconds (default 0.01 s = 100 Hz).
    seed : int
        Random seed for reproducible engine noise.
    real_time : bool
        If True, sleeps between frames to honour wall-clock rate.
        If False (default), runs as fast as possible.
    """

    def __init__(
        self,
        engine_id: str = "AEROTWIN-4-001",
        dt: float = 0.01,
        seed: int = 42,
        real_time: bool = True,
    ):
        self.engine_id = engine_id
        self.dt = dt
        self.seed = seed
        self.real_time = real_time
        self._decoder = CANDecoder(strict=False)
        self._t = 0.0

    # ------------------------------------------------------------------ #
    # Generator API                                                        #
    # ------------------------------------------------------------------ #

    def generate(self) -> Generator[Tuple[int, bytes, float], None, None]:
        """
        Infinite generator of (frame_id, payload, timestamp) tuples.
        """
        # Try to use the real AeroTwin physics engine; fall back to
        # a lightweight analytic model if imports fail.
        try:
            from AeroTwin.simulator import EngineRunner
            runner = EngineRunner(dt=self.dt, seed=self.seed)
            use_runner = True
        except ImportError:
            log.info("AeroTwin EngineRunner not found — using analytic fallback.")
            runner = None
            use_runner = False

        while True:
            # Advance physics by one step
            if use_runner:
                state = runner.step()
                telem_dict = state.to_dict()
            else:
                telem_dict = self._analytic_state(self._t)
            self._t += self.dt

            frames = self._encode(telem_dict)

            wall_ts = time.time()
            for frame_id, payload in frames:
                yield (frame_id, payload, wall_ts)
                if self.real_time:
                    time.sleep(_INTER_FRAME_DELAY_S)
                wall_ts += _INTER_FRAME_DELAY_S

            if self.real_time:
                time.sleep(max(0.0, self.dt - len(frames) * _INTER_FRAME_DELAY_S))

    # ------------------------------------------------------------------ #
    # Encoding                                                             #
    # ------------------------------------------------------------------ #

    def _encode(self, telem: dict) -> list:
        """
        Encode one EngineTelemetry dict into a list of (frame_id, payload).
        """
        frames = []
        for fid in _EMIT_ORDER:
            if fid not in DBC_ENGINE_SIGNALS:
                continue
            _, signals = DBC_ENGINE_SIGNALS[fid]
            payload = bytearray(8)
            for sig in signals:
                value = telem.get(sig.name, 0.0)
                encoded = CANDecoder.encode_signal(sig, value)
                payload[sig.start_byte: sig.start_byte + sig.length_bytes] = encoded
            frames.append((fid, bytes(payload)))
        return frames

    # ------------------------------------------------------------------ #
    # Analytic fallback                                                    #
    # ------------------------------------------------------------------ #

    def _analytic_state(self, t: float) -> dict:
        """
        Minimal analytic engine model for use when AeroTwin is not importable.
        Produces plausible-looking telemetry with sinusoidal variations.
        """
        throttle = 0.65 + 0.1 * math.sin(0.05 * t)
        rpm = 4500 + 500 * throttle + 50 * math.sin(2.0 * t)
        cht = 160 + 40 * throttle + 5 * math.sin(0.3 * t)
        egt = 680 + 80 * throttle + 8 * math.sin(0.25 * t)
        oil_temp = 90 + 15 * throttle
        oil_pres_kpa = 380 - 20 * (1 - throttle) + 3 * math.sin(t)
        fuel_flow_kgs = 0.012 * throttle
        vibration_g = 0.15 + 0.05 * math.sin(10 * t)
        mean_torq = 45 * throttle
        return {
            "timestamp": time.time(),
            "simulation_time": t,
            "engine_id": self.engine_id,
            "operating_mode": "CRUISE",
            "throttle": throttle,
            "rpm": rpm,
            "crank_angle": (rpm / 60 * 720 * t) % 720,
            "mean_torque": mean_torq,
            "instant_torque": mean_torq + 3 * math.sin(4 * t),
            "load_torque": mean_torq * 0.9,
            "friction_torque": mean_torq * 0.05,
            "net_torque": mean_torq * 0.05,
            "cylinder_1_torque": mean_torq * 0.26,
            "cylinder_2_torque": mean_torq * 0.25,
            "cylinder_3_torque": mean_torq * 0.25,
            "cylinder_4_torque": mean_torq * 0.24,
            "cht": cht,
            "egt": egt,
            "oil_temperature": oil_temp,
            "oil_pressure": oil_pres_kpa,
            "oil_pressure_psi": oil_pres_kpa * 0.14504,
            "fuel_flow": fuel_flow_kgs,
            "fuel_flow_lph": fuel_flow_kgs * 3600 / 0.72,
            "fuel_pressure": 280 + 5 * math.sin(t),
            "vibration": vibration_g,
        }
