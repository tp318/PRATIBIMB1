"""
CANDecoder — Low-level CAN frame payload decoder for PRATIBIMB.

Converts raw 8-byte CAN payloads into physical-unit floating-point values
using the DBC signal table. Works with both real hardware (python-can) and
offline byte buffers, making it easy to unit-test without hardware present.

Implementation notes:
  - Byte order: Intel/little-endian throughout (standard for J1939 ECUs).
  - Multi-byte values: int.from_bytes(..., byteorder='little').
  - Signed integers use Python's built-in two's-complement handling.
  - CRC / checksum validation is left to the transport layer (CANBridge).
"""

import struct
import time
import logging
from typing import Dict, Any, Optional

from .dbc import DBC_ENGINE_SIGNALS, EngineDBCSignal

log = logging.getLogger(__name__)


class CANDecodeError(ValueError):
    """Raised when a frame payload cannot be decoded."""


class CANDecoder:
    """
    Stateless CAN frame decoder.

    Given a CAN frame ID and its 8-byte payload, returns a dictionary of
    {signal_name: physical_value} ready to be merged into an EngineTelemetry.

    Parameters
    ----------
    strict : bool
        If True, raises CANDecodeError on unknown frame IDs.
        If False (default), unknown IDs are silently skipped.
    """

    def __init__(self, strict: bool = False):
        self.strict = strict
        self._frame_count: int = 0
        self._error_count: int = 0

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def decode_frame(self, frame_id: int, payload: bytes) -> Dict[str, float]:
        """
        Decode a single CAN frame payload into physical signal values.

        Parameters
        ----------
        frame_id : int
            11-bit CAN frame identifier.
        payload : bytes
            Exactly 8 bytes of CAN frame data (padded with 0x00 if shorter).

        Returns
        -------
        dict
            Mapping of signal name -> physical value.

        Raises
        ------
        CANDecodeError
            If `strict=True` and frame_id is not in the DBC catalogue.
        """
        if frame_id not in DBC_ENGINE_SIGNALS:
            if self.strict:
                raise CANDecodeError(f"Unknown CAN frame ID: 0x{frame_id:03X}")
            return {}

        # Pad payload to 8 bytes if shorter (common with RTR frames)
        if len(payload) < 8:
            payload = payload.ljust(8, b"\x00")

        self._frame_count += 1
        _, signals = DBC_ENGINE_SIGNALS[frame_id]
        result: Dict[str, float] = {}

        for sig in signals:
            try:
                raw = self._extract_raw(payload, sig)
                result[sig.name] = raw * sig.scale + sig.offset
            except (IndexError, struct.error) as exc:
                self._error_count += 1
                log.warning("Decode error for signal %s in frame 0x%03X: %s",
                            sig.name, frame_id, exc)

        return result

    def decode_frame_batch(
        self,
        frames: list,
    ) -> list:
        """
        Decode a list of (frame_id, payload) tuples.

        Returns a list of decoded signal dicts, one per frame.
        """
        return [self.decode_frame(fid, pl) for fid, pl in frames]

    @property
    def stats(self) -> Dict[str, int]:
        return {"frames_decoded": self._frame_count,
                "decode_errors": self._error_count}

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _extract_raw(payload: bytes, sig: EngineDBCSignal) -> int:
        """Extract an integer raw value from the payload bytes."""
        raw_bytes = payload[sig.start_byte: sig.start_byte + sig.length_bytes]
        return int.from_bytes(raw_bytes, byteorder="little", signed=sig.signed)

    # ------------------------------------------------------------------ #
    # Encoder (for simulation / test injection)                           #
    # ------------------------------------------------------------------ #

    @staticmethod
    def encode_signal(signal: EngineDBCSignal, physical_value: float) -> bytes:
        """
        Encode a physical value back to raw bytes for a given DBC signal.
        Used by CANSimulator to generate synthetic CAN frames from
        EngineTelemetry objects.
        """
        raw = round((physical_value - signal.offset) / signal.scale)
        raw = max(0, min(raw, (1 << (signal.length_bytes * 8)) - 1))
        return raw.to_bytes(signal.length_bytes, byteorder="little",
                            signed=signal.signed)


# Expose a module-level singleton for convenience
SID_MAP = {}  # re-exported from dbc.py for external consumers
from .dbc import SID_MAP  # noqa: E402, F401
