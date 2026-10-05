"""
CAN TELEMETRY — PRATIBIMB CAN Bus Integration Layer.

Translates raw MIL-STD-1939 / SAE J1939 CAN frames from the UAV Engine
Control Unit (ECU) into the canonical AeroTwin EngineTelemetry schema,
enabling real hardware telemetry ingestion without touching the core twin.

Usage:
    from CAN_TELEMETRY import CANBridge, CANDecoder, CANSimulator

    bridge = CANBridge(interface="socketcan", channel="can0")
    bridge.start()                      # begins streaming decoded frames
    for frame in bridge.stream():
        print(frame.cht, frame.rpm)     # EngineTelemetry objects
"""

from .decoder import CANDecoder, SID_MAP
from .bridge import CANBridge
from .simulator import CANSimulator
from .dbc import EngineDBCSignal, DBC_ENGINE_SIGNALS

__all__ = [
    "CANDecoder",
    "SID_MAP",
    "CANBridge",
    "CANSimulator",
    "EngineDBCSignal",
    "DBC_ENGINE_SIGNALS",
]
