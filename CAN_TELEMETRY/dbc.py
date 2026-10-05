"""
DBC Signal Definitions for PRATIBIMB CAN Telemetry Layer.

Maps CAN frame IDs (11-bit) to physical engine parameters using a
minimal DBC-like descriptor. Covers the Rotax 914 / indigenous DRDO
aero piston ECU message catalogue.

Signal encoding follows Intel byte-order (little-endian), matching
industry-standard MilCAN and SAE J1939 conventions.

Frame layout (example, 8-byte payload):
  ID 0x601 — RPM / Throttle frame
    [0:1]  RPM         scale=0.25, offset=0, unit=RPM,  range=[0, 9000]
    [2]    Throttle    scale=0.392, offset=0, unit=%,   range=[0, 100]
    [3]    Flags       bitfield (RUNNING, FAULT, STARTER)
"""

from dataclasses import dataclass
from typing import List


@dataclass
class EngineDBCSignal:
    """Describes a single physical signal inside a CAN frame payload."""
    name: str           # Python attribute name in EngineTelemetry
    start_byte: int     # Byte index in the 8-byte CAN payload (0-based)
    length_bytes: int   # Number of bytes encoding this value
    scale: float        # Value = raw * scale + offset
    offset: float
    unit: str
    signed: bool = False


# --------------------------------------------------------------------------- #
# PRATIBIMB Engine CAN Frame Catalogue
# --------------------------------------------------------------------------- #
# Format: {frame_id: (description, [EngineDBCSignal, ...])}
# --------------------------------------------------------------------------- #
DBC_ENGINE_SIGNALS = {

    # --- Frame 0x601: Speed & Throttle -------------------------------------
    0x601: ("RPM and throttle command", [
        EngineDBCSignal("rpm",      start_byte=0, length_bytes=2,
                        scale=0.25, offset=0.0, unit="RPM"),
        EngineDBCSignal("throttle", start_byte=2, length_bytes=1,
                        scale=0.00392, offset=0.0, unit="fraction"),
        # byte 3: status flags — decoded separately by bridge
    ]),

    # --- Frame 0x602: Thermal cluster (CHT, EGT, Oil Temp) -----------------
    0x602: ("Thermal subsystem", [
        EngineDBCSignal("cht",           start_byte=0, length_bytes=2,
                        scale=0.1, offset=-273.15, unit="degC"),
        EngineDBCSignal("egt",           start_byte=2, length_bytes=2,
                        scale=0.1, offset=-273.15, unit="degC"),
        EngineDBCSignal("oil_temperature", start_byte=4, length_bytes=2,
                        scale=0.1, offset=-273.15, unit="degC"),
    ]),

    # --- Frame 0x603: Lubrication & Fuel -----------------------------------
    0x603: ("Lubrication and fuel", [
        EngineDBCSignal("oil_pressure",    start_byte=0, length_bytes=2,
                        scale=0.1, offset=0.0, unit="kPa"),
        EngineDBCSignal("fuel_flow",       start_byte=2, length_bytes=2,
                        scale=0.0001, offset=0.0, unit="kg/s"),
        EngineDBCSignal("fuel_pressure",   start_byte=4, length_bytes=2,
                        scale=0.1, offset=0.0, unit="kPa"),
    ]),

    # --- Frame 0x604: Torque outputs ---------------------------------------
    0x604: ("Torque and load", [
        EngineDBCSignal("mean_torque",    start_byte=0, length_bytes=2,
                        scale=0.01, offset=0.0, unit="Nm", signed=True),
        EngineDBCSignal("instant_torque", start_byte=2, length_bytes=2,
                        scale=0.01, offset=0.0, unit="Nm", signed=True),
        EngineDBCSignal("load_torque",    start_byte=4, length_bytes=2,
                        scale=0.01, offset=0.0, unit="Nm", signed=True),
        EngineDBCSignal("net_torque",     start_byte=6, length_bytes=2,
                        scale=0.01, offset=0.0, unit="Nm", signed=True),
    ]),

    # --- Frame 0x605: Vibration & per-cylinder cylinder torques -----------
    0x605: ("Vibration and cylinder torques", [
        EngineDBCSignal("vibration",          start_byte=0, length_bytes=2,
                        scale=0.0001, offset=0.0, unit="g"),
        EngineDBCSignal("cylinder_1_torque",  start_byte=2, length_bytes=2,
                        scale=0.01, offset=0.0, unit="Nm", signed=True),
        EngineDBCSignal("cylinder_2_torque",  start_byte=4, length_bytes=2,
                        scale=0.01, offset=0.0, unit="Nm", signed=True),
    ]),
    0x606: ("Cylinder torques 3 and 4", [
        EngineDBCSignal("cylinder_3_torque",  start_byte=0, length_bytes=2,
                        scale=0.01, offset=0.0, unit="Nm", signed=True),
        EngineDBCSignal("cylinder_4_torque",  start_byte=2, length_bytes=2,
                        scale=0.01, offset=0.0, unit="Nm", signed=True),
        EngineDBCSignal("crank_angle",        start_byte=4, length_bytes=2,
                        scale=0.1, offset=0.0, unit="deg"),
    ]),

    # --- Frame 0x607: Auxiliary computed fields ----------------------------
    0x607: ("Derived fields", [
        EngineDBCSignal("oil_pressure_psi", start_byte=0, length_bytes=2,
                        scale=0.01, offset=0.0, unit="PSI"),
        EngineDBCSignal("fuel_flow_lph",    start_byte=2, length_bytes=2,
                        scale=0.01, offset=0.0, unit="L/h"),
        EngineDBCSignal("friction_torque",  start_byte=4, length_bytes=2,
                        scale=0.01, offset=0.0, unit="Nm", signed=True),
    ]),
}

# Quick lookup: signal name -> (frame_id, EngineDBCSignal descriptor)
SID_MAP = {}
for _fid, (_desc, _sigs) in DBC_ENGINE_SIGNALS.items():
    for _sig in _sigs:
        SID_MAP[_sig.name] = (_fid, _sig)
