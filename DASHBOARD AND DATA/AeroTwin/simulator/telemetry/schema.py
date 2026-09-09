"""
AeroTwin-4 Canonical Telemetry Schema & Data Contract.

This module defines the standardized EngineTelemetry data contract used
throughout the AeroTwin digital twin system.
"""

import json
from dataclasses import asdict, dataclass
from typing import Dict, Any


@dataclass
class EngineTelemetry:
    """
    Standardized canonical telemetry payload emitted by the engine simulator.
    """

    # Time tracking
    timestamp: float           # Wall-clock timestamp (Unix time)
    simulation_time: float     # Simulation time (seconds)
    engine_id: str             # Engine identifier (e.g. "AEROTWIN-4-001")
    operating_mode: str        # Operating scenario (e.g. "CRUISE")

    # Command & State
    throttle: float            # Throttle command [0.0, 1.0]
    rpm: float                 # Engine speed (RPM)
    crank_angle: float         # Crankshaft angle in degrees [0.0, 720.0)

    # Torques (N*m)
    mean_torque: float         # Mean engine torque capability
    instant_torque: float      # Instantaneous total engine torque
    load_torque: float         # Propeller load torque
    friction_torque: float     # Mechanical friction torque
    net_torque: float          # Net instantaneous torque

    # Per-cylinder torque contributions (N*m)
    cylinder_1_torque: float
    cylinder_2_torque: float
    cylinder_3_torque: float
    cylinder_4_torque: float

    # Thermal Subsystem (°C)
    cht: float                 # Cylinder Head Temperature
    egt: float                 # Exhaust Gas Temperature
    oil_temperature: float     # Oil Temperature

    # Lubrication Subsystem
    oil_pressure: float        # Oil Pressure (kPa)
    oil_pressure_psi: float    # Oil Pressure (PSI)

    # Fuel Subsystem
    fuel_flow: float           # Fuel mass flow rate (kg/s)
    fuel_flow_lph: float       # Fuel volumetric flow rate (L/h)
    fuel_pressure: float       # Fuel line pressure (kPa)

    # Mechanical Vibration (g)
    vibration: float           # Overall RMS mechanical vibration

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert telemetry object to dictionary.
        """
        return asdict(self)

    def to_json(self) -> str:
        """
        Convert telemetry object to JSON string.
        """
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EngineTelemetry":
        """
        Construct telemetry object from dictionary.
        """
        return cls(**data)
