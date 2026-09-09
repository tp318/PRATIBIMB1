"""
AeroTwin-4 Digital Twin State Engine Schemas & Data Contracts.

Defines canonical dataclasses for OperatingState, ExpectedState, ResidualState,
ResidualIndicators, and DigitalTwinFrame.

Strict Channel Separation Rule:
- Metadata (timestamp, simulation_time, engine_id, operating_mode): Preserved context, NO residuals.
- Operating Inputs (throttle, ambient_temperature): Inputs to Healthy Twin.
- Physical Outputs (rpm, torques, temperatures, pressures, vibration): Subject to residual calculation.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional, List


@dataclass
class OperatingState:
    """
    Operating inputs and environmental context extracted from telemetry.
    Does NOT contain ground-truth labels (severity, degradation_type, etc.).
    """
    timestamp: float
    simulation_time: float
    engine_id: str
    operating_mode: str
    throttle: float
    rpm: float
    crank_angle: float = 0.0
    ambient_temperature: float = 25.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ExpectedState:
    """
    Counterfactual healthy engine predictions for physical outputs.
    Excludes metadata fields.
    """
    rpm: float
    crank_angle: float
    mean_torque: float
    instant_torque: float
    load_torque: float
    friction_torque: float
    net_torque: float
    cylinder_1_torque: float
    cylinder_2_torque: float
    cylinder_3_torque: float
    cylinder_4_torque: float
    cht: float
    egt: float
    oil_temperature: float
    oil_pressure: float
    oil_pressure_psi: float
    fuel_flow: float
    fuel_flow_lph: float
    fuel_pressure: float
    vibration: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ResidualState:
    """
    Residual calculations for physical outputs:
    - raw_signed: (Observed - Expected)
    - absolute: abs(Observed - Expected)
    - normalized: (Observed - Expected) / scale_ref
    """
    raw_signed: Dict[str, float]
    absolute: Dict[str, float]
    normalized: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ResidualIndicators:
    """
    Interpretable health indicators derived from residuals.
    These are residual-derived deviation signals, NOT final health scores or diagnoses.
    """
    thermal_deviation: float
    oil_deviation: float
    vibration_deviation: float
    torque_deviation: float
    cylinder_balance_deviation: float
    window_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DigitalTwinFrame:
    """
    Canonical Digital Twin Frame combining context metadata, operating inputs,
    observed physical outputs, counterfactual expected outputs, residuals, and indicators.
    """
    timestamp: float
    simulation_time: float
    engine_id: str
    operating_mode: str
    operating_inputs: Dict[str, float]
    observed_outputs: Dict[str, float]
    expected_outputs: Dict[str, float]
    residuals: ResidualState
    indicators: ResidualIndicators

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "simulation_time": self.simulation_time,
            "engine_id": self.engine_id,
            "operating_mode": self.operating_mode,
            "operating_inputs": self.operating_inputs,
            "observed_outputs": self.observed_outputs,
            "expected_outputs": self.expected_outputs,
            "residuals": self.residuals.to_dict(),
            "indicators": self.indicators.to_dict(),
        }
