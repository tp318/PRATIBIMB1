"""
AeroTwin-4 Degradation Configuration & Enum Definitions.

Defines degradation types (CYLINDER, BEARING, COOLING, LUBRICATION, NONE),
target components, severity levels, trajectory types, and configuration models.

Note on Phenomenological Mappings:
Severity mappings (0.0 = NORMAL to 1.0 = CRITICAL) represent prototype simulator
assumptions mapping severity S in [0.0, 1.0] to physical engine parameters.
They serve as engineering simulation labels rather than clinical/aviation limits.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


class DegradationType(str, Enum):
    NONE = "NONE"
    CYLINDER = "CYLINDER"
    BEARING = "BEARING"
    COOLING = "COOLING"
    LUBRICATION = "LUBRICATION"


class ComponentID(str, Enum):
    NONE = "NONE"
    CYLINDER_1 = "CYLINDER_1"
    CYLINDER_2 = "CYLINDER_2"
    CYLINDER_3 = "CYLINDER_3"
    CYLINDER_4 = "CYLINDER_4"
    BEARING = "BEARING"
    COOLING_SYSTEM = "COOLING_SYSTEM"
    LUBRICATION_SYSTEM = "LUBRICATION_SYSTEM"


class SeverityLevel(float, Enum):
    NORMAL = 0.00
    SLIGHT = 0.20
    MODERATE = 0.40
    SIGNIFICANT = 0.60
    SEVERE = 0.80
    CRITICAL = 1.00


class TrajectoryType(str, Enum):
    CONSTANT = "CONSTANT"
    LINEAR = "LINEAR"
    STEP = "STEP"
    EXPONENTIAL = "EXPONENTIAL"


@dataclass
class ComponentDegradation:
    """
    Degradation configuration for a single engine component.
    """
    degradation_type: DegradationType = DegradationType.NONE
    component_id: ComponentID = ComponentID.NONE
    severity: float = 0.0  # [0.0, 1.0]

    def __post_init__(self):
        if not (0.0 <= self.severity <= 1.0):
            raise ValueError(f"Degradation severity must be in range [0.0, 1.0], got {self.severity}")


@dataclass
class DegradationConfig:
    """
    Full simulation degradation configuration supporting single and combined fault injection.
    """
    degradation_list: List[ComponentDegradation] = field(default_factory=list)
    trajectory_type: TrajectoryType = TrajectoryType.CONSTANT
    start_time: float = 0.0
    ramp_duration: float = 60.0  # seconds for trajectory progression

    @classmethod
    def healthy(cls) -> "DegradationConfig":
        """
        Construct healthy baseline configuration (zero degradation).
        """
        return cls(
            degradation_list=[
                ComponentDegradation(
                    degradation_type=DegradationType.NONE,
                    component_id=ComponentID.NONE,
                    severity=0.0,
                )
            ],
            trajectory_type=TrajectoryType.CONSTANT,
        )

    @classmethod
    def single_fault(
        cls,
        degradation_type: DegradationType,
        component_id: ComponentID,
        severity: float,
        trajectory_type: TrajectoryType = TrajectoryType.CONSTANT,
        start_time: float = 0.0,
        ramp_duration: float = 60.0,
    ) -> "DegradationConfig":
        """
        Construct single-fault degradation configuration.
        """
        return cls(
            degradation_list=[
                ComponentDegradation(
                    degradation_type=degradation_type,
                    component_id=component_id,
                    severity=severity,
                )
            ],
            trajectory_type=trajectory_type,
            start_time=start_time,
            ramp_duration=ramp_duration,
        )
