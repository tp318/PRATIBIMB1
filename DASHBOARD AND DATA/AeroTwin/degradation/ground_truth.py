"""
AeroTwin-4 Ground Truth Data Contract.

Defines dual-granularity ground-truth labels:
1. RunGroundTruth: Scenario-level dataset metadata.
2. SampleGroundTruth: Per-time-step ground-truth telemetry labels.

Ground truth is generated directly from injected degradation parameters
and is strictly isolated from sensor observations.
"""

import json
from dataclasses import asdict, dataclass
from typing import Dict, Any


@dataclass
class RunGroundTruth:
    """
    Scenario/Run-level ground truth metadata.
    """
    run_id: str                  # Unique simulation run identifier
    degradation_type: str        # e.g. "CYLINDER", "BEARING", "COOLING", "LUBRICATION", "NONE"
    target_component: str        # e.g. "CYLINDER_3", "BEARING", "NONE"
    max_severity: float          # Peak injected severity in [0.0, 1.0]
    trajectory_type: str         # "CONSTANT", "LINEAR", "STEP", "EXPONENTIAL"
    seed: int                    # Random seed
    operating_profile: str       # Operating profile name (e.g. "DEFAULT_MISSION")
    sample_rate_hz: float = 100.0
    # Environmental context of the sortie (see degradation.conditions).
    # Recorded so a residual can be interpreted against the air the engine flew in.
    altitude_m: float = 0.0
    ambient_temperature_c: float = 20.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SampleGroundTruth:
    """
    Per-time-step sample ground truth label.
    """
    timestamp: float             # Unix wall-clock time
    simulation_time: float       # Simulation time (seconds)
    degradation_type: str        # Active degradation type
    target_component: str        # Active target component
    active_severity: float       # Injected active severity S(t) in [0.0, 1.0]
    current_health: float        # Health index H(t) = 1.0 - S(t) in [0.0, 1.0]
    is_degraded: bool            # True if active_severity > 0.001

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def healthy_default(cls, timestamp: float, simulation_time: float) -> "SampleGroundTruth":
        """
        Construct healthy default sample ground truth label (zero degradation).
        """
        return cls(
            timestamp=timestamp,
            simulation_time=simulation_time,
            degradation_type="NONE",
            target_component="NONE",
            active_severity=0.0,
            current_health=1.0,
            is_degraded=False,
        )
