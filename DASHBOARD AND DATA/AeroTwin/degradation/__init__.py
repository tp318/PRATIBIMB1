"""
AeroTwin-4 Phase 3 Degradation Physics & Dataset Package.
"""

from .config import (
    DegradationConfig,
    ComponentDegradation,
    DegradationType,
    ComponentID,
    SeverityLevel,
    TrajectoryType,
)
from .mechanisms import PhysicalDegradationMapper
from .trajectory import DegradationTrajectoryCalculator
from .ground_truth import RunGroundTruth, SampleGroundTruth
from .injector import DegradationInjector, SensorNoiseModel
from .dataset import DatasetBuilder, SlidingWindowGenerator
from .validators import DatasetValidator

__all__ = [
    "DegradationConfig",
    "ComponentDegradation",
    "DegradationType",
    "ComponentID",
    "SeverityLevel",
    "TrajectoryType",
    "PhysicalDegradationMapper",
    "DegradationTrajectoryCalculator",
    "RunGroundTruth",
    "SampleGroundTruth",
    "DegradationInjector",
    "SensorNoiseModel",
    "DatasetBuilder",
    "SlidingWindowGenerator",
    "DatasetValidator",
]
