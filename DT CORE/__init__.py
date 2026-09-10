"""
DT CORE — Digital Twin Core Package

This package implements the full physics chain described in Absolutely.docx:

    Environment → Air Path → Fuel → Combustion → Crankshaft → Thermal → Oil
                                                                         ↓
                                                                   Y_predicted
                                                                         ↓
                                                         Y_measured − Y_predicted
                                                                         ↓
                                                                   Residuals (r, z)

Quick start:
    from DT_CORE import DigitalTwinEngine, TelemetryBus

    dt  = DigitalTwinEngine()
    bus = TelemetryBus()
    bus.start_sortie("SORTIE_001")

    # Each 0.1s tick
    result = dt.step(
        throttle    = 0.65,
        altitude_ft = 3000,
        ambient_c   = 15.0,
        measured    = {"RPM": 4500, "EGT": 695, "CHT": 140,
                       "OilPress": 4.1, "OilTemp": 88, "FuelFlow": 22.0},
    )
    bus.publish(result.to_dict())
"""

import os as _os
import sys as _sys

# Ensure this directory is on sys.path so intra-package imports work when
# the package is imported from any location.
_PKG_DIR = _os.path.dirname(_os.path.abspath(__file__))
if _PKG_DIR not in _sys.path:
    _sys.path.insert(0, _PKG_DIR)

from engine    import DigitalTwinEngine, DTStepResult, DT_STEP_S
from bus       import TelemetryBus
from residuals import compute_residuals, ResidualFrame, SIGMA, Z_CAUTION, Z_WARNING
from environment import compute_atmosphere
from air_path    import compute_air_path
from fuel_model  import compute_fuel
from combustion  import compute_combustion
from crankshaft  import CrankshaftState, step_crankshaft
from thermal     import ThermalState, step_thermal
from oil         import OilState, step_oil

__version__ = "1.0.0"
__all__ = [
    "DigitalTwinEngine",
    "DTStepResult",
    "DT_STEP_S",
    "TelemetryBus",
    "compute_residuals",
    "ResidualFrame",
    "SIGMA",
    "Z_CAUTION",
    "Z_WARNING",
]
