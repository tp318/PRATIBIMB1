"""
=============================================================================
AEROTWIN-4 SHARED FAULT PERTURBATION ENGINE
=============================================================================
apply_fault_perturbations() is called IDENTICALLY during:
  1. Dataset generation (generate_dataset.py)
  2. Live inference (xgboost_adapter.py)

Fault class indices:
  0: NORMAL
  1: MISFIRE           — stochastic RPM dip, EGT dip, vibration/kurtosis spike
  2: INJECTOR_FAULT    — high-variance fuel, EGT swing, mild vibration
  3: CODING_DEGRADATION— real engine CHT rise (no perturbation needed)
  4: LUBRICATION_ISSUE — real engine oil-pressure drop (no perturbation needed)
  5: SENSOR_DRIFT      — EGT drift bias (accumulated externally by caller)
  6: COMBUSTION_INST   — oscillating RPM, elevated vibration + kurtosis
  7: OVERHEATING       — real engine CHT/EGT rise (no perturbation needed)
  8: ABNORMAL_VIBRATION— amplified bearing vibration + kurtosis/crest
=============================================================================
"""
import math
from typing import Tuple
import numpy as np

FAULT_CLASS_NAMES = [
    "NORMAL", "MISFIRE", "INJECTOR_ABNORMALITY", "CODING_DEGRADATION",
    "LUBRICATION_ISSUE", "SENSOR_DRIFT_FAILURE", "COMBUSTION_INSTABILITY",
    "OVERHEATING", "ABNORMAL_VIBRATION",
]

# UI fault_type string → fault class index (for live injection)
FAULT_TYPE_TO_CLASS = {
    "CLEAR": 0, "NORMAL": 0,
    "CYLINDER": 1,    # MISFIRE (default cylinder fault = misfire signature)
    "BEARING":  8,    # ABNORMAL_VIBRATION
    "COOLING":  7,    # OVERHEATING (cooling system fault = overheating)
    "LUBRICATION": 4, # LUBRICATION_ISSUE
    "SENSOR":   5,    # SENSOR_DRIFT_FAILURE — instrumentation fault, engine stays healthy
}

# Calibrated noise stds — MUST match generate_dataset.py exactly
NOISE_STD = {
    "rpm":       30.0,
    "cht":        2.0,
    "egt":        8.0,
    "oil_press":  0.08,
    "oil_temp":   1.2,
    "fuel_flow":  0.6,
    "vibration":  0.12,
}


def apply_fault_perturbations(
    fault_class: int,
    ramp_factor: float,
    severity: float,
    sim_time: float,
    rpm: float,
    egt: float,
    fuel: float,
    vib: float,
    rng: np.random.Generator,
) -> Tuple[float, float, float, float, float, float]:
    """
    Apply physics-consistent synthetic perturbations for faults that the
    AeroTwin engine does not natively model in vibration/combustion.

    Returns: (rpm, egt, fuel, vib, vib_kurtosis, vib_crest)
    """
    vib_kurtosis = 3.0
    vib_crest    = 3.5
    eff          = severity * ramp_factor

    if fault_class == 1 and eff > 0:
        # MISFIRE: stochastic combustion dropout
        if rng.random() < eff * 0.70:
            rpm         -= eff * 350.0
            egt         -= eff * 180.0
            vib         += eff * 0.60
            vib_kurtosis += eff * 8.0
            vib_crest    += eff * 4.0
        else:
            vib_kurtosis += eff * 1.5

    elif fault_class == 2 and eff > 0:
        # INJECTOR_ABNORMALITY: stochastic fuel delivery variance
        fuel_err      = float(rng.normal(0.0, eff * 3.5))
        fuel         += fuel_err
        egt          -= fuel_err * 8.0
        vib          += abs(fuel_err) * 0.04
        vib_kurtosis += eff * 1.5

    elif fault_class == 6 and eff > 0:
        # COMBUSTION_INSTABILITY: pressure wave oscillation
        rpm          += eff * 100.0 * math.sin(sim_time * 2.1)
        vib          += eff * 0.35
        vib_kurtosis += eff * 3.5
        vib_crest    += eff * 1.5

    elif fault_class == 8 and eff > 0:
        # ABNORMAL_VIBRATION: bearing surface damage impulses
        vib          += eff * 0.80
        vib_kurtosis += eff * 7.0
        vib_crest    += eff * 4.0

    elif fault_class == 7 and eff > 0:
        # OVERHEATING: severe thermal runaway — EGT amplification
        # Distinguishes from CODING_DEGRADATION (both use COOLING engine fault)
        egt += eff * 45.0

    return rpm, egt, fuel, vib, vib_kurtosis, vib_crest
