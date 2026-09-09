"""
============================================================================
config.py  —  MALE UAV Fault Detection: Centralised Configuration
============================================================================
All hyper-parameters, sensor definitions, and class mappings live here.
Import this module everywhere else to avoid magic numbers scattered through
the codebase.

Design principle: Every number that might need tuning is a named constant.
============================================================================
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import torch

# ── Repository root (all relative paths anchor here) ──────────────────────
ROOT_DIR: Path = Path(__file__).resolve().parent


# ═══════════════════════════════════════════════════════════════════════════
#  SENSOR CHANNEL DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════

# Each channel is the scalar residual produced by the AUKF/MVEM block:
#   residual_i(t) = sensor_reading_i(t) − MVEM_predicted_i(t)
# Near-zero residuals → healthy engine; elevated residuals → deviation.

SENSOR_CHANNELS: List[str] = [
    "rpm_residual",           # 0 — Engine speed deviation          [RPM]
    "cht_residual",           # 1 — Cylinder head temperature dev.  [°C]
    "egt_residual",           # 2 — Exhaust gas temperature dev.    [°C]
    "oil_pressure_residual",  # 3 — Oil pressure deviation          [bar]
    "oil_temp_residual",      # 4 — Oil temperature deviation       [°C]
    "fuel_flow_residual",     # 5 — Fuel-flow rate deviation        [L/h]
    "vibration_residual",     # 6 — Vibration RMS deviation         [g]
    "batt_voltage_residual",  # 7 — Battery / alternator V dev.     [V]
    "inj_timing_residual",    # 8 — Injection timing deviation      [°CA]
]

SENSOR_UNITS: Dict[str, str] = {
    "rpm_residual":           "RPM",
    "cht_residual":           "°C",
    "egt_residual":           "°C",
    "oil_pressure_residual":  "bar",
    "oil_temp_residual":      "°C",
    "fuel_flow_residual":     "L/h",
    "vibration_residual":     "g",
    "batt_voltage_residual":  "V",
    "inj_timing_residual":    "°CA",
}

NUM_SENSORS: int = len(SENSOR_CHANNELS)


# ═══════════════════════════════════════════════════════════════════════════
#  FAULT CLASS DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════

FAULT_CLASSES: List[str] = [
    "Normal operation",
    "Misfire conditions",
    "Injector abnormalities",
    "Cooling degradation",
    "Lubrication issues",
    "Sensor drift / failure",
    "Combustion instability",
    "Overheating trends",
    "Abnormal vibration patterns",
]

NUM_CLASSES: int = len(FAULT_CLASSES)

FAULT_SHORT: Dict[int, str] = {
    0: "HEALTHY",       1: "MISFIRE",    2: "INJECTOR",
    3: "COOLING",       4: "LUBRICATION", 5: "SENSOR_DRIFT",
    6: "COMBUSTION",    7: "OVERHEAT",   8: "VIBRATION",
}

# Maintenance urgency 0=none 1=watch 2=caution 3=warning 4=critical
FAULT_URGENCY: Dict[int, int] = {
    0: 0, 1: 3, 2: 3, 3: 2, 4: 3, 5: 2, 6: 4, 7: 4, 8: 3,
}

URGENCY_LABELS: Dict[int, str] = {
    0: "NONE", 1: "WATCH", 2: "CAUTION", 3: "WARNING", 4: "CRITICAL",
}


# ═══════════════════════════════════════════════════════════════════════════
#  SLIDING WINDOW CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

SAMPLE_RATE_HZ: float = 50.0
WINDOW_SIZE: int = 64   # ~1.28 s at 50 Hz
STRIDE: int = 16        # 75% overlap


# ═══════════════════════════════════════════════════════════════════════════
#  MODEL ARCHITECTURE HYPER-PARAMETERS
# ═══════════════════════════════════════════════════════════════════════════

MODEL_CONFIG = {
    "cnn_filters_1":   64,
    "cnn_filters_2":   128,
    "cnn_kernel_size": 5,
    "cnn_dropout":     0.25,
    "lstm_hidden":     128,
    "lstm_layers":     2,
    "lstm_dropout":    0.30,
    "fc_hidden":       64,
    "fc_dropout":      0.30,
}


# ═══════════════════════════════════════════════════════════════════════════
#  TRAINING HYPER-PARAMETERS
# ═══════════════════════════════════════════════════════════════════════════

BATCH_SIZE:    int   = 64
NUM_EPOCHS:    int   = 50
LEARNING_RATE: float = 1e-3
WEIGHT_DECAY:  float = 1e-4
VAL_SPLIT:     float = 0.15
GRAD_CLIP:     float = 1.0
RANDOM_SEED:   int   = 42
LR_ETA_MIN:    float = 1e-6


# ═══════════════════════════════════════════════════════════════════════════
#  DEEPSHAP CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

SHAP_BACKGROUND_SIZE:    int   = 100
CONFIDENCE_THRESHOLD_PCT: float = 55.0


# ═══════════════════════════════════════════════════════════════════════════
#  FILE PATHS
# ═══════════════════════════════════════════════════════════════════════════

CHECKPOINT_PATH: str = str(ROOT_DIR / "best_fault_detector.pt")
SCALER_PATH:     str = str(ROOT_DIR / "residual_scaler.pkl")


# ═══════════════════════════════════════════════════════════════════════════
#  DEVICE AUTO-DETECTION
# ═══════════════════════════════════════════════════════════════════════════

def get_device() -> torch.device:
    """Select: CUDA GPU > Apple MPS > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE: torch.device = get_device()
