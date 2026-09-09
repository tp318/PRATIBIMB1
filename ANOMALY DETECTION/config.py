"""
============================================================================
config.py  —  MALE UAV Anomaly Detection: Centralised Configuration
============================================================================
All hyperparameters, sensor channel definitions, and operational thresholds
for the unsupervised LSTM Autoencoder Anomaly Detection subsystem.
============================================================================
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import torch

# ── Repository root (all relative paths anchor here) ──────────────────────
ROOT_DIR: Path = Path(__file__).resolve().parent


# ═══════════════════════════════════════════════════════════════════════════
#  SENSOR CHANNEL DEFINITIONS (AUKF/MVEM Residuals)
# ═══════════════════════════════════════════════════════════════════════════

# Must match FAULT DETECTION channels exactly for Digital Twin compatibility
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
#  SLIDING WINDOW CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

SAMPLE_RATE_HZ: float = 50.0
WINDOW_SIZE: int = 64   # ~1.28 s at 50 Hz
STRIDE: int = 16        # 75% overlap for continuous monitoring


# ═══════════════════════════════════════════════════════════════════════════
#  LSTM AUTOENCODER MODEL ARCHITECTURE
# ═══════════════════════════════════════════════════════════════════════════

ENCODER_HIDDEN: int = 64     # First LSTM layer hidden dimension
LATENT_DIM: int = 32         # Information bottleneck dimension
NUM_LAYERS: int = 2          # Stacked LSTM depth
DROPOUT: float = 0.1         # Recurrent dropout between layers


# ═══════════════════════════════════════════════════════════════════════════
#  TRAINING HYPERPARAMETERS
# ═══════════════════════════════════════════════════════════════════════════

BATCH_SIZE: int = 64
LEARNING_RATE: float = 1e-3
WEIGHT_DECAY: float = 1e-4
NUM_EPOCHS: int = 40
PATIENCE: int = 7            # Early stopping patience
TRAIN_SPLIT: float = 0.80    # 80% train / 20% validation split


# ═══════════════════════════════════════════════════════════════════════════
#  ANOMALY SCORING & THRESHOLDS
# ═══════════════════════════════════════════════════════════════════════════

# Dynamic statistical threshold: Threshold = μ + (SIGMA_K * σ) of validation errors
THRESHOLD_SIGMA: float = 3.0

# Secondary nonparametric threshold (percentile of healthy validation reconstruction error)
PERCENTILE_THRESHOLD: float = 99.0

# Exponentially Weighted Moving Average (EWMA) alpha for streaming score smoothing
EWMA_ALPHA: float = 0.25

# Urgency categories based on score = reconstruction_error / threshold
ALERT_LEVELS: Dict[str, float] = {
    "NORMAL":   0.0,   # Score < 1.0
    "WATCH":    1.0,   # 1.0 <= Score < 1.5
    "CAUTION":  1.5,   # 1.5 <= Score < 2.5
    "WARNING":  2.5,   # 2.5 <= Score < 4.0
    "CRITICAL": 4.0,   # Score >= 4.0
}


# ═══════════════════════════════════════════════════════════════════════════
#  FILE ARTIFACT PATHS
# ═══════════════════════════════════════════════════════════════════════════

CHECKPOINT_PATH: Path = ROOT_DIR / "best_anomaly_detector.pt"
SCALER_PATH: Path     = ROOT_DIR / "anomaly_scaler.pkl"
THRESHOLD_PATH: Path  = ROOT_DIR / "anomaly_threshold.json"


# ═══════════════════════════════════════════════════════════════════════════
#  DEVICE CONFIGURATION (Apple Silicon MPS / CUDA / CPU)
# ═══════════════════════════════════════════════════════════════════════════

def get_device() -> torch.device:
    """Select the fastest available compute device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE: torch.device = get_device()
