"""
============================================================================
preprocessing.py  —  Data Preprocessing & Sliding Window Pipeline
============================================================================
Preprocesses continuous multi-channel AUKF/MVEM residual vectors for the
unsupervised LSTM Autoencoder:
  1. ResidualScaler: Z-score normalization fitted strictly on healthy data
  2. create_anomaly_sliding_windows(): Slices (T, F) time-series into (N, F, W)
  3. AnomalyDataset: PyTorch Dataset yielding (X, X) reconstruction pairs
  4. build_anomaly_dataloaders(): Train/val DataLoader factory
  5. generate_synthetic_telemetry(): Realistic AUKF residual generator
============================================================================
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from config import (
    NUM_SENSORS,
    SAMPLE_RATE_HZ,
    SCALER_PATH,
    STRIDE,
    TRAIN_SPLIT,
    WINDOW_SIZE,
)


# ═══════════════════════════════════════════════════════════════════════════
#  1. RESIDUAL SCALER
# ═══════════════════════════════════════════════════════════════════════════

class ResidualScaler:
    """
    Standard Z-score scaler fitted strictly on healthy engine residuals.

    Preserves zero-mean baseline of the AUKF state estimator while scaling
    heterogeneous sensor units (°C, bar, RPM, g) to comparable variance.
    """

    def __init__(self) -> None:
        self.mean_: Optional[np.ndarray] = None
        self.scale_: Optional[np.ndarray] = None
        self.is_fitted: bool = False

    def fit(self, X: np.ndarray) -> "ResidualScaler":
        """
        Compute mean and standard deviation per channel on healthy residuals.

        Args:
            X: Array of shape (T, F) where F = NUM_SENSORS.
        """
        if X.ndim != 2 or X.shape[1] != NUM_SENSORS:
            raise ValueError(f"Expected input shape (T, {NUM_SENSORS}), got {X.shape}")

        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0)
        # Avoid division by zero for inactive/flat channels
        self.scale_ = np.where(self.scale_ < 1e-8, 1.0, self.scale_)
        self.is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply Z-score standardization: (X - mean) / scale."""
        if not self.is_fitted:
            raise RuntimeError("ResidualScaler must be fitted before calling transform().")
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit scaler and transform in a single pass."""
        return self.fit(X).transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Undo Z-score normalization: X * scale + mean."""
        if not self.is_fitted:
            raise RuntimeError("ResidualScaler must be fitted before inverse_transform().")
        return (X * self.scale_) + self.mean_

    def save(self, filepath: Path = SCALER_PATH) -> None:
        """Serialize scaler state to disk."""
        with open(filepath, "wb") as f:
            pickle.dump({"mean_": self.mean_, "scale_": self.scale_}, f)

    @classmethod
    def load(cls, filepath: Path = SCALER_PATH) -> "ResidualScaler":
        """Deserialize scaler state from disk."""
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        scaler = cls()
        scaler.mean_ = data["mean_"]
        scaler.scale_ = data["scale_"]
        scaler.is_fitted = True
        return scaler


# ═══════════════════════════════════════════════════════════════════════════
#  2. SLIDING WINDOW GENERATION
# ═══════════════════════════════════════════════════════════════════════════

def create_anomaly_sliding_windows(
    residuals: np.ndarray,
    labels: Optional[np.ndarray] = None,
    window_size: int = WINDOW_SIZE,
    stride: int = STRIDE,
    healthy_only: bool = False,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Convert continuous residual stream (T, F) into sliding windows (N, F, W).

    Args:
        residuals: (T, F) normalized residual array.
        labels: Optional (T,) ground-truth class labels (0 = Healthy).
        window_size: Timesteps per window (default: 64).
        stride: Timestep step size between consecutive windows (default: 16).
        healthy_only: If True and labels are provided, only extract windows
                      where all timesteps belong to class 0 (Normal).

    Returns:
        X: (N, F, W) array of windows in channels-first format.
        y: (N,) array of window labels (mode label within window), or None.
    """
    T, F = residuals.shape
    if F != NUM_SENSORS:
        raise ValueError(f"Expected {NUM_SENSORS} sensor channels, got {F}")
    if T < window_size:
        raise ValueError(f"Time-series length ({T}) must be >= window_size ({window_size})")

    # Efficient strided view
    num_windows = (T - window_size) // stride + 1
    windows = []
    window_labels = []

    for i in range(num_windows):
        start = i * stride
        end = start + window_size
        win = residuals[start:end, :]  # (W, F)

        if labels is not None:
            win_lbl = labels[start:end]
            # If healthy_only, require window to be purely healthy
            if healthy_only and not np.all(win_lbl == 0):
                continue
            # Majority vote label for the window
            majority_label = int(np.bincount(win_lbl).argmax())
            window_labels.append(majority_label)

        # Transpose to channels-first: (F, W)
        windows.append(win.T)

    X = np.stack(windows, axis=0).astype(np.float32)  # (N, F, W)
    y = np.array(window_labels, dtype=np.int64) if labels is not None else None
    return X, y


# ═══════════════════════════════════════════════════════════════════════════
#  3. PYTORCH DATASET
# ═══════════════════════════════════════════════════════════════════════════

class AnomalyDataset(Dataset):
    """
    PyTorch Dataset for self-supervised Autoencoder reconstruction.
    Yields pairs (X, X) where input and target are identical.
    """

    def __init__(self, windows: np.ndarray) -> None:
        self.windows = torch.tensor(windows, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.windows[idx]
        return x, x  # (F, W) -> reconstruction target is the input itself


# ═══════════════════════════════════════════════════════════════════════════
#  4. DATALOADER FACTORY
# ═══════════════════════════════════════════════════════════════════════════

def build_anomaly_dataloaders(
    windows: np.ndarray,
    batch_size: int = 64,
    train_split: float = TRAIN_SPLIT,
    shuffle: bool = True,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create Train and Validation DataLoaders for the Autoencoder.

    Args:
        windows: (N, F, W) healthy window array.
        batch_size: Mini-batch size.
        train_split: Proportion of data allocated to training (default: 0.8).
        shuffle: Whether to shuffle the training set.
        seed: Random seed for reproducibility.

    Returns:
        (train_loader, val_loader)
    """
    np.random.seed(seed)
    N = len(windows)
    indices = np.random.permutation(N)
    split_idx = int(N * train_split)

    train_idx = indices[:split_idx]
    val_idx = indices[split_idx:]

    train_ds = AnomalyDataset(windows[train_idx])
    val_ds = AnomalyDataset(windows[val_idx])

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    return train_loader, val_loader


# ═══════════════════════════════════════════════════════════════════════════
#  5. SYNTHETIC AUKF TELEMETRY GENERATOR (For verification & testing)
# ═══════════════════════════════════════════════════════════════════════════

def generate_synthetic_telemetry(
    num_timesteps: int = 20000,
    anomaly_fraction: float = 0.15,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate realistic multi-channel AUKF residual time-series.

    Simulates:
      - Healthy baseline: zero-mean Gaussian noise with calibrated physical variance.
      - Anomaly intervals: micro-drifts, thermal deviations, and vibration spikes.

    Returns:
        residuals: (T, NUM_SENSORS) array of simulated residuals.
        labels: (T,) binary labels (0 = Healthy, 1 = Anomaly).
    """
    rng = np.random.default_rng(seed)

    # Baseline standard deviations for healthy engine residuals
    base_sigmas = np.array([
        15.0,   # rpm_residual         [RPM]
        1.2,    # cht_residual         [°C]
        3.5,    # egt_residual         [°C]
        0.08,   # oil_pressure_residual [bar]
        0.8,    # oil_temp_residual    [°C]
        0.12,   # fuel_flow_residual   [L/h]
        0.04,   # vibration_residual   [g]
        0.15,   # batt_voltage_residual [V]
        0.3,    # inj_timing_residual  [°CA]
    ], dtype=np.float32)

    residuals = rng.normal(0.0, base_sigmas, size=(num_timesteps, NUM_SENSORS))
    labels = np.zeros(num_timesteps, dtype=np.int64)

    # Inject anomaly events
    num_anomalies = int((num_timesteps * anomaly_fraction) / 400)
    for _ in range(num_anomalies):
        start = rng.integers(1000, num_timesteps - 500)
        duration = rng.integers(150, 450)
        end = min(start + duration, num_timesteps)

        anomaly_type = rng.integers(0, 3)
        if anomaly_type == 0:
            # Overheating anomaly: CHT + EGT drift upward
            residuals[start:end, 1] += np.linspace(0, 15.0, end - start)
            residuals[start:end, 2] += np.linspace(0, 35.0, end - start)
        elif anomaly_type == 1:
            # Mechanical imbalance: Vibration spike + RPM jitter
            residuals[start:end, 6] += rng.normal(0.35, 0.1, size=end - start)
            residuals[start:end, 0] += rng.normal(60.0, 20.0, size=end - start)
        else:
            # Lubrication degradation: Oil pressure drop + Oil temp rise
            residuals[start:end, 3] -= np.linspace(0, 0.45, end - start)
            residuals[start:end, 4] += np.linspace(0, 8.0, end - start)

        labels[start:end] = 1

    return residuals.astype(np.float32), labels
