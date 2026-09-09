"""
============================================================================
preprocessing.py — Sliding Window Dataset Generator
============================================================================
Converts the continuous AUKF/MVEM residual stream into discrete, fixed-
length windows ready for the 1D-CNN+LSTM model.

Pipeline position:
  [AUKF residuals (T, F)] → [Normalise] → [Sliding windows (N, W, F)]
                          → [ResidualWindowDataset] → [DataLoader]

Key design decisions:
  1. StandardScaler fitted ONLY on training windows (no leakage).
  2. Window label = MODE of timestep labels inside the window (robust to
     brief mis-annotations at fault onset/offset boundaries).
  3. Dataset stores windows in (N, W, F) and permutes to (N, F, W) inside
     __getitem__ so memory layout is compact but Conv1d gets the right shape.
============================================================================
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split

from config import (
    NUM_SENSORS, NUM_CLASSES, SENSOR_CHANNELS,
    WINDOW_SIZE, STRIDE,
    BATCH_SIZE, VAL_SPLIT, RANDOM_SEED,
    DEVICE, SCALER_PATH,
)

logger = logging.getLogger("UAV.PHM.Preprocessing")


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 1 — MANUAL STANDARD SCALER (no sklearn dependency at edge)
# ═══════════════════════════════════════════════════════════════════════════

class ResidualScaler:
    """
    Lightweight per-channel StandardScaler for residual vectors.

    Why not sklearn?  On edge GCS deployments sklearn may not be available.
    This class is serialisable via pickle for persistence between flights.

    Fit once on training data, then transform both train + inference streams.
    The scaler operates on the raw residual stream (T, F) BEFORE windowing,
    which is the correct order.

    Attributes:
        mean_ : (F,) per-channel mean fitted on training data
        std_  : (F,) per-channel std  fitted on training data
        fitted: bool flag
    """

    def __init__(self) -> None:
        self.mean_: Optional[np.ndarray] = None
        self.std_:  Optional[np.ndarray] = None
        self.fitted: bool = False

    def fit(self, residuals: np.ndarray) -> "ResidualScaler":
        """
        Compute per-channel mean and std from a (T, F) residual array.

        Args:
            residuals : np.ndarray, shape (T, F) — raw AUKF residuals
        Returns:
            self (for chaining)
        """
        if residuals.ndim != 2:
            raise ValueError(f"Expected 2-D array, got shape {residuals.shape}.")
        self.mean_ = residuals.mean(axis=0)          # (F,)
        self.std_  = residuals.std(axis=0) + 1e-8    # (F,) — epsilon avoids /0
        self.fitted = True
        logger.info(
            "Scaler fitted on %d timesteps, %d channels.", *residuals.shape
        )
        return self

    def transform(self, residuals: np.ndarray) -> np.ndarray:
        """
        Z-score normalise residuals using fitted statistics.

        Args:
            residuals : (T, F) raw residuals
        Returns:
            (T, F) normalised residuals, float32
        """
        if not self.fitted:
            raise RuntimeError("Scaler must be fitted before calling transform().")
        return ((residuals - self.mean_) / self.std_).astype(np.float32)

    def fit_transform(self, residuals: np.ndarray) -> np.ndarray:
        """Fit and transform in one call (use on training data only)."""
        return self.fit(residuals).transform(residuals)

    def save(self, path: str = SCALER_PATH) -> None:
        """Persist scaler to disk for use at inference time."""
        with open(path, "wb") as f:
            pickle.dump({"mean": self.mean_, "std": self.std_}, f)
        logger.info("Scaler saved to %s.", path)

    @classmethod
    def load(cls, path: str = SCALER_PATH) -> "ResidualScaler":
        """Load a previously saved scaler from disk."""
        with open(path, "rb") as f:
            state = pickle.load(f)
        scaler = cls()
        scaler.mean_ = state["mean"]
        scaler.std_  = state["std"]
        scaler.fitted = True
        logger.info("Scaler loaded from %s.", path)
        return scaler


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 2 — SLIDING WINDOW GENERATOR
# ═══════════════════════════════════════════════════════════════════════════

def create_sliding_windows(
    residuals: np.ndarray,
    labels: np.ndarray,
    window_size: int = WINDOW_SIZE,
    stride: int = STRIDE,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert a continuous residual time-series into fixed-length,
    overlapping windows suitable for sequence deep learning.

    The AUKF/MVEM pipeline produces a residual stream shaped (T, F):
        T = total timesteps in the flight recording
        F = number of sensor residual channels (NUM_SENSORS = 9)

    We slice this into N windows of shape (window_size, F) with a
    configurable stride, producing:
        X : (N, window_size, F)   — windowed residuals
        y : (N,)                  — per-window fault class label

    Window labelling strategy (MODE):
        The label for a window is the most-frequent label among all
        timesteps inside that window.  This is more robust than taking
        the last or first label because:
          a) Fault boundaries are fuzzy in practice (gradual onset).
          b) Brief annotation artefacts near boundaries don't corrupt the
             entire window label.

    Args:
        residuals   : np.ndarray (T, F) — pre-normalised residual stream.
        labels      : np.ndarray (T,)  — integer class at every timestep.
        window_size : int — timesteps per window (default 64 @ 50 Hz = 1.28 s).
        stride      : int — shift between window starts (default 16 = 75% overlap).

    Returns:
        X : np.ndarray (N, window_size, F), dtype float32
        y : np.ndarray (N,),               dtype int64
    """
    # ── Input validation ──────────────────────────────────────────────────
    if residuals.ndim != 2:
        raise ValueError(
            f"residuals must be 2-D (T, F), got shape {residuals.shape}."
        )
    T, F = residuals.shape
    if F != NUM_SENSORS:
        raise ValueError(
            f"Expected {NUM_SENSORS} channels (see SENSOR_CHANNELS in config.py), "
            f"got {F}.  Ensure your AUKF output matches the config."
        )
    if labels.ndim != 1 or len(labels) != T:
        raise ValueError(
            f"labels must be 1-D with length T={T}, got shape {labels.shape}."
        )
    if window_size > T:
        raise ValueError(
            f"window_size={window_size} exceeds series length T={T}."
        )

    windows_X: list = []
    windows_y: list = []

    # ── Core windowing loop ───────────────────────────────────────────────
    for start in range(0, T - window_size + 1, stride):
        end = start + window_size

        # Residual slice: (window_size, F)
        window_data = residuals[start:end, :]

        # MODE label: most frequent class within the window
        window_labels = labels[start:end]                           # (window_size,)
        counts = np.bincount(window_labels, minlength=NUM_CLASSES)  # (NUM_CLASSES,)
        majority_label = int(np.argmax(counts))

        windows_X.append(window_data)
        windows_y.append(majority_label)

    X = np.array(windows_X, dtype=np.float32)   # (N, window_size, F)
    y = np.array(windows_y, dtype=np.int64)     # (N,)

    # ── Logging summary ───────────────────────────────────────────────────
    class_dist = dict(zip(*np.unique(y, return_counts=True)))
    logger.info(
        "Windowing: %d timesteps → %d windows (size=%d, stride=%d).",
        T, len(X), window_size, stride,
    )
    logger.info("Class distribution: %s", class_dist)
    return X, y


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 3 — PYTORCH DATASET
# ═══════════════════════════════════════════════════════════════════════════

class ResidualWindowDataset(Dataset):
    """
    PyTorch Dataset wrapping pre-computed sliding window arrays.

    Memory layout & shape convention:
        Stored internally : (N, window_size, F)   — natural time-series shape
        Returned to model : (F, window_size)       — Conv1d convention

    The permute happens inside __getitem__ so:
      • Slicing X[idx] copies only one window (memory efficient).
      • The transposition cost is negligible per sample.

    Args:
        X : np.ndarray (N, window_size, F) — residual windows (float32)
        y : np.ndarray (N,)               — fault class labels (int64)
    """

    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        # Convert once at construction to avoid repeated numpy→torch overhead
        self.X: torch.Tensor = torch.from_numpy(X.astype(np.float32))
        self.y: torch.Tensor = torch.from_numpy(y.astype(np.int64))

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            x_cnn : (F, window_size) — input for Conv1d (channels-first)
            label : scalar int64 tensor
        """
        # Permute: (window_size, F) → (F, window_size) for Conv1d
        x_cnn = self.X[idx].permute(1, 0)   # (F, window_size)
        return x_cnn, self.y[idx]


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 4 — DATALOADER FACTORY
# ═══════════════════════════════════════════════════════════════════════════

def build_dataloaders(
    X: np.ndarray,
    y: np.ndarray,
    val_split: float = VAL_SPLIT,
    batch_size: int = BATCH_SIZE,
    seed: int = RANDOM_SEED,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    """
    Build train and validation DataLoaders from windowed residual arrays.

    Split strategy: random_split with a fixed seed.  For production with
    labelled flight recording sessions, consider a session-aware split
    (split by flight ID, not by window index) to prevent data leakage
    across overlapping windows.

    Args:
        X          : (N, window_size, F) residual windows
        y          : (N,) integer class labels
        val_split  : fraction reserved for validation (default 15%)
        batch_size : mini-batch size
        seed       : RNG seed for reproducibility
        num_workers: DataLoader worker processes (0 = main process only)

    Returns:
        train_loader : DataLoader (shuffled)
        val_loader   : DataLoader (sequential)
    """
    dataset = ResidualWindowDataset(X, y)
    n_total = len(dataset)
    n_val   = int(n_total * val_split)
    n_train = n_total - n_val

    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=generator)

    # pin_memory moves batches to pinned (page-locked) RAM for faster GPU transfer
    pin = (DEVICE.type == "cuda")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin,
        drop_last=True,    # Avoids single-sample batches that break BatchNorm
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
    )

    logger.info(
        "DataLoaders: train=%d windows (%d batches), val=%d windows.",
        n_train, len(train_loader), n_val,
    )
    return train_loader, val_loader


def compute_class_weights(
    y: np.ndarray,
    num_classes: int = NUM_CLASSES,
) -> torch.Tensor:
    """
    Inverse-frequency class weights to counteract class imbalance.

    In real UAV engine datasets 'Normal' windows vastly outnumber fault
    windows (ratio often 50:1 to 200:1).  Without reweighting the model
    learns to always predict 'Normal' and achieves high accuracy but zero
    diagnostic utility.

    Formula:
        weight_c = 1 / count_c,  then normalised so weights.sum() == num_classes

    This keeps the total loss scale comparable to the unweighted case.

    Args:
        y          : (N,) training labels (train split only — NOT full dataset)
        num_classes: total number of classes

    Returns:
        weights : torch.Tensor (num_classes,) on DEVICE
    """
    counts  = np.bincount(y, minlength=num_classes).astype(np.float32)
    counts  = np.clip(counts, 1.0, None)          # Avoid division by zero
    weights = 1.0 / counts
    weights = weights / weights.sum() * num_classes   # Normalise
    # NOTE: keep on CPU — CrossEntropyLoss(weight=...) requires weight on the
    # same device as the logits, but MPS has a known bug when weight is created
    # directly on MPS.  Move to DEVICE inside train_model() after construction.
    return torch.tensor(weights, dtype=torch.float32)  # CPU tensor


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 5 — SYNTHETIC DATA GENERATOR (for testing / demo)
# ═══════════════════════════════════════════════════════════════════════════

def generate_synthetic_residuals(
    n_timesteps: int = 50_000,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate plausible synthetic AUKF residuals for pipeline validation.

    ► REPLACE THIS WITH REAL AUKF/MVEM OUTPUT IN PRODUCTION ◄

    Statistical model used:
        Normal operation : residuals ~ N(0, 0.08) — small, near-zero noise
        Fault episodes   : specific sensor channels receive a biased,
                           higher-variance signal characteristic of the fault

    Fault-to-sensor mapping (based on engine PHM domain knowledge):
        Misfire          → RPM(0), EGT(2), vibration(6)   — combustion miss
        Injector         → fuel_flow(5), inj_timing(8)    — delivery anomaly
        Cooling degr.    → CHT(1), EGT(2)                 — heat buildup
        Lubrication      → oil_pres(3), oil_temp(4)       — tribological
        Sensor drift     → RPM(0), oil_pres(3), batt(7)   — measurement bias
        Combustion inst. → EGT(2), vibration(6)           — pressure waves
        Overheating      → CHT(1), EGT(2), oil_temp(4)   — thermal runaway
        Vibration        → vibration(6)                   — structural resonance

    Args:
        n_timesteps : total length of the synthetic mission recording
        seed        : NumPy RNG seed

    Returns:
        residuals : (n_timesteps, NUM_SENSORS) float32
        labels    : (n_timesteps,) int64
    """
    rng = np.random.default_rng(seed)

    # Base healthy residuals: small, zero-mean Gaussian
    residuals = rng.normal(0.0, 0.08, size=(n_timesteps, NUM_SENSORS)).astype(np.float32)
    labels    = np.zeros(n_timesteps, dtype=np.int64)   # All healthy by default

    # Each tuple: (t_start, t_end, class_id, [sensor_indices], mean_shift, noise_std)
    fault_episodes = [
        (5_000,  5_500,  1, [0, 2, 6],    2.0,  0.35),  # Misfire
        (10_000, 10_800, 2, [5, 8],        1.8,  0.30),  # Injector
        (18_000, 19_000, 3, [1, 2],        1.5,  0.25),  # Cooling
        (25_000, 25_700, 4, [3, 4],        2.2,  0.40),  # Lubrication
        (30_000, 30_500, 5, [0, 3, 7],     1.2,  0.20),  # Sensor drift
        (35_000, 36_000, 6, [2, 6],        2.5,  0.50),  # Combustion instability
        (40_000, 41_000, 7, [1, 2, 4],     3.0,  0.45),  # Overheating
        (45_000, 46_000, 8, [6],           4.0,  0.60),  # Abnormal vibration
    ]

    for t0, t1, cls, sensors, mu, sigma in fault_episodes:
        length = t1 - t0
        labels[t0:t1] = cls
        for s_idx in sensors:
            # Biased, higher-variance anomaly signal
            residuals[t0:t1, s_idx] += rng.normal(mu, sigma, size=length).astype(np.float32)

    logger.info(
        "Synthetic dataset: %d timesteps, 8 fault episodes, %d sensors.",
        n_timesteps, NUM_SENSORS,
    )
    return residuals, labels
