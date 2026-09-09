"""
============================================================================
training.py  —  Unsupervised Autoencoder Training & Threshold Calibration
============================================================================
Trains the LSTM Autoencoder exclusively on healthy engine residuals, minimizes
reconstruction error (MSE), and calculates operational anomaly thresholds.
============================================================================
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import (
    BATCH_SIZE,
    CHECKPOINT_PATH,
    DEVICE,
    LEARNING_RATE,
    NUM_EPOCHS,
    PATIENCE,
    PERCENTILE_THRESHOLD,
    SCALER_PATH,
    THRESHOLD_PATH,
    THRESHOLD_SIGMA,
    WEIGHT_DECAY,
    WINDOW_SIZE,
)
from architecture import (
    LSTMAutoencoder,
    count_parameters,
    save_model,
)
from preprocessing import (
    ResidualScaler,
    build_anomaly_dataloaders,
    create_anomaly_sliding_windows,
    generate_synthetic_telemetry,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("UAV.PHM.AnomalyTraining")


# ═══════════════════════════════════════════════════════════════════════════
#  EARLY STOPPING
# ═══════════════════════════════════════════════════════════════════════════

class EarlyStopping:
    """Monitors validation loss and terminates training when plateaued."""

    def __init__(self, patience: int = PATIENCE, min_delta: float = 1e-5) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0
        self.early_stop = False

    def step(self, val_loss: float) -> bool:
        """Returns True if this epoch improved over best_loss."""
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False


# ═══════════════════════════════════════════════════════════════════════════
#  TRAIN & EVAL EPOCHS
# ═══════════════════════════════════════════════════════════════════════════

def _run_train_epoch(
    model: LSTMAutoencoder,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Execute one training epoch over mini-batches."""
    model.train()
    total_loss = 0.0

    for x, target in loader:
        x = x.to(device)
        target = target.to(device)

        optimizer.zero_grad()
        x_hat = model(x)
        loss = criterion(x_hat, target)
        loss.backward()

        # Gradient clipping to stabilize recurrent layers
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()

        total_loss += loss.item() * x.size(0)

    return total_loss / len(loader.dataset)


def _run_val_epoch(
    model: LSTMAutoencoder,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Execute one validation epoch."""
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for x, target in loader:
            x = x.to(device)
            target = target.to(device)
            x_hat = model(x)
            loss = criterion(x_hat, target)
            total_loss += loss.item() * x.size(0)

    return total_loss / len(loader.dataset)


# ═══════════════════════════════════════════════════════════════════════════
#  DYNAMIC THRESHOLD CALIBRATION
# ═══════════════════════════════════════════════════════════════════════════

def calibrate_thresholds(
    model: LSTMAutoencoder,
    val_loader: DataLoader,
    device: torch.device = DEVICE,
    save_path: Path = THRESHOLD_PATH,
) -> Dict[str, float]:
    """
    Compute statistical and percentile anomaly thresholds on healthy validation data.

    Thresholds:
      1. Statistical (3-sigma): T = mean + (3.0 * std)
      2. Empirical (Percentile): T = 99th percentile of healthy errors
    """
    model.eval()
    errors: List[float] = []

    with torch.no_grad():
        for x, _ in val_loader:
            x = x.to(device)
            x_hat = model(x)
            total_mse, _ = model.compute_reconstruction_error(x, x_hat)
            errors.extend(total_mse.cpu().numpy().tolist())

    errors_arr = np.array(errors, dtype=np.float64)
    mu = float(np.mean(errors_arr))
    sigma = float(np.std(errors_arr))
    stat_threshold = float(mu + (THRESHOLD_SIGMA * sigma))
    perc_threshold = float(np.percentile(errors_arr, PERCENTILE_THRESHOLD))
    max_error = float(np.max(errors_arr))

    thresholds = {
        "mean_error": round(mu, 6),
        "std_error": round(sigma, 6),
        "statistical_threshold_3sigma": round(stat_threshold, 6),
        "percentile_threshold_p99": round(perc_threshold, 6),
        "max_healthy_error": round(max_error, 6),
        "calibration_samples": len(errors_arr),
    }

    with open(save_path, "w") as f:
        json.dump(thresholds, f, indent=2)

    logger.info(
        f"Calibrated Thresholds -> Mean: {mu:.5f}, Std: {sigma:.5f}, "
        f"3-Sigma: {stat_threshold:.5f}, P99: {perc_threshold:.5f}"
    )
    return thresholds


# ═══════════════════════════════════════════════════════════════════════════
#  TRAINING ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════

def train_anomaly_model(
    model: LSTMAutoencoder,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = NUM_EPOCHS,
    lr: float = LEARNING_RATE,
    weight_decay: float = WEIGHT_DECAY,
    device: torch.device = DEVICE,
    checkpoint_path: Path = CHECKPOINT_PATH,
) -> Tuple[LSTMAutoencoder, Dict[str, List[float]]]:
    """Train Autoencoder with early stopping and save the best checkpoint."""
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )
    criterion = nn.MSELoss()
    early_stopping = EarlyStopping(patience=PATIENCE)

    history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}
    logger.info(f"Starting Autoencoder training on {device.type.upper()} ({num_epochs} max epochs)...")

    start_time = time.time()
    best_epoch = 0

    for epoch in range(1, num_epochs + 1):
        train_loss = _run_train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = _run_val_epoch(model, val_loader, criterion, device)
        prev_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]
        if current_lr < prev_lr:
            logger.info(f"Epoch {epoch:02d}: Learning rate reduced from {prev_lr:.2e} to {current_lr:.2e}")

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        improved = early_stopping.step(val_loss)
        if improved:
            best_epoch = epoch
            save_model(model, checkpoint_path, metadata={"epoch": epoch, "val_loss": val_loss})
            tag = "★ BEST"
        else:
            tag = f"({early_stopping.counter}/{PATIENCE})"

        logger.info(
            f"Epoch [{epoch:02d}/{num_epochs:02d}] "
            f"Train MSE: {train_loss:.6f} | Val MSE: {val_loss:.6f} {tag}"
        )

        if early_stopping.early_stop:
            logger.info(f"Early stopping triggered at epoch {epoch}. Best was epoch {best_epoch}.")
            break

    elapsed = time.time() - start_time
    logger.info(f"Training completed in {elapsed:.1f}s. Best checkpoint saved to: {checkpoint_path}")

    # Load best model weights for threshold calibration
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    return model, history


# ═══════════════════════════════════════════════════════════════════════════
#  HIGH-LEVEL PIPELINE ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

def run_training_pipeline(
    residuals: Optional[np.ndarray] = None,
    labels: Optional[np.ndarray] = None,
    num_epochs: int = NUM_EPOCHS,
    batch_size: int = BATCH_SIZE,
    device: torch.device = DEVICE,
) -> Tuple[LSTMAutoencoder, Dict[str, float]]:
    """
    End-to-end training pipeline for the Anomaly Detection subsystem.
    """
    if residuals is None:
        logger.info("No input residuals provided — generating synthetic flight telemetry...")
        residuals, labels = generate_synthetic_telemetry()

    # Scale residuals
    scaler = ResidualScaler()
    # If labels provided, fit scaler on healthy data only
    if labels is not None:
        healthy_mask = (labels == 0)
        scaler.fit(residuals[healthy_mask])
    else:
        scaler.fit(residuals)
    scaler.save(SCALER_PATH)
    logger.info(f"Fitted ResidualScaler saved to: {SCALER_PATH}")

    norm_residuals = scaler.transform(residuals)

    # Extract healthy windows for training
    X_healthy, _ = create_anomaly_sliding_windows(
        norm_residuals,
        labels=labels,
        healthy_only=(labels is not None),
    )
    logger.info(f"Created {len(X_healthy)} healthy windows (shape: {X_healthy.shape}) for Autoencoder")

    # Build DataLoaders
    train_loader, val_loader = build_anomaly_dataloaders(
        X_healthy,
        batch_size=batch_size,
    )

    # Instantiate Autoencoder
    model = LSTMAutoencoder()
    logger.info(f"LSTM Autoencoder initialized with {count_parameters(model):,} parameters")

    # Train
    trained_model, _ = train_anomaly_model(
        model,
        train_loader,
        val_loader,
        num_epochs=num_epochs,
        device=device,
    )

    # Calibrate thresholds
    thresholds = calibrate_thresholds(trained_model, val_loader, device=device)

    return trained_model, thresholds


if __name__ == "__main__":
    run_training_pipeline()
