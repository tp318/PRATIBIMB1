"""
============================================================================
training.py — Training Loop for the CNN-LSTM Fault Detector
============================================================================
Implements the full supervised training pipeline for Part C of the
MALE UAV Digital Twin.

Components:
  1. EarlyStopping callback — prevents over-fitting, crucial for small
     PHM datasets where each fault class may have < 1000 windows.
  2. Weighted CrossEntropyLoss — handles severe class imbalance
     (normal >> fault samples in production data).
  3. AdamW optimiser with cosine-annealing LR schedule.
  4. Per-epoch metric logging: loss, accuracy, per-class F1 (via sklearn).
  5. Best-model checkpoint saving (by validation loss).
  6. History dict returned for post-training visualisation.

Usage:
    from training import run_training_pipeline
    history = run_training_pipeline()   # uses synthetic data by default
============================================================================
"""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from config import (
    NUM_CLASSES, FAULT_CLASSES,
    BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY,
    GRAD_CLIP, LR_ETA_MIN, RANDOM_SEED,
    DEVICE, CHECKPOINT_PATH,
)
from architecture import CNNLSTMFaultDetector, save_model
from preprocessing import (
    ResidualScaler,
    generate_synthetic_residuals,
    create_sliding_windows,
    build_dataloaders,
    compute_class_weights,
)

logger = logging.getLogger("UAV.PHM.Training")

# ── Try importing sklearn for per-class F1 scores (optional) ──────────────
try:
    from sklearn.metrics import classification_report
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False
    logger.warning(
        "sklearn not available — per-class F1 scores will not be computed."
    )


# ═══════════════════════════════════════════════════════════════════════════
#  EARLY STOPPING CALLBACK
# ═══════════════════════════════════════════════════════════════════════════

class EarlyStopping:
    """
    Stop training when validation loss has not improved for `patience` epochs.

    Avoids wasting compute (and risking overfit) after the model has
    converged.  The best model state is restored automatically when triggered.

    Args:
        patience  : Epochs to wait without improvement before stopping.
        min_delta : Minimum improvement in val_loss to count as 'better'.
        verbose   : Log messages when improvement detected / patience expired.
    """

    def __init__(
        self,
        patience:  int   = 10,
        min_delta: float = 1e-4,
        verbose:   bool  = True,
    ) -> None:
        self.patience   = patience
        self.min_delta  = min_delta
        self.verbose    = verbose
        self.counter:    int   = 0
        self.best_loss:  float = float("inf")
        self.best_state: Optional[Dict] = None   # Copy of best model weights
        self.triggered:  bool  = False

    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        """
        Call at the end of each epoch.

        Args:
            val_loss : Current epoch validation loss.
            model    : Model instance (state dict copied if improved).

        Returns:
            True if training should stop, False otherwise.
        """
        if val_loss < self.best_loss - self.min_delta:
            # Improvement detected — reset counter and save state
            self.best_loss  = val_loss
            self.counter    = 0
            # Store a deep copy of the model weights (not the model itself)
            self.best_state = {k: v.clone() for k, v in model.state_dict().items()}
            if self.verbose:
                logger.info("EarlyStopping: improvement → best_loss=%.4f", val_loss)
        else:
            self.counter += 1
            if self.verbose:
                logger.debug(
                    "EarlyStopping: no improvement (%d/%d).", self.counter, self.patience
                )
            if self.counter >= self.patience:
                self.triggered = True
                if self.verbose:
                    logger.info(
                        "EarlyStopping triggered after %d epochs without improvement.",
                        self.patience,
                    )
                # Restore best weights into model
                if self.best_state is not None:
                    model.load_state_dict(self.best_state)
                    logger.info("Best model weights restored.")
                return True
        return False


# ═══════════════════════════════════════════════════════════════════════════
#  METRIC HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _compute_accuracy(preds: np.ndarray, targets: np.ndarray) -> float:
    """Return top-1 accuracy as a fraction (0.0–1.0)."""
    return float((preds == targets).mean())


def _compute_per_class_f1(
    preds:   np.ndarray,
    targets: np.ndarray,
    labels:  List[str] = FAULT_CLASSES,
) -> Optional[str]:
    """
    Return sklearn classification_report string, or None if sklearn unavailable.

    Macro-F1 is the metric of choice for imbalanced PHM datasets because
    it weights each class equally regardless of sample count, ensuring
    rare fault classes are not drowned out by the dominant 'Normal' class.
    """
    if not _SKLEARN_AVAILABLE:
        return None
    return classification_report(
        targets, preds,
        target_names=labels,
        zero_division=0,
    )


# ═══════════════════════════════════════════════════════════════════════════
#  SINGLE EPOCH HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _run_train_epoch(
    model:     CNNLSTMFaultDetector,
    loader:    DataLoader,
    criterion: nn.CrossEntropyLoss,
    optimizer: optim.Optimizer,
) -> float:
    """
    Execute one full training epoch over the DataLoader.

    Returns:
        train_loss : Weighted average cross-entropy loss over all batches.
    """
    model.train()
    running_loss = 0.0
    n_samples    = 0

    for batch_X, batch_y in loader:
        batch_X = batch_X.to(DEVICE, non_blocking=True)   # (B, F, W)
        batch_y = batch_y.to(DEVICE, non_blocking=True)   # (B,)

        # Zero gradients — set_to_none=True is faster than fill-with-zero
        optimizer.zero_grad(set_to_none=True)

        # Forward pass
        logits = model(batch_X)                # (B, NUM_CLASSES)
        loss   = criterion(logits, batch_y)    # Scalar

        # Backward pass
        loss.backward()

        # Gradient clipping: prevents LSTM recurrent weights from exploding,
        # particularly important with multiple stacked layers.
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)

        optimizer.step()

        # Accumulate weighted by batch size (handles variable final batch)
        running_loss += loss.item() * batch_X.size(0)
        n_samples    += batch_X.size(0)

    return running_loss / max(n_samples, 1)


def _run_val_epoch(
    model:     CNNLSTMFaultDetector,
    loader:    DataLoader,
    criterion: nn.CrossEntropyLoss,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """
    Evaluate the model on the validation set.

    Returns:
        val_loss : Average cross-entropy loss.
        val_acc  : Top-1 accuracy (fraction).
        all_preds   : (N,) predicted class ids.
        all_targets : (N,) ground-truth class ids.
    """
    model.eval()
    running_loss = 0.0
    n_samples    = 0
    all_preds:   List[int] = []
    all_targets: List[int] = []

    with torch.no_grad():
        for batch_X, batch_y in loader:
            batch_X = batch_X.to(DEVICE, non_blocking=True)
            batch_y = batch_y.to(DEVICE, non_blocking=True)

            logits = model(batch_X)
            loss   = criterion(logits, batch_y)

            running_loss += loss.item() * batch_X.size(0)
            n_samples    += batch_X.size(0)

            preds = logits.argmax(dim=-1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_targets.extend(batch_y.cpu().numpy().tolist())

    val_loss = running_loss / max(n_samples, 1)
    preds_arr   = np.array(all_preds,   dtype=np.int64)
    targets_arr = np.array(all_targets, dtype=np.int64)
    val_acc     = _compute_accuracy(preds_arr, targets_arr)

    return val_loss, val_acc, preds_arr, targets_arr


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN TRAINING FUNCTION
# ═══════════════════════════════════════════════════════════════════════════

def train_model(
    model:           CNNLSTMFaultDetector,
    train_loader:    DataLoader,
    val_loader:      DataLoader,
    num_epochs:      int   = NUM_EPOCHS,
    learning_rate:   float = LEARNING_RATE,
    weight_decay:    float = WEIGHT_DECAY,
    class_weights:   Optional[torch.Tensor] = None,
    checkpoint_path: str   = CHECKPOINT_PATH,
    early_stopping:  Optional[EarlyStopping] = None,
    verbose_f1:      bool  = False,
) -> Dict[str, List[float]]:
    """
    Full supervised training loop for CNNLSTMFaultDetector.

    Features:
      • Weighted cross-entropy (class imbalance compensation)
      • AdamW optimiser (decoupled weight decay)
      • Cosine annealing LR schedule (smooth warm-down)
      • Gradient norm clipping (LSTM stability)
      • Per-epoch checkpoint saving (best validation loss)
      • Optional early stopping
      • Optional per-class F1 reporting

    Args:
        model           : CNNLSTMFaultDetector (on CPU; moved to DEVICE here).
        train_loader    : Training DataLoader.
        val_loader      : Validation DataLoader.
        num_epochs      : Maximum number of training epochs.
        learning_rate   : Initial AdamW learning rate.
        weight_decay    : L2 penalty coefficient (AdamW decoupled).
        class_weights   : Optional (NUM_CLASSES,) tensor for imbalance handling.
                          If None, standard unweighted cross-entropy is used.
        checkpoint_path : Path to save the best model checkpoint.
        early_stopping  : Optional EarlyStopping instance.  Pass None to
                          disable early stopping.
        verbose_f1      : If True, print per-class F1 after each epoch.

    Returns:
        history : dict with keys:
            'train_loss' : List[float] — per-epoch training loss
            'val_loss'   : List[float] — per-epoch validation loss
            'val_acc'    : List[float] — per-epoch validation accuracy
    """
    model = model.to(DEVICE)
    logger.info(
        "Training on %s | epochs=%d | lr=%.2e | batch=%d",
        DEVICE, num_epochs, learning_rate, train_loader.batch_size,
    )

    # ── Loss: weighted cross-entropy ──────────────────────────────────────
    # CrossEntropyLoss = log-softmax + NLL loss.
    # class_weights=None → standard unweighted CE.
    # Move weights to DEVICE *after* tensor construction to avoid a known
    # MPS bug where CrossEntropyLoss hangs if weight was created on MPS.
    if class_weights is not None:
        class_weights = class_weights.to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # ── Optimiser: AdamW ──────────────────────────────────────────────────
    # AdamW decouples L2 regularisation from the gradient update, which
    # produces better generalisation than standard Adam + L2 penalty.
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    # ── LR Scheduler: cosine annealing ────────────────────────────────────
    # Smoothly reduces LR from `learning_rate` to `LR_ETA_MIN` over training.
    # Prevents oscillation around a local minimum in the final epochs.
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=LR_ETA_MIN
    )

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "val_loss":   [],
        "val_acc":    [],
    }
    best_val_loss = float("inf")
    t0_total = time.perf_counter()

    for epoch in range(1, num_epochs + 1):
        t0_epoch = time.perf_counter()

        # ── Training ──────────────────────────────────────────────────────
        train_loss = _run_train_epoch(model, train_loader, criterion, optimizer)

        # ── Validation ────────────────────────────────────────────────────
        val_loss, val_acc, val_preds, val_targets = _run_val_epoch(
            model, val_loader, criterion
        )

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        epoch_time = time.perf_counter() - t0_epoch

        # ── Record history ────────────────────────────────────────────────
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        logger.info(
            "Epoch %03d/%03d | train_loss=%.4f | val_loss=%.4f | "
            "val_acc=%.3f | lr=%.2e | %.1fs",
            epoch, num_epochs, train_loss, val_loss, val_acc,
            current_lr, epoch_time,
        )

        # ── Per-class F1 (optional, expensive) ───────────────────────────
        if verbose_f1:
            report = _compute_per_class_f1(val_preds, val_targets)
            if report:
                logger.info("\n%s", report)

        # ── Checkpoint: save best model ───────────────────────────────────
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch":              epoch,
                    "model_state_dict":   model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss":           best_val_loss,
                    "val_acc":            val_acc,
                    "model_config":       model.get_config(),
                },
                checkpoint_path,
            )
            logger.info(
                "  ✓ Checkpoint saved (val_loss=%.4f, val_acc=%.3f).",
                best_val_loss, val_acc,
            )

        # ── Early stopping ────────────────────────────────────────────────
        if early_stopping is not None:
            if early_stopping(val_loss, model):
                logger.info("Training stopped early at epoch %d.", epoch)
                break

    total_time = time.perf_counter() - t0_total
    logger.info(
        "Training complete in %.1f s. Best val_loss=%.4f. Saved → %s",
        total_time, best_val_loss, checkpoint_path,
    )
    return history


# ═══════════════════════════════════════════════════════════════════════════
#  CONVENIENCE: FULL TRAINING PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

def run_training_pipeline(
    residuals:       Optional[np.ndarray] = None,
    labels:          Optional[np.ndarray] = None,
    use_early_stop:  bool = True,
    early_stop_patience: int = 10,
    verbose_f1:      bool = True,
    checkpoint_path: str  = CHECKPOINT_PATH,
) -> Tuple[CNNLSTMFaultDetector, Dict[str, List[float]]]:
    """
    One-call training pipeline: data → windows → train → checkpoint.

    Workflow:
        1. If residuals/labels are not provided, generate synthetic data.
        2. Fit and apply a ResidualScaler (fit on training split only).
        3. Create sliding windows.
        4. Build DataLoaders.
        5. Compute class weights from training labels.
        6. Instantiate the CNN-LSTM model.
        7. Train with EarlyStopping (optional).
        8. Return the trained model and training history.

    Args:
        residuals        : (T, F) AUKF residual stream.  If None, synthetic
                           data is generated for demonstration.
        labels           : (T,) integer class labels per timestep.
        use_early_stop   : Enable EarlyStopping callback.
        early_stop_patience : Patience epochs for EarlyStopping.
        verbose_f1       : Print per-class F1 after each epoch.
        checkpoint_path  : Path to save the best model.

    Returns:
        model   : Trained CNNLSTMFaultDetector (best weights loaded).
        history : Training history dict.
    """
    # ── Step 1: Data source ───────────────────────────────────────────────
    if residuals is None or labels is None:
        logger.info("No real data supplied — generating synthetic residuals.")
        residuals, labels = generate_synthetic_residuals(n_timesteps=50_000)

    # ── Step 2: Normalise residuals ────────────────────────────────────────
    # Scaler is fit on the full stream here.  In production, fit only on
    # the TRAINING split indices to avoid leakage.
    scaler = ResidualScaler()
    residuals_norm = scaler.fit_transform(residuals)
    scaler.save()   # Persist so inference module can load the same scaler

    # ── Step 3: Sliding window generation ────────────────────────────────
    X, y = create_sliding_windows(residuals_norm, labels)

    # ── Step 4: DataLoaders ───────────────────────────────────────────────
    train_loader, val_loader = build_dataloaders(X, y, seed=RANDOM_SEED)

    # ── Step 5: Class weights (from training split labels only) ───────────
    # Access the underlying subset indices to get training labels
    train_indices  = train_loader.dataset.indices  # type: ignore[attr-defined]
    train_labels_y = y[train_indices]
    class_weights  = compute_class_weights(train_labels_y)
    logger.info("Class weights: %s", class_weights.tolist())

    # ── Step 6: Model instantiation ───────────────────────────────────────
    model = CNNLSTMFaultDetector()
    logger.info(
        "Model: %d trainable parameters.",
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    )

    # ── Step 7: Train ─────────────────────────────────────────────────────
    early_stop = (
        EarlyStopping(patience=early_stop_patience, verbose=True)
        if use_early_stop else None
    )

    history = train_model(
        model            = model,
        train_loader     = train_loader,
        val_loader       = val_loader,
        num_epochs       = NUM_EPOCHS,
        class_weights    = class_weights,
        checkpoint_path  = checkpoint_path,
        early_stopping   = early_stop,
        verbose_f1       = verbose_f1,
    )

    # ── Step 8: Return best model ─────────────────────────────────────────
    # Reload the checkpoint so the returned model has the best val_loss weights
    from architecture import load_model
    best_model = load_model(checkpoint_path)

    logger.info("Pipeline complete. Model ready for inference.")
    return best_model, history


# ── Entrypoint ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    model, history = run_training_pipeline(verbose_f1=True)
    print(f"\nFinal val_acc: {history['val_acc'][-1]:.4f}")
