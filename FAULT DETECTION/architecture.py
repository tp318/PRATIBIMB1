"""
============================================================================
architecture.py — 1D-CNN + LSTM Hybrid Fault Detector
============================================================================
Author  : Senior AI/ML Engineer — Aerospace PHM Division
Context : Receives residual windows produced by the AUKF/MVEM pipeline.
          Input shape  : (batch, F, W)  — channels-first after Dataset permute
          Output shape : (batch, C)     — raw logits for C fault classes

Architecture Overview:
  [Residual Window (B,F,W)]
      ↓
  [CNN Block-1: Conv1D → BN → GELU → Dropout]    ← cross-sensor correlations
      ↓
  [CNN Block-2: Conv1D → BN → GELU → Dropout]    ← richer fault signatures
      ↓
  [MaxPool1D(2)]                                  ← reduce temporal resolution
      ↓
  [Permute → (B, W//2, C2)]
      ↓
  [Stacked LSTM (2 layers)]                       ← temporal fault evolution
      ↓
  [Last hidden state h_n[-1]]
      ↓
  [FC → GELU → Dropout → FC]                     ← classifier head
      ↓
  [Raw logits (B, NUM_CLASSES)]

Why this architecture?
  • 1D-CNN extracts short-duration cross-sensor patterns (e.g., simultaneous
    EGT spike + RPM drop + vibration burst = misfire signature).
  • LSTM accumulates evidence over the compressed feature sequence so slowly
    evolving faults (cooling degradation, lubrication wear) that need tens of
    windows to become significant are captured.
  • Causal (unidirectional) LSTM enables online / real-time inference without
    needing future timesteps.
  • BatchNorm + GELU replaces the original BatchNorm + ReLU for smoother
    gradients in the presence of the small-residual input distributions.
============================================================================
"""

from __future__ import annotations

import logging
from typing import Dict

import torch
import torch.nn as nn

from config import (
    NUM_SENSORS, NUM_CLASSES, WINDOW_SIZE,
    MODEL_CONFIG, DEVICE,
)

logger = logging.getLogger("UAV.PHM.Architecture")


# ═══════════════════════════════════════════════════════════════════════════
#  CNN BUILDING BLOCK
# ═══════════════════════════════════════════════════════════════════════════

class Conv1DBlock(nn.Module):
    """
    Reusable 1D convolution block: Conv1d → BatchNorm → GELU → Dropout.

    Args:
        in_channels  : Number of input channels (sensor count or prior filters).
        out_channels : Number of output feature maps.
        kernel_size  : Temporal receptive field (number of timesteps per filter).
        dropout      : Spatial dropout fraction applied after activation.
    """

    def __init__(
        self,
        in_channels:  int,
        out_channels: int,
        kernel_size:  int,
        dropout:      float = 0.25,
    ) -> None:
        super().__init__()
        # 'same' padding: output length == input length (no temporal shrinkage)
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),            # Smoother gradient than ReLU; preferred for PHM
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (batch, in_channels, seq_len)
        Returns:
            (batch, out_channels, seq_len)
        """
        return self.block(x)


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN MODEL CLASS
# ═══════════════════════════════════════════════════════════════════════════

class CNNLSTMFaultDetector(nn.Module):
    """
    Hybrid 1D-CNN → LSTM fault detector for MALE UAV piston-engine PHM.

    This model is the core of Part C of the Digital Twin framework.

    Input shape  : (batch_size, num_sensors, window_size)
                   where window_size is the number of AUKF residual timesteps
                   and num_sensors = 9 (one channel per residual).

    Output shape : (batch_size, num_classes) — raw logits
                   Apply softmax for probabilities (done in inference module).

    Model is compatible with DeepSHAP because:
      • All operations are differentiable (no argmax inside forward).
      • Dropout layers are disabled in .eval() mode automatically.
      • BatchNorm uses running statistics in eval mode (no batch dependency).

    Constructor Args (all have sensible defaults from MODEL_CONFIG):
        num_sensors    : Input channels = number of AUKF residuals (9).
        window_size    : Temporal length of each window (64 @ 50 Hz = 1.28 s).
        num_classes    : Output classes = number of fault states (9).
        cnn_filters_1  : Feature maps in first CNN block.
        cnn_filters_2  : Feature maps in second CNN block.
        cnn_kernel_size: Temporal receptive field per filter.
        cnn_dropout    : Dropout after each CNN block.
        lstm_hidden    : LSTM hidden state dimension.
        lstm_layers    : Number of stacked LSTM layers.
        lstm_dropout   : Inter-layer LSTM dropout (active only if layers > 1).
        fc_hidden      : Bottleneck neurons before final classification layer.
        fc_dropout     : Dropout in classifier head.
    """

    def __init__(
        self,
        num_sensors:    int   = NUM_SENSORS,
        window_size:    int   = WINDOW_SIZE,
        num_classes:    int   = NUM_CLASSES,
        cnn_filters_1:  int   = MODEL_CONFIG["cnn_filters_1"],
        cnn_filters_2:  int   = MODEL_CONFIG["cnn_filters_2"],
        cnn_kernel_size:int   = MODEL_CONFIG["cnn_kernel_size"],
        cnn_dropout:    float = MODEL_CONFIG["cnn_dropout"],
        lstm_hidden:    int   = MODEL_CONFIG["lstm_hidden"],
        lstm_layers:    int   = MODEL_CONFIG["lstm_layers"],
        lstm_dropout:   float = MODEL_CONFIG["lstm_dropout"],
        fc_hidden:      int   = MODEL_CONFIG["fc_hidden"],
        fc_dropout:     float = MODEL_CONFIG["fc_dropout"],
    ) -> None:
        super().__init__()

        # Store for checkpoint serialisation and model reloading
        self.num_sensors  = num_sensors
        self.window_size  = window_size
        self.num_classes  = num_classes
        self.lstm_hidden  = lstm_hidden
        self.lstm_layers  = lstm_layers

        # ── Stage 1: CNN feature extraction ──────────────────────────────
        # Block-1: learns cross-sensor short-duration patterns
        #   in_channels = num_sensors (9 residuals as parallel channels)
        self.cnn_block1 = Conv1DBlock(
            in_channels=num_sensors,
            out_channels=cnn_filters_1,
            kernel_size=cnn_kernel_size,
            dropout=cnn_dropout,
        )

        # Block-2: composes simple patterns into richer fault signatures
        self.cnn_block2 = Conv1DBlock(
            in_channels=cnn_filters_1,
            out_channels=cnn_filters_2,
            kernel_size=cnn_kernel_size,
            dropout=cnn_dropout,
        )

        # Halve temporal resolution: reduces LSTM sequence length → faster
        # and forces the LSTM to focus on larger-scale temporal patterns.
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self._pooled_len = window_size // 2     # 64 // 2 = 32 timesteps

        # ── Stage 2: LSTM temporal modelling ─────────────────────────────
        # Input to LSTM: (batch, seq_len, input_size)
        #   After CNN+pool: (B, cnn_filters_2, W//2)
        #   After permute : (B, W//2, cnn_filters_2)   ← LSTM sees this
        #
        # Causal (unidirectional) LSTM for online/real-time compatibility.
        # bidirectional=True would require the full window to be available
        # before producing output — not suitable for streaming inference.
        self.lstm = nn.LSTM(
            input_size=cnn_filters_2,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=lstm_dropout if lstm_layers > 1 else 0.0,
            bidirectional=False,    # Causal: no look-ahead required
        )

        # ── Stage 3: Classifier head ──────────────────────────────────────
        # Bottleneck FC → GELU → Dropout → output logits
        # NOTE: No Softmax here.  CrossEntropyLoss includes log-softmax
        #       during training.  Softmax is applied at inference explicitly.
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden, fc_hidden),
            nn.GELU(),
            nn.Dropout(fc_dropout),
            nn.Linear(fc_hidden, num_classes),
        )

        # ── Weight initialisation ─────────────────────────────────────────
        self._init_weights()

        logger.info(
            "CNNLSTMFaultDetector initialised — params: {:,}".format(
                sum(p.numel() for p in self.parameters() if p.requires_grad)
            )
        )

    # ── Weight initialisation ─────────────────────────────────────────────
    def _init_weights(self) -> None:
        """
        Apply sensible initial weights:
          • LSTM input weights   → Xavier uniform (balances input variance)
          • LSTM recurrent weights → Orthogonal (prevents vanishing gradients)
          • LSTM biases          → Zero (including forget-gate bias = 0)
          • Linear weights       → Kaiming normal (for GELU non-linearity)
          • Conv weights         → Kaiming normal (default for Conv1d in PyTorch)
        """
        for name, param in self.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
        # Classifier linear layers
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    # ── Forward pass ──────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        End-to-end forward pass through CNN → LSTM → classifier.

        Args:
            x : torch.Tensor, shape (batch, num_sensors, window_size)
                Permuted residual window from ResidualWindowDataset.__getitem__.

        Returns:
            logits : torch.Tensor, shape (batch, num_classes)
                Raw scores.  Apply softmax for probabilities.
        """
        # ── CNN Stage ─────────────────────────────────────────────────────
        # (B, F, W) → (B, C1, W)
        x = self.cnn_block1(x)

        # (B, C1, W) → (B, C2, W)
        x = self.cnn_block2(x)

        # (B, C2, W) → (B, C2, W//2)   — MaxPool halves temporal length
        x = self.pool(x)

        # ── Reshape for LSTM ──────────────────────────────────────────────
        # Conv output (B, C2, W//2) → LSTM input (B, W//2, C2)
        x = x.permute(0, 2, 1)

        # ── LSTM Stage ────────────────────────────────────────────────────
        # lstm_out : (B, W//2, lstm_hidden) — all intermediate hidden states
        # h_n      : (num_layers, B, lstm_hidden)
        # c_n      : (num_layers, B, lstm_hidden) — cell states (unused)
        lstm_out, (h_n, _) = self.lstm(x)

        # Extract the last layer's final hidden state.
        # h_n[-1] = (B, lstm_hidden) — the complete sequence summary.
        last_hidden = h_n[-1]

        # ── Classifier ────────────────────────────────────────────────────
        logits = self.classifier(last_hidden)   # (B, num_classes)
        return logits

    # ── Convenience helpers ───────────────────────────────────────────────
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return softmax class probabilities for a batch of windows.

        Args:
            x : (B, F, W) tensor on any device

        Returns:
            probs : (B, num_classes) probabilities summing to 1.0 per sample
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
        return torch.softmax(logits, dim=-1)

    def get_config(self) -> Dict:
        """Serialisable config dict for checkpoint saving."""
        return {
            "num_sensors":    self.num_sensors,
            "window_size":    self.window_size,
            "num_classes":    self.num_classes,
            "lstm_hidden":    self.lstm_hidden,
            "lstm_layers":    self.lstm_layers,
        }


# ═══════════════════════════════════════════════════════════════════════════
#  MODEL PERSISTENCE UTILITIES
# ═══════════════════════════════════════════════════════════════════════════

def save_model(model: CNNLSTMFaultDetector, path: str) -> None:
    """
    Save model state dict + config to a single .pt file.

    Saving config alongside the weights allows reloading without
    reconstructing constructor arguments from memory.

    Args:
        model : Trained CNNLSTMFaultDetector.
        path  : File path for the checkpoint (e.g., 'best_fault_detector.pt').
    """
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config":     model.get_config(),
        },
        path,
    )
    logger.info("Model saved → %s", path)


def load_model(
    path: str,
    device: torch.device = DEVICE,
) -> CNNLSTMFaultDetector:
    """
    Load a CNNLSTMFaultDetector from a checkpoint file.

    Args:
        path   : Path to the .pt checkpoint.
        device : Target device for the loaded model.

    Returns:
        model  : CNNLSTMFaultDetector in eval() mode, on `device`.
    """
    checkpoint = torch.load(path, map_location=device)
    cfg = checkpoint.get("model_config", {})

    model = CNNLSTMFaultDetector(
        num_sensors=cfg.get("num_sensors", NUM_SENSORS),
        window_size=cfg.get("window_size", WINDOW_SIZE),
        num_classes=cfg.get("num_classes", NUM_CLASSES),
    )
    state_dict = checkpoint["model_state_dict"]
    # Remap keys if saved without .block sub-module
    adapted_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("cnn_block1.") and not k.startswith("cnn_block1.block."):
            adapted_state_dict[k.replace("cnn_block1.", "cnn_block1.block.")] = v
        elif k.startswith("cnn_block2.") and not k.startswith("cnn_block2.block."):
            adapted_state_dict[k.replace("cnn_block2.", "cnn_block2.block.")] = v
        else:
            adapted_state_dict[k] = v
    model.load_state_dict(adapted_state_dict)
    model = model.to(device).eval()
    logger.info("Model loaded ← %s", path)
    return model
