"""
============================================================================
architecture.py  —  LSTM Autoencoder for Temporal Anomaly Detection
============================================================================
Author  : Senior AI/ML Engineer — Aerospace PHM Division
Purpose : Unsupervised reconstruction model for multi-channel AUKF/MVEM
          residual sequences.
          
Model Topology:
  Input (B, F=9, W=64)
    │  Transpose to (B, W=64, F=9)
    ▼
  [Encoder LSTM-1] (F=9 → hidden=64)
    │
  [Encoder LSTM-2] (64 → latent=32)
    │  Bottleneck latent vector z: (B, 32)
    ▼
  [RepeatVector] (B, 32) → (B, W=64, 32)
    │
  [Decoder LSTM-1] (32 → 64)
    │
  [Decoder LSTM-2] (64 → 64)
    │
  [Linear Projection] (64 → F=9)
    │  Transpose back to (B, F=9, W=64)
    ▼
  Reconstructed Window x̂: (B, F=9, W=64)
============================================================================
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from config import (
    CHECKPOINT_PATH,
    DEVICE,
    DROPOUT,
    ENCODER_HIDDEN,
    LATENT_DIM,
    NUM_LAYERS,
    NUM_SENSORS,
    WINDOW_SIZE,
)


class Encoder(nn.Module):
    """
    Encodes a multi-channel sequence into a compact latent vector.
    """

    def __init__(
        self,
        num_sensors: int = NUM_SENSORS,
        hidden_dim: int = ENCODER_HIDDEN,
        latent_dim: int = LATENT_DIM,
        dropout: float = DROPOUT,
    ) -> None:
        super().__init__()
        self.lstm1 = nn.LSTM(
            input_size=num_sensors,
            hidden_size=hidden_dim,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=latent_dim,
            batch_first=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (B, W, F)
        Returns:
            z: Latent vector of shape (B, latent_dim)
        """
        out, _ = self.lstm1(x)       # (B, W, hidden_dim)
        out = self.dropout(out)
        _, (h_n, _) = self.lstm2(out) # h_n: (1, B, latent_dim)
        z = h_n[-1]                   # (B, latent_dim)
        return z


class Decoder(nn.Module):
    """
    Decodes the latent bottleneck vector back to the original sequence shape.
    """

    def __init__(
        self,
        latent_dim: int = LATENT_DIM,
        hidden_dim: int = ENCODER_HIDDEN,
        num_sensors: int = NUM_SENSORS,
        window_size: int = WINDOW_SIZE,
        dropout: float = DROPOUT,
    ) -> None:
        super().__init__()
        self.window_size = window_size
        self.lstm1 = nn.LSTM(
            input_size=latent_dim,
            hidden_size=hidden_dim,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            batch_first=True,
        )
        self.fc_out = nn.Linear(hidden_dim, num_sensors)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: Latent vector of shape (B, latent_dim)
        Returns:
            x_hat: Reconstructed sequence of shape (B, W, F)
        """
        B = z.size(0)
        # RepeatVector: expand z to (B, W, latent_dim)
        z_repeated = z.unsqueeze(1).repeat(1, self.window_size, 1)

        out, _ = self.lstm1(z_repeated)  # (B, W, hidden_dim)
        out = self.dropout(out)
        out, _ = self.lstm2(out)         # (B, W, hidden_dim)
        x_hat = self.fc_out(out)         # (B, W, num_sensors)
        return x_hat


class LSTMAutoencoder(nn.Module):
    """
    Complete sequence-to-sequence LSTM Autoencoder for residual reconstruction.
    """

    def __init__(
        self,
        num_sensors: int = NUM_SENSORS,
        window_size: int = WINDOW_SIZE,
        hidden_dim: int = ENCODER_HIDDEN,
        latent_dim: int = LATENT_DIM,
        dropout: float = DROPOUT,
    ) -> None:
        super().__init__()
        self.num_sensors = num_sensors
        self.window_size = window_size

        self.encoder = Encoder(
            num_sensors=num_sensors,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            dropout=dropout,
        )
        self.decoder = Decoder(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            num_sensors=num_sensors,
            window_size=window_size,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct the input residual window.

        Args:
            x: Input tensor of shape (B, F, W) — channels-first.
        Returns:
            x_hat: Reconstructed tensor of shape (B, F, W).
        """
        # Convert channels-first (B, F, W) -> sequence-first (B, W, F)
        x_seq = x.permute(0, 2, 1)

        # Encode to bottleneck
        z = self.encoder(x_seq)

        # Decode back to sequence
        x_hat_seq = self.decoder(z)

        # Convert back to channels-first (B, F, W)
        x_hat = x_hat_seq.permute(0, 2, 1)
        return x_hat

    def compute_reconstruction_error(
        self,
        x: torch.Tensor,
        x_hat: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate reconstruction MSE across entire window and per sensor.

        Args:
            x: Original tensor of shape (B, F, W).
            x_hat: Reconstructed tensor of shape (B, F, W).

        Returns:
            total_mse: Scalar MSE per sample of shape (B,).
            sensor_mse: Per-sensor MSE of shape (B, F).
        """
        # Squared error per element: (B, F, W)
        sq_err = (x - x_hat) ** 2

        # Average over time: (B, F)
        sensor_mse = sq_err.mean(dim=2)

        # Average over all sensors and time: (B,)
        total_mse = sq_err.mean(dim=(1, 2))

        return total_mse, sensor_mse


# ═══════════════════════════════════════════════════════════════════════════
#  UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def count_parameters(model: nn.Module) -> int:
    """Return total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_model(
    model: LSTMAutoencoder,
    filepath: Path = CHECKPOINT_PATH,
    metadata: Optional[Dict] = None,
) -> None:
    """Save model weights and architectural hyperparameters."""
    checkpoint = {
        "state_dict": model.state_dict(),
        "num_sensors": model.num_sensors,
        "window_size": model.window_size,
        "metadata": metadata or {},
    }
    torch.save(checkpoint, filepath)


def load_model(
    filepath: Path = CHECKPOINT_PATH,
    device: torch.device = DEVICE,
) -> LSTMAutoencoder:
    """Load model checkpoint into memory."""
    checkpoint = torch.load(filepath, map_location=device)
    model = LSTMAutoencoder(
        num_sensors=checkpoint["num_sensors"],
        window_size=checkpoint["window_size"],
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model
