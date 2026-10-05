"""
MLPAutoencoder — Lightweight NumPy-only MLP autoencoder for FL clients.

Deliberately implemented in pure NumPy (no PyTorch/TF dependency) so it
can run on resource-constrained GCS laptops and be serialized as plain
dict-of-arrays for network transmission between FL nodes.

Architecture:
    Encoder:  input_dim -> hidden_dim -> bottleneck (hidden_dim // 2)
    Decoder:  bottleneck -> hidden_dim -> input_dim

Activation: ReLU (encoder), Linear (decoder output)
Loss:       Mean Squared Error (reconstruction)

For edge deployment the weights are exported to a plain dict and can be
quantized to INT8 by the EDGE_ML quantizer without modification.
"""

import math
import numpy as np
from typing import Dict, Tuple


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def _relu_grad(x: np.ndarray) -> np.ndarray:
    return (x > 0).astype(x.dtype)


def autoencoder_loss(x: np.ndarray, recon: np.ndarray) -> float:
    """Mean Squared Error reconstruction loss."""
    return float(np.mean((x - recon) ** 2))


class MLPAutoencoder:
    """
    Pure-NumPy MLP Autoencoder.

    Parameters
    ----------
    input_dim : int
        Number of sensor features (9 for PRATIBIMB default).
    hidden_dim : int
        Width of encoder hidden layer (bottleneck = hidden_dim // 2).
    """

    def __init__(self, input_dim: int = 9, hidden_dim: int = 32):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.bottleneck_dim = max(4, hidden_dim // 2)
        rng = np.random.default_rng(0)
        # Xavier initialization
        self.W1 = rng.normal(0, math.sqrt(2.0 / input_dim),
                             (input_dim, hidden_dim)).astype(np.float32)
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)
        self.W2 = rng.normal(0, math.sqrt(2.0 / hidden_dim),
                             (hidden_dim, self.bottleneck_dim)).astype(np.float32)
        self.b2 = np.zeros(self.bottleneck_dim, dtype=np.float32)
        self.W3 = rng.normal(0, math.sqrt(2.0 / self.bottleneck_dim),
                             (self.bottleneck_dim, hidden_dim)).astype(np.float32)
        self.b3 = np.zeros(hidden_dim, dtype=np.float32)
        self.W4 = rng.normal(0, math.sqrt(2.0 / hidden_dim),
                             (hidden_dim, input_dim)).astype(np.float32)
        self.b4 = np.zeros(input_dim, dtype=np.float32)

    # ------------------------------------------------------------------ #
    # Forward pass                                                         #
    # ------------------------------------------------------------------ #

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass — returns reconstructed input.
        x : (batch, input_dim)
        """
        h1 = _relu(x @ self.W1 + self.b1)
        h2 = _relu(h1 @ self.W2 + self.b2)
        h3 = _relu(h2 @ self.W3 + self.b3)
        return h3 @ self.W4 + self.b4  # linear output

    # ------------------------------------------------------------------ #
    # Forward + Backward (for local training)                             #
    # ------------------------------------------------------------------ #

    def forward_backward(
        self, x: np.ndarray
    ) -> Tuple[float, Dict[str, np.ndarray]]:
        """
        Compute MSE loss and gradients w.r.t. all weights.

        Returns
        -------
        loss : float
        grads : dict of weight-name -> gradient array
        """
        batch = x.shape[0]

        # --- Forward ---
        z1 = x @ self.W1 + self.b1
        h1 = _relu(z1)
        z2 = h1 @ self.W2 + self.b2
        h2 = _relu(z2)
        z3 = h2 @ self.W3 + self.b3
        h3 = _relu(z3)
        recon = h3 @ self.W4 + self.b4

        loss = autoencoder_loss(x, recon)

        # --- Backward (MSE gradient = 2*(recon - x) / batch) ---
        d_recon = 2.0 * (recon - x) / batch

        # Layer 4
        dW4 = h3.T @ d_recon
        db4 = d_recon.sum(0)
        dh3 = d_recon @ self.W4.T

        # Layer 3
        dz3 = dh3 * _relu_grad(z3)
        dW3 = h2.T @ dz3
        db3 = dz3.sum(0)
        dh2 = dz3 @ self.W3.T

        # Layer 2
        dz2 = dh2 * _relu_grad(z2)
        dW2 = h1.T @ dz2
        db2 = dz2.sum(0)
        dh1 = dz2 @ self.W2.T

        # Layer 1
        dz1 = dh1 * _relu_grad(z1)
        dW1 = x.T @ dz1
        db1 = dz1.sum(0)

        grads = {
            "W1": dW1, "b1": db1,
            "W2": dW2, "b2": db2,
            "W3": dW3, "b3": db3,
            "W4": dW4, "b4": db4,
        }
        return loss, grads

    # ------------------------------------------------------------------ #
    # Weight I/O                                                           #
    # ------------------------------------------------------------------ #

    def get_weights(self) -> Dict[str, np.ndarray]:
        return {k: getattr(self, k).copy()
                for k in ("W1", "b1", "W2", "b2", "W3", "b3", "W4", "b4")}

    def set_weights(self, weights: Dict[str, np.ndarray]):
        for k, v in weights.items():
            setattr(self, k, v.copy().astype(np.float32))

    def apply_gradients(
        self, grads: Dict[str, np.ndarray], lr: float
    ):
        for k in grads:
            setattr(self, k, getattr(self, k) - lr * grads[k])

    def parameter_count(self) -> int:
        return sum(v.size for v in self.get_weights().values())
