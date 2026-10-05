"""
FederatedClient — Per-UAV local model trainer for PRATIBIMB FL system.

Each client encapsulates:
  1. A local copy of the shared baseline anomaly detector (MLP autoencoder).
  2. Local healthy-flight data (sortie windows normalized per-client).
  3. Local SGD training with DP gradient clipping.
  4. A method to export gradient updates back to the server.

The local model is a compact MLP Autoencoder. Its reconstruction error on
a 9-sensor sliding window is the anomaly score fed into the persistence gate.
The weights learned from each UAV's operating environment (altitude, ambient
temperature, engine serial characteristics) are federated to improve the
global model without sharing raw data.
"""

import math
import random
import logging
import copy
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import FLConfig
from .privacy import DPGradientClipper
from .model import MLPAutoencoder, autoencoder_loss

log = logging.getLogger(__name__)


class FederatedClient:
    """
    Represents one UAV ground-station node in the FL federation.

    Parameters
    ----------
    client_id : str
        Unique identifier (e.g. "UAV-TAPAS-007").
    config : FLConfig
        Shared federated learning configuration.
    local_data : np.ndarray, optional
        Healthy-flight telemetry windows of shape (N, model_dim).
        If None, synthetic data is generated for testing.
    """

    def __init__(
        self,
        client_id: str,
        config: FLConfig,
        local_data: Optional[np.ndarray] = None,
    ):
        self.client_id = client_id
        self.config = config
        self.model = MLPAutoencoder(config.model_dim, config.hidden_dim)
        self._clipper = DPGradientClipper(
            max_norm=config.dp_max_grad_norm,
            noise_multiplier=config.dp_noise_multiplier,
        )
        self.local_data = (
            local_data
            if local_data is not None
            else self._generate_synthetic_data()
        )
        self._round_losses: List[float] = []
        log.info("Client %s initialized with %d local samples.",
                 client_id, len(self.local_data))

    # ------------------------------------------------------------------ #
    # Federation Protocol                                                  #
    # ------------------------------------------------------------------ #

    def receive_global_weights(self, global_weights: Dict[str, np.ndarray]):
        """Pull current global model weights from the server."""
        self.model.set_weights(global_weights)

    def local_update(self) -> Tuple[Dict[str, np.ndarray], int]:
        """
        Run E epochs of local SGD on healthy-flight data.

        Returns
        -------
        delta_weights : dict
            Weight delta = (trained_weights - received_weights), after
            DP clipping and noise addition.
        n_samples : int
            Number of local data points used (for weighted aggregation).
        """
        initial_weights = copy.deepcopy(self.model.get_weights())
        data = self.local_data
        n = len(data)
        rng = np.random.default_rng(
            abs(hash(self.client_id)) % (2**31)
        )

        epoch_losses = []
        for epoch in range(self.config.local_epochs):
            indices = rng.permutation(n)
            batch_losses = []
            for start in range(0, n, self.config.local_batch_size):
                batch_idx = indices[start: start + self.config.local_batch_size]
                batch = data[batch_idx]
                loss, grads = self.model.forward_backward(batch)
                # Apply DP gradient clipping + noise
                grads = self._clipper.clip_and_noise(grads, len(batch_idx))
                # Simple SGD weight update
                self.model.apply_gradients(grads, self.config.learning_rate)
                batch_losses.append(loss)
            epoch_losses.append(float(np.mean(batch_losses)))

        round_loss = float(np.mean(epoch_losses))
        self._round_losses.append(round_loss)
        log.debug("Client %s local update: loss=%.5f over %d epochs",
                  self.client_id, round_loss, self.config.local_epochs)

        # Compute weight delta
        trained_weights = self.model.get_weights()
        delta = {k: trained_weights[k] - initial_weights[k]
                 for k in trained_weights}
        return delta, n

    def compute_anomaly_score(self, window: np.ndarray) -> float:
        """
        Compute the reconstruction error (anomaly score) for a telemetry window.
        Used by the health pipeline to get a locally-personalized anomaly score.

        Parameters
        ----------
        window : np.ndarray
            Shape (model_dim,) — one normalized sensor observation vector.

        Returns
        -------
        float
            Mean squared reconstruction error (higher = more anomalous).
        """
        recon = self.model.forward(window.reshape(1, -1))
        return float(np.mean((window - recon.squeeze()) ** 2))

    # ------------------------------------------------------------------ #
    # Synthetic data generation (for offline testing)                     #
    # ------------------------------------------------------------------ #

    def _generate_synthetic_data(self) -> np.ndarray:
        """
        Generate synthetic healthy-flight windows with client-specific
        unit-to-unit variation to simulate real fleet diversity.
        """
        rng = np.random.default_rng(abs(hash(self.client_id)) % (2**31))
        n = rng.integers(400, 800)
        # Base healthy operating point
        base = np.array([4500, 160, 680, 90, 380, 0.012, 0.15, 0.65, 2000],
                        dtype=np.float32)
        # Per-client engine personality offset (unit-to-unit variation)
        personality = rng.normal(0, 0.05 * base)
        data = rng.normal(
            loc=base + personality,
            scale=0.02 * base,
            size=(n, self.config.model_dim),
        ).astype(np.float32)
        # Normalize to [0, 1] using per-client min/max
        data = (data - data.min(0)) / (data.max(0) - data.min(0) + 1e-8)
        return data
