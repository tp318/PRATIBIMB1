"""
FederatedServer — Central FL coordinator for the PRATIBIMB fleet.

Orchestrates the full FedAvg training loop:
    for each round:
        1. Randomly select C clients from the fleet pool.
        2. Broadcast current global model weights.
        3. Collect local updates from participating clients.
        4. Aggregate updates via FedAvg (or FedMedian).
        5. Apply aggregated delta to global model.
        6. Log round metrics and check convergence.

After training, the global model is exported and can be:
    - Pushed to the EDGE_ML quantizer for INT8 deployment.
    - Loaded by any FederatedClient as a starting point.
    - Saved to disk as a plain NumPy .npz checkpoint.

Usage:
    config = FLConfig(num_rounds=20, clients_per_round=4)
    server = FederatedServer(config)
    clients = [FederatedClient(f"UAV-{i}", config) for i in range(10)]
    history = server.fit(clients)
    server.save_checkpoint("global_model.npz")
"""

import random
import logging
import copy
import time
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np

from .config import FLConfig
from .client import FederatedClient
from .model import MLPAutoencoder
from .aggregator import select_aggregator

log = logging.getLogger(__name__)


class FederatedServer:
    """
    Central FL server / aggregator.

    Parameters
    ----------
    config : FLConfig
        Shared training configuration.
    initial_weights : dict, optional
        Pre-trained starting weights. If None, fresh random weights are used.
    """

    def __init__(
        self,
        config: FLConfig,
        initial_weights: Optional[Dict[str, np.ndarray]] = None,
    ):
        self.config = config
        self.global_model = MLPAutoencoder(config.model_dim, config.hidden_dim)
        if initial_weights is not None:
            self.global_model.set_weights(initial_weights)
        self._aggregator = select_aggregator(config.aggregation)
        self._round_logs: List[Dict] = []
        self._rng = random.Random(config.seed)
        log.info(
            "FL Server initialized | strategy=%s | rounds=%d | clients/round=%d",
            config.aggregation, config.num_rounds, config.clients_per_round,
        )

    # ------------------------------------------------------------------ #
    # Main training loop                                                   #
    # ------------------------------------------------------------------ #

    def fit(self, all_clients: List[FederatedClient]) -> List[Dict]:
        """
        Run the full federated training loop.

        Parameters
        ----------
        all_clients : list of FederatedClient
            Full pool of available UAV clients (any may be selected each round).

        Returns
        -------
        list of dict
            Round-by-round metrics log.
        """
        eligible = [c for c in all_clients
                    if len(c.local_data) >= self.config.min_data_points]
        if not eligible:
            raise RuntimeError("No clients have sufficient local data to train.")

        log.info("Starting FL training: %d eligible clients, %d rounds.",
                 len(eligible), self.config.num_rounds)

        prev_loss = float("inf")
        for rnd in range(1, self.config.num_rounds + 1):
            t0 = time.perf_counter()

            # 1. Select clients for this round
            k = min(self.config.clients_per_round, len(eligible))
            selected = self._rng.sample(eligible, k)

            # 2. Broadcast global weights
            global_weights = self.global_model.get_weights()
            for client in selected:
                client.receive_global_weights(global_weights)

            # 3. Collect local updates
            updates = []
            for client in selected:
                delta, n = client.local_update()
                updates.append((delta, n))

            # 4. Aggregate
            agg_delta = self._aggregator(updates)

            # 5. Apply aggregated delta
            new_weights = {
                k: global_weights[k] + agg_delta[k]
                for k in global_weights
            }
            self.global_model.set_weights(new_weights)

            # 6. Log
            elapsed = time.perf_counter() - t0
            round_loss = self._estimate_global_loss(selected)
            improvement = prev_loss - round_loss
            prev_loss = round_loss

            entry = {
                "round": rnd,
                "loss": round(round_loss, 6),
                "improvement": round(improvement, 8),
                "n_clients": k,
                "elapsed_s": round(elapsed, 3),
                "total_samples": sum(n for _, n in updates),
            }
            self._round_logs.append(entry)
            log.info(
                "Round %3d/%d | loss=%.5f | Δloss=%+.6f | clients=%d | %.2fs",
                rnd, self.config.num_rounds, round_loss, improvement, k, elapsed,
            )

            # Early stopping
            if abs(improvement) < self.config.convergence_tol and rnd > 5:
                log.info("Converged at round %d (Δloss < %.1e). Stopping.",
                         rnd, self.config.convergence_tol)
                break

        log.info("FL training complete. Final loss: %.5f", prev_loss)
        return self._round_logs

    # ------------------------------------------------------------------ #
    # Utilities                                                            #
    # ------------------------------------------------------------------ #

    def _estimate_global_loss(self, clients: List[FederatedClient]) -> float:
        """Compute mean reconstruction loss across selected clients."""
        losses = []
        w = self.global_model.get_weights()
        for client in clients:
            client.model.set_weights(w)
            recon = client.model.forward(client.local_data)
            loss = float(np.mean((client.local_data - recon) ** 2))
            losses.append(loss)
        return float(np.mean(losses))

    def save_checkpoint(self, path: str):
        """Save global model weights to a NumPy .npz file."""
        weights = self.global_model.get_weights()
        np.savez(path, **weights)
        log.info("Global model checkpoint saved: %s", path)

    @classmethod
    def load_checkpoint(cls, path: str, config: FLConfig) -> "FederatedServer":
        """Load a previously saved checkpoint."""
        data = np.load(path)
        weights = {k: data[k] for k in data}
        return cls(config, initial_weights=weights)

    @property
    def round_history(self) -> List[Dict]:
        return self._round_logs

    def summary(self) -> str:
        if not self._round_logs:
            return "No training rounds completed."
        first = self._round_logs[0]["loss"]
        last = self._round_logs[-1]["loss"]
        total_rounds = len(self._round_logs)
        return (
            f"FederatedServer | {total_rounds} rounds | "
            f"loss: {first:.5f} → {last:.5f} "
            f"({100*(first-last)/first:.1f}% reduction)"
        )
