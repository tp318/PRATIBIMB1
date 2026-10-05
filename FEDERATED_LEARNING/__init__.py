"""
FEDERATED_LEARNING — PRATIBIMB Fleet-Wide Federated Anomaly Detector.

Implements FedAvg (McMahan et al., 2017) over the PRATIBIMB statistical
anomaly detector, allowing multiple UAV ground stations to collaboratively
improve the shared baseline model without sharing raw sensor telemetry.

Architecture:
    Each UAV (client) holds local sortie data.
    Each round:
        1. Server sends current global model weights to all active clients.
        2. Each client runs E epochs of local SGD on its own data.
        3. Clients return clipped, noise-added gradient updates.
        4. Server aggregates updates weighted by dataset size (FedAvg).
        5. Global model is updated; clients pull the new weights.

Privacy:
    Differential privacy via Gaussian noise addition and gradient clipping
    (DP-SGD, Abadi et al., 2016) at the client side.
    Raw telemetry never leaves the client machine.

Usage:
    from FEDERATED_LEARNING import FederatedServer, FederatedClient, FLConfig

    config = FLConfig(num_rounds=20, clients_per_round=5, local_epochs=3)
    server = FederatedServer(config)
    clients = [FederatedClient(f"UAV-{i}", config) for i in range(10)]
    server.fit(clients)
    aggregated_model = server.global_model
"""

from .config import FLConfig
from .client import FederatedClient
from .server import FederatedServer
from .aggregator import fedavg_aggregate, weighted_mean_params
from .privacy import DPGradientClipper

__all__ = [
    "FLConfig",
    "FederatedClient",
    "FederatedServer",
    "fedavg_aggregate",
    "weighted_mean_params",
    "DPGradientClipper",
]
