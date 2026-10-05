"""
FLConfig — Federated Learning Hyperparameter Configuration.

All global and per-client hyperparameters live here to ensure every
component reads from a single source of truth.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class FLConfig:
    """
    Federated learning configuration for PRATIBIMB fleet deployment.

    Parameters
    ----------
    num_rounds : int
        Number of federated communication rounds.
    clients_per_round : int
        Number of clients randomly selected per round (C in FedAvg).
    local_epochs : int
        Number of local training epochs each client runs per round (E).
    local_batch_size : int
        Mini-batch size for local SGD.
    learning_rate : float
        Local SGD learning rate.
    model_dim : int
        Input feature dimension of the shared model (number of sensors).
    hidden_dim : int
        Number of hidden units in the baseline MLP anomaly detector.
    dp_max_grad_norm : float
        L2 norm clip bound for DP gradient clipping (C in DP-SGD).
    dp_noise_multiplier : float
        Gaussian noise multiplier σ for differential privacy.
        Set to 0.0 to disable DP.
    min_data_points : int
        Minimum local samples a client must have to participate.
    aggregation : str
        Aggregation strategy: "fedavg" (weighted) or "fedmedian" (robust).
    seed : int
        Global random seed for reproducibility.
    eval_every : int
        Evaluate global model on held-out validation set every N rounds.
    convergence_tol : float
        Stop early if round-over-round loss improvement < tol.
    """

    num_rounds: int = 30
    clients_per_round: int = 5
    local_epochs: int = 3
    local_batch_size: int = 64
    learning_rate: float = 1e-3
    model_dim: int = 9          # RPM, CHT, EGT, oil_temp, oil_pres, fuel_flow,
                                #  vibration, throttle, altitude
    hidden_dim: int = 32
    dp_max_grad_norm: float = 1.0
    dp_noise_multiplier: float = 0.5
    min_data_points: int = 200
    aggregation: str = "fedavg"  # "fedavg" | "fedmedian"
    seed: int = 42
    eval_every: int = 5
    convergence_tol: float = 1e-5

    # Sensor feature names (must match EngineTelemetry field names)
    sensor_features: List[str] = field(default_factory=lambda: [
        "rpm", "cht", "egt", "oil_temperature", "oil_pressure",
        "fuel_flow", "vibration", "throttle", "altitude",
    ])
