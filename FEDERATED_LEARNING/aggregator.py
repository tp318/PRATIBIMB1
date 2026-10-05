"""
FedAvg Aggregation — server-side weight aggregation for PRATIBIMB FL.

Implements two strategies:
  1. fedavg_aggregate  — weighted mean (McMahan et al., 2017)
  2. fedmedian_aggregate — coordinate-wise median (robust to Byzantine clients,
                           Li et al., 2019)

Both take a list of (weight_delta, n_samples) pairs and return a single
aggregated weight delta that the server applies to the global model.
"""

import numpy as np
from typing import Dict, List, Tuple


WeightDict = Dict[str, np.ndarray]
ClientUpdate = Tuple[WeightDict, int]   # (delta_weights, n_samples)


def weighted_mean_params(updates: List[ClientUpdate]) -> WeightDict:
    """
    Compute the dataset-size-weighted mean of weight deltas.

    Parameters
    ----------
    updates : list of (weight_delta_dict, n_samples)

    Returns
    -------
    WeightDict
        Aggregated weight delta, weighted by n_samples.
    """
    total_samples = sum(n for _, n in updates)
    if total_samples == 0:
        raise ValueError("Total samples across all clients is zero.")

    keys = list(updates[0][0].keys())
    aggregated: WeightDict = {}
    for k in keys:
        aggregated[k] = sum(
            (n / total_samples) * delta[k] for delta, n in updates
        ).astype(np.float32)
    return aggregated


def fedavg_aggregate(updates: List[ClientUpdate]) -> WeightDict:
    """
    FedAvg aggregation (Algorithm 1 of McMahan et al., 2017).

    Computes ∑ (n_k / n_total) · Δw_k  for each weight tensor.
    """
    return weighted_mean_params(updates)


def fedmedian_aggregate(updates: List[ClientUpdate]) -> WeightDict:
    """
    Coordinate-wise median aggregation (Byzantine-robust).

    Ignores n_samples weighting — treats each client equally.
    Robust against up to floor((K-1)/2) adversarial or corrupted clients.
    """
    if not updates:
        raise ValueError("No client updates provided.")
    keys = list(updates[0][0].keys())
    aggregated: WeightDict = {}
    for k in keys:
        stacked = np.stack([delta[k] for delta, _ in updates], axis=0)
        aggregated[k] = np.median(stacked, axis=0).astype(np.float32)
    return aggregated


def select_aggregator(strategy: str):
    """Return the aggregation function matching the strategy name."""
    registry = {
        "fedavg": fedavg_aggregate,
        "fedmedian": fedmedian_aggregate,
    }
    if strategy not in registry:
        raise ValueError(f"Unknown aggregation strategy '{strategy}'. "
                         f"Choose from: {list(registry)}")
    return registry[strategy]
