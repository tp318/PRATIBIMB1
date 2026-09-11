"""
gru_correction.py
=================
GRU correction to the physics degradation rate (Deliverable E, Part 12).

The network does NOT predict RUL, and it does not predict the degradation state.
It predicts a bounded multiplicative correction to the physics-identified wear
rate for each of the six health parameters:

    rate_pred_k = rate_phys_k * exp(1.6 * tanh(u_k))  +  softplus(v_k) * floor

The two terms do different jobs:

  * the MULTIPLICATIVE term corrects a rate the physics model already believes
    in. Bounded to [0.20, 4.95] so the network can rescale the physics but never
    replace it. If the head saturates often, that is evidence the physics term is
    wrong and is reported rather than hidden.

  * the ADDITIVE term exists for one specific failure mode: a mechanism whose
    incubation has just ended. There the identified coefficient a_k is still
    near zero, so a purely multiplicative correction cannot lift the rate off
    zero no matter how large it gets. The additive floor is small (scaled to
    ~1/4 of a typical rate) so it cannot dominate a healthy prediction.

Target
------
The label is the TRUE smoothed degradation rate from the simulator. That is
legitimate supervised learning, not leakage: labels are ground truth by
definition, and every model INPUT is drawn from the REAL_ENGINE or DIGITAL_TWIN
classes of the feature manifest. At inference no label is required.

Loss
----
Rates span orders of magnitude (zero during incubation, large near EOL), so the
loss is computed in log space against a reference scale. A plain MSE on raw
rates would be dominated entirely by the fastest-wearing engines near their end
of life and would learn nothing about early detection, which is the part that
actually matters operationally.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn as nn

from ..config import HEALTH_PARAMS, SNAPSHOT_HOURS

# Typical rate: a full life consumed over 1500 snapshots. Used to non-dimensionalise
# the loss and to scale the additive floor.
RATE_SCALE = 1.0 / (1500.0 * SNAPSHOT_HOURS)
LOG_EPS = 0.02 * RATE_SCALE


class DegradationRateGRU(nn.Module):
    """Small GRU producing a bounded correction to the physics rate.

    Deliberately small. It runs on the ground station rather than the aircraft,
    but the whole point of a physics-informed design is that the learned part is
    a correction, and a correction that needs a million parameters is not a
    correction. Roughly 30k parameters at the default width.
    """

    def __init__(self, n_features: int, n_params: int = len(HEALTH_PARAMS),
                 hidden: int = 64, layers: int = 1, dropout: float = 0.10,
                 max_log_correction: float = 1.6):
        super().__init__()
        self.n_params = n_params
        self.max_log_correction = max_log_correction
        self.gru = nn.GRU(n_features, hidden, num_layers=layers,
                          batch_first=True,
                          dropout=dropout if layers > 1 else 0.0)
        self.norm = nn.LayerNorm(hidden)
        self.drop = nn.Dropout(dropout)
        self.head_mult = nn.Linear(hidden, n_params)
        self.head_add = nn.Linear(hidden, n_params)
        # Start life as the identity correction: zero multiplicative log-shift
        # and a negligible additive term. The untrained model is then exactly the
        # physics model, which is the right place to start from.
        nn.init.zeros_(self.head_mult.weight)
        nn.init.zeros_(self.head_mult.bias)
        nn.init.zeros_(self.head_add.weight)
        nn.init.constant_(self.head_add.bias, -4.0)

    def forward(self, x: torch.Tensor, rate_phys: torch.Tensor) -> Dict[str, torch.Tensor]:
        h, _ = self.gru(x)
        h = self.drop(self.norm(h[:, -1, :]))
        u = self.head_mult(h)
        v = self.head_add(h)
        mult = torch.exp(self.max_log_correction * torch.tanh(u))
        add = torch.nn.functional.softplus(v) * (0.25 * RATE_SCALE)
        rate = rate_phys * mult + add
        return {"rate": rate, "mult": mult, "add": add, "log_mult": u}


def rate_loss(pred: torch.Tensor, target: torch.Tensor,
              weight: torch.Tensor | None = None) -> torch.Tensor:
    """Log-space error between predicted and true degradation rate."""
    lp = torch.log(torch.clamp(pred, min=0.0) + LOG_EPS)
    lt = torch.log(torch.clamp(target, min=0.0) + LOG_EPS)
    err = (lp - lt) ** 2
    if weight is not None:
        err = err * weight
    return err.mean()


# --------------------------------------------------------------------------- #
# Direct-RUL GRU (Baseline 2) - same backbone, different head
# --------------------------------------------------------------------------- #

class DirectRULGRU(nn.Module):
    """Telemetry sequence -> RUL, the conventional deep-learning approach.

    Included as an honest baseline. It shares the backbone width with the
    correction network so that any difference in results is attributable to the
    architecture of the PIPELINE rather than to one model simply being bigger.
    Predicts in a normalised RUL space and returns hours.
    """

    def __init__(self, n_features: int, hidden: int = 64, layers: int = 2,
                 dropout: float = 0.15, rul_scale: float = 400.0):
        super().__init__()
        self.rul_scale = rul_scale
        self.gru = nn.GRU(n_features, hidden, num_layers=layers,
                          batch_first=True,
                          dropout=dropout if layers > 1 else 0.0)
        self.norm = nn.LayerNorm(hidden)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(),
                                  nn.Linear(32, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.gru(x)
        h = self.drop(self.norm(h[:, -1, :]))
        # softplus keeps RUL non-negative, which is free information the model
        # should not have to learn.
        return torch.nn.functional.softplus(self.head(h).squeeze(-1)) * self.rul_scale


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_size_bytes(model: nn.Module) -> int:
    return sum(p.numel() * p.element_size() for p in model.parameters())
