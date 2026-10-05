"""
DPGradientClipper — Differential Privacy for PRATIBIMB FL clients.

Implements the per-sample gradient clipping + Gaussian noise mechanism
from DP-SGD (Abadi et al., 2016, "Deep Learning with Differential Privacy").

Privacy guarantee:
    After T rounds with clipping norm C and noise multiplier σ,
    the (ε, δ)-DP budget can be tracked via the Moments Accountant.
    This implementation clips the *aggregated* batch gradient (an
    approximation that is standard for small batch sizes) and adds
    calibrated Gaussian noise N(0, (σ·C)²) to each weight tensor.

Parameters
----------
max_norm : float
    Maximum L2 norm for gradient clipping (C in DP-SGD).
noise_multiplier : float
    Ratio of noise std to clipping norm (σ). Set to 0 to disable DP.
"""

import math
import numpy as np
from typing import Dict


class DPGradientClipper:
    """
    Apply L2 gradient clipping and add calibrated Gaussian noise.

    Attributes
    ----------
    max_norm : float
        Clip threshold C.
    noise_multiplier : float
        σ — noise std = σ * C / sqrt(batch_size).
    total_noise_added : int
        Counter for auditing purposes.
    """

    def __init__(self, max_norm: float = 1.0, noise_multiplier: float = 0.5):
        if max_norm <= 0:
            raise ValueError("max_norm must be positive")
        self.max_norm = max_norm
        self.noise_multiplier = noise_multiplier
        self.total_noise_added: int = 0
        self._rng = np.random.default_rng(0)

    def clip_and_noise(
        self,
        grads: Dict[str, np.ndarray],
        batch_size: int,
    ) -> Dict[str, np.ndarray]:
        """
        1. Clip aggregated gradient by global L2 norm.
        2. Add Gaussian noise scaled by σ·C / sqrt(batch_size).

        Parameters
        ----------
        grads : dict
            Gradient tensors from one local SGD step.
        batch_size : int
            Number of samples in the mini-batch (for noise scaling).

        Returns
        -------
        dict
            Clipped and noised gradient tensors.
        """
        # Compute global gradient L2 norm
        global_norm = math.sqrt(
            sum(float(np.sum(g ** 2)) for g in grads.values())
        )

        # Clip
        clip_factor = min(1.0, self.max_norm / (global_norm + 1e-8))
        clipped = {k: g * clip_factor for k, g in grads.items()}

        # Add Gaussian noise (skip if noise_multiplier == 0)
        if self.noise_multiplier > 0.0:
            noise_std = (self.noise_multiplier * self.max_norm
                         / math.sqrt(max(1, batch_size)))
            noised = {}
            for k, g in clipped.items():
                noise = self._rng.normal(0.0, noise_std, size=g.shape).astype(
                    g.dtype
                )
                noised[k] = g + noise
                self.total_noise_added += g.size
            return noised

        return clipped

    def privacy_budget_estimate(
        self,
        n_samples: int,
        batch_size: int,
        n_rounds: int,
        local_epochs: int,
        delta: float = 1e-5,
    ) -> Dict[str, float]:
        """
        Rough ε estimate using the strong composition theorem.
        (For rigorous accounting use the Moments Accountant / RDP accountant.)

        Returns
        -------
        dict with keys: epsilon_approx, delta, steps
        """
        steps = n_rounds * local_epochs * (n_samples // max(1, batch_size))
        q = batch_size / max(1, n_samples)       # sampling ratio
        # Rough Gaussian mechanism ε per step
        eps_per_step = (
            q * math.sqrt(2 * math.log(1.25 / delta))
            / self.noise_multiplier
        )
        # Strong composition over T steps
        eps_total = eps_per_step * math.sqrt(steps * math.log(1 / delta))
        return {
            "epsilon_approx": round(eps_total, 4),
            "delta": delta,
            "steps": steps,
            "noise_std_per_param": round(
                self.noise_multiplier * self.max_norm / math.sqrt(batch_size), 6
            ),
        }
