"""
baselines.py
============
Baseline RUL models (Part 14, Deliverable G).

Four reference points, ordered by how much of the physics pipeline they use:

  B1  XGBoost      window-summary telemetry -> RUL         (no twin, no filter)
  B2  GRU          telemetry sequence -> RUL               (no twin, no filter)
  B3  GRU          UKF health sequence -> RUL              (filter, no dynamics)
  B4  PHYSICS-ONLY identified wear rate propagated to EOL  (no learning at all)

B4 matters more than it looks. It is the null hypothesis for the entire
exercise: if the full hybrid does not beat a pure physics extrapolation, the
learned components are decoration. It is reported alongside the others rather
than quietly omitted.

FAIRNESS
--------
All learned baselines see the same window geometry, the same runs, the same
splits and comparable model capacity. B1 is given WINDOW SUMMARIES rather than a
single row so that it has access to the same temporal information as the
sequence models - trend, level and dispersion over the window - and any
difference reflects the model class rather than an information handicap. Giving
a tabular model only the latest row and then declaring sequence models superior
would be a rigged comparison.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np


def window_summary(X: np.ndarray, feature_names: Sequence[str]
                   ) -> Tuple[np.ndarray, List[str]]:
    """Collapse (N, L, F) sequences into (N, 4F) tabular summaries.

    Four statistics per feature, chosen to carry what a sequence model can read
    off a window:
        last   the current value
        mean   the level over the window
        std    the dispersion (combustion roughness, transient activity)
        slope  the trend per snapshot (the primary degradation signal)
    """
    N, L, F = X.shape
    t = np.arange(L, dtype=np.float32)
    t = t - t.mean()
    denom = float((t ** 2).sum())

    last = X[:, -1, :]
    mean = X.mean(axis=1)
    std = X.std(axis=1)
    slope = np.einsum("nlf,l->nf", X - mean[:, None, :], t) / denom

    out = np.concatenate([last, mean, std, slope], axis=1).astype(np.float32)
    names = ([f"{n}__last" for n in feature_names]
             + [f"{n}__mean" for n in feature_names]
             + [f"{n}__std" for n in feature_names]
             + [f"{n}__slope" for n in feature_names])
    return out, names


def train_xgboost_rul(X_tr: np.ndarray, y_tr: np.ndarray,
                      X_va: np.ndarray, y_va: np.ndarray,
                      feature_names: Sequence[str],
                      seed: int = 0, n_estimators: int = 1200):
    """Gradient-boosted RUL regressor with early stopping on the validation runs."""
    import xgboost as xgb

    model = xgb.XGBRegressor(
        n_estimators=n_estimators,
        max_depth=6,
        learning_rate=0.035,
        subsample=0.85,
        colsample_bytree=0.65,
        min_child_weight=8.0,
        reg_lambda=2.5,
        reg_alpha=0.2,
        objective="reg:squarederror",
        tree_method="hist",
        random_state=seed,
        early_stopping_rounds=60,
        n_jobs=8,
    )
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    return model


# --------------------------------------------------------------------------- #
# B4: pure physics extrapolation
# --------------------------------------------------------------------------- #

def physics_only_rul(phi_now: np.ndarray,
                     rate_coeffs: np.ndarray,
                     stress_recent: np.ndarray,
                     surrogate,
                     baseline=None,
                     dt_h: float = 1.0,
                     horizon_h: float = 900.0) -> float:
    """Deterministic RUL from the identified wear rate alone.

    No filter uncertainty, no learned correction, no duty-cycle sampling: the
    engine is assumed to keep wearing at its identified rate under its average
    recent stress until a limit is reached. This is what a competent engineer
    would do with a trend plot and a spreadsheet, and it is the bar the hybrid
    architecture has to clear to justify itself.
    """
    from .degradation_model import shape_term

    phi = np.asarray(phi_now, dtype=float).copy()[None, :]
    s_mean = np.asarray(stress_recent, dtype=float).mean(axis=0)[None, :]
    a = np.asarray(rate_coeffs, dtype=float)[None, :]

    n_steps = int(np.ceil(horizon_h / dt_h))
    prev = float(surrogate.worst_margin(phi, baseline)[0])
    if prev >= 1.0:
        return 0.0
    for step in range(n_steps):
        phi = phi + np.maximum(a * s_mean * shape_term(phi), 0.0) * dt_h
        m = float(surrogate.worst_margin(phi, baseline)[0])
        if m >= 1.0:
            frac = np.clip((1.0 - prev) / max(m - prev, 1e-9), 0.0, 1.0)
            return float((step + frac) * dt_h)
        prev = m
    return float(horizon_h)
