"""
plots.py
========
Physical-validation and diagnostic figures (Part 19).

The point of these figures is not decoration. Metrics can look excellent while
the underlying behaviour is nonsense - a model that has learned "RUL falls with
operating hours" scores well on MAE and R^2 and is useless. These plots are the
check that the system is doing what it claims: that estimated health tracks true
health, that residuals grow as the engine degrades, and that RUL falls
monotonically toward zero at the right time.
"""

from __future__ import annotations

import os
from typing import Dict, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..config import FIGURE_DIR, HEALTH_PARAMS

PARAM_LABEL = {
    "eta_inj": r"injector efficiency  $\eta_{inj}$",
    "eta_comb": r"combustion efficiency  $\eta_{comb}$",
    "h_cool": r"cooling effectiveness  $h_{cool}$",
    "friction_mult": "friction multiplier",
    "lub_health": "lubrication health",
    "eta_vol": r"volumetric efficiency  $\eta_{vol}$",
}

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 130, "font.size": 9,
    "axes.grid": True, "grid.alpha": 0.25, "axes.spines.top": False,
    "axes.spines.right": False, "figure.autolayout": False,
})


def _save(fig, name: str) -> str:
    os.makedirs(FIGURE_DIR, exist_ok=True)
    path = os.path.join(FIGURE_DIR, name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# 1 / 7 / 8 / 9  health parameter tracking
# --------------------------------------------------------------------------- #

def plot_health_tracking(df_run: pd.DataFrame, run_id: str,
                         name: str = "health_tracking.png") -> str:
    """True vs UKF-estimated health parameters over one engine's life."""
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 6.6), sharex=True)
    t = df_run.operating_hours.to_numpy()
    for ax, k in zip(axes.ravel(), HEALTH_PARAMS):
        ax.plot(t, df_run[f"true_{k}"], color="#1b3a6b", lw=1.9, label="true", zorder=3)
        e = df_run[f"est_{k}"].to_numpy()
        s = df_run[f"est_{k}_std"].to_numpy() if f"est_{k}_std" in df_run else None
        ax.plot(t, e, color="#c1440e", lw=1.4, label="UKF estimate", zorder=2)
        if s is not None:
            ax.fill_between(t, e - 2 * s, e + 2 * s, color="#c1440e", alpha=0.16,
                            lw=0, label=r"$\pm 2\sigma$", zorder=1)
        ax.set_title(PARAM_LABEL[k], fontsize=9)
        ax.set_xlabel("operating hours")
    axes[0, 0].legend(fontsize=8, loc="best")
    fig.suptitle(f"Latent health tracking - {run_id}", fontsize=11, y=1.00)
    fig.tight_layout()
    return _save(fig, name)


def plot_health_scatter(df: pd.DataFrame, name: str = "health_scatter.png") -> str:
    """Estimated vs true health across the whole test set, per parameter."""
    fig, axes = plt.subplots(2, 3, figsize=(13.0, 7.0))
    for ax, k in zip(axes.ravel(), HEALTH_PARAMS):
        x = df[f"true_{k}"].to_numpy()
        y = df[f"est_{k}"].to_numpy()
        sub = np.random.default_rng(0).choice(len(x), size=min(6000, len(x)), replace=False)
        ax.scatter(x[sub], y[sub], s=2.0, alpha=0.16, color="#1b3a6b", lw=0)
        lo = float(min(np.nanmin(x), np.nanmin(y)))
        hi = float(max(np.nanmax(x), np.nanmax(y)))
        ax.plot([lo, hi], [lo, hi], color="#c1440e", lw=1.2, ls="--")
        r = float(np.corrcoef(x, y)[0, 1])
        mae = float(np.mean(np.abs(x - y)))
        ax.set_title(f"{PARAM_LABEL[k]}\nr={r:.3f}  MAE={mae:.4f}", fontsize=8.5)
        ax.set_xlabel("true"); ax.set_ylabel("estimated")
    fig.suptitle("UKF health estimation across the held-out fleet", fontsize=11)
    fig.tight_layout()
    return _save(fig, name)


# --------------------------------------------------------------------------- #
# 2  degradation
# --------------------------------------------------------------------------- #

def plot_degradation(df_run: pd.DataFrame, run_id: str,
                     name: str = "degradation.png") -> str:
    """True vs modelled damage accumulation and wear rate."""
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.0))
    t = df_run.operating_hours.to_numpy()

    ax = axes[0]
    for k in HEALTH_PARAMS:
        ax.plot(t, df_run[f"phi_true_{k}"], lw=1.6, label=k)
    ax.axhline(1.0, color="k", ls=":", lw=1.0)
    ax.set_title("true damage $\\phi$ (1.0 = that mechanism alone ends life)")
    ax.set_xlabel("operating hours"); ax.set_ylabel(r"$\phi$")
    ax.legend(fontsize=6.5, ncol=2)

    ax = axes[1]
    for k in HEALTH_PARAMS:
        ax.plot(t, df_run[f"phi_{k}"], lw=1.6, label=k)
    ax.axhline(1.0, color="k", ls=":", lw=1.0)
    ax.set_title("estimated damage $\\hat{\\phi}$ from the UKF")
    ax.set_xlabel("operating hours")

    ax = axes[2]
    dom = max(HEALTH_PARAMS, key=lambda k: float(df_run[f"phi_true_{k}"].iloc[-1]))
    ax.plot(t, df_run[f"rate_true_{dom}"], color="#1b3a6b", lw=1.8, label="true rate")
    ax.plot(t, df_run[f"rate_phys_{dom}"], color="#2e7d32", lw=1.3, label="physics rate")
    if f"rate_pred_{dom}" in df_run.columns:
        ax.plot(t, df_run[f"rate_pred_{dom}"], color="#c1440e", lw=1.3,
                label="physics + GRU")
    ax.set_title(f"wear rate, dominant mechanism ({dom})")
    ax.set_xlabel("operating hours"); ax.set_ylabel(r"$d\phi/dt$  [1/h]")
    ax.legend(fontsize=7.5)

    fig.suptitle(f"Degradation modelling - {run_id}", fontsize=11)
    fig.tight_layout()
    return _save(fig, name)


# --------------------------------------------------------------------------- #
# 3 / 4  RUL
# --------------------------------------------------------------------------- #

def plot_rul_trajectories(preds: pd.DataFrame, run_ids: Sequence[str],
                          name: str = "rul_trajectories.png") -> str:
    """Predicted RUL with its uncertainty band against the truth, per run."""
    n = len(run_ids)
    ncol = min(3, n)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.7 * ncol, 3.4 * nrow),
                             squeeze=False)
    for ax, rid in zip(axes.ravel(), run_ids):
        g = preds[preds.run_id == rid].sort_values("operating_hours")
        t = g.operating_hours.to_numpy()
        ax.plot(t, g.rul_true, color="#1b3a6b", lw=2.0, label="true RUL", zorder=3)
        ax.plot(t, g.rul_pred, color="#c1440e", lw=1.6, label="predicted (median)",
                zorder=2)
        if "rul_p10" in g.columns:
            ax.fill_between(t, g.rul_p10, g.rul_p90, color="#c1440e", alpha=0.18,
                            lw=0, label="P10-P90", zorder=1)
        ax.set_title(rid, fontsize=9)
        ax.set_xlabel("operating hours"); ax.set_ylabel("RUL (h)")
        ax.set_ylim(bottom=0)
    for ax in axes.ravel()[n:]:
        ax.set_visible(False)
    axes[0, 0].legend(fontsize=7.5)
    fig.suptitle("Probabilistic RUL against ground truth", fontsize=11)
    fig.tight_layout()
    return _save(fig, name)


def plot_rul_scatter(preds: Dict[str, pd.DataFrame],
                     name: str = "rul_scatter.png") -> str:
    """Predicted vs true RUL for every model, on the same axes limits."""
    n = len(preds)
    fig, axes = plt.subplots(1, n, figsize=(3.6 * n, 3.6), squeeze=False)
    rng = np.random.default_rng(0)
    for ax, (label, df) in zip(axes.ravel(), preds.items()):
        x = df.rul_true.to_numpy(); y = df.rul_pred.to_numpy()
        sub = rng.choice(len(x), size=min(5000, len(x)), replace=False)
        ax.scatter(x[sub], y[sub], s=2.5, alpha=0.14, color="#1b3a6b", lw=0)
        hi = float(np.nanpercentile(np.concatenate([x, y]), 99.5))
        ax.plot([0, hi], [0, hi], color="#c1440e", ls="--", lw=1.1)
        mae = float(np.mean(np.abs(y - x)))
        ax.set_title(f"{label}\nMAE = {mae:.1f} h", fontsize=9)
        ax.set_xlabel("true RUL (h)"); ax.set_ylabel("predicted RUL (h)")
        ax.set_xlim(0, hi); ax.set_ylim(0, hi)
    fig.tight_layout()
    return _save(fig, name)


# --------------------------------------------------------------------------- #
# 5 / 6  twin and residuals
# --------------------------------------------------------------------------- #

def plot_twin_vs_measured(df_run: pd.DataFrame, run_id: str,
                          name: str = "twin_vs_measured.png") -> str:
    """Digital Twin prediction against measurement over a slice of one run."""
    chans = [("rpm", "rpm_pred", "RPM"), ("cht", "cht_pred", "CHT (C)"),
             ("egt", "egt_pred", "EGT (C)"),
             ("oil_pressure", "oil_pressure_pred", "oil pressure (bar)"),
             ("oil_temperature", "oil_temperature_pred", "oil temp (C)"),
             ("fuel_flow", "fuel_flow_pred", "fuel flow (L/h)")]
    fig, axes = plt.subplots(3, 2, figsize=(13.0, 7.6), sharex=True)
    n = len(df_run)
    sl = slice(0, min(n, 600))
    t = df_run.operating_hours.to_numpy()[sl]
    for ax, (m, p, lab) in zip(axes.ravel(), chans):
        ax.plot(t, df_run[m].to_numpy()[sl], color="#1b3a6b", lw=1.0,
                label="measured")
        ax.plot(t, df_run[p].to_numpy()[sl], color="#c1440e", lw=1.0,
                label="twin prediction", alpha=0.85)
        ax.set_ylabel(lab, fontsize=8)
    axes[0, 0].legend(fontsize=7.5)
    for ax in axes[-1]:
        ax.set_xlabel("operating hours")
    fig.suptitle(f"Digital Twin vs measured telemetry (first 150 h) - {run_id}",
                 fontsize=11)
    fig.tight_layout()
    return _save(fig, name)


def plot_residual_trajectories(df_run: pd.DataFrame, run_id: str,
                               name: str = "residual_trajectories.png") -> str:
    """Normalised residuals over life: the raw degradation signal."""
    zc = [c for c in df_run.columns if c.endswith("_z") and not c.endswith("__z")]
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.2))
    t = df_run.operating_hours.to_numpy()

    ax = axes[0]
    for c in zc:
        roll = pd.Series(df_run[c]).rolling(60, min_periods=8).median()
        ax.plot(t, roll, lw=1.3, label=c.replace("_z", ""))
    ax.axhline(0, color="k", lw=0.8)
    for lv in (-3, 3):
        ax.axhline(lv, color="k", ls=":", lw=0.8)
    ax.set_title("normalised residuals (15 h rolling median)")
    ax.set_xlabel("operating hours"); ax.set_ylabel("z")
    ax.legend(fontsize=6.5, ncol=2)

    ax = axes[1]
    ax.plot(t, df_run.true_health_index, color="#1b3a6b", lw=1.8,
            label="true health index")
    if "z_sq_sum_roll" in df_run.columns:
        ax2 = ax.twinx()
        ax2.plot(t, df_run.z_sq_sum_roll, color="#c1440e", lw=1.3,
                 label=r"$\sum z^2$ (rolling)")
        ax2.set_ylabel(r"$\sum z_i^2$", color="#c1440e")
        ax2.grid(False)
    ax.set_title("health index vs aggregate residual energy")
    ax.set_xlabel("operating hours"); ax.set_ylabel("health index")
    ax.legend(fontsize=7.5, loc="lower left")

    fig.suptitle(f"Residual behaviour - {run_id}", fontsize=11)
    fig.tight_layout()
    return _save(fig, name)


# --------------------------------------------------------------------------- #
# Calibration
# --------------------------------------------------------------------------- #

def plot_calibration(cal: pd.DataFrame, pit: np.ndarray,
                     name: str = "calibration.png") -> str:
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.1))
    ax = axes[0]
    ax.plot([0, 1], [0, 1], color="k", ls="--", lw=1.0, label="perfect")
    ax.plot(cal.nominal, cal.empirical, "o-", color="#c1440e", lw=1.6,
            label="observed")
    ax.set_xlabel("nominal interval level"); ax.set_ylabel("empirical coverage")
    ax.set_title("Prediction-interval calibration")
    ax.legend(fontsize=8)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    ax = axes[1]
    ax.hist(pit, bins=20, range=(0, 1), color="#1b3a6b", alpha=0.85,
            edgecolor="white")
    ax.axhline(len(pit) / 20.0, color="#c1440e", ls="--", lw=1.2,
               label="uniform (calibrated)")
    ax.set_xlabel("PIT value"); ax.set_ylabel("count")
    ax.set_title("Probability integral transform\n(U-shape = intervals too narrow)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return _save(fig, name)


def plot_early_prediction(tables: Dict[str, pd.DataFrame],
                          name: str = "early_prediction.png") -> str:
    """RUL error as a function of remaining life, per model."""
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    order = None
    for label, tab in tables.items():
        tab = tab.sort_values("life_fraction_lo")
        if order is None:
            order = tab.band.tolist()
        ax.plot(range(len(tab)), tab.mae, "o-", lw=1.7, label=label)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=18, ha="right")
    ax.set_ylabel("MAE (h)")
    ax.set_title("Early-prediction capability: RUL error vs remaining life")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return _save(fig, name)


def plot_model_comparison(table: pd.DataFrame, name: str = "model_comparison.png") -> str:
    """Bar chart of MAE by model and test partition."""
    piv = table.pivot(index="model", columns="split", values="mae")
    fig, ax = plt.subplots(figsize=(9.6, 4.4))
    piv.plot(kind="bar", ax=ax, width=0.78,
             color=["#1b3a6b", "#c1440e", "#2e7d32", "#8e6c00"][:piv.shape[1]])
    ax.set_ylabel("RUL MAE (h)")
    ax.set_xlabel("")
    ax.set_title("RUL accuracy by model and test partition (lower is better)")
    ax.legend(fontsize=8, title="")
    plt.setp(ax.get_xticklabels(), rotation=16, ha="right")
    fig.tight_layout()
    return _save(fig, name)


def plot_monotonicity(preds: pd.DataFrame, name: str = "monotonicity.png") -> str:
    """Distribution of step-to-step change in predicted RUL."""
    d = []
    for _, g in preds.groupby("run_id"):
        g = g.sort_values("operating_hours")
        dt = np.diff(g.operating_hours.to_numpy())
        dr = np.diff(g.rul_pred.to_numpy())
        ok = dt > 0
        d.extend((dr[ok] + dt[ok]).tolist())     # 0 means perfect countdown
    d = np.array(d)
    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    ax.hist(np.clip(d, -60, 60), bins=70, color="#1b3a6b", alpha=0.85,
            edgecolor="white")
    ax.axvline(0, color="#c1440e", lw=1.5, ls="--",
               label="ideal (RUL falls exactly with elapsed time)")
    ax.set_xlabel(r"$\Delta$RUL + $\Delta$t   (h)")
    ax.set_ylabel("count")
    ax.set_title(f"RUL countdown behaviour\n"
                 f"{100*np.mean(d > 5):.1f}% of steps increase RUL by more than 5 h")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return _save(fig, name)
