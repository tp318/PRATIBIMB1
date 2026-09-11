"""
metrics.py
==========
RUL evaluation metrics (Part 18) plus the monotonicity and sanity checks
(Part 20).

Design notes that affect how the numbers should be read:

  * EFFECTIVE SAMPLE SIZE IS RUNS, NOT WINDOWS. Windows within a run share an
    engine and a realised life, so a standard error computed over windows is
    optimistic by roughly sqrt(windows per run). Every aggregate here is
    therefore also reported per run, and confidence intervals are computed by
    bootstrapping OVER RUNS.

  * R^2 IS REPORTED BUT NOT TRUSTED ALONE. On a run-to-failure dataset RUL is
    dominated by a near-deterministic countdown within each run, so even a
    useless model that has merely learned "RUL decreases with age" scores a high
    R^2. The metrics that discriminate are the error near end of life and the
    early-prediction error at fixed life fractions.

  * ASYMMETRY MATTERS. Predicting more life than remains is an airworthiness
    problem; predicting less is a cost problem. Both a signed bias and the
    NASA-style asymmetric scoring function are reported.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Point metrics
# --------------------------------------------------------------------------- #

def point_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ok = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[ok], y_pred[ok]
    if y_true.size == 0:
        return {k: float("nan") for k in
                ("mae", "rmse", "medae", "r2", "bias", "mape", "smape", "nasa_score")}

    err = y_pred - y_true
    ss_res = float((err ** 2).sum())
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())

    # Percentage error only where the denominator is meaningful; MAPE explodes
    # as RUL approaches zero, which is exactly where the model matters most, so
    # it is computed on RUL > 10 h and sMAPE is given as the robust companion.
    big = y_true > 10.0
    mape = float(np.mean(np.abs(err[big] / y_true[big])) * 100.0) if big.any() else float("nan")
    denom = np.abs(y_true) + np.abs(y_pred)
    smape = float(np.mean(2.0 * np.abs(err) / np.maximum(denom, 1e-9)) * 100.0)

    # NASA asymmetric prognostic score: late predictions (over-estimating life)
    # are penalised harder than early ones.
    d = err
    score = float(np.sum(np.where(d < 0, np.exp(-d / 13.0) - 1.0, np.exp(d / 10.0) - 1.0)))

    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "medae": float(np.median(np.abs(err))),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        "bias": float(np.mean(err)),
        "mape": mape,
        "smape": smape,
        "nasa_score": score,
        "n": int(y_true.size),
    }


def per_run_metrics(df: pd.DataFrame, true_col: str = "rul_true",
                    pred_col: str = "rul_pred") -> pd.DataFrame:
    rows = []
    for rid, g in df.groupby("run_id"):
        m = point_metrics(g[true_col].to_numpy(), g[pred_col].to_numpy())
        m["run_id"] = rid
        rows.append(m)
    return pd.DataFrame(rows)


def bootstrap_ci_over_runs(df: pd.DataFrame, metric: str = "mae",
                           true_col: str = "rul_true", pred_col: str = "rul_pred",
                           n_boot: int = 2000, seed: int = 0) -> Dict[str, float]:
    """Bootstrap a metric by RESAMPLING RUNS, not rows.

    This is the only honest way to put an interval on these numbers: the
    independent units are engines.
    """
    rng = np.random.default_rng(seed)
    runs = df.run_id.unique()
    groups = {r: g for r, g in df.groupby("run_id")}
    stats = []
    for _ in range(n_boot):
        pick = rng.choice(runs, size=len(runs), replace=True)
        sub = pd.concat([groups[r] for r in pick], ignore_index=True)
        stats.append(point_metrics(sub[true_col].to_numpy(), sub[pred_col].to_numpy())[metric])
    stats = np.array(stats, dtype=float)
    return {"mean": float(np.nanmean(stats)),
            "lo95": float(np.nanpercentile(stats, 2.5)),
            "hi95": float(np.nanpercentile(stats, 97.5))}


# --------------------------------------------------------------------------- #
# Uncertainty / calibration
# --------------------------------------------------------------------------- #

def interval_metrics(y_true: np.ndarray, lo: np.ndarray, hi: np.ndarray,
                     nominal: float = 0.80) -> Dict[str, float]:
    """Prediction-interval coverage, width and the coverage error."""
    y_true = np.asarray(y_true, dtype=float)
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    ok = np.isfinite(y_true) & np.isfinite(lo) & np.isfinite(hi)
    y_true, lo, hi = y_true[ok], lo[ok], hi[ok]
    if y_true.size == 0:
        return {"picp": float("nan"), "mpiw": float("nan"),
                "coverage_error": float("nan"), "nmpiw": float("nan"),
                "winkler": float("nan")}

    inside = (y_true >= lo) & (y_true <= hi)
    picp = float(inside.mean())
    width = hi - lo
    mpiw = float(width.mean())

    # Winkler / interval score: width plus a penalty for every miss, scaled by
    # the nominal level. A model can always get perfect coverage with infinite
    # intervals; this is the metric that stops that being rewarded.
    alpha = 1.0 - nominal
    pen_lo = 2.0 / alpha * np.maximum(lo - y_true, 0.0)
    pen_hi = 2.0 / alpha * np.maximum(y_true - hi, 0.0)
    winkler = float(np.mean(width + pen_lo + pen_hi))

    rng_true = float(np.ptp(y_true)) if y_true.size > 1 else 1.0
    return {
        "picp": picp,
        "mpiw": mpiw,
        "nmpiw": mpiw / max(rng_true, 1e-9),
        "coverage_error": picp - nominal,
        "winkler": winkler,
        "n": int(y_true.size),
    }


def calibration_curve(y_true: np.ndarray, samples: Sequence[np.ndarray],
                      levels: Iterable[float] = (0.1, 0.2, 0.3, 0.4, 0.5,
                                                 0.6, 0.7, 0.8, 0.9, 0.95)
                      ) -> pd.DataFrame:
    """Empirical coverage of central intervals at several nominal levels.

    A perfectly calibrated probabilistic forecast lies on the diagonal. Any
    departure is directly readable as over- or under-confidence.
    """
    rows = []
    for lv in levels:
        lo_q, hi_q = (1.0 - lv) / 2.0, 1.0 - (1.0 - lv) / 2.0
        inside = []
        for yt, s in zip(y_true, samples):
            s = np.asarray(s, dtype=float)
            lo, hi = np.percentile(s, [100 * lo_q, 100 * hi_q])
            inside.append(bool(lo <= yt <= hi))
        rows.append({"nominal": lv, "empirical": float(np.mean(inside)),
                     "n": len(inside)})
    return pd.DataFrame(rows)


def pit_values(y_true: np.ndarray, samples: Sequence[np.ndarray]) -> np.ndarray:
    """Probability integral transform: the predictive CDF evaluated at the truth.

    Under a perfectly calibrated forecast these are uniform on [0, 1]. A
    U-shaped histogram means the intervals are too narrow; a hump in the middle
    means they are too wide.
    """
    out = []
    for yt, s in zip(y_true, samples):
        s = np.asarray(s, dtype=float)
        out.append(float((s <= yt).mean()))
    return np.array(out)


def crps_from_samples(y_true: np.ndarray, samples: Sequence[np.ndarray]) -> float:
    """Continuous ranked probability score, estimated from samples.

    Proper scoring rule: rewards a forecast for being both sharp and calibrated,
    so unlike coverage it cannot be gamed by widening the interval.
    """
    vals = []
    for yt, s in zip(y_true, samples):
        s = np.sort(np.asarray(s, dtype=float))
        n = len(s)
        term1 = np.mean(np.abs(s - yt))
        # E|X - X'| via the sorted-sample identity, O(n) instead of O(n^2)
        i = np.arange(1, n + 1)
        term2 = 2.0 / (n * n) * np.sum((2 * i - n - 1) * s)
        vals.append(term1 - 0.5 * term2)
    return float(np.mean(vals))


# --------------------------------------------------------------------------- #
# Early prediction (Part 18)
# --------------------------------------------------------------------------- #

def early_prediction_table(df: pd.DataFrame,
                           life_bands: Sequence[tuple] = ((0.8, 1.01, "<=20% life left"),
                                                          (0.7, 0.8, "20-30% left"),
                                                          (0.5, 0.7, "30-50% left"),
                                                          (0.25, 0.5, "50-75% left"),
                                                          (0.0, 0.25, ">75% left")),
                           true_col: str = "rul_true",
                           pred_col: str = "rul_pred") -> pd.DataFrame:
    """Error as a function of how far through its life the engine is.

    `life_fraction` = elapsed / total, so 0.8 means 20% of life remains. This is
    the table that answers "how early can the system say something useful",
    which a single pooled MAE cannot.
    """
    rows = []
    for lo, hi, label in life_bands:
        sub = df[(df.life_fraction >= lo) & (df.life_fraction < hi)]
        if len(sub) == 0:
            continue
        m = point_metrics(sub[true_col].to_numpy(), sub[pred_col].to_numpy())
        m["band"] = label
        m["life_fraction_lo"] = lo
        m["life_fraction_hi"] = hi
        m["runs"] = int(sub.run_id.nunique())
        if "rul_p10" in sub.columns and "rul_p90" in sub.columns:
            m.update({f"iv_{k}": v for k, v in
                      interval_metrics(sub[true_col].to_numpy(),
                                       sub.rul_p10.to_numpy(),
                                       sub.rul_p90.to_numpy()).items()})
        rows.append(m)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Monotonicity and sanity (Part 20)
# --------------------------------------------------------------------------- #

def monotonicity_report(df: pd.DataFrame, pred_col: str = "rul_pred",
                        tolerance_h: float = 5.0) -> Dict[str, float]:
    """How often does predicted RUL go UP as the engine gets older?

    RUL should fall by dt between consecutive predictions. Small increases are
    noise; the tolerance separates that from a genuine failure of the model to
    behave physically. Reported both as a fraction of steps and as the worst
    single increase, because one 100-hour jump is operationally far worse than
    many 1-hour wobbles.
    """
    viol, total, worst, jumps = 0, 0, 0.0, []
    spearman = []
    for _, g in df.groupby("run_id"):
        g = g.sort_values("operating_hours")
        p = g[pred_col].to_numpy()
        if len(p) < 3:
            continue
        d = np.diff(p)
        total += len(d)
        v = d > tolerance_h
        viol += int(v.sum())
        if v.any():
            jumps.extend(d[v].tolist())
            worst = max(worst, float(d.max()))
        # rank correlation between predicted RUL and time: should be strongly negative
        t = g.operating_hours.to_numpy()
        if len(t) > 5 and np.std(p) > 1e-9:
            rt = pd.Series(t).rank().to_numpy()
            rp = pd.Series(p).rank().to_numpy()
            spearman.append(float(np.corrcoef(rt, rp)[0, 1]))

    return {
        "violation_rate": float(viol / max(total, 1)),
        "n_violations": int(viol),
        "n_steps": int(total),
        "worst_increase_h": float(worst),
        "mean_violation_h": float(np.mean(jumps)) if jumps else 0.0,
        "median_spearman_time_vs_rul": float(np.median(spearman)) if spearman else float("nan"),
        "runs_with_positive_trend": int(sum(1 for s in spearman if s > 0)),
        "n_runs": int(df.run_id.nunique()),
    }


def condition_confounding_check(df: pd.DataFrame, pred_col: str = "rul_pred",
                                true_col: str = "rul_true") -> pd.DataFrame:
    """Does the model read hot or high operating conditions as degradation?

    Regresses the RUL ERROR on ambient temperature and altitude. A model that
    confuses environment with health shows a systematic negative error (it
    under-predicts life) as conditions get harsher. Because the twin already
    accounts for the environment, the slope should be near zero; a large slope
    is direct evidence the physics layer is being bypassed.
    """
    rows = []
    err = df[pred_col].to_numpy() - df[true_col].to_numpy()
    for var, label in (("ambient_temperature_c", "ambient temperature (C)"),
                       ("altitude_ft", "altitude (kft)")):
        x = df[var].to_numpy()
        if var == "altitude_ft":
            x = x / 1000.0
        ok = np.isfinite(x) & np.isfinite(err)
        if ok.sum() < 20:
            continue
        A = np.stack([x[ok], np.ones(ok.sum())], axis=1)
        slope, intercept = np.linalg.lstsq(A, err[ok], rcond=None)[0]
        r = float(np.corrcoef(x[ok], err[ok])[0, 1])
        rows.append({"variable": label, "slope_h_per_unit": float(slope),
                     "correlation": r, "n": int(ok.sum())})
    return pd.DataFrame(rows)
