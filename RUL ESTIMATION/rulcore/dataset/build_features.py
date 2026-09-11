"""
build_features.py
=================
Turn raw run-to-failure trajectories into the modelling dataset (Deliverable B).

Responsibilities
----------------
  1. Split by COMPLETE RUN, stratified by mechanism and rate archetype (Part 15).
  2. Run the Digital Twin over every run and form residuals (Parts 4, 5).
  3. Fit the healthy residual dispersion sigma on TRAINING runs only, and only on
     early-life snapshots. Fitting sigma on all data - or on the test set - would
     be a subtle but real leak, because sigma would then encode how degraded
     engines behave.
  4. Add causal temporal features (rolling level, slope, dispersion of the
     normalised residuals). Every one of them is computed with a backward-looking
     window inside a single run, so no future information and no cross-run
     information ever enters a row.
  5. Write per-split feature tables plus a feature manifest classifying every
     column (Part 17).

SPLIT POLICY
------------
  MAIN     -> train / val / test, 70/15/15 by run
  STRESS   -> test_stress only        (hot-and-high, outside training support)
  XENG     -> test_cross_engine only  (different engine builds and biases)

No run appears in more than one partition, and no window ever spans two runs.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from ..config import (DATA_DIR, FEATURE_DIR, HEALTH_PARAMS, RANDOM_SEED,
                      RAW_DIR, SPLIT_FRACTIONS)
from ..estimation.twin import RESIDUAL_CHANNELS, DigitalTwin, SigmaModel

# Snapshots below this life fraction are treated as "known healthy" for the
# purpose of estimating sigma. Early life is when nothing has degraded yet.
HEALTHY_LIFE_FRACTION = 0.08

# Rolling windows in snapshots (0.25 h each): 5 h, 15 h, 40 h of history.
ROLL_SHORT, ROLL_MED, ROLL_LONG = 20, 60, 160

Z_CHANNELS = [f"{c}_z" for c in RESIDUAL_CHANNELS] + ["injection_command_z"]


# --------------------------------------------------------------------------- #
# Splits
# --------------------------------------------------------------------------- #

def assign_splits(index: pd.DataFrame, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Stratified split over complete runs."""
    rng = np.random.default_rng(seed)
    index = index.copy()
    index["split"] = ""

    index.loc[index.population == "STRESS", "split"] = "test_stress"
    index.loc[index.population == "XENG", "split"] = "test_cross_engine"

    main = index[index.population == "MAIN"]
    for _, grp in main.groupby(["mechanism", "rate_archetype"]):
        ids = grp.run_id.to_numpy().copy()
        rng.shuffle(ids)
        n = len(ids)
        n_tr = max(1, int(round(SPLIT_FRACTIONS["train"] * n)))
        n_va = max(1, int(round(SPLIT_FRACTIONS["val"] * n))) if n >= 3 else 0
        if n_tr + n_va >= n and n >= 3:
            n_tr = n - 2
            n_va = 1
        parts = (["train"] * n_tr + ["val"] * n_va +
                 ["test"] * max(n - n_tr - n_va, 0))
        for rid, part in zip(ids, parts):
            index.loc[index.run_id == rid, "split"] = part

    return index


# --------------------------------------------------------------------------- #
# Per-run feature construction
# --------------------------------------------------------------------------- #

def _rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Backward-looking temporal features for a SINGLE run.

    Three views of each normalised residual:
      level  - rolling median, robust to the injected outliers
      slope  - least-squares trend per operating hour, the primary degradation
               signal (a drifting residual matters far more than a large one)
      spread - rolling IQR, which rises when combustion becomes unstable
    """
    out = {}
    x = np.arange(ROLL_MED, dtype=float)
    x = x - x.mean()
    denom = float((x ** 2).sum())

    for ch in Z_CHANNELS:
        if ch not in df.columns:
            continue
        s = df[ch]
        out[f"{ch}_roll{ROLL_SHORT}"] = s.rolling(ROLL_SHORT, min_periods=3).median()
        out[f"{ch}_roll{ROLL_LONG}"] = s.rolling(ROLL_LONG, min_periods=8).median()
        out[f"{ch}_iqr{ROLL_MED}"] = (s.rolling(ROLL_MED, min_periods=8).quantile(0.75)
                                      - s.rolling(ROLL_MED, min_periods=8).quantile(0.25))
        # slope per operating hour (window spans ROLL_MED * 0.25 h)
        cov = s.rolling(ROLL_MED, min_periods=8).apply(
            lambda v: float(np.dot(v - v.mean(), (np.arange(len(v)) - (len(v) - 1) / 2.0))),
            raw=True)
        out[f"{ch}_slope"] = cov / denom / 0.25

    res = pd.DataFrame(out, index=df.index)
    return res


def build_run_features(df: pd.DataFrame, twin: DigitalTwin) -> pd.DataFrame:
    """Twin prediction + raw residuals for one run (z-scores added later)."""
    pred = twin.predict(df)
    resid = twin.residuals(df, pred)
    return pd.concat([df.reset_index(drop=True),
                      pred.reset_index(drop=True),
                      resid.reset_index(drop=True)], axis=1)


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #

def build_dataset(index: pd.DataFrame, verbose: bool = True) -> Tuple[Dict[str, pd.DataFrame], SigmaModel]:
    twin = DigitalTwin()
    frames: Dict[str, pd.DataFrame] = {}

    if verbose:
        print("  computing twin predictions and residuals ...", flush=True)
    per_run: Dict[str, pd.DataFrame] = {}
    for i, row in index.iterrows():
        raw = pd.read_parquet(os.path.join(RAW_DIR, f"{row.run_id}.parquet"))
        per_run[row.run_id] = build_run_features(raw, twin)
        if verbose and (i + 1) % 50 == 0:
            print(f"    {i+1}/{len(index)}", flush=True)

    # ---- sigma from TRAINING runs, early life only ------------------------ #
    train_ids = index.loc[index.split == "train", "run_id"].tolist()
    healthy = pd.concat(
        [per_run[r][per_run[r].life_fraction <= HEALTHY_LIFE_FRACTION] for r in train_ids],
        ignore_index=True)
    if verbose:
        print(f"  fitting sigma on {len(healthy):,} healthy training snapshots "
              f"from {len(train_ids)} runs", flush=True)
    sigma = SigmaModel().fit(healthy)

    # ---- normalise, add temporal features --------------------------------- #
    if verbose:
        print("  normalising residuals and building temporal features ...", flush=True)
    for rid, d in per_run.items():
        z = sigma.normalise(d)
        d = pd.concat([d, z], axis=1)
        d = pd.concat([d, _rolling_features(d)], axis=1)

        # Aggregate physics-informed health scores available without any label.
        zcols = [c for c in Z_CHANNELS if c in d.columns]
        zmat = d[zcols].to_numpy()
        d["z_absmax"] = np.nanmax(np.abs(zmat), axis=1)
        d["z_sq_sum"] = np.nansum(zmat ** 2, axis=1)
        d["z_sq_sum_roll"] = pd.Series(d["z_sq_sum"]).rolling(ROLL_MED, min_periods=8).median()
        per_run[rid] = d

    for split in index.split.unique():
        ids = index.loc[index.split == split, "run_id"].tolist()
        frames[split] = pd.concat([per_run[r] for r in ids], ignore_index=True)

    return frames, sigma


def feature_manifest(columns: List[str]) -> pd.DataFrame:
    """Classify every column for the leakage audit (Part 17).

    Three classes:
      REAL_ENGINE     measurable or commanded on the actual aircraft
      DIGITAL_TWIN    computed by the twin from REAL_ENGINE inputs alone
      GROUND_TRUTH    simulation truth - labels and diagnostics only
      BOOKKEEPING     identifiers and split metadata, never a model input
    """
    rows = []
    real_engine = {
        "operating_hours", "sortie_index", "mission_phase", "snapshot_index",
        "altitude_ft", "ambient_temperature_c", "ambient_pressure_kpa", "humidity",
        "throttle", "engine_load", "injection_command", "injection_timing_deg",
        "load_factor", "airspeed_factor",
        "rpm", "cht", "egt", "oil_pressure", "oil_temperature", "fuel_flow",
        "battery_voltage", "alternator_current", "manifold_pressure",
    }
    for c in columns:
        if c.startswith("true_") or c in ("RUL_hours", "RUL_hours_capped", "life_fraction"):
            cls = "GROUND_TRUTH"
        elif c in ("run_id", "split", "population"):
            cls = "BOOKKEEPING"
        elif c in real_engine or c.startswith("vibration_") or c.endswith("_window_std"):
            cls = "REAL_ENGINE"
        elif (c.endswith("_pred") or c.endswith("_residual") or c.endswith("_z")
              or "_z_roll" in c or "_z_iqr" in c or c.endswith("_slope")
              or c.startswith("z_") or c.startswith("est_")
              or c in ("torque_pred", "power_pred_w", "bsfc_pred", "afr_pred")):
            cls = "DIGITAL_TWIN"
        else:
            cls = "UNCLASSIFIED"
        rows.append({"feature": c, "availability": cls})
    return pd.DataFrame(rows)


def save_dataset(frames: Dict[str, pd.DataFrame], sigma: SigmaModel,
                 index: pd.DataFrame, verbose: bool = True):
    os.makedirs(FEATURE_DIR, exist_ok=True)
    summary = {}
    for split, df in frames.items():
        path = os.path.join(FEATURE_DIR, f"{split}.parquet")
        df.to_parquet(path, index=False)
        summary[split] = {"runs": int(df.run_id.nunique()), "rows": int(len(df))}
        if verbose:
            print(f"    {split:20s} {df.run_id.nunique():4d} runs  {len(df):8,d} rows"
                  f"  -> {os.path.basename(path)}", flush=True)

    with open(os.path.join(FEATURE_DIR, "sigma_model.json"), "w", encoding="utf-8") as f:
        json.dump(sigma.to_dict(), f, indent=1)

    any_df = next(iter(frames.values()))
    manifest = feature_manifest(list(any_df.columns))
    manifest.to_csv(os.path.join(FEATURE_DIR, "feature_manifest.csv"), index=False)

    index.to_csv(os.path.join(DATA_DIR, "run_index_split.csv"), index=False)

    with open(os.path.join(FEATURE_DIR, "dataset_summary.json"), "w", encoding="utf-8") as f:
        json.dump({
            "splits": summary,
            "healthy_life_fraction_for_sigma": HEALTHY_LIFE_FRACTION,
            "rolling_windows_snapshots": {"short": ROLL_SHORT, "medium": ROLL_MED,
                                          "long": ROLL_LONG},
            "n_features_by_availability":
                manifest.availability.value_counts().to_dict(),
        }, f, indent=1)
    return manifest
