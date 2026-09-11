"""Stage 6: evaluate every model on every test partition (Deliverable H).

    python scripts/evaluate.py [--checkpoint-stride 40] [--particles 600]

Produces:
    outputs/reports/rul_predictions_<split>.parquet
    outputs/reports/model_comparison.csv
    outputs/reports/early_prediction_<model>.csv
    outputs/reports/calibration.csv, monotonicity.json, confounding.csv
    outputs/figures/*.png
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from rulcore.config import (ARTIFACT_DIR, FEATURE_DIR, FIGURE_DIR, HEALTH_PARAMS,
                            REPORT_DIR, RUL_CAP_H, SNAPSHOT_HOURS)
from rulcore.models.baselines import physics_only_rul, window_summary
from rulcore.models.degradation_model import PHI_RANGE
from rulcore.models.rul_propagation import HealthIndexSurrogate, propagate_rul
from rulcore.models.windows import (FEATURE_SETS, StandardScaler3D,
                                    available_features, build_sequences)

TEST_SPLITS = ["test", "test_stress", "test_cross_engine"]
BASELINE_SKIP, BASELINE_SNAPSHOTS = 12, 40

MODEL_LABELS = {
    "hybrid": "Hybrid (physics+UKF+GRU+MC)",
    "physics_only": "B4 Physics only",
    "xgb": "B1 XGBoost telemetry",
    "gru_telem": "B2 GRU telemetry",
    "gru_health": "B3 GRU health",
}


# --------------------------------------------------------------------------- #
# Hybrid + physics-only predictions (per run, parallelised)
# --------------------------------------------------------------------------- #

def _hybrid_for_run(args):
    split, rid, stride, seq_len, n_particles, gru_std = args
    import torch

    from rulcore.models.gru_correction import DegradationRateGRU

    sur = HealthIndexSurrogate.from_dict(
        json.load(open(os.path.join(ARTIFACT_DIR, "health_index_surrogate.json"),
                       encoding="utf-8")))

    d = pd.read_parquet(os.path.join(FEATURE_DIR, "_runs", f"{split}__{rid}.parquet"))
    d = d.sort_values("snapshot_index").reset_index(drop=True)
    n = len(d)

    phi = np.stack([d[f"phi_{k}"].to_numpy() for k in HEALTH_PARAMS], axis=1)
    phi_std = np.stack([d[f"est_{k}_std"].to_numpy() / abs(PHI_RANGE[k])
                        for k in HEALTH_PARAMS], axis=1)
    coeffs = np.stack([d[f"a_{k}"].to_numpy() for k in HEALTH_PARAMS], axis=1)
    stress = np.stack([d[f"stress_{k}"].to_numpy() for k in HEALTH_PARAMS], axis=1)

    # Per-engine commissioning baseline from the filter's own early-life output.
    if n > BASELINE_SNAPSHOTS:
        baseline = sur.baseline_from_phi(
            np.median(phi[BASELINE_SKIP:BASELINE_SNAPSHOTS], axis=0))
    else:
        baseline = None

    ends = np.arange(max(seq_len, BASELINE_SNAPSHOTS) - 1, n, stride, dtype=int)
    if len(ends) == 0:
        return pd.DataFrame()

    # ---- batched GRU multipliers ------------------------------------------ #
    mults = np.ones((len(ends), len(HEALTH_PARAMS)))
    rp = os.path.join(ARTIFACT_DIR, "gru_rate_correction.pt")
    if os.path.exists(rp):
        ck = torch.load(rp, map_location="cpu", weights_only=False)
        feats, sl = ck["features"], ck["seq_len"]
        scaler = StandardScaler3D.from_dict(ck["scaler"])
        model = DegradationRateGRU(n_features=len(feats))
        model.load_state_dict(ck["state_dict"])
        model.eval()
        F = np.nan_to_num(d[feats].to_numpy(dtype=np.float32), nan=0.0,
                          posinf=0.0, neginf=0.0)
        rphys = d[[f"rate_phys_{k}" for k in HEALTH_PARAMS]].to_numpy(dtype=np.float32)
        idx = ends[:, None] - np.arange(sl - 1, -1, -1)[None, :]
        idx = np.clip(idx, 0, n - 1)
        X = scaler.transform(F[idx])
        with torch.no_grad():
            out = model(torch.from_numpy(X), torch.from_numpy(rphys[ends]))
        mults = out["mult"].numpy()

    rows = []
    for j, i in enumerate(ends):
        lo = max(0, i - 400)
        res = propagate_rul(phi[i], np.diag(np.maximum(phi_std[i], 1e-4) ** 2),
                            coeffs[i], stress[lo:i + 1], sur,
                            gru_multiplier=mults[j], baseline=baseline,
                            gru_rel_sigma=gru_std, n_particles=n_particles,
                            seed=int(i))
        phys = physics_only_rul(phi[i], coeffs[i], stress[lo:i + 1], sur,
                                baseline=baseline)
        rows.append({
            "run_id": rid, "split": split, "end_row": int(i),
            "operating_hours": float(d.operating_hours.iloc[i]),
            "life_fraction": float(d.life_fraction.iloc[i]),
            "rul_true": float(d.RUL_hours.iloc[i]),
            "rul_true_capped": float(d.RUL_hours_capped.iloc[i]),
            "hybrid": res["rul_median_hours"],
            "hybrid_p10": res["rul_p10_hours"],
            "hybrid_p90": res["rul_p90_hours"],
            "hybrid_censored": res["censored_fraction"],
            "p_below_100h": res["probability_rul_below_100h"],
            "health_index_est": res["health_index_now"],
            "physics_only": phys,
            "ambient_temperature_c": float(d.ambient_temperature_c.iloc[i]),
            "altitude_ft": float(d.altitude_ft.iloc[i]),
            "rul_samples": res["rul_samples"].astype(np.float32),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Learned baselines (batched over a whole split)
# --------------------------------------------------------------------------- #

def _baseline_predictions(split_df: pd.DataFrame, ends_lookup: dict,
                          seq_len: int) -> pd.DataFrame:
    """Predict B1/B2/B3 at exactly the checkpoints the hybrid used."""
    import torch

    from rulcore.models.gru_correction import DirectRULGRU

    split_df = split_df.sort_values(["run_id", "snapshot_index"]).reset_index(drop=True)
    offsets, absolute, keys = {}, [], []
    off = 0
    for rid, g in split_df.groupby("run_id", sort=False):
        offsets[rid] = off
        for e in ends_lookup.get(rid, []):
            absolute.append(off + e)
            keys.append((rid, e))
        off += len(g)
    if not absolute:
        return pd.DataFrame()
    absolute = np.array(absolute, dtype=int)

    out = pd.DataFrame({"run_id": [k[0] for k in keys],
                        "end_row": [k[1] for k in keys]})

    def seqs(fset):
        f = available_features(split_df, FEATURE_SETS[fset])
        M = np.nan_to_num(split_df[f].to_numpy(dtype=np.float32), nan=0.0,
                          posinf=0.0, neginf=0.0)
        idx = absolute[:, None] - np.arange(seq_len - 1, -1, -1)[None, :]
        idx = np.clip(idx, 0, len(M) - 1)
        return f, M[idx]

    # B1 XGBoost
    xp = os.path.join(ARTIFACT_DIR, "xgb_rul_baseline.json")
    if os.path.exists(xp):
        import xgboost as xgb
        meta = json.load(open(os.path.join(ARTIFACT_DIR, "xgb_rul_features.json"),
                              encoding="utf-8"))
        _, X = seqs("telemetry")
        S, _ = window_summary(X, meta["base_features"])
        m = xgb.XGBRegressor()
        m.load_model(xp)
        out["xgb"] = m.predict(S)

    # B2 / B3 GRUs
    for tag, fset, fname in (("gru_telem", "telemetry", "gru_rul_telemetry.pt"),
                             ("gru_health", "health_only", "gru_rul_health.pt")):
        p = os.path.join(ARTIFACT_DIR, fname)
        if not os.path.exists(p):
            continue
        ck = torch.load(p, map_location="cpu", weights_only=False)
        f = ck["features"]
        M = np.nan_to_num(split_df[f].to_numpy(dtype=np.float32), nan=0.0,
                          posinf=0.0, neginf=0.0)
        idx = absolute[:, None] - np.arange(ck["seq_len"] - 1, -1, -1)[None, :]
        idx = np.clip(idx, 0, len(M) - 1)
        X = StandardScaler3D.from_dict(ck["scaler"]).transform(M[idx])
        model = DirectRULGRU(n_features=len(f))
        model.load_state_dict(ck["state_dict"])
        model.eval()
        preds = []
        with torch.no_grad():
            for i in range(0, len(X), 4096):
                preds.append(model(torch.from_numpy(X[i:i + 4096])).numpy())
        out[tag] = np.concatenate(preds)

    return out


# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint-stride", type=int, default=40)
    ap.add_argument("--particles", type=int, default=600)
    ap.add_argument("--seq-len", type=int, default=60)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    gru_std = 0.25
    tsum = os.path.join(REPORT_DIR, "training_summary.json")
    if os.path.exists(tsum):
        try:
            gru_std = float(json.load(open(tsum, encoding="utf-8"))["rate_gru"]["log_mult_std"])
        except Exception:
            pass

    all_preds = {}
    for split in TEST_SPLITS:
        path = os.path.join(FEATURE_DIR, f"{split}_est.parquet")
        if not os.path.exists(path):
            print(f"  !! missing {path}, skipping")
            continue
        sdf = pd.read_parquet(path)
        rids = sorted(sdf.run_id.unique())
        print(f"\n=== {split}: {len(rids)} runs, {len(sdf):,} rows ===", flush=True)

        jobs = [(split, r, args.checkpoint_stride, args.seq_len, args.particles, gru_std)
                for r in rids]
        t0 = time.time()
        parts = []
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for n, p in enumerate(ex.map(_hybrid_for_run, jobs), 1):
                parts.append(p)
                if n % 10 == 0 or n == len(jobs):
                    print(f"    hybrid {n:3d}/{len(jobs)}  {time.time()-t0:6.1f}s",
                          flush=True)
        hyb = pd.concat([p for p in parts if len(p)], ignore_index=True)

        ends_lookup = {r: g.end_row.tolist() for r, g in hyb.groupby("run_id")}
        base = _baseline_predictions(sdf, ends_lookup, args.seq_len)
        preds = hyb.merge(base, on=["run_id", "end_row"], how="left")
        all_preds[split] = preds

        samples = preds.pop("rul_samples")
        np.save(os.path.join(REPORT_DIR, f"rul_samples_{split}.npy"),
                np.stack(samples.to_numpy()))
        preds.to_parquet(os.path.join(REPORT_DIR, f"rul_predictions_{split}.parquet"),
                         index=False)
        preds["rul_samples"] = samples
        print(f"    {len(preds):,} checkpoints  ({time.time()-t0:.1f}s)")

    with open(os.path.join(REPORT_DIR, "evaluation_config.json"), "w",
              encoding="utf-8") as f:
        json.dump(vars(args), f, indent=1)
    print("\nPredictions written. Run scripts/report.py to produce metrics and figures.")


if __name__ == "__main__":
    main()
