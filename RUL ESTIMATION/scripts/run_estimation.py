"""Stage 4b: run the UKF and the physics rate identification over every run.

Appends est_*, phi_*, stress_*, rate_phys_*, a_* and the ground-truth rate
labels to each split's feature table.

    python scripts/run_estimation.py [--workers 8]
"""
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from rulcore.config import ARTIFACT_DIR, FEATURE_DIR, HEALTH_PARAMS, REPORT_DIR, UKF_MEAS
from rulcore.estimation.twin import SigmaModel
from rulcore.models.degradation_model import fit_run_rates, true_rates

SPLITS = ["train", "val", "test", "test_stress", "test_cross_engine"]

_best_path = os.path.join(REPORT_DIR, "ukf_tuning_best.json")
if os.path.exists(_best_path):
    _best = json.load(open(_best_path, encoding="utf-8"))
    Q_SCALE = float(_best["q_scale"])
    RECOVERY = float(_best["recovery"])
else:
    Q_SCALE, RECOVERY = 1.0, 0.35

_sm = SigmaModel.from_dict(json.load(open(os.path.join(FEATURE_DIR, "sigma_model.json"),
                                          encoding="utf-8")))
SIGMA = {c: _sm.global_sigma[f"{c}_residual"] for c in UKF_MEAS}


def process_run(args):
    split, rid = args
    import rulcore.config as cfg
    cfg.UKF_RECOVERY_FRAC = RECOVERY
    import importlib

    from rulcore.estimation import ukf as ukf_mod
    importlib.reload(ukf_mod)

    df = pd.read_parquet(os.path.join(FEATURE_DIR, f"{split}.parquet"))
    d = df[df.run_id == rid].sort_values("snapshot_index").reset_index(drop=True)

    est = ukf_mod.run_ukf_on_frame(d, SIGMA, q_scale=Q_SCALE)
    for k, v in est.items():
        d[k] = v
    rates = fit_run_rates(d)
    d = pd.concat([d, rates], axis=1)
    d = pd.concat([d, true_rates(d)], axis=1)

    path = os.path.join(FEATURE_DIR, "_runs", f"{split}__{rid}.parquet")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    d.to_parquet(path, index=False)
    return {"split": split, "run_id": rid, "rows": len(d),
            "gated_frac": float(np.mean(est["ukf_gated"]))}


if __name__ == "__main__":
    workers = 8
    if "--workers" in sys.argv:
        workers = int(sys.argv[sys.argv.index("--workers") + 1])

    print(f"UKF settings: q_scale={Q_SCALE}  recovery={RECOVERY}\n")

    jobs = []
    for split in SPLITS:
        ids = pd.read_parquet(os.path.join(FEATURE_DIR, f"{split}.parquet"),
                              columns=["run_id"]).run_id.unique()
        jobs += [(split, r) for r in ids]
    print(f"Running estimation over {len(jobs)} runs with {workers} workers")

    t0 = time.time()
    results = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for n, r in enumerate(ex.map(process_run, jobs), 1):
            results.append(r)
            if n % 20 == 0 or n == len(jobs):
                el = time.time() - t0
                print(f"  {n:4d}/{len(jobs)}  {el:7.1f}s  "
                      f"eta {(len(jobs)-n)/max(n/el,1e-9):6.1f}s", flush=True)

    print("\nRe-assembling split tables ...")
    for split in SPLITS:
        parts = [r for r in results if r["split"] == split]
        dfs = [pd.read_parquet(os.path.join(FEATURE_DIR, "_runs", f"{split}__{r['run_id']}.parquet"))
               for r in parts]
        out = pd.concat(dfs, ignore_index=True)
        out.to_parquet(os.path.join(FEATURE_DIR, f"{split}_est.parquet"), index=False)
        print(f"  {split:20s} {out.run_id.nunique():4d} runs  {len(out):8,d} rows  "
              f"{out.shape[1]:4d} cols")

    gated = float(np.mean([r["gated_frac"] for r in results]))
    print(f"\nMean innovation-gating rate: {gated*100:.2f}% of snapshots")
    print(f"Total time: {time.time()-t0:.1f}s")

    with open(os.path.join(REPORT_DIR, "estimation_summary.json"), "w", encoding="utf-8") as f:
        json.dump({"q_scale": Q_SCALE, "recovery": RECOVERY,
                   "mean_gated_fraction": gated, "n_runs": len(results)}, f, indent=1)
