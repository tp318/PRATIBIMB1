"""Stage 4a: tune the UKF on VALIDATION runs only.

Two knobs are swept: the process-noise scale and the monotonicity recovery
allowance. Selection is on validation runs; the test partitions are never
touched here, so the reported test numbers stay honest.

Scoring balances two failure modes that pull in opposite directions:
  * tracking error  - mean absolute error on the health parameters
  * end-of-life bias - signed error at the last snapshot, which is what the
                       monotone ratchet biases and what RUL depends on most

    python scripts/tune_ukf.py
"""
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from rulcore.config import (FEATURE_DIR, HEALTH_EOL, HEALTH_NOMINAL,
                            HEALTH_PARAMS, REPORT_DIR, UKF_MEAS)
from rulcore.estimation.twin import SigmaModel
from rulcore.estimation.ukf import run_ukf_on_frame

N_VAL_RUNS = 10          # enough to rank settings without costing an hour

sm = SigmaModel.from_dict(json.load(open(os.path.join(FEATURE_DIR, "sigma_model.json"),
                                         encoding="utf-8")))
SIGMA = {c: sm.global_sigma[f"{c}_residual"] for c in UKF_MEAS}
RANGE = {k: abs(HEALTH_EOL[k] - HEALTH_NOMINAL[k]) for k in HEALTH_PARAMS}


_CACHE = {}


def _val_run(rid):
    """Cache the validation table per worker process; re-reading a 48k-row
    parquet for every grid point dominated the runtime of the first sweep."""
    if "df" not in _CACHE:
        _CACHE["df"] = pd.read_parquet(os.path.join(FEATURE_DIR, "val.parquet"))
    df = _CACHE["df"]
    return df[df.run_id == rid].reset_index(drop=True)


def score_run(args):
    rid, q_scale, recovery, monotone = args
    import importlib

    import rulcore.config as cfg
    cfg.UKF_RECOVERY_FRAC = recovery
    cfg.UKF_MONOTONE = monotone
    from rulcore.estimation import ukf as ukf_mod
    importlib.reload(ukf_mod)

    d = _val_run(rid)
    out = ukf_mod.run_ukf_on_frame(d, SIGMA, q_scale=q_scale)

    mae, endbias = {}, {}
    for k in HEALTH_PARAMS:
        e = out[f"est_{k}"]
        t = d[f"true_{k}"].to_numpy()
        # normalise by the parameter's usable range so parameters are comparable
        mae[k] = float(np.mean(np.abs(e - t)) / RANGE[k])
        endbias[k] = float((e[-1] - t[-1]) / RANGE[k])
    return {"run_id": rid, "q_scale": q_scale, "recovery": recovery,
            "monotone": monotone, "mae": mae, "endbias": endbias}


if __name__ == "__main__":
    val = pd.read_parquet(os.path.join(FEATURE_DIR, "val.parquet"),
                          columns=["run_id"])
    rids = sorted(val.run_id.unique())[:N_VAL_RUNS]
    print(f"Tuning on {len(rids)} validation runs\n")

    grid = [(q, r, True) for q in (0.25, 0.4, 0.6) for r in (2.0, 6.0)]
    grid += [(q, 0.0, False) for q in (0.25, 0.4)]

    # ONE pool for every (setting, run) job. The first version created a fresh
    # pool per grid point, which threw away the per-worker parquet cache and
    # made the sweep an order of magnitude slower than the filtering itself.
    jobs = [(rid, q, r, m) for (q, r, m) in grid for rid in rids]
    print(f"{len(jobs)} jobs over {len(grid)} settings", flush=True)
    with ProcessPoolExecutor(max_workers=8) as ex:
        res_all = list(ex.map(score_run, jobs, chunksize=2))

    rows = []
    for (q, r, mono) in grid:
        res = [x for x in res_all if x["q_scale"] == q and x["recovery"] == r
               and x["monotone"] == mono]
        mae = float(np.mean([np.mean(list(x["mae"].values())) for x in res]))
        bias = float(np.mean([np.mean(list(x["endbias"].values())) for x in res]))
        absbias = float(np.mean([np.mean(np.abs(list(x["endbias"].values()))) for x in res]))
        rows.append({"q_scale": q, "recovery": r, "monotone": mono, "norm_mae": mae,
                     "signed_end_bias": bias, "abs_end_bias": absbias,
                     "score": mae + absbias})
        tag = f"recovery={r:4.1f}" if mono else "ratchet OFF "
        print(f"  q_scale={q:4.2f}  {tag}  "
              f"normMAE={mae:.4f}  endBias={bias:+.4f}  |endBias|={absbias:.4f}  "
              f"score={mae+absbias:.4f}", flush=True)

    tab = pd.DataFrame(rows).sort_values("score")
    best = tab.iloc[0]
    print("\nBest setting:")
    print(f"  q_scale={best.q_scale}  recovery={best.recovery}  "
          f"monotone={best.monotone}  score={best.score:.4f}")

    os.makedirs(REPORT_DIR, exist_ok=True)
    tab.to_csv(os.path.join(REPORT_DIR, "ukf_tuning.csv"), index=False)
    with open(os.path.join(REPORT_DIR, "ukf_tuning_best.json"), "w", encoding="utf-8") as f:
        json.dump({"q_scale": float(best.q_scale), "recovery": float(best.recovery),
                   "monotone": bool(best.monotone), "n_val_runs": len(rids)}, f, indent=1)
    print(f"\nWrote {os.path.join(REPORT_DIR, 'ukf_tuning.csv')}")
