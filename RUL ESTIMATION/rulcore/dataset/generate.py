"""
generate.py
===========
Build the full run-to-failure dataset (Deliverable A / B).

The fleet is composed of three deliberately separate populations:

  MAIN (200 runs)         nominal build scatter, nominal operating envelope.
                          This is the only population that training may see.

  STRESS (30 runs)        hot-and-high envelope that is under-represented in
                          training (15000-20000 ft, 35-45 C). TEST ONLY.
                          Part 15's "harder test set".

  CROSS_ENGINE (20 runs)  build scatter widened by 2.5x plus larger sensor
                          calibration errors, so efficiency, friction, thermal
                          constants and biases all differ from anything in
                          training. TEST ONLY. Part 21's generalisation probe.

WHY THIS MANY RUNS
------------------
The effective sample size of a prognostics problem is the number of independent
run-to-failure trajectories, not the number of rows. Every row inside a run
shares one engine, one degradation plan and one realised life, so rows are
massively correlated: 300 000 rows drawn from 12 engines carry roughly 12
independent observations of "how an engine dies".

200 training-eligible runs was chosen so that after a 70/15/15 split by run
there are 140 training / 30 validation / 30 test runs. With ~7 degradation
mechanisms x 4 rate archetypes = 28 mechanism-rate cells, 140 training runs
gives ~5 runs per cell, which is the minimum at which a model can distinguish a
mechanism from a single engine's idiosyncrasy. 30 test runs gives a standard
error on test MAE of roughly MAE/sqrt(30) ~ 18% - loose, but honest, and small
enough to separate the architectures if the differences are real. Going much
below this makes the test set incapable of distinguishing the models; going much
above costs simulation time without changing any conclusion, because the
dominant uncertainty is the realism of the simulator, not the sample count.

For reference, NASA's C-MAPSS FD001 ships 100 training and 100 test units, and
the published spread of results on it is wide precisely because 100 units is not
many. 200 is a deliberate step up from that, not an arbitrary number.
"""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List

import numpy as np
import pandas as pd

from ..config import (MECHANISMS, MECHANISM_WEIGHTS, N_RUNS_CROSS_ENGINE,
                      N_RUNS_MAIN, N_RUNS_STRESS, RANDOM_SEED, RAW_DIR,
                      RATE_WEIGHTS, CROSS_ENGINE_SCATTER_MULT, DATA_DIR)
from .simulate_run import simulate_run

SENSOR_DRIFT_FRACTION = 0.12      # share of runs carrying an instrument-drift fault


def build_manifest(seed: int = RANDOM_SEED) -> List[Dict]:
    """Plan every run before simulating any of them.

    Mechanism and rate archetype are drawn here, and the operating envelope is
    drawn INSIDE simulate_run from a separate stream, so the two are independent
    by construction. Stratifying the mechanism/rate assignment (rather than
    sampling it i.i.d.) guarantees every mechanism-rate cell is populated even
    though the fleet is small.
    """
    rng = np.random.default_rng(seed)
    jobs: List[Dict] = []

    mech_p = np.array(MECHANISM_WEIGHTS, dtype=float)
    mech_p = mech_p / mech_p.sum()
    rate_names = list(RATE_WEIGHTS)
    rate_p = np.array([RATE_WEIGHTS[n] for n in rate_names], dtype=float)
    rate_p = rate_p / rate_p.sum()

    def stratified(n: int) -> List[tuple]:
        """Round-robin over mechanism x rate, then shuffle."""
        cells = [(m, r) for m in MECHANISMS for r in rate_names]
        weights = np.array([mech_p[MECHANISMS.index(m)] * rate_p[rate_names.index(r)]
                            for m, r in cells])
        counts = np.floor(weights / weights.sum() * n).astype(int)
        # distribute the remainder to the largest fractional parts
        rem = n - counts.sum()
        if rem > 0:
            frac = weights / weights.sum() * n - counts
            for i in np.argsort(-frac)[:rem]:
                counts[i] += 1
        out = []
        for (m, r), c in zip(cells, counts):
            out.extend([(m, r)] * int(c))
        rng.shuffle(out)
        return out[:n]

    populations = [
        ("MAIN", N_RUNS_MAIN, "nominal", 1.0),
        ("STRESS", N_RUNS_STRESS, "stress", 1.0),
        ("XENG", N_RUNS_CROSS_ENGINE, "nominal", CROSS_ENGINE_SCATTER_MULT),
    ]

    for pop, n, envelope, scatter in populations:
        assignments = stratified(n)
        for i, (mech, rate) in enumerate(assignments):
            jobs.append({
                "run_id": f"{pop}_{i:03d}",
                "population": pop,
                "seed": int(rng.integers(0, 2**31 - 1)),
                "envelope": envelope,
                "scatter_mult": scatter,
                "mechanism": mech,
                "rate_archetype": rate,
                "sensor_drift_fault": bool(rng.random() < SENSOR_DRIFT_FRACTION),
                "noise_scale": float(np.clip(rng.normal(1.0, 0.12), 0.7, 1.5)),
            })
    return jobs


def _simulate_one(job: Dict) -> Dict:
    """Worker: simulate one run and persist it. Returns metadata only."""
    attempt = 0
    seed = job["seed"]
    while attempt < 4:
        cols, meta = simulate_run(
            run_id=job["run_id"],
            seed=seed,
            envelope=job["envelope"],
            scatter_mult=job["scatter_mult"],
            mechanism=job["mechanism"],
            rate_archetype=job["rate_archetype"],
            sensor_drift_fault=job["sensor_drift_fault"],
            noise_scale=job["noise_scale"],
        )
        if meta.get("status") == "ok":
            df = pd.DataFrame(cols)
            path = os.path.join(RAW_DIR, f"{job['run_id']}.parquet")
            df.to_parquet(path, index=False)
            meta["population"] = job["population"]
            meta["path"] = path
            return meta
        # An engine that refuses to die within the horizon gets a new draw.
        attempt += 1
        seed = seed + 7919
    return {"run_id": job["run_id"], "status": "failed", "population": job["population"]}


def generate_dataset(jobs: List[Dict] | None = None, workers: int = 8,
                     verbose: bool = True) -> pd.DataFrame:
    """Simulate every run in the manifest, in parallel, and write the index."""
    jobs = jobs or build_manifest()
    os.makedirs(RAW_DIR, exist_ok=True)
    metas: List[Dict] = []
    t0 = time.time()

    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_simulate_one, j): j for j in jobs}
        for n_done, fut in enumerate(as_completed(futures), 1):
            meta = fut.result()
            metas.append(meta)
            if verbose and (n_done % 10 == 0 or n_done == len(jobs)):
                el = time.time() - t0
                rate = n_done / max(el, 1e-6)
                eta = (len(jobs) - n_done) / max(rate, 1e-9)
                ok = sum(1 for m in metas if m.get("status") == "ok")
                print(f"  {n_done:4d}/{len(jobs)}  ok={ok:4d}  "
                      f"{el:6.1f}s elapsed  ~{eta:5.1f}s remaining", flush=True)

    index = pd.DataFrame([m for m in metas if m.get("status") == "ok"])
    index = index.sort_values("run_id").reset_index(drop=True)

    # Persist the run index in two forms: a flat CSV for eyeballing and a JSON
    # that keeps the nested engine parameters and sensor calibrations.
    flat_cols = [c for c in index.columns
                 if c not in ("engine_params", "sensor_calibration",
                              "commissioning_baseline", "theta0", "theta_eol",
                              "sortie_mix")]
    index[flat_cols].to_csv(os.path.join(DATA_DIR, "run_index.csv"), index=False)
    with open(os.path.join(DATA_DIR, "run_index.json"), "w", encoding="utf-8") as f:
        json.dump([_jsonable(m) for m in metas], f, indent=1)

    if verbose:
        failed = [m for m in metas if m.get("status") != "ok"]
        print(f"\nGenerated {len(index)} runs in {time.time()-t0:.1f}s "
              f"({len(failed)} failed)")
    return index


def _jsonable(d: Dict) -> Dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, (np.integer,)):
            out[k] = int(v)
        elif isinstance(v, (np.floating,)):
            out[k] = float(v)
        elif isinstance(v, np.ndarray):
            out[k] = v.tolist()
        elif isinstance(v, dict):
            out[k] = _jsonable(v)
        else:
            out[k] = v
    return out
