"""
AeroTwin-4 RUL Trainer & Evaluator CLI.

Two stages:
  1. HealthEstimator  - residual features -> health index H(t)
  2. RULProjector     - health trend -> remaining useful life with an interval

Held out by ENGINE UNIT (U02), not by time, so the test asks whether the model
transfers to an engine it has never seen rather than merely interpolating within
a run it already knows.

Usage:
  .venv/Scripts/python.exe scripts/train_rul_model.py
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

_script_dir = os.path.dirname(os.path.abspath(__file__))
_root_dir = os.path.dirname(_script_dir)
_aerotwin_dir = os.path.join(_root_dir, "AeroTwin")

for _p in [_aerotwin_dir, _root_dir]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from AeroTwin.degradation.conditions import RunConditionSampler
from AeroTwin.health.engine import DigitalTwinStateEngine
from AeroTwin.ml.anomaly.features import FeatureExtractor
from AeroTwin.ml.anomaly.preprocessing import FeatureScaler
from AeroTwin.ml.rul.health_estimator import HealthEstimator
from AeroTwin.ml.rul.projector import DEFAULT_FAILURE_THRESHOLD, RULProjector


def build_residuals(raw_file: str) -> pd.DataFrame:
    """Run one raw RUL telemetry file through the condition-matched Digital Twin."""
    run_name = os.path.basename(raw_file).replace("_raw.csv", "")
    meta_path = raw_file.replace("_raw.csv", "_metadata.json")
    seed = 42
    if os.path.exists(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            seed = int(json.load(f).get("seed", 42))

    sampler = RunConditionSampler()
    conditions = sampler.sample(run_name, seed)
    twin_params = sampler.build_engine_parameters(conditions)

    engine = DigitalTwinStateEngine(
        dt=0.01, seed=seed, mode="COUNTERFACTUAL", engine_parameters=twin_params
    )

    df_raw = pd.read_csv(raw_file)
    gt_cols = [
        "gt_active_severity",
        "gt_current_health",
        "gt_is_degraded",
        "gt_degradation_type",
        "gt_target_component",
    ]

    rows = []
    for _, row in df_raw.iterrows():
        frame = engine.process_telemetry(row.to_dict())
        rec = {"simulation_time": frame.simulation_time}
        for k, v in frame.observed_outputs.items():
            rec["obs_" + k] = v
        for k, v in frame.expected_outputs.items():
            rec["exp_" + k] = v
        for k, v in frame.residuals.raw_signed.items():
            rec["res_signed_" + k] = v
        for k, v in frame.residuals.normalized.items():
            rec["res_norm_" + k] = v
        # Indicator names must match the Phase 4 pipeline exactly, or the feature
        # extractor sees different columns here than it did at training time.
        rec["ind_thermal_dev"] = frame.indicators.thermal_deviation
        rec["ind_oil_dev"] = frame.indicators.oil_deviation
        rec["ind_vibration_dev"] = frame.indicators.vibration_deviation
        rec["ind_torque_dev"] = frame.indicators.torque_deviation
        rec["ind_cylinder_balance_dev"] = frame.indicators.cylinder_balance_deviation
        for gt in gt_cols:
            if gt in df_raw.columns:
                rec[gt] = row[gt]
        rows.append(rec)

    out = pd.DataFrame(rows)
    out["run_id"] = run_name
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="residual", choices=["raw", "residual", "hybrid"])
    ap.add_argument("--failure-threshold", type=float, default=DEFAULT_FAILURE_THRESHOLD)
    args = ap.parse_args()

    print("=" * 68)
    print("AeroTwin-4 RUL Trainer  (health regression + trend projection)")
    print("=" * 68)

    rul_dir = os.path.join(_root_dir, "data", "generated", "rul")
    raw_files = sorted(glob.glob(os.path.join(rul_dir, "**", "*_raw.csv"), recursive=True))
    if not raw_files:
        raise SystemExit("No RUL runs under " + rul_dir + ". Run generate_rul_dataset.py first.")
    print("Found " + str(len(raw_files)) + " progressive-degradation runs.\n")

    cache_dir = os.path.join(rul_dir, "_residuals")
    os.makedirs(cache_dir, exist_ok=True)

    extractor = FeatureExtractor(config_type=args.config.upper())
    X_parts, meta_parts = [], []

    for i, rf in enumerate(raw_files, 1):
        run_name = os.path.basename(rf).replace("_raw.csv", "")
        cached = os.path.join(cache_dir, run_name + "_residuals.csv")
        if os.path.exists(cached):
            df_res = pd.read_csv(cached)
        else:
            print("  [" + str(i) + "/" + str(len(raw_files)) + "] twin-processing " + run_name + "...")
            df_res = build_residuals(rf)
            df_res.to_csv(cached, index=False)
        Xi, mi = extractor.extract_dataset(df_res, window_size_sec=5.0, stride_sec=1.0)
        X_parts.append(Xi)
        meta_parts.append(mi)

    X = pd.concat(X_parts, ignore_index=True)
    meta = pd.concat(meta_parts, ignore_index=True)
    print("\nFeature matrix: " + str(X.shape[0]) + " windows x " + str(X.shape[1]) + " features")

    # Hold out engine unit U02 entirely.
    is_test = meta["run_id"].str.endswith("U02")
    X_tr, m_tr = X[~is_test].reset_index(drop=True), meta[~is_test].reset_index(drop=True)
    X_te, m_te = X[is_test].reset_index(drop=True), meta[is_test].reset_index(drop=True)
    print("  train: " + str(len(X_tr)) + " windows (" + str(m_tr.run_id.nunique()) + " runs, unit U01)")
    print("  test : " + str(len(X_te)) + " windows (" + str(m_te.run_id.nunique()) + " runs, UNSEEN unit U02)")

    scaler = FeatureScaler()
    X_tr_s = pd.DataFrame(scaler.fit_transform(X_tr), columns=scaler.feature_names)
    X_te_s = pd.DataFrame(scaler.transform(X_te), columns=scaler.feature_names)

    print("\nStage 1: training health estimator...")
    est = HealthEstimator().fit(X_tr_s, m_tr["gt_current_health"])
    print("  training residual sigma: %.4f" % est.residual_std)

    h_pred = est.predict(X_te_s)
    h_true = m_te["gt_current_health"].to_numpy(dtype=float)
    mae = float(np.mean(np.abs(h_pred - h_true)))
    rmse = float(np.sqrt(np.mean((h_pred - h_true) ** 2)))
    print("  HELD-OUT UNIT health MAE=%.4f  RMSE=%.4f" % (mae, rmse))

    print("\nStage 2: projecting RUL per held-out run...")
    projector = RULProjector(
        failure_threshold=args.failure_threshold, health_noise_std=est.residual_std
    )

    # Evaluating RUL only at the final window is meaningless: by then true health
    # has already crossed the threshold, so both the estimate and the truth are 0
    # and any model "passes". The useful question is asked mid-flight - standing at
    # time t with health still above threshold, how long until it crosses?
    results = []
    per_run = []

    for run_id in sorted(m_te["run_id"].unique()):
        mask = (m_te["run_id"] == run_id).to_numpy()
        times_all = m_te.loc[mask, "simulation_end"].to_numpy(dtype=float)
        h_est_all = h_pred[mask]
        h_true_all = h_true[mask]
        order = np.argsort(times_all)
        times_all, h_est_all, h_true_all = times_all[order], h_est_all[order], h_true_all[order]

        # True crossing time of the failure threshold for this run.
        below = np.where(h_true_all <= args.failure_threshold)[0]
        t_fail = float(times_all[below[0]]) if len(below) else None

        run_errors = []
        # Step through the sortie, predicting from progressively more history.
        for i in range(projector.min_points, len(times_all)):
            t_now = float(times_all[i])
            if t_fail is not None and t_now >= t_fail:
                break  # already failed; nothing left to predict

            est_rul = projector.estimate(times_all[: i + 1], h_est_all[: i + 1])
            true_rul = (t_fail - t_now) if t_fail is not None else None

            rec = {
                "run_id": run_id,
                "t_now_s": round(t_now, 2),
                "health_true": round(float(h_true_all[i]), 4),
                "health_est": round(float(h_est_all[i]), 4),
                "rul_est_s": est_rul.rul_seconds,
                "rul_lo_s": est_rul.rul_lower_seconds,
                "rul_hi_s": est_rul.rul_upper_seconds,
                "rul_true_s": None if true_rul is None else round(true_rul, 2),
                "conf": round(est_rul.confidence, 3),
                "decaying": est_rul.is_decaying,
            }
            results.append(rec)
            if true_rul is not None and est_rul.rul_seconds is not None:
                run_errors.append(abs(est_rul.rul_seconds - true_rul))

        healthy_run = t_fail is None
        per_run.append(
            {
                "run_id": run_id,
                "is_healthy_run": healthy_run,
                "n_predictions": int(sum(1 for r in results if r["run_id"] == run_id)),
                "rul_mae_s": round(float(np.mean(run_errors)), 2) if run_errors else None,
                "true_failure_time_s": None if t_fail is None else round(t_fail, 2),
            }
        )

    df_res = pd.DataFrame(results)
    df_run = pd.DataFrame(per_run)

    print()
    print(df_run.to_string(index=False))

    # A healthy engine must NOT be given a finite RUL - that is a false alarm that
    # would ground a serviceable aircraft.
    healthy_rows = df_res[df_res["run_id"].str.contains("HEALTHY")]
    if len(healthy_rows):
        false_decay = int(healthy_rows["decaying"].sum())
        print(
            "\nHealthy-engine false decay calls: %d / %d predictions"
            % (false_decay, len(healthy_rows))
        )

    degraded = df_res[~df_res["run_id"].str.contains("HEALTHY")].dropna(
        subset=["rul_true_s", "rul_est_s"]
    )
    if len(degraded):
        err = (degraded["rul_est_s"] - degraded["rul_true_s"]).abs()
        print("\nDegraded-run RUL error over %d mid-flight predictions:" % len(degraded))
        print("  MAE  = %.2f s" % err.mean())
        print("  RMSE = %.2f s" % float(np.sqrt((err ** 2).mean())))
        hi = degraded["rul_hi_s"].fillna(np.inf)
        inside = int(
            ((degraded["rul_true_s"] >= degraded["rul_lo_s"]) & (degraded["rul_true_s"] <= hi)).sum()
        )
        print(
            "  trend-band coverage  = %d/%d (%.1f%%)   [band is uncalibrated, not 95%%]"
            % (inside, len(degraded), 100.0 * inside / len(degraded))
        )

    out_models = os.path.join(_root_dir, "models", "rul", args.config)
    est.save(out_models)
    scaler.save(os.path.join(out_models, "scaler.json"))

    out_data = os.path.join(_root_dir, "data", "generated", "rul", "evaluation")
    os.makedirs(out_data, exist_ok=True)
    df_res.to_csv(os.path.join(out_data, "rul_projection_" + args.config + ".csv"), index=False)
    df_run.to_csv(os.path.join(out_data, "rul_per_run_" + args.config + ".csv"), index=False)
    with open(os.path.join(out_data, "rul_report_" + args.config + ".json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "feature_config": args.config,
                "failure_threshold": args.failure_threshold,
                "health_mae_heldout_unit": mae,
                "health_rmse_heldout_unit": rmse,
                "training_residual_std": est.residual_std,
                "per_run": per_run,
                "n_predictions": len(results),
                "units_note": "RUL is in SIMULATION seconds, not engine flight hours.",
            },
            f,
            indent=2,
        )

    print("\nSaved model  -> " + out_models)
    print("Saved report -> " + out_data)


if __name__ == "__main__":
    main()
