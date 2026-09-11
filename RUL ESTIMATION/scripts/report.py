"""Stage 7: metrics, calibration, sanity checks and every figure.

    python scripts/report.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from rulcore.config import (FEATURE_DIR, FIGURE_DIR, HEALTH_PARAMS, REPORT_DIR,
                            RUL_CAP_H)
from rulcore.evaluation import plots as P
from rulcore.evaluation.metrics import (bootstrap_ci_over_runs,
                                        calibration_curve,
                                        condition_confounding_check,
                                        crps_from_samples,
                                        early_prediction_table,
                                        interval_metrics, monotonicity_report,
                                        per_run_metrics, pit_values,
                                        point_metrics)

TEST_SPLITS = ["test", "test_stress", "test_cross_engine"]
MODELS = ["hybrid", "physics_only", "xgb", "gru_telem", "gru_health"]
LABELS = {
    "hybrid": "Hybrid (physics+UKF+GRU+MC)",
    "physics_only": "B4 Physics only",
    "xgb": "B1 XGBoost telemetry",
    "gru_telem": "B2 GRU telemetry",
    "gru_health": "B3 GRU health",
}


def load_preds(split):
    p = os.path.join(REPORT_DIR, f"rul_predictions_{split}.parquet")
    if not os.path.exists(p):
        return None
    df = pd.read_parquet(p)
    s = os.path.join(REPORT_DIR, f"rul_samples_{split}.npy")
    if os.path.exists(s):
        df["_samples_idx"] = np.arange(len(df))
    return df


def main():
    os.makedirs(FIGURE_DIR, exist_ok=True)
    rows, early_tables, figures = [], {}, []

    # ================================================================== #
    # Comparison table
    # ================================================================== #
    print("=" * 104)
    print("RUL ACCURACY BY MODEL AND TEST PARTITION")
    print("(labels are capped at %.0f h, the standard piecewise-linear "
          "prognostic target)" % RUL_CAP_H)
    print("=" * 104)
    header = (f"{'partition':<20s}{'model':<30s}{'MAE':>8s}{'RMSE':>8s}"
              f"{'MedAE':>8s}{'R2':>8s}{'bias':>8s}{'sMAPE':>8s}{'runs':>6s}")
    print(header)
    print("-" * 104)

    for split in TEST_SPLITS:
        df = load_preds(split)
        if df is None:
            continue
        yt = np.minimum(df.rul_true.to_numpy(), RUL_CAP_H)
        for m in MODELS:
            if m not in df.columns or df[m].isna().all():
                continue
            yp = np.minimum(df[m].to_numpy(), RUL_CAP_H)
            met = point_metrics(yt, yp)
            met.update({"split": split, "model": m, "label": LABELS[m],
                        "runs": int(df.run_id.nunique())})
            sub = df.assign(rul_true=yt, rul_pred=yp)
            ci = bootstrap_ci_over_runs(sub, "mae", n_boot=600)
            met["mae_lo95"], met["mae_hi95"] = ci["lo95"], ci["hi95"]
            rows.append(met)
            print(f"{split:<20s}{LABELS[m]:<30s}{met['mae']:8.1f}{met['rmse']:8.1f}"
                  f"{met['medae']:8.1f}{met['r2']:8.3f}{met['bias']:+8.1f}"
                  f"{met['smape']:8.1f}{met['runs']:6d}")
        print("-" * 104)

    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(REPORT_DIR, "model_comparison.csv"), index=False)

    print("\n95% bootstrap CI on MAE (resampled over RUNS, the independent unit):")
    for _, r in table.iterrows():
        print(f"   {r.split:<20s} {r.label:<30s} "
              f"MAE {r.mae:6.1f}  [{r.mae_lo95:6.1f}, {r.mae_hi95:6.1f}]")

    # ================================================================== #
    # Uncertainty calibration (hybrid only - it is the only probabilistic model)
    # ================================================================== #
    print("\n" + "=" * 104)
    print("UNCERTAINTY CALIBRATION - hybrid model")
    print("=" * 104)
    cal_rows = []
    for split in TEST_SPLITS:
        df = load_preds(split)
        if df is None:
            continue
        yt = np.minimum(df.rul_true.to_numpy(), RUL_CAP_H)
        iv = interval_metrics(yt, df.hybrid_p10.to_numpy(), df.hybrid_p90.to_numpy(),
                              nominal=0.80)
        spath = os.path.join(REPORT_DIR, f"rul_samples_{split}.npy")
        crps = float("nan")
        cal = None
        if os.path.exists(spath):
            S = np.load(spath)
            S = np.minimum(S, RUL_CAP_H)
            cal = calibration_curve(yt, list(S))
            crps = crps_from_samples(yt, list(S))
            pit = pit_values(yt, list(S))
            if split == "test":
                figures.append(P.plot_calibration(cal, pit))
                cal.to_csv(os.path.join(REPORT_DIR, "calibration.csv"), index=False)
        print(f"  {split:<20s} P10-P90 coverage {iv['picp']*100:5.1f}% "
              f"(nominal 80%)   mean width {iv['mpiw']:6.1f} h   "
              f"Winkler {iv['winkler']:7.1f}   CRPS {crps:6.2f}   "
              f"censored {df.hybrid_censored.mean()*100:4.1f}%")
        cal_rows.append({"split": split, **iv, "crps": crps,
                         "censored": float(df.hybrid_censored.mean())})
    pd.DataFrame(cal_rows).to_csv(os.path.join(REPORT_DIR, "interval_metrics.csv"),
                                  index=False)

    # ================================================================== #
    # Early prediction
    # ================================================================== #
    print("\n" + "=" * 104)
    print("EARLY PREDICTION - RUL MAE (h) by remaining life, partition 'test'")
    print("=" * 104)
    df = load_preds("test")
    if df is not None:
        yt = np.minimum(df.rul_true.to_numpy(), RUL_CAP_H)
        bands = None
        for m in MODELS:
            if m not in df.columns or df[m].isna().all():
                continue
            sub = df.assign(rul_true=yt,
                            rul_pred=np.minimum(df[m].to_numpy(), RUL_CAP_H))
            if m == "hybrid":
                sub["rul_p10"] = df.hybrid_p10
                sub["rul_p90"] = df.hybrid_p90
            tab = early_prediction_table(sub)
            early_tables[LABELS[m]] = tab
            tab.to_csv(os.path.join(REPORT_DIR, f"early_prediction_{m}.csv"), index=False)
            if bands is None:
                bands = tab.band.tolist()
                print(f"{'model':<30s}" + "".join(f"{b:>17s}" for b in bands))
            vals = {r.band: r.mae for _, r in tab.iterrows()}
            print(f"{LABELS[m]:<30s}" + "".join(f"{vals.get(b, np.nan):17.1f}"
                                                for b in bands))
        if early_tables:
            figures.append(P.plot_early_prediction(early_tables))

        hyb = early_tables.get(LABELS["hybrid"])
        if hyb is not None and "iv_picp" in hyb.columns:
            print("\n  hybrid interval coverage by band:")
            for _, r in hyb.iterrows():
                print(f"     {r.band:<18s} coverage {r.iv_picp*100:5.1f}%  "
                      f"width {r.iv_mpiw:6.1f} h")

    # ================================================================== #
    # Monotonicity and confounding
    # ================================================================== #
    print("\n" + "=" * 104)
    print("MONOTONICITY AND SANITY CHECKS (Part 20)")
    print("=" * 104)
    mono_all = {}
    for split in TEST_SPLITS:
        df = load_preds(split)
        if df is None:
            continue
        for m in MODELS:
            if m not in df.columns or df[m].isna().all():
                continue
            rep = monotonicity_report(df.assign(rul_pred=df[m]))
            mono_all[f"{split}/{m}"] = rep
            if split == "test":
                print(f"  {LABELS[m]:<30s} RUL increases >5 h in "
                      f"{rep['violation_rate']*100:5.2f}% of steps   "
                      f"worst +{rep['worst_increase_h']:6.1f} h   "
                      f"median Spearman(t, RUL) {rep['median_spearman_time_vs_rul']:+.3f}")
    with open(os.path.join(REPORT_DIR, "monotonicity.json"), "w", encoding="utf-8") as f:
        json.dump(mono_all, f, indent=1)

    print("\n  Does the model read harsh conditions as degradation?")
    print("  (slope of RUL error against condition; near zero = twin is doing its job)")
    conf_rows = []
    for split in TEST_SPLITS:
        df = load_preds(split)
        if df is None:
            continue
        for m in ("hybrid", "gru_telem", "xgb"):
            if m not in df.columns or df[m].isna().all():
                continue
            c = condition_confounding_check(df.assign(rul_pred=df[m],
                                                      rul_true=df.rul_true))
            c["split"] = split
            c["model"] = LABELS[m]
            conf_rows.append(c)
            if split == "test":
                for _, r in c.iterrows():
                    print(f"     {LABELS[m]:<30s} {r.variable:<26s} "
                          f"slope {r.slope_h_per_unit:+7.2f} h/unit   r={r.correlation:+.3f}")
    if conf_rows:
        pd.concat(conf_rows, ignore_index=True).to_csv(
            os.path.join(REPORT_DIR, "confounding.csv"), index=False)

    # ================================================================== #
    # Figures
    # ================================================================== #
    print("\n" + "=" * 104)
    print("FIGURES")
    print("=" * 104)

    df = load_preds("test")
    if df is not None:
        figures.append(P.plot_model_comparison(table[table.split.isin(TEST_SPLITS)]
                                               .assign(model=lambda d: d.label)))
        scat = {}
        for m in MODELS:
            if m in df.columns and not df[m].isna().all():
                scat[LABELS[m]] = df.assign(rul_pred=df[m])
        figures.append(P.plot_rul_scatter(scat))
        figures.append(P.plot_monotonicity(df.assign(rul_pred=df.hybrid)))

        # Pick runs spanning short / medium / long lives for the trajectory plot.
        lives = df.groupby("run_id").rul_true.max().sort_values()
        picks = [lives.index[0], lives.index[len(lives) // 2], lives.index[-1]]
        traj = df.assign(rul_pred=df.hybrid, rul_p10=df.hybrid_p10,
                         rul_p90=df.hybrid_p90)
        figures.append(P.plot_rul_trajectories(traj, picks))

    # Physics-validation figures from a representative test run
    est_path = os.path.join(FEATURE_DIR, "test_est.parquet")
    if os.path.exists(est_path):
        te = pd.read_parquet(est_path)
        figures.append(P.plot_health_scatter(te))
        rid = te.groupby("run_id").size().sort_values().index[len(te.run_id.unique()) // 2]
        run = te[te.run_id == rid].sort_values("snapshot_index").reset_index(drop=True)
        figures.append(P.plot_health_tracking(run, rid))
        figures.append(P.plot_degradation(run, rid))
        figures.append(P.plot_twin_vs_measured(run, rid))
        figures.append(P.plot_residual_trajectories(run, rid))

        print("\n  UKF health estimation on held-out runs:")
        for k in HEALTH_PARAMS:
            t, e = te[f"true_{k}"].to_numpy(), te[f"est_{k}"].to_numpy()
            print(f"     {k:<16s} r={np.corrcoef(t, e)[0,1]:+.3f}  "
                  f"MAE={np.mean(np.abs(t-e)):.4f}  bias={np.mean(e-t):+.4f}")

    for f in figures:
        print(f"  wrote {os.path.relpath(f, os.path.dirname(REPORT_DIR))}")


if __name__ == "__main__":
    main()
