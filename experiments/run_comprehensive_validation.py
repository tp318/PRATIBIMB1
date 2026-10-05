# -*- coding: utf-8 -*-
"""
================================================================================
AeroTwin-4 Advanced Validation Suite:
  1. Classifier Re-evaluation (Per-Class Precision/Recall & Thermal Merging)
  2. Severity Sensitivity Curve (0.5%, 1%, 2%, 5%, 10% with Unit-to-Unit Noise)
  3. Ramp-Rate Sweep (1, 5, 10, 30 min ramps)
  4. Stronger Baseline (Learned Regression vs. Physics Digital Twin)
================================================================================
"""

from __future__ import annotations

import copy
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT  = SCRIPT_DIR.parent
RESULTS_DIR = SCRIPT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

for p in [str(REPO_ROOT / "FAULT DETECTION"), str(REPO_ROOT / "DASHBOARD AND DATA" / "AeroTwin")]:
    if p not in sys.path:
        sys.path.insert(0, p)

from run_unified_suite import (
    GLOBAL_SEED, DT, WINDOW_SIZE, EVAL_STRIDE, THRESHOLD_PCT, PERSISTENCE_COUNT,
    FAULT_DEFS, FAULT_CLASSES, OPERATING_POINTS, REDLINE,
    SimulatedEngine, CalibratedAnomalyDetector, GatedFaultClassifier,
)


# ─────────────────────────────────────────────────────────────────────────────
#  TASK 1: CLASSIFIER WITH PER-CLASS PRECISION/RECALL & THERMAL MERGING
# ─────────────────────────────────────────────────────────────────────────────

def run_classifier_evaluation(classifier: GatedFaultClassifier) -> Tuple[pd.DataFrame, pd.DataFrame]:
    print("\n" + "="*80)
    print("TASK 1 — Classifier Evaluation: Per-Class Precision/Recall & Thermal Merging")
    print("="*80)

    # Classes: 1=Misfire, 2=Injector, 3=Cooling, 4=Lubrication, 5=Drift, 6=Combustion, 7=Overheat, 8=Vib
    # Merged Thermal: Class 3 (Cooling) and Class 7 (Overheating) merged into 3 ("Thermal Degradation")
    n_runs = 25
    cm_8 = np.zeros((9, 9), dtype=int)

    # Calibrate scaler
    eng_h = SimulatedEngine("cruise_altitude", GLOBAL_SEED)
    _, h_res, _ = eng_h.generate_run(500.0, DT)
    mu_h  = h_res.mean(axis=0)
    std_h = h_res.std(axis=0) + 1e-6

    for true_fid in range(1, 9):
        for rep in range(n_runs):
            eng = SimulatedEngine("cruise_altitude", noise_seed=7000 + true_fid * 100 + rep)
            _, res, _ = eng.generate_run(600.0, DT, fault_id=true_fid)
            t = int(0.75 * len(res))
            w_res = res[t - WINDOW_SIZE:t]
            w_scaled = ((w_res - mu_h) / std_h) * 0.08
            pred = classifier.classify_with_drift_gate(w_scaled)
            cm_8[true_fid, pred] += 1

    # 8-Class Metrics
    metrics_8 = []
    for c in range(1, 9):
        tp = cm_8[c, c]
        fp = cm_8[1:9, c].sum() - tp
        fn = cm_8[c, 1:9].sum() - tp
        prec = (tp / (tp + fp)) * 100.0 if (tp + fp) > 0 else 0.0
        rec  = (tp / (tp + fn)) * 100.0 if (tp + fn) > 0 else 0.0
        f1   = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        metrics_8.append({
            "Class_ID": c,
            "Class_Name": FAULT_CLASSES[c],
            "Precision_Pct": round(prec, 1),
            "Recall_Pct": round(rec, 1),
            "F1_Score": round(f1, 1),
            "Support": n_runs,
        })
    df_metrics_8 = pd.DataFrame(metrics_8)

    # Merged 7-Class Evaluation: Merge 3 (Cooling) and 7 (Overheat) -> "Thermal Degradation"
    # Mapping: {1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:3, 8:7}
    merged_names = {
        1: "Misfire", 2: "Injector", 3: "Thermal Degradation",
        4: "Lubrication", 5: "Sensor Drift", 6: "Combustion Instability", 7: "Vibration"
    }
    remap = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 3, 8: 7}

    cm_merged = np.zeros((8, 8), dtype=int)
    for true_c in range(1, 9):
        new_true = remap[true_c]
        for pred_c in range(1, 9):
            new_pred = remap[pred_c]
            cm_merged[new_true, new_pred] += cm_8[true_c, pred_c]

    metrics_merged = []
    for c in range(1, 8):
        tp = cm_merged[c, c]
        fp = cm_merged[1:8, c].sum() - tp
        fn = cm_merged[c, 1:8].sum() - tp
        prec = (tp / (tp + fp)) * 100.0 if (tp + fp) > 0 else 0.0
        rec  = (tp / (tp + fn)) * 100.0 if (tp + fn) > 0 else 0.0
        f1   = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        metrics_merged.append({
            "Class_ID": c,
            "Diagnosis": merged_names[c],
            "Precision_Pct": round(prec, 1),
            "Recall_Pct": round(rec, 1),
            "F1_Score": round(f1, 1),
            "Support": n_runs * 2 if c == 3 else n_runs,
        })
    df_metrics_merged = pd.DataFrame(metrics_merged)

    df_metrics_8.to_csv(RESULTS_DIR / "classifier_metrics_8class.csv", index=False)
    df_metrics_merged.to_csv(RESULTS_DIR / "classifier_metrics_merged_thermal.csv", index=False)

    print("\nMerged 7-Class Diagnosis Performance (Cooling + Overheating combined):")
    print(df_metrics_merged.to_string(index=False))
    return df_metrics_8, df_metrics_merged


# ─────────────────────────────────────────────────────────────────────────────
#  TASK 2: SENSITIVITY CURVE (0.5%, 1%, 2%, 5%, 10% WITH UNIT-TO-UNIT NOISE)
# ─────────────────────────────────────────────────────────────────────────────

def run_sensitivity_curve(detector: CalibratedAnomalyDetector) -> pd.DataFrame:
    print("\n" + "="*80)
    print("TASK 2 — Severity Sensitivity Curve (Fixed Severity with Unit-to-Unit Spread)")
    print("="*80)

    SEVERITY_LEVELS = [0.005, 0.010, 0.020, 0.050, 0.100]  # 0.5%, 1%, 2%, 5%, 10%
    DURATION_S      = 1800.0  # 30 minutes
    N_REPEATS       = 20
    FAULTS_TESTED   = [1, 3, 4, 8]  # Representative fault suite

    records = []

    for sev in SEVERITY_LEVELS:
        detected_cnt = 0
        detect_times = []
        total_runs   = 0

        for fid in FAULTS_TESTED:
            for rep in range(N_REPEATS):
                total_runs += 1
                seed = 12000 + int(sev * 10000) + fid * 100 + rep
                rng  = np.random.default_rng(seed)

                # Realistic Unit-to-Unit Engine Variation (±3% sensor bias, ±5% efficiency variation)
                unit_variation = {
                    "rpm":       rng.uniform(0.97, 1.03),
                    "cht":       rng.uniform(0.95, 1.05),
                    "egt":       rng.uniform(0.96, 1.04),
                    "oil_p_psi": rng.uniform(0.97, 1.03),
                    "oil_t":     rng.uniform(0.95, 1.05),
                    "ff":        rng.uniform(0.97, 1.03),
                    "vib":       rng.uniform(0.92, 1.08),
                    "batt":      rng.uniform(0.98, 1.02),
                    "inj_t":     rng.uniform(0.97, 1.03),
                }

                eng = SimulatedEngine("cruise_altitude", noise_seed=seed, twin_params=unit_variation)
                # Generate fixed severity run (constant shift instead of ramp)
                readings, residuals, _ = eng.generate_run(DURATION_S, DT, fault_id=fid, fault_ramp_start=0.0)

                # Apply fixed severity scale (generate_run ramps 0->1, so scale fault shift to fixed sev)
                prog_scale = sev
                T = len(residuals)
                t_det = None
                consec = 0

                for t in range(WINDOW_SIZE, T, EVAL_STRIDE):
                    w_res = residuals[t - WINDOW_SIZE:t] * prog_scale / max(1e-5, (t / T))
                    if detector.is_anomalous(detector.score(w_res)):
                        consec += 1
                        if consec >= PERSISTENCE_COUNT:
                            t_det = t * DT
                            break
                    else:
                        consec = 0

                if t_det is not None:
                    detected_cnt += 1
                    detect_times.append(t_det)

        pd_val = (detected_cnt / total_runs) * 100.0
        med_t  = float(np.median(detect_times)) if detect_times else None
        p10_t  = float(np.percentile(detect_times, 10)) if detect_times else None
        p90_t  = float(np.percentile(detect_times, 90)) if detect_times else None

        records.append({
            "Severity_Pct": round(sev * 100.0, 1),
            "Detection_Probability_Pd_Pct": round(pd_val, 1),
            "Median_Time_to_Detect_s": round(med_t, 1) if med_t else "-",
            "P10_Time_to_Detect_s": round(p10_t, 1) if p10_t else "-",
            "P90_Time_to_Detect_s": round(p90_t, 1) if p90_t else "-",
            "Total_Evaluated_Runs": total_runs,
        })

        print(f"  Severity: {sev*100:4.1f}% | Pd: {pd_val:5.1f}% | Time to Detect: {med_t if med_t else '> 1800':>6} s "
              f"(P10={p10_t if p10_t else '-'}, P90={p90_t if p90_t else '-'})")

    df_sens = pd.DataFrame(records)
    df_sens.to_csv(RESULTS_DIR / "sensitivity_curve.csv", index=False)
    return df_sens


# ─────────────────────────────────────────────────────────────────────────────
#  TASK 3: RAMP-RATE SWEEP (1, 5, 10, 30 MINUTE RAMPS)
# ─────────────────────────────────────────────────────────────────────────────

def run_ramp_rate_sweep(detector: CalibratedAnomalyDetector) -> pd.DataFrame:
    print("\n" + "="*80)
    print("TASK 3 — Ramp-Rate Sweep (1 min, 5 min, 10 min, 30 min)")
    print("="*80)

    RAMP_DURATIONS_S = [60.0, 300.0, 600.0, 1800.0]  # 1, 5, 10, 30 min
    FAULTS_TO_SWEEP  = [1, 3, 4, 7, 8]  # Representative redline faults
    N_REPEATS        = 10

    records = []

    for ramp_s in RAMP_DURATIONS_S:
        ramp_min = ramp_s / 60.0
        leads_s  = []
        leads_pct_ramp = []
        det_sevs = []

        for fid in FAULTS_TO_SWEEP:
            for rep in range(N_REPEATS):
                seed = 15000 + int(ramp_s) + fid * 10 + rep
                eng = SimulatedEngine("cruise_altitude", noise_seed=seed)
                readings, residuals, rl_flags = eng.generate_run(ramp_s, DT, fault_id=fid)

                # Physical redline time
                rl_idx = np.where(rl_flags[:, [1, 2, 3, 6]].any(axis=1))[0]
                if len(rl_idx) == 0: continue
                t_rl = rl_idx[0] * DT

                t_det = None
                consec = 0
                for t in range(WINDOW_SIZE, len(residuals), EVAL_STRIDE):
                    w_res = residuals[t - WINDOW_SIZE:t]
                    if detector.is_anomalous(detector.score(w_res)):
                        consec += 1
                        if consec >= PERSISTENCE_COUNT:
                            t_det = t * DT
                            break
                    else:
                        consec = 0

                if t_det is not None and t_det < t_rl:
                    lead = t_rl - t_det
                    leads_s.append(lead)
                    leads_pct_ramp.append((lead / ramp_s) * 100.0)
                    det_sevs.append(t_det / ramp_s)

        med_lead_s = float(np.median(leads_s)) if leads_s else 0.0
        p10_lead_s = float(np.percentile(leads_s, 10)) if leads_s else 0.0
        med_lead_pct = float(np.median(leads_pct_ramp)) if leads_pct_ramp else 0.0
        med_sev = float(np.median(det_sevs)) if det_sevs else 0.0

        records.append({
            "Ramp_Duration_min": round(ramp_min, 1),
            "Ramp_Duration_s": round(ramp_s, 0),
            "Median_Lead_Time_s": round(med_lead_s, 1),
            "P10_Lead_Time_s": round(p10_lead_s, 1),
            "Lead_Time_Pct_of_Ramp": round(med_lead_pct, 1),
            "Median_Severity_at_Detection": round(med_sev, 3),
        })

        print(f"  Ramp: {ramp_min:4.1f} min ({ramp_s:4.0f} s) | Lead Time: +{med_lead_s:5.1f} s ({med_lead_pct:4.1f}% of ramp) | "
              f"P10: +{p10_lead_s:5.1f} s | Detection Severity: {med_sev:.3f}")

    df_sweep = pd.DataFrame(records)
    df_sweep.to_csv(RESULTS_DIR / "ramp_rate_sweep.csv", index=False)
    return df_sweep


# ─────────────────────────────────────────────────────────────────────────────
#  TASK 4: STRONGER BASELINE (LEARNED REGRESSION VS. PHYSICS DIGITAL TWIN)
# ─────────────────────────────────────────────────────────────────────────────

def run_learned_baseline_comparison(detector_twin: CalibratedAnomalyDetector) -> pd.DataFrame:
    print("\n" + "="*80)
    print("TASK 4 — Competitor Baseline: Learned Regression Model vs. Physics Digital Twin")
    print("="*80)

    # 1. Fit Multi-Variable Polynomial Regression Model on Training Envelope (0 - 4000 m)
    X_train = []; Y_train = []
    for alt in [500, 1500, 2500, 3500]:
        for isa in [-5, 0, +5]:
            eng = SimulatedEngine("climb" if alt < 2000 else "cruise_altitude", 42, alt_m=alt, isa_offset=isa)
            r, _, _ = eng.generate_run(400.0, DT)
            thr = 0.85 if alt < 2000 else 0.70
            for row in r[::10]:
                X_train.append([thr, alt / 1000.0, isa / 10.0])
                Y_train.append(row)

    X_tr = np.array(X_train); Y_tr = np.array(Y_train)
    X_tr_poly = np.hstack([np.ones((len(X_tr), 1)), X_tr, X_tr**2])
    W_reg = np.linalg.lstsq(X_tr_poly, Y_tr, rcond=None)[0]

    # Fit detector on learned regression residuals in-distribution
    eng_in = SimulatedEngine("cruise_altitude", 100, alt_m=2000, isa_offset=0)
    r_in, _, _ = eng_in.generate_run(800.0, DT)
    X_in = np.array([[0.70, 2.0, 0.0]] * len(r_in))
    pred_in = np.hstack([np.ones((len(X_in), 1)), X_in, X_in**2]) @ W_reg
    res_reg_in = r_in - pred_in
    detector_reg = CalibratedAnomalyDetector(threshold_pct=THRESHOLD_PCT, window_size=WINDOW_SIZE).fit(res_reg_in)

    # Test Scenarios
    scenarios = [
        ("Nominal In-Distribution (2,000m Cruise)", 2000, 0.0, None),
        ("High-Altitude Expansion (6,000m Cruise)", 6000, 0.0, None),
        ("Extreme Envelope Excursion (7,500m Hot-Day)", 7500, +15.0, None),
        ("Engine Component Mismatch (+15% Aging)", 3000, 0.0, {"cht": 1.15, "rpm": 1.08, "oil_t": 1.10}),
    ]

    records = []

    for name, alt, isa, tp in scenarios:
        eng = SimulatedEngine("cruise_altitude", 20000, alt_m=alt, isa_offset=isa, twin_params=tp)
        readings, res_twin, _ = eng.generate_run(1800.0, DT, fault_id=None)

        # Regression residual
        thr = 0.70
        X_eval = np.array([[thr, alt / 1000.0, isa / 10.0]] * len(readings))
        pred_eval = np.hstack([np.ones((len(X_eval), 1)), X_eval, X_eval**2]) @ W_reg
        res_reg = readings - pred_eval

        # Measure residual bias & false alert rate
        twin_bias = float(np.abs(res_twin.mean()))
        reg_bias  = float(np.abs(res_reg.mean()))

        fa_twin = 0; fa_reg = 0
        c_tw = 0; c_rg = 0
        total_w = 0

        for t in range(WINDOW_SIZE, len(readings), EVAL_STRIDE):
            total_w += 1
            w_tw = res_twin[t - WINDOW_SIZE:t]
            w_rg = res_reg[t - WINDOW_SIZE:t]

            if detector_twin.is_anomalous(detector_twin.score(w_tw)):
                c_tw += 1
                if c_tw == PERSISTENCE_COUNT: fa_twin += 1
            else: c_tw = 0

            if detector_reg.is_anomalous(detector_reg.score(w_rg)):
                c_rg += 1
                if c_rg == PERSISTENCE_COUNT: fa_reg += 1
            else: c_rg = 0

        h = 1800.0 / 3600.0
        fa_rate_twin = (fa_twin / h) * 100.0
        fa_rate_reg  = (fa_reg / h) * 100.0

        records.append({
            "Evaluation_Scenario": name,
            "Physics_Twin_Bias": round(twin_bias, 4),
            "Learned_Reg_Bias": round(reg_bias, 4),
            "Physics_Twin_FA_per_100h": round(fa_rate_twin, 1),
            "Learned_Reg_FA_per_100h": round(fa_rate_reg, 1),
            "Winner": "Tie (Both clean)" if (fa_rate_twin == 0 and fa_rate_reg == 0) else ("Physics Twin (Robust)" if fa_rate_twin < fa_rate_reg else "Regression"),
        })

        print(f"  {name:<44} | Reg Bias={reg_bias:6.3f} vs Twin={twin_bias:6.3f} | "
              f"FA/100h: Reg={fa_rate_reg:5.1f} vs Twin={fa_rate_twin:4.1f} -> {records[-1]['Winner']}")

    df_comp = pd.DataFrame(records)
    df_comp.to_csv(RESULTS_DIR / "learned_baseline_comparison.csv", index=False)
    return df_comp


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("="*80)
    print("AEROTWIN-4 COMPREHENSIVE EXPERIMENT EXTENSIONS")
    print("="*80)

    t0 = time.perf_counter()

    eng_fit = SimulatedEngine("cruise_altitude", GLOBAL_SEED)
    _, h_res, _ = eng_fit.generate_run(800.0, DT)
    detector   = CalibratedAnomalyDetector(threshold_pct=THRESHOLD_PCT, window_size=WINDOW_SIZE).fit(h_res)
    classifier = GatedFaultClassifier()

    df_m8, df_mm  = run_classifier_evaluation(classifier)
    df_sens       = run_sensitivity_curve(detector)
    df_sweep      = run_ramp_rate_sweep(detector)
    df_comp       = run_learned_baseline_comparison(detector)

    elapsed = time.perf_counter() - t0
    print(f"\n[DONE] All 4 analyses completed in {elapsed:.1f} seconds.")
    print(f"Outputs saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
