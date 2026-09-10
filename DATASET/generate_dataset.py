"""
=============================================================================
AEROTWIN-4 CORRECTED DATASET GENERATOR v2 — HYBRID PHYSICS MODEL
=============================================================================
Uses the ACTUAL AeroTwin physics engine for all thermodynamic outputs
(RPM, CHT, EGT, oil pressure/temp, fuel flow). For faults where the engine
physics model lacks the required signature (vibration, misfire oscillation,
injector variance), adds PHYSICS-CONSISTENT SYNTHETIC PERTURBATIONS that
are applied IDENTICALLY in xgboost_adapter.py during live inference.

This guarantees zero training/inference domain gap.

Fault signatures (measured from real engine + added perturbations):
  0: NORMAL          — real engine, no perturbation
  1: MISFIRE         — CYLINDER fault → real RPM drop + synthetic vib & EGT dip
  2: INJECTOR_FAULT  — CYLINDER fault → real RPM change + synthetic fuel/EGT variance
  3: COOLING_DEGRAD  — COOLING fault  → real CHT/oil_temp rise (engine accurate)
  4: LUBRICATION     — LUBRICATION fault → real oil pressure drop (engine accurate)
  5: SENSOR_DRIFT    — healthy engine + EGT bias drift (synthetic, no engine change)
  6: COMBUSTION_INST — CYLINDER fault → real changes + synthetic oscillation
  7: OVERHEATING     — COOLING fault severe → real CHT rise (engine accurate)
  8: ABNORMAL_VIB    — BEARING fault → real small vib + amplified synthetic vib
=============================================================================
"""
import os
import sys
import json
import math
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

_BASE = r"e:\PRATIBIMB\DASHBOARD AND DATA"
if _BASE not in sys.path:
    sys.path.insert(0, _BASE)

from AeroTwin.simulator.runner import EngineRunner
from AeroTwin.api.pipeline import LiveAssessmentPipeline, default_engine_parameters
from AeroTwin.degradation.config import (
    DegradationConfig, DegradationType, ComponentID, TrajectoryType
)
from AeroTwin.degradation.injector import DegradationInjector
from AeroTwin.mission.risk import MissionProfile

# ============================================================
# Constants — MUST match xgboost_adapter.py exactly
# ============================================================
OUTPUT_DIR  = r"e:\PRATIBIMB\DATASET"
FEAT_DIR    = os.path.join(OUTPUT_DIR, "features")
RAW_DIR     = os.path.join(OUTPUT_DIR, "raw")

FAULT_NAMES = [
    "NORMAL", "MISFIRE", "INJECTOR_ABNORMALITY", "CODING_DEGRADATION",
    "LUBRICATION_ISSUE", "SENSOR_DRIFT_FAILURE", "COMBUSTION_INSTABILITY",
    "OVERHEATING", "ABNORMAL_VIBRATION",
]

# Calibrated from REAL engine output distributions.
# MUST MATCH xgboost_adapter.py NOISE_STD exactly.
NOISE_STD = {
    "rpm":       30.0,
    "cht":        2.0,
    "egt":        8.0,
    "oil_press":  0.08,  # bar
    "oil_temp":   1.2,
    "fuel_flow":  0.6,
    "vibration":  0.12,  # calibrated: real engine vib std ~0.25-0.30
}

# Healthy vibration baseline — calibrated from real engine at 0.65 throttle
# This is what the DT expects. Used to compute vibration_residual correctly.
VIB_BASELINE = 0.05   # minimum (idle)
VIB_RPM_COEFF = 0.25  # per 1000 rpm

def healthy_vib_expected(rpm: float) -> float:
    """Compute expected healthy vibration from RPM — matches adapter formula."""
    return VIB_BASELINE + (rpm / 1000.0) * VIB_RPM_COEFF

_ROOT_DIR   = r"e:\PRATIBIMB\DASHBOARD AND DATA"
PROFILE_NAMES = ["CLIMB_CRUISE", "EXTENDED_CRUISE", "LOITER", "DESCENT", "HIGH_THROTTLE"]
MISSION_PROFILES: Dict[str, List[Tuple]] = {
    "CLIMB_CRUISE":    [(0, 0.15, 0, 15), (60, 0.90, 1000, 13), (240, 0.70, 8000, 8)],
    "EXTENDED_CRUISE": [(0, 0.65, 8000, 8), (120, 0.68, 9000, 7), (240, 0.65, 8500, 8)],
    "LOITER":          [(0, 0.45, 5000, 11), (120, 0.48, 5200, 11), (240, 0.44, 5000, 11)],
    "DESCENT":         [(0, 0.65, 10000, 9), (60, 0.45, 6000, 12), (240, 0.20, 500, 14)],
    "HIGH_THROTTLE":   [(0, 0.85, 3000, 12), (180, 0.90, 3500, 11), (240, 0.80, 3000, 12)],
}


def _interp(t, wps):
    times = [w[0] for w in wps]; thrs = [w[1] for w in wps]
    alts  = [w[2] for w in wps]; ambs = [w[3] for w in wps]
    if t <= times[0]:  return thrs[0], alts[0], ambs[0]
    if t >= times[-1]: return thrs[-1], alts[-1], ambs[-1]
    for i in range(len(times)-1):
        if times[i] <= t <= times[i+1]:
            f = (t-times[i])/(times[i+1]-times[i])
            return thrs[i]+f*(thrs[i+1]-thrs[i]), alts[i]+f*(alts[i+1]-alts[i]), ambs[i]+f*(ambs[i+1]-ambs[i])
    return thrs[-1], alts[-1], ambs[-1]


def _ramp(sim_time, onset_s, ramp_s):
    if sim_time < onset_s: return 0.0
    return min(1.0, (sim_time - onset_s) / max(1.0, ramp_s))


def _make_fault_config(fault_class, severity, onset_s, ramp_s, rng):
    cyl = [ComponentID.CYLINDER_1, ComponentID.CYLINDER_2,
           ComponentID.CYLINDER_3, ComponentID.CYLINDER_4][int(rng.integers(0,4))]
    if fault_class == 0: return DegradationConfig.healthy()
    elif fault_class == 1:  # MISFIRE — abrupt cylinder
        return DegradationConfig.single_fault(DegradationType.CYLINDER, cyl, severity,
            TrajectoryType.STEP, start_time=onset_s, ramp_duration=8.0)
    elif fault_class == 2:  # INJECTOR_ABNORMALITY — gradual cylinder
        return DegradationConfig.single_fault(DegradationType.CYLINDER, cyl, severity*0.6,
            TrajectoryType.LINEAR, start_time=onset_s, ramp_duration=ramp_s)
    elif fault_class == 3:  # CODING_DEGRADATION (cooling)
        return DegradationConfig.single_fault(DegradationType.COOLING, ComponentID.COOLING_SYSTEM,
            severity*0.85, TrajectoryType.LINEAR, start_time=onset_s, ramp_duration=ramp_s)
    elif fault_class == 4:  # LUBRICATION_ISSUE
        return DegradationConfig.single_fault(DegradationType.LUBRICATION, ComponentID.LUBRICATION_SYSTEM,
            severity, TrajectoryType.LINEAR, start_time=onset_s, ramp_duration=ramp_s)
    elif fault_class == 5: return DegradationConfig.healthy()  # SENSOR_DRIFT — engine healthy
    elif fault_class == 6:  # COMBUSTION_INSTABILITY
        return DegradationConfig.single_fault(DegradationType.CYLINDER, cyl, severity*0.5,
            TrajectoryType.CONSTANT, start_time=onset_s, ramp_duration=ramp_s)
    elif fault_class == 7:  # OVERHEATING — severe cooling
        return DegradationConfig.single_fault(DegradationType.COOLING, ComponentID.COOLING_SYSTEM,
            min(1.0, severity*1.1), TrajectoryType.EXPONENTIAL, start_time=onset_s, ramp_duration=ramp_s*1.5)
    elif fault_class == 8:  # ABNORMAL_VIBRATION — bearing
        return DegradationConfig.single_fault(DegradationType.BEARING, ComponentID.BEARING,
            severity, TrajectoryType.LINEAR, start_time=onset_s, ramp_duration=ramp_s)
    return DegradationConfig.healthy()


def apply_fault_perturbations(
    fault_class: int, ramp_factor: float, severity: float, sim_time: float,
    rpm: float, egt: float, fuel: float, vib: float,
    drift_bias: float, rng: np.random.Generator
) -> Tuple[float, float, float, float, float, float, float]:
    """
    Apply physics-consistent synthetic perturbations for faults where the engine
    physics model does not produce the correct signature.

    Returns: (rpm, egt, fuel, vib, vib_kurtosis, vib_crest, drift_bias)

    CRITICAL: This function is MIRRORED EXACTLY in xgboost_adapter.py.
    Any change here MUST be made there too.
    """
    vib_kurtosis = 3.0
    vib_crest    = 3.5

    if fault_class == 1 and ramp_factor > 0:
        # MISFIRE: stochastic RPM dips, EGT dip (unburned mixture), high vibration kurtosis
        p_misfire = severity * ramp_factor * 0.7
        if rng.random() < p_misfire:
            rpm    -= severity * ramp_factor * 350.0
            egt    -= severity * ramp_factor * 180.0
            vib    += severity * ramp_factor * 0.60
            vib_kurtosis += severity * ramp_factor * 8.0
            vib_crest    += severity * ramp_factor * 4.0
        else:
            vib_kurtosis += severity * ramp_factor * 1.5   # mild even between events

    elif fault_class == 2 and ramp_factor > 0:
        # INJECTOR_ABNORMALITY: high-variance fuel delivery, EGT swings
        fuel_err  = float(rng.normal(0, severity * ramp_factor * 3.5))
        fuel     += fuel_err
        egt      -= fuel_err * 8.0   # lean -> high EGT, rich -> low EGT
        vib      += abs(fuel_err) * 0.04   # slight mechanical sympathetic vibration
        vib_kurtosis += severity * ramp_factor * 1.5

    elif fault_class == 6 and ramp_factor > 0:
        # COMBUSTION_INSTABILITY: oscillating RPM, elevated vibration
        osc       = severity * ramp_factor * 100.0 * math.sin(sim_time * 2.1)
        rpm      += osc
        vib      += severity * ramp_factor * 0.35
        vib_kurtosis += severity * ramp_factor * 3.5
        vib_crest    += severity * ramp_factor * 1.5

    elif fault_class == 8 and ramp_factor > 0:
        # ABNORMAL_VIBRATION: bearing wear — amplify the existing vibration signal
        vib          += severity * ramp_factor * 0.80
        vib_kurtosis += severity * ramp_factor * 7.0
        vib_crest    += severity * ramp_factor * 4.0

    elif fault_class == 7 and ramp_factor > 0:
        # OVERHEATING: severe thermal runaway — CHT and EGT amplification
        # This distinguishes OVERHEATING from CODING_DEGRADATION (both use COOLING fault)
        # OVERHEATING: faster, more severe CHT rise + EGT increase
        # Coding degradation only gets the real engine's moderate CHT rise
        egt += severity * ramp_factor * 45.0   # exhaust temp rises in thermal runaway

    elif fault_class == 5 and ramp_factor > 0:
        # SENSOR_DRIFT: EGT thermocouple drift — accumulated separately in the caller
        pass  # handled via drift_bias outside

    return rpm, egt, fuel, vib, vib_kurtosis, vib_crest, drift_bias


def simulate_run_real_engine(run_id, fault_class, duration_s=300.0, seed=42):
    rng = np.random.Generator(np.random.PCG64(seed))
    severity = 0.0; onset_s = duration_s + 10.0
    if fault_class != 0:
        severity = float(rng.uniform(0.55, 0.92))
        onset_s  = float(rng.uniform(20.0, 55.0))
    ramp_s = float(rng.uniform(35.0, 90.0))

    profile_wps = MISSION_PROFILES[PROFILE_NAMES[run_id % len(PROFILE_NAMES)]]
    fault_cfg   = _make_fault_config(fault_class, severity, onset_s, ramp_s, rng)
    params      = default_engine_parameters(f"RUN_{run_id:04d}", seed)
    runner      = EngineRunner(dt=0.01, seed=seed, engine_parameters=params)
    injector    = DegradationInjector(config=fault_cfg, runner=runner,
                                      run_id=f"RUN_{run_id:04d}", noise_enabled=True)
    pipeline    = LiveAssessmentPipeline(root_dir=_ROOT_DIR, engine_parameters=params, seed=seed,
                                         dt=0.01, mission=MissionProfile(name="DS", required_duration_s=duration_s))

    drift_bias = 0.0
    records    = []
    step_count = 0
    SIM_HZ     = 100

    while True:
        sim_t = injector.runner.clock.simulation_time
        if sim_t >= duration_s:
            break

        thr, alt_ft, amb_c = _interp(sim_t, profile_wps)
        thr = float(np.clip(thr + rng.normal(0, 0.01), 0.10, 1.0))
        runner.set_throttle(thr)

        tel, gt  = injector.step()
        step_count += 1
        sim_time  = tel.simulation_time

        if step_count % SIM_HZ != 0:
            continue

        d   = tel.to_dict()
        res = pipeline.ingest(d)
        exp = res.get("expected", {})

        ramp_factor = _ramp(sim_time, onset_s, ramp_s)

        # Raw engine outputs
        rpm   = d["rpm"]
        cht   = d["cht"]
        egt   = d["egt"]
        op_psi = d.get("oil_pressure_psi", d.get("oil_pressure", 58.0))
        oil_p  = op_psi * 0.0689476 if op_psi > 15.0 else op_psi
        oil_t  = d["oil_temperature"]
        fuel   = d.get("fuel_flow_lph", d.get("fuel_flow", 6.0))
        vib    = d["vibration"]
        throttle = d["throttle"]

        # Sensor drift accumulation for class 5
        if fault_class == 5 and ramp_factor > 0:
            drift_bias += float(rng.normal(0.25, 0.12)) * severity * ramp_factor
            drift_bias  = min(drift_bias, 100.0)
        else:
            drift_bias = 0.0

        # Apply synthetic perturbations
        vib_kurtosis = 3.0
        vib_crest    = 3.5
        rpm, egt, fuel, vib, vib_kurtosis, vib_crest, drift_bias = apply_fault_perturbations(
            fault_class, ramp_factor, severity, sim_time,
            rpm, egt, fuel, vib, drift_bias, rng
        )
        egt_measured = egt + drift_bias   # class 5 EGT drift on top

        # DT expected from pipeline (healthy twin)
        e_rpm   = exp.get("rpm",   rpm)
        e_cht   = exp.get("cht",   cht)
        e_egt   = exp.get("egt",   d["egt"])   # DT does NOT know about sensor drift
        ep_psi  = exp.get("oil_pressure_psi", exp.get("oil_pressure", op_psi))
        e_op    = ep_psi * 0.0689476 if ep_psi > 15.0 else ep_psi
        e_ot    = exp.get("oil_temperature", oil_t)
        e_fuel  = exp.get("fuel_flow_lph", exp.get("fuel_flow", d.get("fuel_flow_lph", fuel)))
        e_vib   = exp.get("vibration", d["vibration"])   # DT expected healthy vibration

        # Residuals
        res_rpm   = rpm         - e_rpm
        res_cht   = cht         - e_cht
        res_egt   = egt_measured - e_egt
        res_op    = oil_p       - e_op
        res_ot    = oil_t       - e_ot
        res_fuel  = fuel        - e_fuel
        res_vib   = vib         - e_vib

        # Z-scores
        z_rpm   = res_rpm   / NOISE_STD["rpm"]
        z_cht   = res_cht   / NOISE_STD["cht"]
        z_egt   = res_egt   / NOISE_STD["egt"]
        z_op    = res_op    / NOISE_STD["oil_press"]
        z_ot    = res_ot    / NOISE_STD["oil_temp"]
        z_fuel  = res_fuel  / NOISE_STD["fuel_flow"]
        z_vib   = res_vib   / NOISE_STD["vibration"]

        alt_m    = alt_ft * 0.3048
        p_kpa    = 101.325 * ((1.0 - 2.25577e-5 * alt_m) ** 5.25588)
        load_pct = min(100.0, max(15.0, throttle * 100.0 * (p_kpa / 101.325)))
        label    = fault_class if (ramp_factor > 0 and fault_class != 0) else 0

        records.append({
            "run_id": run_id, "timestamp_s": sim_time,
            "fault_class": label, "fault_name": FAULT_NAMES[label],
            "fault_severity": gt.active_severity,
            "altitude_ft": alt_ft, "ambient_temperature_c": amb_c,
            "ambient_pressure_kpa": p_kpa, "humidity_pct": 50.0,
            "throttle": throttle, "engine_load": load_pct,
            "injection_command": throttle * 0.95 + 0.05,
            "injection_timing": 22.0 + throttle * 6.0,
            "rpm": rpm, "cht": cht, "egt": egt_measured,
            "oil_pressure": oil_p, "oil_temperature": oil_t, "fuel_flow": fuel,
            "battery_voltage": d.get("battery_voltage", 13.8 + 0.4 * (rpm / 5800.0)),
            "alternator_current": d.get("alternator_current", 15.0 + throttle * 25.0),
            "injection_timing_deg": 22.0 + throttle * 6.0,
            "vibration_rms": vib, "vibration_std": vib * 0.25,
            "vibration_kurtosis": vib_kurtosis, "vibration_crest_factor": vib_crest,
            "vibration_peak_frequency": rpm / 60.0,
            "vibration_1x": vib * 0.6, "vibration_2x": vib * 0.2, "vibration_3x": vib * 0.08,
            "rpm_residual": res_rpm, "cht_residual": res_cht, "egt_residual": res_egt,
            "oil_pressure_residual": res_op, "oil_temperature_residual": res_ot,
            "fuel_flow_residual": res_fuel, "vibration_residual": res_vib,
            "rpm_z": z_rpm, "cht_z": z_cht, "egt_z": z_egt,
            "oil_pressure_z": z_op, "oil_temperature_z": z_ot,
            "fuel_flow_z": z_fuel, "vibration_z": z_vib,
        })

    return pd.DataFrame(records)


def extract_feature_windows(df_run, window_size=30, stride=10):
    n = len(df_run)
    if n < window_size: return pd.DataFrame()
    windows = []; run_id = int(df_run["run_id"].iloc[0])

    for start in range(0, n - window_size + 1, stride):
        sub   = df_run.iloc[start:start + window_size]
        last  = sub.iloc[-1]; first = sub.iloc[0]; dt = float(window_size)
        vib_kurt = sub["vibration_kurtosis"].mean()

        feat = {
            "run_id": run_id, "window_start_s": float(first["timestamp_s"]),
            "window_end_s": float(last["timestamp_s"]),
            "fault_name": str(last["fault_name"]), "fault_severity": float(last["fault_severity"]),
            "altitude_ft": float(last["altitude_ft"]),
            "ambient_temperature_c": float(last["ambient_temperature_c"]),
            "ambient_pressure_kpa": float(last["ambient_pressure_kpa"]),
            "humidity_pct": float(last["humidity_pct"]),
            "throttle": float(last["throttle"]), "engine_load": float(last["engine_load"]),
            "injection_command": float(last["injection_command"]),
            "injection_timing": float(last["injection_timing"]),
            "rpm": float(last["rpm"]), "cht": float(last["cht"]), "egt": float(last["egt"]),
            "oil_pressure": float(last["oil_pressure"]),
            "oil_temperature": float(last["oil_temperature"]),
            "fuel_flow": float(last["fuel_flow"]),
            "battery_voltage": float(last["battery_voltage"]),
            "alternator_current": float(last["alternator_current"]),
            "injection_timing_deg": float(last["injection_timing_deg"]),
            "vibration_rms": float(last["vibration_rms"]),
            "vibration_std": float(last["vibration_std"]),
            "vibration_kurtosis": float(vib_kurt),
            "vibration_crest_factor": float(last["vibration_crest_factor"]),
            "vibration_peak_frequency": float(last["vibration_peak_frequency"]),
            "vibration_1x": float(last["vibration_1x"]),
            "vibration_2x": float(last["vibration_2x"]),
            "vibration_3x": float(last["vibration_3x"]),
            "rpm_residual": float(last["rpm_residual"]),
            "cht_residual": float(last["cht_residual"]),
            "egt_residual": float(last["egt_residual"]),
            "oil_pressure_residual": float(last["oil_pressure_residual"]),
            "oil_temperature_residual": float(last["oil_temperature_residual"]),
            "fuel_flow_residual": float(last["fuel_flow_residual"]),
            "vibration_residual": float(last["vibration_residual"]),
            "rpm_z": float(last["rpm_z"]), "cht_z": float(last["cht_z"]),
            "egt_z": float(last["egt_z"]), "oil_pressure_z": float(last["oil_pressure_z"]),
            "oil_temperature_z": float(last["oil_temperature_z"]),
            "fuel_flow_z": float(last["fuel_flow_z"]),
            "vibration_z": float(last["vibration_z"]),
            "cht_slope":           (last["cht"] - first["cht"]) / dt,
            "egt_slope":           (last["egt"] - first["egt"]) / dt,
            "oil_pressure_slope":  (last["oil_pressure"] - first["oil_pressure"]) / dt,
            "oil_temperature_slope":(last["oil_temperature"] - first["oil_temperature"]) / dt,
            "fuel_flow_slope":     (last["fuel_flow"] - first["fuel_flow"]) / dt,
            "vibration_slope":     (last["vibration_rms"] - first["vibration_rms"]) / dt,
            "cht_residual_slope":  (last["cht_residual"] - first["cht_residual"]) / dt,
            "egt_residual_slope":  (last["egt_residual"] - first["egt_residual"]) / dt,
            "oil_pressure_residual_slope": (last["oil_pressure_residual"] - first["oil_pressure_residual"]) / dt,
            "fuel_residual_slope": (last["fuel_flow_residual"] - first["fuel_flow_residual"]) / dt,
            "vibration_residual_slope": (last["vibration_residual"] - first["vibration_residual"]) / dt,
            "egt_residual_mean":   float(sub["egt_residual"].mean()),
            "egt_residual_std":    float(sub["egt_residual"].std() or 0.0),
            "cht_residual_mean":   float(sub["cht_residual"].mean()),
            "cht_residual_std":    float(sub["cht_residual"].std() or 0.0),
            "oil_pressure_residual_mean": float(sub["oil_pressure_residual"].mean()),
            "oil_pressure_residual_std":  float(sub["oil_pressure_residual"].std() or 0.0),
            "fuel_residual_mean":  float(sub["fuel_flow_residual"].mean()),
            "fuel_residual_std":   float(sub["fuel_flow_residual"].std() or 0.0),
            "vibration_mean":      float(sub["vibration_rms"].mean()),
            "vibration_kurtosis_mean": float(vib_kurt),
            "fault_class": int(last["fault_class"]),
        }
        windows.append(feat)
    return pd.DataFrame(windows)


def generate_master_dataset(duration_s=300.0):
    os.makedirs(FEAT_DIR, exist_ok=True); os.makedirs(RAW_DIR, exist_ok=True)
    class_alloc = {0:30, 1:20, 2:20, 3:20, 4:20, 5:20, 6:20, 7:20, 8:20}
    total = sum(class_alloc.values())
    print(f"[*] Generating {total} runs (real engine + hybrid perturbations)...")

    all_windows, runs_meta, run_counter = [], [], 0

    for f_class, n_runs in class_alloc.items():
        print(f"  -> Class {f_class} ({FAULT_NAMES[f_class]}): {n_runs} runs", end="", flush=True)
        for i in range(n_runs):
            run_counter += 1; seed = run_counter * 31 + 7
            df  = simulate_run_real_engine(run_counter, f_class, duration_s, seed)
            if i == 0:
                df.to_csv(os.path.join(RAW_DIR, f"run_{run_counter:03d}_{FAULT_NAMES[f_class]}.csv"), index=False)
            wdf = extract_feature_windows(df)
            if len(wdf) > 0: all_windows.append(wdf)
            runs_meta.append({"run_id": run_counter, "fault_class": f_class,
                               "fault_name": FAULT_NAMES[f_class], "num_windows": len(wdf)})
            print(".", end="", flush=True)
        print()

    full_df = pd.concat(all_windows, ignore_index=True)
    runs_meta_df = pd.DataFrame(runs_meta)
    print(f"\n[OK] Total windows: {len(full_df)}")
    print(f"     Class distribution:\n{full_df['fault_class'].value_counts().sort_index()}")

    print("\n=== Calibration: vibration_z by class (NORMAL should be ~0) ===")
    for fc, fn in enumerate(FAULT_NAMES):
        sub = full_df[full_df['fault_class']==fc]
        if len(sub):
            vz = sub['vibration_z']
            print(f"  {fn:<30}: z mean={vz.mean():.2f}  std={vz.std():.2f}  max={vz.max():.2f}")

    print("\n=== Calibration: egt_z by class (SENSOR_DRIFT should be high) ===")
    for fc, fn in enumerate(FAULT_NAMES):
        sub = full_df[full_df['fault_class']==fc]
        if len(sub):
            ez = sub['egt_z']
            print(f"  {fn:<30}: egt_z mean={ez.mean():.2f}  std={ez.std():.2f}")

    # Run-level split 70/15/15
    rng_s = np.random.Generator(np.random.PCG64(2028))
    train_runs, val_runs, test_runs = [], [], []
    for f_class, n_runs in class_alloc.items():
        ids = runs_meta_df[runs_meta_df["fault_class"]==f_class]["run_id"].values.copy()
        rng_s.shuffle(ids)
        n_tr = max(1, int(round(0.70*len(ids)))); n_va = max(1, int(round(0.15*len(ids))))
        if n_tr+n_va >= len(ids): n_va = max(1, len(ids)-n_tr-1)
        train_runs.extend(ids[:n_tr]); val_runs.extend(ids[n_tr:n_tr+n_va]); test_runs.extend(ids[n_tr+n_va:])

    train_df = full_df[full_df["run_id"].isin(train_runs)].copy().reset_index(drop=True)
    val_df   = full_df[full_df["run_id"].isin(val_runs)].copy().reset_index(drop=True)
    test_df  = full_df[full_df["run_id"].isin(test_runs)].copy().reset_index(drop=True)
    print(f"\n[*] Split: Train {len(train_df)} | Val {len(val_df)} | Test {len(test_df)}")

    for name, df in [("train",train_df),("validation",val_df),("test",test_df)]:
        df.to_csv(os.path.join(FEAT_DIR, f"{name}.csv"), index=False)
        try: df.to_parquet(os.path.join(FEAT_DIR, f"{name}.parquet"), index=False)
        except Exception as e: print(f"  parquet: {e}")

    meta_cols     = {"run_id","window_start_s","window_end_s","fault_name","fault_severity","fault_class"}
    feature_names = [c for c in train_df.columns if c not in meta_cols]
    groups        = {
        "environment":          ["altitude_ft","ambient_temperature_c","ambient_pressure_kpa","humidity_pct"],
        "commands":             ["throttle","engine_load","injection_command","injection_timing"],
        "sensors":              ["rpm","cht","egt","oil_pressure","oil_temperature","fuel_flow",
                                 "battery_voltage","alternator_current","injection_timing_deg"],
        "vibration":            ["vibration_rms","vibration_std","vibration_kurtosis","vibration_crest_factor",
                                 "vibration_peak_frequency","vibration_1x","vibration_2x","vibration_3x"],
        "physics_residuals":    ["rpm_residual","cht_residual","egt_residual","oil_pressure_residual",
                                 "oil_temperature_residual","fuel_flow_residual","vibration_residual"],
        "normalized_residuals": ["rpm_z","cht_z","egt_z","oil_pressure_z","oil_temperature_z","fuel_flow_z","vibration_z"],
        "trends":               ["cht_slope","egt_slope","oil_pressure_slope","oil_temperature_slope","fuel_flow_slope",
                                 "vibration_slope","cht_residual_slope","egt_residual_slope",
                                 "oil_pressure_residual_slope","fuel_residual_slope","vibration_residual_slope"],
        "rolling_statistics":   ["egt_residual_mean","egt_residual_std","cht_residual_mean","cht_residual_std",
                                 "oil_pressure_residual_mean","oil_pressure_residual_std","fuel_residual_mean",
                                 "fuel_residual_std","vibration_mean","vibration_kurtosis_mean"],
    }
    metadata = {
        "dataset_name": "AeroTwin-4 Hybrid Physics Dataset v2",
        "generation_note": "Real engine thermodynamics + physics-consistent synthetic perturbations for vibration/combustion faults",
        "noise_std_calibrated": NOISE_STD,
        "total_runs": run_counter, "total_windows": len(full_df),
        "window_size_seconds": 30, "stride_seconds": 10,
        "classes": {i: n for i, n in enumerate(FAULT_NAMES)},
        "splits": {"train_runs": len(train_runs), "train_windows": len(train_df),
                   "val_runs": len(val_runs), "val_windows": len(val_df),
                   "test_runs": len(test_runs), "test_windows": len(test_df)},
        "feature_groups": groups,
        "num_features": len(feature_names),
        "target": "fault_class",
    }
    with open(os.path.join(FEAT_DIR, "dataset_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\n[OK] Dataset saved to {FEAT_DIR}")
    return metadata, feature_names


if __name__ == "__main__":
    generate_master_dataset()
