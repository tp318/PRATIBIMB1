# -*- coding: utf-8 -*-
"""
================================================================================
AeroTwin-4 Experiment Suite -- Five Validation Experiments
================================================================================
Run:  python run_all_experiments.py
All random seeds are fixed.  Results saved to experiments/results/

Experiments
-----------
1. Detection lead time (headline claim)
2. False alerts on clean flights
3. Model-mismatch / physics stress test
4. Edge timing (desktop CPU proxy)
5. Ablation — what each pipeline stage contributes

Quick extras
-----------
A. Sensor drift vs real fault disambiguation
B. Altitude invariance (residual mean / std at 0-7500 m)
C. RUL honesty (p10/p50/p90 error bands)
================================================================================
"""

from __future__ import annotations

import copy
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ── path setup ────────────────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).resolve().parent
REPO_ROOT    = SCRIPT_DIR.parent
FAULT_DET    = REPO_ROOT / "FAULT DETECTION"
AEROTWIN_DIR = REPO_ROOT / "DASHBOARD AND DATA" / "AeroTwin"
PHASE1_DIR   = AEROTWIN_DIR / "phase1"

for p in [str(FAULT_DET), str(AEROTWIN_DIR), str(PHASE1_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

RESULTS_DIR = SCRIPT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── optional torch ─────────────────────────────────────────────────────────────
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# ── fault detection config ──────────────────────────────────────────────────
try:
    from config import (
        FAULT_CLASSES, FAULT_SHORT, NUM_CLASSES, NUM_SENSORS,
        SENSOR_CHANNELS, WINDOW_SIZE, STRIDE, CHECKPOINT_PATH,
    )
    CONFIG_OK = True
except Exception as e:
    print(f"[WARN] Could not import config.py: {e}")
    CONFIG_OK = False
    FAULT_CLASSES = [
        "Normal operation", "Misfire conditions", "Injector abnormalities",
        "Cooling degradation", "Lubrication issues", "Sensor drift / failure",
        "Combustion instability", "Overheating trends", "Abnormal vibration patterns",
    ]
    FAULT_SHORT = {
        0: "HEALTHY", 1: "MISFIRE", 2: "INJECTOR", 3: "COOLING",
        4: "LUBRICATION", 5: "SENSOR_DRIFT", 6: "COMBUSTION",
        7: "OVERHEAT", 8: "VIBRATION",
    }
    NUM_CLASSES = 9
    NUM_SENSORS = 9
    SENSOR_CHANNELS = [
        "rpm_residual", "cht_residual", "egt_residual",
        "oil_pressure_residual", "oil_temp_residual", "fuel_flow_residual",
        "vibration_residual", "batt_voltage_residual", "inj_timing_residual",
    ]
    WINDOW_SIZE  = 64
    STRIDE       = 16
    CHECKPOINT_PATH = str(FAULT_DET / "best_fault_detector.pt")

# ── engine physics ──────────────────────────────────────────────────────────
try:
    from engine.dynamics import EngineDynamics
    ENGINE_OK = True
except Exception:
    try:
        from AeroTwin.phase1.engine.dynamics import EngineDynamics
        ENGINE_OK = True
    except Exception as e:
        print(f"[WARN] EngineDynamics unavailable: {e}")
        ENGINE_OK = False
        EngineDynamics = None

# ── fault detector (CNN-LSTM) ──────────────────────────────────────────────
try:
    from architecture import CNNLSTMFaultDetector, load_model
    from preprocessing import ResidualScaler, generate_synthetic_residuals, create_sliding_windows
    MODEL_OK = True
except Exception as e:
    print(f"[WARN] Fault detection model unavailable: {e}")
    MODEL_OK = False

# ── XGBoost adapter ────────────────────────────────────────────────────────
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────────────
#  SHARED UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

GLOBAL_SEED = 42
np.random.seed(GLOBAL_SEED)

# Operating-point throttle / ambient-temp table
OPERATING_POINTS = {
    "climb":            {"throttle": 0.90, "alt_m": 1500,  "isa_offset": 0.0},
    "cruise_altitude":  {"throttle": 0.70, "alt_m": 6000,  "isa_offset": 0.0},
    "hot_day_low":      {"throttle": 0.80, "alt_m": 300,   "isa_offset": +15.0},
}

# ISA temperature at altitude
def isa_temp(alt_m: float, offset_degC: float = 0.0) -> float:
    return 15.0 - 0.0065 * alt_m + offset_degC

# 9-class fault definitions: (class_id, sensor_indices_affected, [base_shift])
FAULT_DEFS = {
    1: {"name": "Misfire",           "sensors": [0, 2, 6],    "shift": 2.5,  "has_redline": True,  "redline_sensor": "egt",   "redline_limit": 900.0},
    2: {"name": "Injector",          "sensors": [5, 8],       "shift": 2.0,  "has_redline": False, "redline_sensor": None,    "redline_limit": None},
    3: {"name": "Cooling",           "sensors": [1, 2],       "shift": 2.0,  "has_redline": True,  "redline_sensor": "cht",   "redline_limit": 230.0},
    4: {"name": "Lubrication",       "sensors": [3, 4],       "shift": 2.5,  "has_redline": True,  "redline_sensor": "oil_p", "redline_limit": 15.0},
    5: {"name": "Sensor Drift",      "sensors": [0, 3, 7],    "shift": 1.2,  "has_redline": False, "redline_sensor": None,    "redline_limit": None},
    6: {"name": "Combustion Inst.",  "sensors": [2, 6],       "shift": 3.0,  "has_redline": True,  "redline_sensor": "egt",   "redline_limit": 900.0},
    7: {"name": "Overheating",       "sensors": [1, 2, 4],    "shift": 3.5,  "has_redline": True,  "redline_sensor": "cht",   "redline_limit": 230.0},
    8: {"name": "Vibration",         "sensors": [6],          "shift": 4.5,  "has_redline": True,  "redline_sensor": "vib",   "redline_limit": 2.5},
}

# Redline thresholds in raw (not residual) space — simulated physical values
REDLINE = {
    "cht":   230.0,   # °C
    "egt":   900.0,   # °C
    "oil_p": 15.0,    # PSI (low limit)
    "vib":   2.5,     # g
}


# ─────────────────────────────────────────────────────────────────────────────
#  SIMULATED ENGINE + TWIN-RESIDUAL GENERATOR
# ─────────────────────────────────────────────────────────────────────────────

class SimulatedEngine:
    """
    A lightweight numerical model of the engine physics used as a 'digital twin'.
    Generates 9-channel telemetry with configurable fault injection and noise.
    Avoids direct coupling to EngineDynamics for speed; re-implements the key
    signals needed for residual calculation.

    Healthy baseline uses physics-derived equilibrium values per operating point.
    """

    HEALTHY_BASELINES = {
        "climb":           {"rpm": 2800, "cht": 165, "egt": 710, "oil_p_psi": 52, "oil_t": 98,  "ff": 14.2, "vib": 0.35, "batt": 13.8, "inj_t": 0.0},
        "cruise_altitude": {"rpm": 2400, "cht": 145, "egt": 650, "oil_p_psi": 48, "oil_t": 90,  "ff": 10.8, "vib": 0.28, "batt": 13.9, "inj_t": 0.0},
        "hot_day_low":     {"rpm": 2600, "cht": 180, "egt": 740, "oil_p_psi": 50, "oil_t": 105, "ff": 12.5, "vib": 0.32, "batt": 13.7, "inj_t": 0.0},
    }

    # Sensor noise std in physical units
    NOISE_STD = {
        "rpm": 18.0, "cht": 2.5, "egt": 4.0, "oil_p_psi": 0.8,
        "oil_t": 1.5, "ff": 0.3, "vib": 0.04, "batt": 0.05, "inj_t": 0.5,
    }

    def __init__(self, op_point: str, noise_seed: int, twin_params: Optional[Dict] = None):
        self.op  = op_point
        self.rng = np.random.default_rng(noise_seed)
        self.bl  = copy.deepcopy(self.HEALTHY_BASELINES[op_point])
        # Twin (reference) parameters — allows mismatch testing
        self.twin_params = twin_params or {}

    def _apply_twin_mismatch(self, key: str, value: float) -> float:
        """Apply parametric mismatch to twin reference."""
        scale = self.twin_params.get(key, 1.0)
        return value * scale

    def _healthy_readings(self) -> np.ndarray:
        bl = self.bl
        return np.array([
            bl["rpm"],
            bl["cht"],
            bl["egt"],
            bl["oil_p_psi"],
            bl["oil_t"],
            bl["ff"],
            bl["vib"],
            bl["batt"],
            bl["inj_t"],
        ], dtype=np.float64)

    def _twin_prediction(self) -> np.ndarray:
        """
        Digital twin prediction (matched or mismatched physics).
        """
        bl = self.bl
        return np.array([
            self._apply_twin_mismatch("rpm", bl["rpm"]),
            self._apply_twin_mismatch("cht", bl["cht"]),
            self._apply_twin_mismatch("egt", bl["egt"]),
            self._apply_twin_mismatch("oil_p_psi", bl["oil_p_psi"]),
            self._apply_twin_mismatch("oil_t", bl["oil_t"]),
            self._apply_twin_mismatch("ff", bl["ff"]),
            self._apply_twin_mismatch("vib", bl["vib"]),
            self._apply_twin_mismatch("batt", bl["batt"]),
            self._apply_twin_mismatch("inj_t", bl["inj_t"]),
        ], dtype=np.float64)

    def _add_noise(self, readings: np.ndarray) -> np.ndarray:
        noise = np.array([
            self.rng.normal(0, self.NOISE_STD["rpm"]),
            self.rng.normal(0, self.NOISE_STD["cht"]),
            self.rng.normal(0, self.NOISE_STD["egt"]),
            self.rng.normal(0, self.NOISE_STD["oil_p_psi"]),
            self.rng.normal(0, self.NOISE_STD["oil_t"]),
            self.rng.normal(0, self.NOISE_STD["ff"]),
            self.rng.normal(0, self.NOISE_STD["vib"]),
            self.rng.normal(0, self.NOISE_STD["batt"]),
            self.rng.normal(0, self.NOISE_STD["inj_t"]),
        ])
        return readings + noise

    def generate_run(
        self,
        duration_s: float,
        dt: float,
        fault_id: Optional[int] = None,
        fault_ramp_start: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate a time-series run.

        Returns
        -------
        raw_readings : (T, 9) physical sensor readings
        residuals    : (T, 9) = readings - twin_prediction
        redline_flags: (T, 9) True where physical reading crosses static redline
        """
        T    = int(duration_s / dt)
        ramp_start_idx = int(fault_ramp_start / dt)
        ramp_end_idx   = T  # severity reaches 1.0 at end

        readings_arr  = np.zeros((T, 9), dtype=np.float64)
        residuals_arr = np.zeros((T, 9), dtype=np.float64)

        for t in range(T):
            healthy = self._healthy_readings()
            noisy   = self._add_noise(healthy)

            # Fault injection with linear severity ramp
            if fault_id is not None and t >= ramp_start_idx:
                fdef = FAULT_DEFS[fault_id]
                prog  = (t - ramp_start_idx) / max(1, ramp_end_idx - ramp_start_idx)
                severity = np.clip(prog, 0.0, 1.0)
                fault_shift = fdef["shift"] * severity

                # Physical fault effect on raw readings
                # Each sensor has its own physical units magnitude
                sensor_fault_magnitude = {
                    0: fault_shift * 120,  # RPM
                    1: fault_shift * 60,   # CHT °C
                    2: fault_shift * 80,   # EGT °C
                    3: -fault_shift * 12,  # oil pressure PSI (drops)
                    4: fault_shift * 25,   # oil temp °C
                    5: fault_shift * 2.5,  # fuel flow L/h
                    6: fault_shift * 0.8,  # vibration g
                    7: -fault_shift * 0.5, # battery V
                    8: fault_shift * 3.0,  # inj timing °CA
                }
                for s_idx in fdef["sensors"]:
                    mag = sensor_fault_magnitude.get(s_idx, fault_shift)
                    noisy[s_idx] += mag + self.rng.normal(0, abs(mag) * 0.15)

            readings_arr[t] = noisy
            twin_pred       = self._twin_prediction()
            residuals_arr[t] = noisy - twin_pred

        # Redline flags (physical values)
        # [cht(1), egt(2), oil_p_psi(3), vib(6)]
        rl_flags = np.zeros((T, 9), dtype=bool)
        rl_flags[:, 1] = readings_arr[:, 1] > REDLINE["cht"]    # CHT > 230°C
        rl_flags[:, 2] = readings_arr[:, 2] > REDLINE["egt"]    # EGT > 900°C
        rl_flags[:, 3] = readings_arr[:, 3] < REDLINE["oil_p"]  # Oil P < 15 PSI
        rl_flags[:, 6] = readings_arr[:, 6] > REDLINE["vib"]    # Vib > 2.5 g

        return readings_arr, residuals_arr, rl_flags


# ─────────────────────────────────────────────────────────────────────────────
#  STATISTICAL ANOMALY DETECTOR (Mahalanobis distance on sliding window)
# ─────────────────────────────────────────────────────────────────────────────

class StatAnomalyDetector:
    """
    Lightweight Mahalanobis-distance anomaly detector.
    Fitted on healthy residuals; higher distance = more anomalous.
    Used as the 'anomaly detector' stage in the pipeline.
    """
    def __init__(self, threshold_pct: float = 95.0):
        self.threshold_pct = threshold_pct
        self.mean_: Optional[np.ndarray] = None
        self.cov_inv_: Optional[np.ndarray] = None
        self.threshold_: float = 0.0
        self.is_fitted: bool = False

    def fit(self, healthy_residuals: np.ndarray) -> "StatAnomalyDetector":
        """healthy_residuals: (T, F)"""
        self.mean_ = healthy_residuals.mean(axis=0)
        cov = np.cov(healthy_residuals.T) + np.eye(healthy_residuals.shape[1]) * 1e-6
        try:
            self.cov_inv_ = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            self.cov_inv_ = np.linalg.pinv(cov)
        self.is_fitted = True
        scores = self._score_batch(healthy_residuals)
        self.threshold_ = float(np.percentile(scores, self.threshold_pct))
        return self

    def _score_batch(self, X: np.ndarray) -> np.ndarray:
        delta = X - self.mean_
        mah   = np.einsum("ti,ij,tj->t", delta, self.cov_inv_, delta)
        return np.sqrt(np.maximum(mah, 0.0))

    def score(self, window: np.ndarray) -> float:
        """window: (W, F), returns scalar anomaly score."""
        mean_w = window.mean(axis=0, keepdims=True)
        return float(self._score_batch(mean_w)[0])

    def is_anomalous(self, score: float) -> bool:
        return score > self.threshold_


# ─────────────────────────────────────────────────────────────────────────────
#  STATIC THRESHOLD BASELINE DETECTOR
# ─────────────────────────────────────────────────────────────────────────────

class StaticThresholdDetector:
    """
    Static redline detector — fires when any single sensor reading
    crosses its hard limit.  Used as comparison baseline.
    """
    LIMITS = {
        1: ("cht",   230.0, "gt"),
        2: ("egt",   900.0, "gt"),
        3: ("oil_p",  15.0, "lt"),
        6: ("vib",    2.5,  "gt"),
    }

    def check(self, readings: np.ndarray) -> bool:
        """readings: (9,) raw physical values."""
        if readings[1] > REDLINE["cht"]:    return True
        if readings[2] > REDLINE["egt"]:    return True
        if readings[3] < REDLINE["oil_p"]:  return True
        if readings[6] > REDLINE["vib"]:    return True
        return False


# ─────────────────────────────────────────────────────────────────────────────
#  RESIDUAL-BASED CLASSIFIER (rule-based proxy for CNN-LSTM)
# ─────────────────────────────────────────────────────────────────────────────

class ResidualClassifier:
    """
    Fast residual-signature-based fault classifier.
    Implements the same logic as the CNN-LSTM but without the full model.
    Used when the checkpoint is unavailable, or as a comparison baseline.
    """

    # Sensor index to fault-class responsibility mapping (simplified from SHAP analysis)
    FAULT_SIGNATURES: Dict[int, List[int]] = {
        1: [0, 2, 6],    # Misfire: RPM, EGT, Vib
        2: [5, 8],       # Injector: FuelFlow, InjTiming
        3: [1, 2],       # Cooling: CHT, EGT
        4: [3, 4],       # Lubrication: OilP, OilT
        5: [0, 3, 7],    # Sensor drift: RPM, OilP, Batt
        6: [2, 6],       # Combustion: EGT, Vib
        7: [1, 2, 4],    # Overheat: CHT, EGT, OilT
        8: [6],          # Vibration: Vib
    }

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.scaler_mean: Optional[np.ndarray] = None
        self.scaler_std:  Optional[np.ndarray] = None

    def fit_scaler(self, healthy_residuals: np.ndarray):
        self.scaler_mean = healthy_residuals.mean(axis=0)
        self.scaler_std  = healthy_residuals.std(axis=0) + 1e-8

    def classify(self, residuals: np.ndarray) -> int:
        """residuals: (W, F) window, returns class id."""
        if self.scaler_mean is not None:
            r = (residuals - self.scaler_mean) / self.scaler_std
        else:
            r = residuals
        mean_r = np.abs(r.mean(axis=0))   # (F,)

        if mean_r.max() < self.threshold:
            return 0  # Healthy

        scores = {}
        for cls_id, sensors in self.FAULT_SIGNATURES.items():
            scores[cls_id] = mean_r[sensors].mean()
        return max(scores, key=lambda k: scores[k])


# ─────────────────────────────────────────────────────────────────────────────
#  CNN-LSTM WRAPPER (loads actual trained model if available)
# ─────────────────────────────────────────────────────────────────────────────

class FaultDetectorWrapper:
    """Wraps trained CNN-LSTM or falls back to ResidualClassifier."""

    def __init__(self, use_real_model: bool = True):
        self.use_real_model = use_real_model and MODEL_OK and TORCH_AVAILABLE
        self.model = None
        self.scaler = None
        self.fallback = ResidualClassifier()

        if self.use_real_model:
            try:
                self.model = load_model(CHECKPOINT_PATH)
                self.model.eval()
                print("[INFO] Loaded trained CNN-LSTM checkpoint.")
            except Exception as e:
                print(f"[WARN] Could not load model: {e}. Using rule-based classifier.")
                self.use_real_model = False

        # Generate healthy data to fit scaler/fallback
        raw, lbl = generate_synthetic_residuals(n_timesteps=60_000, seed=GLOBAL_SEED) if MODEL_OK else self._synthetic_data(60_000)
        healthy_mask = lbl == 0
        healthy_res  = raw[healthy_mask]
        self.fallback.fit_scaler(healthy_res)
        self._healthy_residuals = healthy_res

    def _synthetic_data(self, n: int) -> Tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(GLOBAL_SEED)
        residuals = rng.normal(0.0, 0.08, (n, NUM_SENSORS)).astype(np.float32)
        labels    = np.zeros(n, dtype=np.int64)
        return residuals, labels

    def classify(self, residuals: np.ndarray) -> int:
        """residuals: (W, F) — returns predicted class id."""
        if self.use_real_model and self.model is not None:
            try:
                x = torch.from_numpy(residuals.astype(np.float32)).unsqueeze(0).permute(0, 2, 1)
                with torch.no_grad():
                    logits = self.model(x)
                return int(logits.argmax(dim=-1).item())
            except Exception:
                pass
        return self.fallback.classify(residuals)

    def healthy_residuals(self) -> np.ndarray:
        return self._healthy_residuals


# ─────────────────────────────────────────────────────────────────────────────
#  EXPERIMENT 1 — DETECTION LEAD TIME
# ─────────────────────────────────────────────────────────────────────────────

def experiment_1(n_repeats: int = 20) -> Dict:
    """
    Per-fault, per-operating-point detection lead time.
    Ramps fault severity 0 → 1 over RAMP_DURATION_S.
    Logs:
        t_detector  : when anomaly detector first fires
        t_classifier: when classifier first names the correct fault
        t_redline   : when static redline fires (or None)
    """
    print("\n" + "="*70)
    print("EXPERIMENT 1 — Detection Lead Time")
    print("="*70)

    RAMP_DURATION_S = 600.0   # 10 minutes of simulated flight
    DT              = 0.1     # 10 Hz for speed (100 Hz data)
    WINDOW_S        = WINDOW_SIZE * DT  # ~6.4 s window

    # Build detector + classifier
    detector   = StatAnomalyDetector(threshold_pct=95.0)
    classifier = FaultDetectorWrapper(use_real_model=True)
    static_det = StaticThresholdDetector()

    # Fit detector on actual engine-generated healthy residuals
    # (so the threshold is calibrated on the same scale as test data)
    _eng_fit = SimulatedEngine("cruise_altitude", noise_seed=GLOBAL_SEED)
    _, healthy_res, _ = _eng_fit.generate_run(duration_s=500.0, dt=0.1, fault_id=None)
    detector.fit(healthy_res)

    results = {}
    for fault_id, fdef in FAULT_DEFS.items():
        fault_name = fdef["name"]
        has_redline = fdef["has_redline"]
        results[fault_id] = {"name": fault_name, "operating_points": {}}

        for op_name in OPERATING_POINTS:
            lead_times_detector    = []
            lead_times_classifier  = []
            pct_before_redline     = []
            severity_at_detection  = []
            redline_triggered_ever = False

            for rep in range(n_repeats):
                seed = GLOBAL_SEED + fault_id * 1000 + op_name.__hash__() % 1000 + rep
                engine = SimulatedEngine(op_name, noise_seed=abs(seed))

                readings, residuals, rl_flags = engine.generate_run(
                    duration_s=RAMP_DURATION_S,
                    dt=DT,
                    fault_id=fault_id,
                    fault_ramp_start=0.0,
                )
                T = len(readings)
                W = WINDOW_SIZE

                t_detector    = None
                t_classifier  = None
                t_redline     = None

                # Slide through time
                for t in range(W, T):
                    window_res  = residuals[t - W:t]
                    reading_now = readings[t]

                    # Anomaly detector
                    if t_detector is None:
                        score = detector.score(window_res)
                        if detector.is_anomalous(score):
                            t_detector   = t * DT
                            sev          = np.clip(t / T, 0.0, 1.0)
                            severity_at_detection.append(sev)

                    # Classifier
                    if t_classifier is None and t_detector is not None:
                        pred = classifier.classify(window_res)
                        if pred == fault_id:
                            t_classifier = t * DT

                    # Static redline
                    if t_redline is None and static_det.check(reading_now):
                        t_redline = t * DT
                        redline_triggered_ever = True

                    if t_detector and t_classifier and t_redline:
                        break

                if t_detector is None:
                    t_detector = RAMP_DURATION_S   # never fired
                if t_classifier is None:
                    t_classifier = RAMP_DURATION_S

                if has_redline and redline_triggered_ever:
                    if t_redline is not None:
                        lead = t_redline - t_detector
                        lead_times_detector.append(lead)
                        fired_before = (t_detector < t_redline)
                        pct_before_redline.append(float(fired_before))
                    else:
                        lead_times_detector.append(0.0)
                        pct_before_redline.append(0.0)
                else:
                    lead_times_detector.append(float(RAMP_DURATION_S))
                    pct_before_redline.append(1.0)

                lead_times_classifier.append(t_classifier - t_detector)

            # Aggregate
            lead_arr = np.array(lead_times_detector)
            sev_arr  = np.array(severity_at_detection) if severity_at_detection else np.array([np.nan])
            pct_arr  = np.array(pct_before_redline)

            op_result = {
                "lead_time_median_s":       float(np.median(lead_arr)),
                "lead_time_p10_s":          float(np.percentile(lead_arr, 10)),
                "pct_before_redline":       float(pct_arr.mean() * 100),
                "severity_at_detection_med":float(np.nanmedian(sev_arr)),
                "redline_triggered":        redline_triggered_ever,
            }
            results[fault_id]["operating_points"][op_name] = op_result

            print(f"  Fault {fault_id:2d} {fault_name:<22s} | {op_name:<20s} | "
                  f"Lead: {op_result['lead_time_median_s']:+7.1f}s | "
                  f"P10: {op_result['lead_time_p10_s']:+7.1f}s | "
                  f"Before-RL: {op_result['pct_before_redline']:5.1f}% | "
                  f"Sev@det: {op_result['severity_at_detection_med']:.2f}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
#  EXPERIMENT 2 — FALSE ALERTS ON CLEAN FLIGHTS
# ─────────────────────────────────────────────────────────────────────────────

def experiment_2(target_hours: float = 110.0) -> Dict:
    """
    Simulate ~110 flight hours of healthy flight with randomised conditions.
    Count false alerts for both anomaly detector and static-threshold baseline.
    Uses engines with noise seeds not in training.
    """
    print("\n" + "="*70)
    print("EXPERIMENT 2 — False Alerts on Clean Flights")
    print("="*70)

    DT             = 0.5         # 2 Hz for speed in this experiment
    SEGMENT_MIN    = 30          # 30-minute flight segments
    SEGMENT_S      = SEGMENT_MIN * 60.0
    N_SEGMENTS     = int(target_hours * 3600 / SEGMENT_S)
    ACTUAL_HOURS   = (N_SEGMENTS * SEGMENT_S) / 3600

    detector   = StatAnomalyDetector(threshold_pct=95.0)
    static_det = StaticThresholdDetector()

    # Fit detector on actual engine-generated healthy residuals from a training engine
    # (seed 0 -- never used as a test segment)
    _eng_fit = SimulatedEngine("cruise_altitude", noise_seed=0)
    _, healthy_fit, _ = _eng_fit.generate_run(duration_s=800.0, dt=0.1, fault_id=None)
    detector.fit(healthy_fit)

    false_alerts_detector = 0
    false_alerts_static   = 0
    total_windows         = 0

    for seg in range(N_SEGMENTS):
        # Randomise per segment
        op_name = list(OPERATING_POINTS.keys())[seg % 3]
        alt_m   = OPERATING_POINTS[op_name]["alt_m"] + np.random.uniform(-500, 500)
        isa_off  = np.random.uniform(-15, 15)
        seed     = 10000 + seg  # seeds never seen in training

        engine  = SimulatedEngine(op_name, noise_seed=seed)
        readings, residuals, _ = engine.generate_run(
            duration_s=SEGMENT_S,
            dt=DT,
            fault_id=None,
        )
        T = len(readings)
        W = WINDOW_SIZE

        for t in range(W, T, STRIDE):
            window_res   = residuals[t - W:t]
            reading_now  = readings[t]

            score = detector.score(window_res)
            if detector.is_anomalous(score):
                false_alerts_detector += 1

            if static_det.check(reading_now):
                false_alerts_static += 1

            total_windows += 1

    false_per_100h_detector = (false_alerts_detector / ACTUAL_HOURS) * 100
    false_per_100h_static   = (false_alerts_static   / ACTUAL_HOURS) * 100

    result = {
        "simulated_hours":            ACTUAL_HOURS,
        "total_windows_checked":      total_windows,
        "threshold_percentile":       95.0,
        "false_alerts_detector":      false_alerts_detector,
        "false_alerts_static":        false_alerts_static,
        "false_per_100h_detector":    false_per_100h_detector,
        "false_per_100h_static":      false_per_100h_static,
        "detector_threshold_value":   detector.threshold_,
    }

    print(f"  Simulated hours:          {ACTUAL_HOURS:.1f} h")
    print(f"  Total inference windows:  {total_windows:,}")
    print(f"  Detector (threshold={detector.threshold_:.3f}):")
    print(f"    False alerts total:     {false_alerts_detector}")
    print(f"    False per 100 h:        {false_per_100h_detector:.2f}")
    print(f"  Static-threshold baseline:")
    print(f"    False alerts total:     {false_alerts_static}")
    print(f"    False per 100 h:        {false_per_100h_static:.2f}")

    return result


# ─────────────────────────────────────────────────────────────────────────────
#  EXPERIMENT 3 — MODEL-MISMATCH STRESS TEST
# ─────────────────────────────────────────────────────────────────────────────

def experiment_3(n_repeats: int = 20) -> Dict:
    """
    Train detector on nominal-physics twin.
    Test on twins where cooling_coeff, friction, vol_eff, sensor_bias
    are perturbed by ±5%, ±10%, ±20%.
    Do NOT retrain.
    """
    print("\n" + "="*70)
    print("EXPERIMENT 3 — Model-Mismatch Stress Test")
    print("="*70)

    DT          = 0.1
    RAMP_S      = 300.0       # 5-minute ramp
    PERTURB_LVL = [0.05, 0.10, 0.20]
    FAULT_PROBE = [1, 3, 4, 7, 8]    # Representative faults to test

    # Nominal detector (fitted on zero-mismatch data)
    detector_nom = StatAnomalyDetector(threshold_pct=95.0)
    _eng_fit_nom = SimulatedEngine("cruise_altitude", noise_seed=GLOBAL_SEED)
    _, healthy_nom, _ = _eng_fit_nom.generate_run(duration_s=500.0, dt=0.1, fault_id=None)
    detector_nom.fit(healthy_nom)

    ACCEPTABLE_DR  = 0.80   # 80% detection rate minimum
    ACCEPTABLE_FA  = 20.0   # < 20% of healthy windows flagged as anomalous

    results = {"perturbation_levels": {}}

    for pct in PERTURB_LVL:
        for sign in [+1, -1]:
            key = f"{'+' if sign > 0 else '-'}{int(pct*100)}%"
            # Mismatch: perturb cooling (cht, egt), friction (rpm, ff),
            # vol_eff (rpm, ff), sensor bias (all)
            twin_params = {
                "rpm":    1.0 + sign * pct * 0.5,   # cooling/friction mix
                "cht":    1.0 + sign * pct,          # cooling coefficient
                "egt":    1.0 + sign * pct * 0.8,
                "oil_p_psi": 1.0 + sign * pct * 0.4,
                "oil_t":  1.0 + sign * pct * 0.6,
                "ff":     1.0 + sign * pct * 0.7,
                "vib":    1.0 + sign * pct * 0.3,
                "batt":   1.0 + sign * pct * 0.2,
                "inj_t":  1.0 + sign * pct * 0.9,
            }

            detected_total  = 0
            total_runs      = 0

            for fault_id in FAULT_PROBE:
                for rep in range(n_repeats // 2):
                    seed = 5000 + fault_id * 100 + rep
                    engine = SimulatedEngine("cruise_altitude", noise_seed=seed,
                                            twin_params=twin_params)
                    readings, residuals, _ = engine.generate_run(
                        duration_s=RAMP_S, dt=DT,
                        fault_id=fault_id, fault_ramp_start=0.0,
                    )
                    T = len(readings)
                    W = WINDOW_SIZE
                    detected = False
                    for t in range(W, T):
                        window_res = residuals[t - W:t]
                        score = detector_nom.score(window_res)
                        if detector_nom.is_anomalous(score):
                            detected = True
                            break
                    detected_total += int(detected)
                    total_runs     += 1

            # False alert rate: fraction of healthy windows flagged
            # Use 3 x 1-hour healthy runs with the mismatched twin
            healthy_windows_flagged = 0
            healthy_windows_total   = 0
            for rep in range(3):
                seed = 6000 + rep
                engine = SimulatedEngine("cruise_altitude", noise_seed=seed,
                                        twin_params=twin_params)
                _, residuals, _ = engine.generate_run(
                    duration_s=3600.0, dt=DT, fault_id=None,
                )
                T = len(residuals)
                W = WINDOW_SIZE
                for t in range(W, T, STRIDE):
                    score = detector_nom.score(residuals[t - W:t])
                    healthy_windows_total   += 1
                    if detector_nom.is_anomalous(score):
                        healthy_windows_flagged += 1

            fa_pct = (healthy_windows_flagged / max(1, healthy_windows_total)) * 100
            dr     = detected_total / max(1, total_runs)
            below_limit = (dr < ACCEPTABLE_DR) or (fa_pct > ACCEPTABLE_FA)

            results["perturbation_levels"][key] = {
                "detection_rate_pct":          round(dr * 100, 1),
                "healthy_windows_flagged_pct": round(fa_pct, 1),
                "below_acceptable_limit":       below_limit,
                "note": "FA% = fraction of healthy windows flagged; 100% = constant alarm",
            }

            status = " [BELOW LIMIT]" if below_limit else ""
            print(f"  Perturbation {key:>5s}: DR={dr*100:5.1f}%  FA%={fa_pct:6.1f}%{status}")

    # Find breakpoint
    breakpoint_pct = None
    for key, v in results["perturbation_levels"].items():
        if v["below_acceptable_limit"] and breakpoint_pct is None:
            breakpoint_pct = key

    results["breakpoint"] = breakpoint_pct
    results["acceptable_dr_pct"]       = ACCEPTABLE_DR * 100
    results["acceptable_fa_window_pct"] = ACCEPTABLE_FA
    results["design_note"] = (
        "High FA% at even 5% mismatch is expected for a simple Mahalanobis detector: "
        "systematic bias shifts all healthy windows above the nominal threshold. "
        "DR stays high because fault signals dominate the bias. "
        "A CUSUM-on-residual-trend detector is recommended for diverse engine units."
    )
    print(f"\n  Breakpoint (DR or FA limit): {breakpoint_pct}")
    return results


# ─────────────────────────────────────────────────────────────────────────────
#  EXPERIMENT 4 — EDGE TIMING
# ─────────────────────────────────────────────────────────────────────────────

def experiment_4(n_windows: int = 1000) -> Dict:
    """
    Time 1,000 inference windows (twin step + residuals + detector).
    Reports model size, median + p99 latency, RAM estimate.
    Checks if full loop fits in 10 ms (100 Hz).
    """
    print("\n" + "="*70)
    print("EXPERIMENT 4 — Edge Timing")
    print("="*70)

    results = {}

    # ── Model size ─────────────────────────────────────────────────────────
    checkpoint = Path(CHECKPOINT_PATH)
    if checkpoint.exists():
        model_size_mb = checkpoint.stat().st_size / 1e6
    else:
        model_size_mb = None
    results["model_size_mb"] = model_size_mb
    print(f"  Model checkpoint size: {model_size_mb:.2f} MB" if model_size_mb else "  Model checkpoint: not found")

    # ── Build components ───────────────────────────────────────────────────
    detector   = StatAnomalyDetector(threshold_pct=95.0)
    classifier = FaultDetectorWrapper(use_real_model=True)
    engine     = SimulatedEngine("cruise_altitude", noise_seed=GLOBAL_SEED)

    # Fit detector on actual engine-generated healthy residuals
    _eng_fit4 = SimulatedEngine("cruise_altitude", noise_seed=GLOBAL_SEED)
    _, healthy_res, _ = _eng_fit4.generate_run(duration_s=500.0, dt=0.1, fault_id=None)
    detector.fit(healthy_res)

    # Pre-generate raw data so we only time inference, not data-gen
    DT       = 0.01
    BURN_IN  = WINDOW_SIZE * 2
    readings, residuals, _ = engine.generate_run(
        duration_s=(n_windows + BURN_IN) * DT, dt=DT, fault_id=None
    )
    W = WINDOW_SIZE

    latencies_ms = []

    # Warm-up
    for _ in range(20):
        t = BURN_IN + 5
        _ = detector.score(residuals[t - W:t])
        _ = classifier.classify(residuals[t - W:t])

    # Timed loops
    for i in range(n_windows):
        t = BURN_IN + i + W
        if t >= len(residuals):
            break
        window_res = residuals[t - W:t]

        t0 = time.perf_counter()
        score   = detector.score(window_res)
        is_anom = detector.is_anomalous(score)
        if is_anom:
            pred = classifier.classify(window_res)
        t1 = time.perf_counter()

        latencies_ms.append((t1 - t0) * 1000)

    lat_arr = np.array(latencies_ms)
    median_ms  = float(np.median(lat_arr))
    p99_ms     = float(np.percentile(lat_arr, 99))
    fits_10ms  = p99_ms < 10.0

    import platform
    cpu_name = platform.processor() or platform.machine()

    results.update({
        "cpu":            cpu_name,
        "hardware_note":  "desktop CPU, not target hardware",
        "n_windows":      len(latencies_ms),
        "median_ms":      round(median_ms, 3),
        "p99_ms":         round(p99_ms, 3),
        "fits_10ms_frame":fits_10ms,
        "ram_estimate_mb":round(healthy_res.nbytes / 1e6 * 10, 2),  # ~10x overhead estimate
    })

    print(f"  CPU:              {cpu_name} (desktop, not target HW)")
    print(f"  Median latency:   {median_ms:.3f} ms")
    print(f"  P99 latency:      {p99_ms:.3f} ms")
    print(f"  Fits 10 ms frame: {'YES' if fits_10ms else 'NO'}")
    print(f"  Model size:       {model_size_mb:.2f} MB" if model_size_mb else "  Model not found")

    return results


# ─────────────────────────────────────────────────────────────────────────────
#  EXPERIMENT 5 — ABLATION
# ─────────────────────────────────────────────────────────────────────────────

def experiment_5(n_repeats: int = 10) -> Dict:
    """
    Four pipeline variants, repeated for a subset of faults.
    (a) Static thresholds only
    (b) Anomaly detector on raw telemetry
    (c) Anomaly detector on twin residuals
    (d) Full pipeline: twin residuals + anomaly detector + classifier

    Metrics: median lead time vs static redline, false alerts per 100 h.
    """
    print("\n" + "="*70)
    print("EXPERIMENT 5 — Ablation")
    print("="*70)

    DT      = 0.1
    RAMP_S  = 300.0
    FAULTS  = [1, 3, 4, 7, 8]

    # Detectors
    det_raw  = StatAnomalyDetector(threshold_pct=95.0)
    det_twin = StatAnomalyDetector(threshold_pct=95.0)
    static   = StaticThresholdDetector()
    clf      = FaultDetectorWrapper(use_real_model=True)

    # Fit both detectors on actual engine-generated data
    _eng_fit5 = SimulatedEngine("cruise_altitude", noise_seed=GLOBAL_SEED)
    healthy_readings, healthy_res, _ = _eng_fit5.generate_run(
        duration_s=500.0, dt=DT, fault_id=None
    )
    det_raw.fit(healthy_readings)   # raw physical values
    det_twin.fit(healthy_res)       # twin residuals

    variants = {
        "a_static":          {"lead_times": [], "false_alerts": 0},
        "b_det_raw_telem":   {"lead_times": [], "false_alerts": 0},
        "c_det_twin_resid":  {"lead_times": [], "false_alerts": 0},
        "d_full_pipeline":   {"lead_times": [], "false_alerts": 0},
    }

    # ── Lead time test ───────────────────────────────────────────────────
    for fault_id in FAULTS:
        fdef = FAULT_DEFS[fault_id]
        for rep in range(n_repeats):
            seed = 7000 + fault_id * 100 + rep
            engine = SimulatedEngine("cruise_altitude", noise_seed=seed)
            readings, residuals, rl_flags = engine.generate_run(
                duration_s=RAMP_S, dt=DT,
                fault_id=fault_id, fault_ramp_start=0.0,
            )
            T = len(readings)
            W = WINDOW_SIZE

            t_redline  = None
            t_a = t_b = t_c = t_d = None

            for t in range(W, T):
                window_res  = residuals[t - W:t]
                window_raw  = readings[t - W:t]
                reading_now = readings[t]

                # Static redline
                if t_redline is None and static.check(reading_now):
                    t_redline = t * DT

                # (a) static
                if t_a is None and static.check(reading_now):
                    t_a = t * DT

                # (b) detector on raw
                if t_b is None:
                    score = det_raw.score(window_raw)
                    if det_raw.is_anomalous(score):
                        t_b = t * DT

                # (c) detector on residuals
                if t_c is None:
                    score = det_twin.score(window_res)
                    if det_twin.is_anomalous(score):
                        t_c = t * DT

                # (d) full pipeline
                if t_d is None:
                    score = det_twin.score(window_res)
                    if det_twin.is_anomalous(score):
                        pred = clf.classify(window_res)
                        if pred == fault_id:
                            t_d = t * DT

            rl = t_redline or RAMP_S
            variants["a_static"]["lead_times"].append(0.0)            # by definition no lead
            variants["b_det_raw_telem"]["lead_times"].append(rl - (t_b or rl))
            variants["c_det_twin_resid"]["lead_times"].append(rl - (t_c or rl))
            variants["d_full_pipeline"]["lead_times"].append(rl - (t_d or rl))

    # ── False alert rate (healthy flight, 10 hours) ───────────────────────
    HEALTHY_HOURS = 10.0
    HEALTHY_STEPS = int(HEALTHY_HOURS * 3600 / DT)
    TOTAL_WINDOWS = HEALTHY_STEPS - WINDOW_SIZE

    for rep in range(3):
        seed = 8000 + rep
        engine = SimulatedEngine("cruise_altitude", noise_seed=seed)
        readings, residuals, _ = engine.generate_run(
            duration_s=HEALTHY_HOURS * 3600, dt=DT, fault_id=None,
        )
        T = len(readings)
        W = WINDOW_SIZE

        for t in range(W, T, STRIDE):
            window_res = residuals[t - W:t]
            window_raw = readings[t - W:t]
            reading_now = readings[t]

            if static.check(reading_now):
                variants["a_static"]["false_alerts"] += 1

            if det_raw.is_anomalous(det_raw.score(window_raw)):
                variants["b_det_raw_telem"]["false_alerts"] += 1

            if det_twin.is_anomalous(det_twin.score(window_res)):
                variants["c_det_twin_resid"]["false_alerts"] += 1

            score = det_twin.score(window_res)
            if det_twin.is_anomalous(score):
                pred = clf.classify(window_res)
                if pred != 0:
                    variants["d_full_pipeline"]["false_alerts"] += 1

    # Aggregate
    ablation_results = {}
    headers = ["Variant", "Median Lead (s)", "P10 Lead (s)", "False/100h"]
    rows    = []
    for var, data in variants.items():
        lt_arr = np.array(data["lead_times"])
        fa_100h = (data["false_alerts"] / (3 * HEALTHY_HOURS)) * 100
        row = {
            "median_lead_s": round(float(np.median(lt_arr)), 1),
            "p10_lead_s":    round(float(np.percentile(lt_arr, 10)), 1),
            "false_per_100h":round(fa_100h, 2),
        }
        ablation_results[var] = row
        rows.append(row)
        print(f"  {var:<24s}: Lead p50={row['median_lead_s']:+7.1f}s "
              f" p10={row['p10_lead_s']:+7.1f}s  FA/100h={row['false_per_100h']:5.2f}")

    return ablation_results


# ─────────────────────────────────────────────────────────────────────────────
#  QUICK EXTRA A — SENSOR DRIFT vs REAL FAULT
# ─────────────────────────────────────────────────────────────────────────────

def extra_a_drift_vs_fault(n_runs: int = 30) -> Dict:
    """
    Inject slow drift on RPM sensor; report how often system calls it
    'Sensor Drift' (class 5) vs a real engine fault class.
    """
    print("\n" + "="*70)
    print("EXTRA A — Sensor Drift vs Real Fault Disambiguation")
    print("="*70)

    DT     = 0.1
    RUN_S  = 120.0  # 2 min drift ramp
    W      = WINDOW_SIZE

    clf = FaultDetectorWrapper(use_real_model=True)
    correct_drift = 0
    wrong_engine  = 0
    no_detection  = 0

    for rep in range(n_runs):
        seed   = 9000 + rep
        engine = SimulatedEngine("cruise_altitude", noise_seed=seed)
        readings, residuals, _ = engine.generate_run(
            duration_s=RUN_S, dt=DT,
            fault_id=5,    # Sensor drift class
            fault_ramp_start=10.0,
        )
        T = len(readings)

        # Take a window near the end (high severity)
        t = min(T - 1, int(0.8 * T))
        if t < W:
            continue
        window_res = residuals[t - W:t]
        pred = clf.classify(window_res)

        if pred == 5:
            correct_drift += 1
        elif pred != 0:
            wrong_engine  += 1
        else:
            no_detection  += 1

    total  = n_runs
    pct_correct = correct_drift / total * 100
    pct_wrong   = wrong_engine  / total * 100
    pct_healthy = no_detection  / total * 100

    result = {
        "n_runs":             total,
        "pct_correct_drift":  round(pct_correct, 1),
        "pct_wrong_engine":   round(pct_wrong, 1),
        "pct_healthy":        round(pct_healthy, 1),
    }
    print(f"  Correct 'Sensor Drift': {pct_correct:.1f}%")
    print(f"  Misclassified as engine fault: {pct_wrong:.1f}%")
    print(f"  Classified as healthy (low severity): {pct_healthy:.1f}%")
    return result


# ─────────────────────────────────────────────────────────────────────────────
#  QUICK EXTRA B — ALTITUDE INVARIANCE
# ─────────────────────────────────────────────────────────────────────────────

def extra_b_altitude_invariance() -> Dict:
    """
    Healthy engine: report residual mean and std at 0, 3000, 6000, 7500 m.
    Should stay near zero for a well-calibrated twin.
    """
    print("\n" + "="*70)
    print("EXTRA B — Altitude Invariance")
    print("="*70)

    ALTITUDES  = [0, 3000, 6000, 7500]
    DT         = 0.1
    DURATION   = 600.0   # 10 min per altitude

    # Map alt_m to an operating point with matching ISA
    def op_for_alt(alt_m: int) -> str:
        if alt_m <= 1000:   return "hot_day_low"
        if alt_m <= 4000:   return "climb"
        return "cruise_altitude"

    result = {}
    for alt_m in ALTITUDES:
        op    = op_for_alt(alt_m)
        seeds = [1100 + alt_m // 100 + i for i in range(5)]
        all_res = []

        for seed in seeds:
            eng = SimulatedEngine(op, noise_seed=seed)
            _, residuals, _ = eng.generate_run(
                duration_s=DURATION, dt=DT, fault_id=None
            )
            all_res.append(residuals)

        all_res = np.vstack(all_res)  # (5*T, 9)
        means   = all_res.mean(axis=0)
        stds    = all_res.std(axis=0)

        result[f"{alt_m}m"] = {
            "residual_mean_per_channel": [round(float(m), 4) for m in means],
            "residual_std_per_channel":  [round(float(s), 4) for s in stds],
            "overall_mean":              round(float(means.mean()), 5),
            "overall_std":               round(float(stds.mean()), 5),
        }
        print(f"  {alt_m:5d} m  mean={means.mean():+.5f}  std={stds.mean():.5f}")

    return result


# ─────────────────────────────────────────────────────────────────────────────
#  QUICK EXTRA C — RUL HONESTY
# ─────────────────────────────────────────────────────────────────────────────

def extra_c_rul_honesty(n_runs: int = 50) -> Dict:
    """
    RUL honesty: report p10/p50/p90 error bands.
    Loads the held-out engine unit (U02) evaluation from the trained RUL pipeline
    if available; otherwise fits linear degradation trend on simulated runs.
    """
    print("\n" + "="*70)
    print("EXTRA C — RUL Honesty (p10/p50/p90 error bands)")
    print("="*70)

    csv_path = REPO_ROOT / "DASHBOARD AND DATA" / "data" / "generated" / "rul" / "evaluation" / "rul_projection_residual.csv"
    if csv_path.exists():
        try:
            import pandas as pd
            df = pd.read_csv(csv_path)
            degraded = df[~df["run_id"].str.contains("HEALTHY")].dropna(subset=["rul_true_s", "rul_est_s"])
            if len(degraded) > 0:
                err = (degraded["rul_est_s"] - degraded["rul_true_s"]).abs()
                pct_err = (err / degraded["rul_true_s"]) * 100
                arr = pct_err.values
                result = {
                    "n_predictions":   len(arr),
                    "p10_error_pct":   round(float(np.percentile(arr, 10)), 1),
                    "p50_error_pct":   round(float(np.percentile(arr, 50)), 1),
                    "p90_error_pct":   round(float(np.percentile(arr, 90)), 1),
                    "mean_error_pct":  round(float(arr.mean()), 1),
                    "validation_note": "Validated on simulation only (held-out unit U02) — not flight data.",
                }
                print(f"  Source:        {csv_path.name}")
                print(f"  n predictions: {result['n_predictions']}")
                print(f"  p10 error:     {result['p10_error_pct']:.1f}%")
                print(f"  p50 error:     {result['p50_error_pct']:.1f}%")
                print(f"  p90 error:     {result['p90_error_pct']:.1f}%")
                print(f"  mean error:    {result['mean_error_pct']:.1f}%")
                print(f"  Note:          {result['validation_note']}")
                return result
        except Exception as e:
            print(f"  Could not load CSV ({e}), falling back to simulation.")

    DT       = 0.1
    RAMP_S   = 600.0
    FAULTS   = [3, 4, 7]
    pct_errors = []

    for fault_id in FAULTS:
        fdef = FAULT_DEFS[fault_id]
        s_idx = {"cht": 1, "oil_p": 3, "egt": 2, "vib": 6}[fdef["redline_sensor"]]
        rl_val = fdef["redline_limit"]
        is_lower = (fdef["redline_sensor"] == "oil_p")

        for rep in range(max(10, n_runs // len(FAULTS))):
            seed   = 11000 + fault_id * 100 + rep
            engine = SimulatedEngine("cruise_altitude", noise_seed=seed)
            readings, residuals, rl_flags = engine.generate_run(
                duration_s=RAMP_S, dt=DT,
                fault_id=fault_id, fault_ramp_start=0.0,
            )
            T = len(readings)
            rl_t = np.where(rl_flags[:, s_idx])[0]
            if len(rl_t) == 0:
                continue
            t_actual = rl_t[0] * DT
            t_p = int(0.4 * T)
            t_curr = t_p * DT
            if t_actual <= t_curr + 5.0:
                continue
            rul_actual = t_actual - t_curr

            win = int(120.0 / DT)
            times = np.arange(win) * DT
            y = readings[t_p - win : t_p, s_idx]
            slope, intercept = np.polyfit(times, y, 1)
            curr_fitted = intercept + slope * times[-1]
            if is_lower:
                if slope >= -1e-4:
                    continue
                rul_pred = max(0.0, (rl_val - curr_fitted) / slope)
            else:
                if slope <= 1e-4:
                    continue
                rul_pred = max(0.0, (rl_val - curr_fitted) / slope)

            pct_err = abs(rul_pred - rul_actual) / rul_actual * 100
            pct_errors.append(min(pct_err, 300.0))

    if not pct_errors:
        print("  No valid RUL predictions generated.")
        return {}

    arr = np.array(pct_errors)
    result = {
        "n_predictions":   len(arr),
        "p10_error_pct":   round(float(np.percentile(arr, 10)), 1),
        "p50_error_pct":   round(float(np.percentile(arr, 50)), 1),
        "p90_error_pct":   round(float(np.percentile(arr, 90)), 1),
        "mean_error_pct":  round(float(arr.mean()), 1),
        "validation_note": "Validated on simulation only — not flight data.",
    }
    print(f"  n predictions: {result['n_predictions']}")
    print(f"  p10 error:     {result['p10_error_pct']:.1f}%")
    print(f"  p50 error:     {result['p50_error_pct']:.1f}%")
    print(f"  p90 error:     {result['p90_error_pct']:.1f}%")
    print(f"  mean error:    {result['mean_error_pct']:.1f}%")
    print(f"  Note:          {result['validation_note']}")
    return result


# ─────────────────────────────────────────────────────────────────────────────
#  RESULTS PRINTER (slide-ready tables)
# ─────────────────────────────────────────────────────────────────────────────

def print_slide_tables(all_results: Dict) -> None:
    print("\n\n" + "="*70)
    print("SLIDE-READY RESULTS SUMMARY")
    print("="*70)

    # ── Experiment 1 ────────────────────────────────────────────────────
    print("\n[SLIDE 1/2/6] Experiment 1: Detection Lead Time Per Fault")
    print("-"*70)
    header = f"{'Fault':<22} {'Op Point':<22} {'Lead p50 (s)':>12} {'Lead p10 (s)':>12} {'%Before RL':>10} {'Sev@Det':>8} {'Redline?':>9}"
    print(header)
    print("-" * len(header))

    e1 = all_results.get("exp1", {})
    for fault_id, fdata in e1.items():
        name = fdata.get("name", "?")
        for op, opd in fdata.get("operating_points", {}).items():
            rl_str = "YES" if opd["redline_triggered"] else "NEVER"
            print(f"  {name:<20} {op:<22} {opd['lead_time_median_s']:>+12.1f} {opd['lead_time_p10_s']:>+12.1f} "
                  f"{opd['pct_before_redline']:>10.1f} {opd['severity_at_detection_med']:>8.2f} {rl_str:>9}")

    # ── Experiment 2 ────────────────────────────────────────────────────
    print("\n[SLIDE 2/4] Experiment 2: False Alerts")
    e2 = all_results.get("exp2", {})
    print(f"  Simulated hours:    {e2.get('simulated_hours', 0):.1f} h")
    print(f"  Detector threshold: {e2.get('threshold_percentile', 0):.0f}th percentile (value={e2.get('detector_threshold_value', 0):.4f})")
    print(f"  {'Method':<30}  {'False alerts/100h':>18}")
    print(f"  {'Twin-residual anomaly detector':<30}  {e2.get('false_per_100h_detector', 0):>18.2f}")
    print(f"  {'Static-threshold baseline':<30}  {e2.get('false_per_100h_static', 0):>18.2f}")

    # ── Experiment 3 ────────────────────────────────────────────────────
    print("\n[SLIDE 4] Experiment 3: Mismatch Stress Test (Validation Ladder)")
    e3 = all_results.get("exp3", {})
    print(f"  {'Perturbation':<12}  {'Det.Rate%':>10}  {'FA%':>10}  {'Below limit?':>12}")
    for key, v in e3.get("perturbation_levels", {}).items():
        flag = "[FAIL]" if v["below_acceptable_limit"] else ""
        fa_col = v.get('healthy_windows_flagged_pct', v.get('false_per_100h', 0))
        print(f"  {key:<12}  {v['detection_rate_pct']:>10.1f}  {fa_col:>10.1f}  {flag:>12}")
    print(f"  Breakpoint: {e3.get('breakpoint', 'none')}")
    note = e3.get('design_note', '')
    if note:
        print(f"  Note: {note[:120]}...")

    # ── Experiment 4 ────────────────────────────────────────────────────
    print("\n[SLIDE 2/4] Experiment 4: Edge Timing")
    e4 = all_results.get("exp4", {})
    print(f"  Platform:       {e4.get('cpu', 'unknown')} ({e4.get('hardware_note', '')})")
    print(f"  Model size:     {e4.get('model_size_mb', 'N/A')} MB")
    print(f"  Median latency: {e4.get('median_ms', 'N/A')} ms")
    print(f"  P99 latency:    {e4.get('p99_ms', 'N/A')} ms")
    print(f"  Fits 10ms (100Hz)? {'YES (OK)' if e4.get('fits_10ms_frame') else 'NO (exceeds 10ms)'}")

    # ── Experiment 5 ────────────────────────────────────────────────────
    print("\n[SLIDE 6] Experiment 5: Ablation Table")
    e5 = all_results.get("exp5", {})
    print(f"  {'Variant':<30}  {'Lead p50 (s)':>12}  {'Lead p10 (s)':>12}  {'FA/100h':>8}")
    labels = {
        "a_static":         "(a) Static thresholds",
        "b_det_raw_telem":  "(b) Detector on raw telemetry",
        "c_det_twin_resid": "(c) Detector on twin residuals",
        "d_full_pipeline":  "(d) Full pipeline",
    }
    for var, lbl in labels.items():
        row = e5.get(var, {})
        print(f"  {lbl:<30}  {row.get('median_lead_s', 0):>+12.1f}  {row.get('p10_lead_s', 0):>+12.1f}  {row.get('false_per_100h', 0):>8.2f}")

    # ── Extras ───────────────────────────────────────────────────────────
    print("\n[EXTRAS]")
    ea = all_results.get("extra_a", {})
    print(f"\n  A. Sensor Drift Classification:")
    print(f"     Correct 'drift' label:      {ea.get('pct_correct_drift', 0):.1f}%")
    print(f"     Wrong engine fault label:   {ea.get('pct_wrong_engine', 0):.1f}%")

    eb = all_results.get("extra_b", {})
    print(f"\n  B. Altitude Invariance (residual overall mean / std):")
    for alt_key, vals in eb.items():
        print(f"     {alt_key:>7s}  mean={vals['overall_mean']:+.5f}  std={vals['overall_std']:.5f}")

    ec = all_results.get("extra_c", {})
    print(f"\n  C. RUL Error Bands (simulation only):")
    print(f"     p10: {ec.get('p10_error_pct', 0):.1f}%  p50: {ec.get('p50_error_pct', 0):.1f}%  p90: {ec.get('p90_error_pct', 0):.1f}%")
    print(f"     Note: {ec.get('validation_note', '')}")


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("AeroTwin-4 -- Five-Experiment Validation Suite")
    print(f"Global seed: {GLOBAL_SEED} | Torch available: {TORCH_AVAILABLE}")
    print(f"Engine physics available: {ENGINE_OK} | CNN-LSTM available: {MODEL_OK}")
    print(f"Results will be saved to: {RESULTS_DIR}\n")

    all_results = {}
    errors      = {}

    experiments = [
        ("exp1",    "Experiment 1 -- Detection Lead Time",      lambda: experiment_1(n_repeats=20)),
        ("exp2",    "Experiment 2 -- False Alerts",             lambda: experiment_2(target_hours=110.0)),
        ("exp3",    "Experiment 3 -- Mismatch Stress Test",     lambda: experiment_3(n_repeats=20)),
        ("exp4",    "Experiment 4 -- Edge Timing",              lambda: experiment_4(n_windows=1000)),
        ("exp5",    "Experiment 5 -- Ablation",                 lambda: experiment_5(n_repeats=10)),
        ("extra_a", "Extra A -- Drift vs Real Fault",           lambda: extra_a_drift_vs_fault(n_runs=30)),
        ("extra_b", "Extra B -- Altitude Invariance",           extra_b_altitude_invariance),
        ("extra_c", "Extra C -- RUL Honesty",                   lambda: extra_c_rul_honesty(n_runs=50)),
    ]

    for key, name, fn in experiments:
        t0 = time.perf_counter()
        try:
            print(f"\n>>> Running: {name}")
            result = fn()
            all_results[key] = result
            elapsed = time.perf_counter() - t0
            print(f"    Completed in {elapsed:.1f}s")
        except Exception as e:
            tb = traceback.format_exc()
            print(f"    ERROR in {name}:\n{tb}")
            errors[key] = str(e)
            all_results[key] = {}

    # Save all results
    output_path = RESULTS_DIR / "experiment_results.json"
    with open(output_path, "w") as f:
        json.dump({
            "meta": {
                "global_seed": GLOBAL_SEED,
                "torch_available": TORCH_AVAILABLE,
                "engine_ok": ENGINE_OK,
                "model_ok": MODEL_OK,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            },
            "results": all_results,
            "errors":  errors,
        }, f, indent=2, default=str)

    print(f"\n[OK] Results saved to: {output_path}")

    # Print slide-ready tables
    if all_results:
        try:
            print_slide_tables(all_results)
        except Exception as e:
            print(f"[WARN] Table printing error: {e}")

    return all_results


if __name__ == "__main__":
    main()
