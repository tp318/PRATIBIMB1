"""
=============================================================================
AEROTWIN-4 LIVE XGBOOST DIAGNOSTICS ADAPTER  (v2 — Hybrid Physics)
=============================================================================
Buffers real-time telemetry frames, applies the SAME physics-consistent
synthetic perturbations used during dataset generation (fault_perturbations.py),
extracts 30-second window features, and runs live inference using the
trained 9-class XGBoost model.

Key design:
  - Perturbations are applied based on the currently-injected fault type
    so the model sees features identical to training.
  - Classes 3 (COOLING), 4 (LUBRICATION), 7 (OVERHEATING) use real engine
    physics only — no perturbation needed.
  - Classes 1 (MISFIRE), 2 (INJECTOR), 6 (COMBUSTION_INST), 8 (BEARING_VIB)
    receive synthetic perturbations matching the dataset generator.
=============================================================================
"""

import os
import sys
import math
from collections import deque
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

# Ensure FAULT DETECTION directory is on path
_BASE_DIR  = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
_FAULT_DIR = os.path.join(_BASE_DIR, "FAULT DETECTION")
if _FAULT_DIR not in sys.path:
    sys.path.insert(0, _FAULT_DIR)

try:
    from xgboost_inference import XGBoostFaultPredictor
except ImportError:
    XGBoostFaultPredictor = None

# Import shared perturbation function — same logic used in dataset generation
from AeroTwin.api.fault_perturbations import (
    apply_fault_perturbations,
    NOISE_STD,
    FAULT_TYPE_TO_CLASS,
    FAULT_CLASS_NAMES,
)


class LiveXGBoostAdapter:
    """
    Online streaming buffer and inference adapter for the 9-class XGBoost model.
    Applies hybrid perturbations that match training-time feature distributions.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        meta_path: Optional[str] = None,
        buffer_seconds: int = 30,
    ):
        if model_path is None:
            model_path = os.path.join(_FAULT_DIR, "best_xgboost_fault_detector.json")
        if meta_path is None:
            meta_path = os.path.join(_FAULT_DIR, "xgboost_model_features.json")

        self.loaded = False
        self.predictor = None
        self.last_diagnosis = None
        self.buffer_seconds = buffer_seconds
        self.buffer = deque(maxlen=buffer_seconds)
        self.last_snapshot_time = -1.0

        # Active fault context — set by server.py when fault is injected
        self.active_fault_class: int = 0           # 0 = NORMAL
        self.active_fault_severity: float = 0.0
        self.active_fault_onset_time: float = -1.0
        self.active_fault_ramp_s: float = 30.0
        # Per-adapter drift accumulator (SENSOR_DRIFT class 5)
        self._drift_bias: float = 0.0
        # Stable RNG seeded once per sortie — keeps perturbations deterministic
        self._rng = np.random.Generator(np.random.PCG64(12345))

        if XGBoostFaultPredictor is not None and os.path.exists(model_path) and os.path.exists(meta_path):
            try:
                self.predictor = XGBoostFaultPredictor(model_path=model_path, meta_path=meta_path)
                self.loaded = True
                print(f"[XGBoostAdapter] Loaded model from {model_path}")
            except Exception as e:
                print(f"[XGBoostAdapter] Warning: Could not initialize model: {e}")

    def reset(self):
        """Called when a new sortie starts."""
        self.buffer.clear()
        self.last_snapshot_time = -1.0
        self.last_diagnosis = None
        self.active_fault_class = 0
        self.active_fault_severity = 0.0
        self.active_fault_onset_time = -1.0
        self.active_fault_ramp_s = 30.0
        self._drift_bias = 0.0
        self._rng = np.random.Generator(np.random.PCG64(12345))

    def set_active_fault(
        self,
        fault_type: str,
        severity: float,
        sim_time: float,
        ramp_duration_s: float = 30.0,
    ):
        """
        Called by server.py inject_fault / clear_fault to update the
        active fault context so perturbations are applied correctly.
        """
        cls = FAULT_TYPE_TO_CLASS.get(fault_type.upper(), 0)
        self.active_fault_class = cls
        self.active_fault_severity = float(severity)
        self.active_fault_onset_time = float(sim_time)
        self.active_fault_ramp_s = float(ramp_duration_s)
        self._drift_bias = 0.0   # reset drift on fault change
        print(f"[XGBoostAdapter] Active fault: {FAULT_CLASS_NAMES[cls]} (class {cls}), sev={severity:.2f}, onset={sim_time:.1f}s")

    def clear_fault(self):
        """Remove active fault — return to NORMAL state."""
        self.active_fault_class = 0
        self.active_fault_severity = 0.0
        self.active_fault_onset_time = -1.0
        self._drift_bias = 0.0

    def _get_ramp_factor(self, sim_time: float) -> float:
        if self.active_fault_class == 0 or self.active_fault_onset_time < 0:
            return 0.0
        if sim_time < self.active_fault_onset_time:
            return 0.0
        return min(1.0, (sim_time - self.active_fault_onset_time) / max(1.0, self.active_fault_ramp_s))

    def ingest_frame(
        self,
        telemetry: Dict[str, Any],
        expected: Dict[str, Any],
        controls: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Ingests a frame. Samples at ~1 Hz into the 30-second rolling window.
        Returns the latest diagnosis dictionary when ready.
        """
        if not self.loaded or self.predictor is None:
            return None

        t = float(telemetry.get("simulation_time", 0.0))
        if t - self.last_snapshot_time < 0.95 and len(self.buffer) > 0:
            return self.last_diagnosis
        self.last_snapshot_time = t

        ctrl    = controls or {}
        alt_ft  = float(ctrl.get("altitude_ft", 0.0))
        temp_c  = float(ctrl.get("ambient_c", 15.0))
        throttle = float(telemetry.get("throttle", ctrl.get("throttle", 0.5)))

        # Raw engine measurements
        rpm      = float(telemetry.get("rpm", 800.0))
        cht      = float(telemetry.get("cht", 75.0))
        egt      = float(telemetry.get("egt", 350.0))
        oil_p_raw = telemetry.get("oil_pressure_psi", telemetry.get("oil_pressure", 58.0))
        oil_p    = float(oil_p_raw) * 0.0689476 if float(oil_p_raw) > 15.0 else float(oil_p_raw)
        oil_t    = float(telemetry.get("oil_temperature", 65.0))
        fuel_flow = float(telemetry.get("fuel_flow_lph", telemetry.get("fuel_flow", 6.0)))
        vib      = float(telemetry.get("vibration", telemetry.get("vibration_rms", 0.05)))
        batt_v   = float(telemetry.get("battery_voltage", 13.8))
        alt_i    = float(telemetry.get("alternator_current", 20.0))

        # === Apply hybrid perturbations for faults that need synthetic signatures ===
        ramp_factor = self._get_ramp_factor(t)
        if ramp_factor > 0 and self.active_fault_class in (1, 2, 6, 7, 8):
            rpm, egt, fuel_flow, vib, vib_kurtosis, vib_crest = apply_fault_perturbations(
                self.active_fault_class, ramp_factor, self.active_fault_severity,
                t, rpm, egt, fuel_flow, vib, self._rng
            )
        else:
            vib_kurtosis = 3.0
            vib_crest    = 3.5

        # Sensor drift accumulation for class 5 (SENSOR_DRIFT)
        if self.active_fault_class == 5 and ramp_factor > 0:
            self._drift_bias += float(self._rng.normal(0.25, 0.12)) * self.active_fault_severity * ramp_factor
            self._drift_bias  = min(self._drift_bias, 100.0)
        elif self.active_fault_class != 5:
            self._drift_bias = 0.0
        egt_measured = egt + self._drift_bias

        # DT expected values
        exp_rpm   = float(expected.get("rpm",   rpm))
        exp_cht   = float(expected.get("cht",   cht))
        exp_egt   = float(expected.get("egt",   telemetry.get("egt", egt)))  # DT unaware of sensor drift
        exp_op_raw = expected.get("oil_pressure_psi", expected.get("oil_pressure", oil_p_raw))
        exp_op    = float(exp_op_raw) * 0.0689476 if float(exp_op_raw) > 15.0 else float(exp_op_raw)
        exp_ot    = float(expected.get("oil_temperature", oil_t))
        exp_fuel  = float(expected.get("fuel_flow_lph", expected.get("fuel_flow", fuel_flow)))
        exp_vib   = float(expected.get("vibration", telemetry.get("vibration", vib)))

        # Physics residuals (perturbed measured - healthy DT expected)
        res_rpm   = rpm          - exp_rpm
        res_cht   = cht          - exp_cht
        res_egt   = egt_measured - exp_egt
        res_oil_p = oil_p        - exp_op
        res_oil_t = oil_t        - exp_ot
        res_fuel  = fuel_flow    - exp_fuel
        res_vib   = vib          - exp_vib

        # Normalized residuals (z-scores)
        z_rpm   = res_rpm   / NOISE_STD["rpm"]
        z_cht   = res_cht   / NOISE_STD["cht"]
        z_egt   = res_egt   / NOISE_STD["egt"]
        z_oil_p = res_oil_p / NOISE_STD["oil_press"]
        z_oil_t = res_oil_t / NOISE_STD["oil_temp"]
        z_fuel  = res_fuel  / NOISE_STD["fuel_flow"]
        z_vib   = res_vib   / NOISE_STD["vibration"]

        # Atmospheric
        alt_m    = alt_ft * 0.3048
        p_kpa    = 101.325 * ((1.0 - 2.25577e-5 * alt_m) ** 5.25588)
        load_pct = min(100.0, max(15.0, throttle * 100.0 * (p_kpa / 101.325)))

        snapshot = {
            "t": t,
            "altitude_ft": alt_ft, "ambient_temperature_c": temp_c,
            "ambient_pressure_kpa": p_kpa, "humidity_pct": 50.0,
            "throttle": throttle, "engine_load": load_pct,
            "injection_command": throttle * 0.95 + 0.05,
            "injection_timing": 22.0 + throttle * 6.0,
            "rpm": rpm, "cht": cht, "egt": egt_measured,
            "oil_pressure": oil_p, "oil_temperature": oil_t,
            "fuel_flow": fuel_flow,
            "battery_voltage": batt_v, "alternator_current": alt_i,
            "injection_timing_deg": 22.0 + throttle * 6.0,
            "vibration_rms": vib, "vibration_std": vib * 0.25,
            "vibration_kurtosis": vib_kurtosis,
            "vibration_crest_factor": vib_crest,
            "vibration_peak_frequency": rpm / 60.0,
            "vibration_1x": vib * 0.6, "vibration_2x": vib * 0.2, "vibration_3x": vib * 0.08,
            "rpm_residual": res_rpm, "cht_residual": res_cht, "egt_residual": res_egt,
            "oil_pressure_residual": res_oil_p, "oil_temperature_residual": res_oil_t,
            "fuel_flow_residual": res_fuel, "vibration_residual": res_vib,
            "rpm_z": z_rpm, "cht_z": z_cht, "egt_z": z_egt,
            "oil_pressure_z": z_oil_p, "oil_temperature_z": z_oil_t,
            "fuel_flow_z": z_fuel, "vibration_z": z_vib,
        }

        self.buffer.append(snapshot)
        return self._evaluate()

    def _evaluate(self) -> Dict[str, Any]:
        """Computes slopes, rolling statistics, and evaluates the XGBoost model."""
        if len(self.buffer) < 3:
            return self._warmup_response()

        first = self.buffer[0]
        last  = self.buffer[-1]
        dt    = max(1.0, last["t"] - first["t"])

        feat_dict = dict(last)

        # Trends
        for key_pair in [
            ("cht",           "cht_slope"),
            ("egt",           "egt_slope"),
            ("oil_pressure",  "oil_pressure_slope"),
            ("oil_temperature","oil_temperature_slope"),
            ("fuel_flow",     "fuel_flow_slope"),
            ("vibration_rms", "vibration_slope"),
            ("cht_residual",  "cht_residual_slope"),
            ("egt_residual",  "egt_residual_slope"),
            ("oil_pressure_residual", "oil_pressure_residual_slope"),
            ("fuel_flow_residual",    "fuel_residual_slope"),
            ("vibration_residual",    "vibration_residual_slope"),
        ]:
            src, dst = key_pair
            feat_dict[dst] = (last[src] - first[src]) / dt

        # Rolling statistics
        def arr(key): return [s[key] for s in self.buffer]
        feat_dict["egt_residual_mean"]          = float(np.mean(arr("egt_residual")))
        feat_dict["egt_residual_std"]           = float(np.std(arr("egt_residual")))
        feat_dict["cht_residual_mean"]          = float(np.mean(arr("cht_residual")))
        feat_dict["cht_residual_std"]           = float(np.std(arr("cht_residual")))
        feat_dict["oil_pressure_residual_mean"] = float(np.mean(arr("oil_pressure_residual")))
        feat_dict["oil_pressure_residual_std"]  = float(np.std(arr("oil_pressure_residual")))
        feat_dict["fuel_residual_mean"]         = float(np.mean(arr("fuel_flow_residual")))
        feat_dict["fuel_residual_std"]          = float(np.std(arr("fuel_flow_residual")))
        feat_dict["vibration_mean"]             = float(np.mean(arr("vibration_rms")))
        feat_dict["vibration_kurtosis_mean"]    = float(np.mean(arr("vibration_kurtosis")))

        pred = self.predictor.predict(feat_dict)

        # Compute TreeSHAP explainability
        shap_explanation = None
        if hasattr(self.predictor, "explain"):
            try:
                shap_explanation = self.predictor.explain(feat_dict, top_k=8)
            except Exception as e:
                print(f"[XGBoostAdapter] SHAP explain error: {e}")

        probs       = pred["probabilities"]
        sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        top_name, top_prob     = sorted_probs[0]
        ru_name, ru_prob       = sorted_probs[1] if len(sorted_probs) > 1 else ("NONE", 0.0)
        margin = round(top_prob - ru_prob, 4)

        # Top feature deviations for operator explainability
        last_snap = self.buffer[-1]
        deviations = sorted([
            {"feature": "CHT z-score",       "value": round(last_snap["cht_z"], 2)},
            {"feature": "EGT z-score",       "value": round(last_snap["egt_z"], 2)},
            {"feature": "Oil Pres z-score",  "value": round(last_snap["oil_pressure_z"], 2)},
            {"feature": "Vibration RMS",     "value": round(last_snap["vibration_rms"], 3)},
            {"feature": "Vibration z-score", "value": round(last_snap["vibration_z"], 2)},
            {"feature": "RPM Residual",      "value": round(last_snap["rpm_residual"], 1)},
        ], key=lambda x: abs(x["value"]), reverse=True)

        res = {
            "model_name":     "PRATIBIMB XGBoost 9-Class Physics Digital Twin",
            "predicted_fault": top_name,
            "predicted_class": pred["predicted_class"],
            "confidence":     pred["confidence"],
            "is_anomaly":     pred["is_anomaly"],
            "runner_up":      ru_name,
            "margin":         margin,
            "probabilities":  probs,
            "top_deviations": deviations[:3],
            "shap_explanation": shap_explanation,
            "window_seconds": len(self.buffer),
            "active_fault_context": {
                "fault_class": self.active_fault_class,
                "fault_name":  FAULT_CLASS_NAMES[self.active_fault_class],
                "severity":    self.active_fault_severity,
                "ramp_factor": self._get_ramp_factor(last["t"]),
            },
        }
        self.last_diagnosis = res
        return res

    def _warmup_response(self) -> Dict[str, Any]:
        zero_probs = {n: 0.0 for n in FAULT_CLASS_NAMES}
        zero_probs["NORMAL"] = 1.0
        return {
            "model_name": "PRATIBIMB XGBoost 9-Class Physics Digital Twin",
            "predicted_fault": "NORMAL", "predicted_class": 0,
            "confidence": 1.0, "is_anomaly": False,
            "runner_up": "NONE", "margin": 1.0,
            "probabilities": zero_probs,
            "top_deviations": [],
            "shap_explanation": {
                "predicted_class": 0,
                "fault_name": "NORMAL",
                "confidence": 1.0,
                "base_value": 0.0,
                "output_margin": 0.0,
                "summary": "TreeSHAP: System initializing observation window; nominal baseline established.",
                "positive_drivers": [],
                "negative_suppressors": [],
                "top_attributions": [],
                "framework": "TreeSHAP / DeepSHAP Physics Feature Attribution",
            },
            "window_seconds": len(self.buffer),
            "status": "warming_up",
        }

