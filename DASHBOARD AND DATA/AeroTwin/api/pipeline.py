"""
AeroTwin-4 Live Inference Pipeline.

Single place where a raw telemetry frame becomes a complete engine assessment:

    telemetry -> Digital Twin residuals -> [anomaly | diagnosis | health -> RUL]
              -> mission risk -> operator decision

The API layer owns transport; this owns the analytics, so the same pipeline can be
driven by a websocket, a REST call, or a batch script without duplicating logic.

Models are loaded lazily and each stage degrades independently: if the diagnosis
artifacts are missing, the pipeline still streams telemetry, residuals and anomaly
scores rather than failing outright. A partially-trained deployment should still
fly the parts that do work.
"""

import os
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from AeroTwin.degradation.conditions import RunConditionSampler
from AeroTwin.health.engine import DigitalTwinStateEngine
from AeroTwin.ml.anomaly.features import FeatureExtractor
from AeroTwin.ml.anomaly.preprocessing import FeatureScaler
from AeroTwin.ml.anomaly.statistical import StatisticalAnomalyDetector
from AeroTwin.ml.diagnosis.classifier import FaultDiagnosisClassifier
from AeroTwin.ml.rul.health_estimator import HealthEstimator
from AeroTwin.ml.rul.projector import RULProjector
from AeroTwin.mission.advisory import MaintenanceAdvisor
from AeroTwin.mission.reporting import AlertLog, EfficiencyTracker, MissionReport
from AeroTwin.mission.risk import MissionProfile, MissionRiskAssessor

# 5 s window at 100 Hz, matching the training-time feature contract exactly.
WINDOW_SECONDS = 5.0
SAMPLE_RATE_HZ = 100.0
WINDOW_SAMPLES = int(WINDOW_SECONDS * SAMPLE_RATE_HZ)

# How often a fresh window is scored, in samples (1 s at 100 Hz).
SCORE_STRIDE_SAMPLES = 100


class LiveAssessmentPipeline:
    """Streams telemetry frames through the full analytics stack."""

    def __init__(
        self,
        root_dir: str,
        feature_config: str = "residual",
        engine_parameters: Optional[Dict[str, Any]] = None,
        seed: int = 42,
        dt: float = 0.01,
        mission: Optional[MissionProfile] = None,
    ):
        self.root_dir = root_dir
        self.feature_config = feature_config
        self.seed = seed

        self.twin = DigitalTwinStateEngine(
            dt=dt, seed=seed, mode="COUNTERFACTUAL", engine_parameters=engine_parameters
        )
        self.extractor = FeatureExtractor(config_type=feature_config.upper())
        self.mission = mission or MissionProfile(name="ISR_SORTIE", required_duration_s=600.0)
        self.assessor = MissionRiskAssessor()
        self.advisor = MaintenanceAdvisor()

        # Operator-facing history: efficiency trend, discrete events, sortie record.
        self.efficiency = EfficiencyTracker()
        self.alerts = AlertLog()
        self.report = MissionReport()
        self.report.reset(self.mission.name, self.mission.required_duration_s)
        self.alerts.sortie_started(0.0, f"Mission {self.mission.name}, {self.mission.required_duration_s:.0f} s required.")

        # Rolling buffer of twin-derived rows; one scored window is the last N rows.
        self._buffer: Deque[Dict[str, Any]] = deque(maxlen=WINDOW_SAMPLES)
        self._samples_seen = 0
        self._health_history: List[Tuple[float, float]] = []

        self.anomaly_scaler: Optional[FeatureScaler] = None
        self.anomaly_model: Optional[StatisticalAnomalyDetector] = None
        self.anomaly_threshold: Optional[float] = None
        self.diagnoser: Optional[FaultDiagnosisClassifier] = None
        self.diagnosis_scaler: Optional[FeatureScaler] = None
        self.health_estimator: Optional[HealthEstimator] = None
        self.health_scaler: Optional[FeatureScaler] = None
        self.projector = RULProjector()

        self.loaded: Dict[str, bool] = {}
        self._load_models()

    # ------------------------------------------------------------------ loading

    def _load_models(self):
        cfg = self.feature_config

        # Anomaly: statistical detector is used live because it needs no torch
        # runtime and scored best on residual features in the Phase 5 ablation.
        try:
            scaler_path = os.path.join(self.root_dir, "models", "phase5", "preprocessing", cfg, "scaler.json")
            model_path = os.path.join(self.root_dir, "models", "phase5", "statistical", cfg, "model.json")
            if os.path.exists(scaler_path) and os.path.exists(model_path):
                self.anomaly_scaler = FeatureScaler().load(scaler_path)
                self.anomaly_model = StatisticalAnomalyDetector().load(model_path)
                self.anomaly_threshold = getattr(self.anomaly_model, "threshold", None)
                self.loaded["anomaly"] = True
            else:
                self.loaded["anomaly"] = False
        except Exception:
            self.loaded["anomaly"] = False

        try:
            diag_dir = os.path.join(self.root_dir, "models", "diagnosis", cfg)
            if os.path.exists(os.path.join(diag_dir, "model.joblib")):
                self.diagnoser = FaultDiagnosisClassifier().load(diag_dir)
                self.diagnosis_scaler = FeatureScaler().load(os.path.join(diag_dir, "scaler.json"))
                self.loaded["diagnosis"] = True
            else:
                self.loaded["diagnosis"] = False
        except Exception:
            self.loaded["diagnosis"] = False

        try:
            rul_dir = os.path.join(self.root_dir, "models", "rul", cfg)
            if os.path.exists(os.path.join(rul_dir, "health_model.joblib")):
                self.health_estimator = HealthEstimator().load(rul_dir)
                self.health_scaler = FeatureScaler().load(os.path.join(rul_dir, "scaler.json"))
                self.projector = RULProjector(health_noise_std=self.health_estimator.residual_std)
                self.loaded["rul"] = True
            else:
                self.loaded["rul"] = False
        except Exception:
            self.loaded["rul"] = False

    # ------------------------------------------------------------------ ingest

    def _twin_row(self, telemetry: Dict[str, Any]) -> Dict[str, Any]:
        """One telemetry frame -> one flat residual row, matching the Phase 4 schema."""
        frame = self.twin.process_telemetry(telemetry)
        row: Dict[str, Any] = {"simulation_time": frame.simulation_time}
        for k, v in frame.observed_outputs.items():
            row["obs_" + k] = v
        for k, v in frame.expected_outputs.items():
            row["exp_" + k] = v
        for k, v in frame.residuals.raw_signed.items():
            row["res_signed_" + k] = v
        for k, v in frame.residuals.normalized.items():
            row["res_norm_" + k] = v
        row["ind_thermal_dev"] = frame.indicators.thermal_deviation
        row["ind_oil_dev"] = frame.indicators.oil_deviation
        row["ind_vibration_dev"] = frame.indicators.vibration_deviation
        row["ind_torque_dev"] = frame.indicators.torque_deviation
        row["ind_cylinder_balance_dev"] = frame.indicators.cylinder_balance_deviation
        return row

    def ingest(self, telemetry: Dict[str, Any]) -> Dict[str, Any]:
        """
        Push one telemetry frame. Returns the frame-level twin state, plus a full
        assessment whenever a complete window is due.
        """
        row = self._twin_row(telemetry)
        self._buffer.append(row)
        self._samples_seen += 1

        out: Dict[str, Any] = {
            "simulation_time": row["simulation_time"],
            "observed": {k[4:]: v for k, v in row.items() if k.startswith("obs_")},
            "expected": {k[4:]: v for k, v in row.items() if k.startswith("exp_")},
            "indicators": {k[4:]: v for k, v in row.items() if k.startswith("ind_")},
            "window_ready": len(self._buffer) == WINDOW_SAMPLES,
            "assessment": None,
        }

        observed = out["observed"]
        expected = out["expected"]
        eff = self.efficiency.update(observed, expected, row["simulation_time"])
        out["efficiency"] = eff.to_dict()

        due = (
            len(self._buffer) == WINDOW_SAMPLES
            and self._samples_seen % SCORE_STRIDE_SAMPLES == 0
        )
        if due:
            out["assessment"] = self.assess_current_window()
        # Note: alerts.evaluate() / report.update() are NOT called here. This
        # pipeline's own diagnosis/anomaly stage only ever sees real physics
        # residuals, so a sensor-only fault (which never touches the twin)
        # would be invisible to it. The caller (server.py) drives self.alerts
        # and self.report from the fully-merged assessment - which also
        # carries the live XGBoost diagnosis - once per scored window, so
        # every fault family (including instrumentation/sensor faults) is
        # actually observable in the alert log and mission report.
        return out

    # ------------------------------------------------------------------ scoring

    def assess_current_window(self) -> Optional[Dict[str, Any]]:
        """Score the buffered window through every available stage."""
        if len(self._buffer) < WINDOW_SAMPLES:
            return None

        df_win = pd.DataFrame(list(self._buffer))
        feats = self.extractor.extract_window_features(df_win)
        X = pd.DataFrame([feats]).fillna(0.0).replace([np.inf, -np.inf], 0.0)
        t_now = float(df_win["simulation_time"].iloc[-1])

        result: Dict[str, Any] = {"simulation_time": t_now}

        # --- anomaly ---------------------------------------------------------
        anomaly_flagged = False
        score_ratio = 0.0
        if self.loaded.get("anomaly") and self.anomaly_scaler and self.anomaly_model:
            try:
                Xs = pd.DataFrame(
                    self.anomaly_scaler.transform(X), columns=self.anomaly_scaler.feature_names
                )
                score = float(self.anomaly_model.compute_anomaly_score(Xs)[0])
                thr = self.anomaly_threshold
                anomaly_flagged = bool(thr is not None and score > thr)
                score_ratio = float(score / thr) if thr else 0.0
                result["anomaly"] = {
                    "score": round(score, 6),
                    "threshold": None if thr is None else round(float(thr), 6),
                    "flagged": anomaly_flagged,
                    "score_ratio": round(score_ratio, 4),
                }
            except Exception as exc:
                result["anomaly"] = {"error": str(exc)}

        # --- diagnosis -------------------------------------------------------
        predicted_fault, fault_conf = "HEALTHY", 0.0
        if self.loaded.get("diagnosis") and self.diagnoser and self.diagnosis_scaler:
            try:
                Xd = pd.DataFrame(
                    self.diagnosis_scaler.transform(X), columns=self.diagnosis_scaler.feature_names
                )
                diag = self.diagnoser.diagnose(Xd).iloc[0]
                predicted_fault = str(diag["predicted_fault"])
                fault_conf = float(diag["confidence"])
                result["diagnosis"] = {
                    "predicted_fault": predicted_fault,
                    "confidence": round(fault_conf, 4),
                    "runner_up": str(diag["runner_up_fault"]),
                    "margin": round(float(diag["margin"]), 4),
                }
            except Exception as exc:
                result["diagnosis"] = {"error": str(exc)}

        # --- health + RUL ----------------------------------------------------
        health = 1.0
        rul_point: Optional[float] = None
        rul_lower: Optional[float] = None
        rul_conf = 1.0
        if self.loaded.get("rul") and self.health_estimator and self.health_scaler:
            try:
                Xh = pd.DataFrame(
                    self.health_scaler.transform(X), columns=self.health_scaler.feature_names
                )
                health = float(self.health_estimator.predict(Xh)[0])
                self._health_history.append((t_now, health))

                times = [t for t, _ in self._health_history]
                healths = [h for _, h in self._health_history]
                est = self.projector.estimate(times, healths)
                rul_point = est.rul_seconds
                rul_lower = est.rul_lower_seconds
                rul_conf = est.confidence
                result["health"] = {"health_index": round(health, 4)}
                result["rul"] = est.to_dict()
            except Exception as exc:
                result["rul"] = {"error": str(exc)}

        # --- mission risk ----------------------------------------------------
        try:
            assessment = self.assessor.assess(
                mission=self.mission,
                health_index=health,
                rul_seconds=rul_point,
                rul_lower_seconds=rul_lower,
                predicted_fault=predicted_fault,
                fault_confidence=fault_conf,
                anomaly_flagged=anomaly_flagged,
                anomaly_score_ratio=score_ratio,
                rul_confidence=rul_conf,
            )
            result["mission_risk"] = assessment.to_dict()
        except Exception as exc:
            result["mission_risk"] = {"error": str(exc)}

        try:
            advisory = self.advisor.advise(
                predicted_fault=predicted_fault,
                fault_confidence=fault_conf,
                health_index=health,
                rul_seconds=rul_lower,
                mission_required_s=self.mission.total_required_s,
                anomaly_flagged=anomaly_flagged,
            )
            result["maintenance"] = [a.to_dict() for a in advisory]
        except Exception as exc:
            result["maintenance"] = [{"error": str(exc)}]

        return result

    def status(self) -> Dict[str, Any]:
        return {
            "feature_config": self.feature_config,
            "models_loaded": self.loaded,
            "samples_seen": self._samples_seen,
            "window_samples_required": WINDOW_SAMPLES,
            "window_ready": len(self._buffer) == WINDOW_SAMPLES,
            "mission": {
                "name": self.mission.name,
                "required_duration_s": self.mission.required_duration_s,
                "reserve_duration_s": self.mission.reserve_duration_s,
            },
        }

    def efficiency_series(self, limit: int = 240):
        return self.efficiency.series()[-limit:]

    def mission_report(self):
        return self.report.build(
            efficiency=self.efficiency.summary(), alert_counts=self.alerts.counts()
        )

    def reset(self):
        self._buffer.clear()
        self._samples_seen = 0
        self._health_history.clear()
        self.twin.reset(seed=self.seed)
        self.efficiency.reset()
        self.alerts.reset()
        self.report.reset(self.mission.name, self.mission.required_duration_s)


def default_engine_parameters(run_id: str = "LIVE_001", seed: int = 42) -> Dict[str, Any]:
    """Build a condition-matched parameter set for a live engine unit."""
    sampler = RunConditionSampler()
    return sampler.build_engine_parameters(sampler.sample(run_id, seed))
