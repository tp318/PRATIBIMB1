"""
============================================================================
inference.py  —  Real-Time Anomaly Inference & Sensor Attribution Engine
============================================================================
Author  : Senior AI/ML Engineer — Aerospace PHM Division
Purpose : Production inference module for UAV engine anomaly detection.
          Takes a continuous stream or single residual window, producing:
            (a) Continuous Anomaly Score (0.0–1.0) and Anomaly Ratio
            (b) Alert Level ("NORMAL", "WATCH", "CAUTION", "CRITICAL")
            (c) Per-sensor reconstruction attribution (% contribution)
            (d) Operator-ready JSON / CLI audit reports
============================================================================
"""

from __future__ import annotations

import json
import logging
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np
import torch

from config import (
    ALERT_LEVELS,
    CHECKPOINT_PATH,
    DEVICE,
    EWMA_ALPHA,
    NUM_SENSORS,
    SCALER_PATH,
    SENSOR_CHANNELS,
    SENSOR_UNITS,
    STRIDE,
    THRESHOLD_PATH,
    WINDOW_SIZE,
)
from architecture import LSTMAutoencoder, load_model
from preprocessing import ResidualScaler

logger = logging.getLogger("UAV.PHM.AnomalyInference")


# ═══════════════════════════════════════════════════════════════════════════
#  ANOMALY REPORT DATA CLASS
# ═══════════════════════════════════════════════════════════════════════════

class AnomalyReport:
    """
    Structured report for a single window anomaly evaluation.
    """

    def __init__(
        self,
        anomaly_score: float,
        anomaly_ratio: float,
        is_anomaly: bool,
        alert_level: str,
        reconstruction_error: float,
        threshold_used: float,
        sensor_contributions_pct: Dict[str, float],
        top_anomalous_sensors: List[Tuple[str, float]],
        inference_latency_ms: float,
    ) -> None:
        self.timestamp = datetime.now(timezone.utc).isoformat()
        self.anomaly_score = round(float(anomaly_score), 4)
        self.anomaly_ratio = round(float(anomaly_ratio), 4)
        self.is_anomaly = bool(is_anomaly)
        self.alert_level = alert_level
        self.reconstruction_error = round(float(reconstruction_error), 6)
        self.threshold_used = round(float(threshold_used), 6)
        self.sensor_contributions_pct = sensor_contributions_pct
        self.top_anomalous_sensors = top_anomalous_sensors
        self.inference_latency_ms = round(float(inference_latency_ms), 2)

    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp,
            "anomaly_score": self.anomaly_score,
            "anomaly_ratio": self.anomaly_ratio,
            "is_anomaly": self.is_anomaly,
            "alert_level": self.alert_level,
            "reconstruction_error": self.reconstruction_error,
            "threshold_used": self.threshold_used,
            "top_anomalous_sensors": [
                {"sensor": s, "contribution_pct": p} for s, p in self.top_anomalous_sensors
            ],
            "sensor_contributions_pct": self.sensor_contributions_pct,
            "inference_latency_ms": self.inference_latency_ms,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def pretty_print(self) -> None:
        """Format report for operator console."""
        bar_len = 30
        filled = int(self.anomaly_score * bar_len)
        score_bar = "█" * filled + "░" * (bar_len - filled)

        status_color = "\033[92m" if not self.is_anomaly else "\033[91m"
        reset_color = "\033[0m"

        print("\n" + "=" * 70)
        print(f"  MALE UAV ANOMALY DETECTION REPORT — {self.timestamp}")
        print("=" * 70)
        print(f"  Status       : {status_color}{self.alert_level}{reset_color} "
              f"(Anomaly Flag: {self.is_anomaly})")
        print(f"  Anomaly Score: [{score_bar}] {self.anomaly_score * 100:.1f}%")
        print(f"  Error Ratio  : {self.anomaly_ratio:.2f}x of baseline threshold")
        print(f"  Raw MSE Loss : {self.reconstruction_error:.6f} (Threshold: {self.threshold_used:.6f})")
        print(f"  Latency      : {self.inference_latency_ms:.2f} ms")
        print("-" * 70)
        print("  Sensor Residual Reconstruction Divergence:")
        for sensor, pct in self.top_anomalous_sensors[:5]:
            unit = SENSOR_UNITS.get(sensor, "")
            sub_bar = "█" * int(pct / 100 * 20) + "░" * (20 - int(pct / 100 * 20))
            print(f"    {sensor:<24} [{sub_bar}] {pct:>5.1f}% [{unit}]")
        print("=" * 70 + "\n")


# ═══════════════════════════════════════════════════════════════════════════
#  INFERENCE ENGINE
# ═══════════════════════════════════════════════════════════════════════════

class AnomalyInferenceEngine:
    """
    Evaluates individual residual windows for reconstruction anomalies.
    """

    def __init__(
        self,
        model: LSTMAutoencoder,
        scaler: ResidualScaler,
        threshold: float,
        device: torch.device = DEVICE,
    ) -> None:
        self.model = model.to(device)
        self.model.eval()
        self.scaler = scaler
        self.threshold = threshold
        self.device = device

    @classmethod
    def from_artifacts(
        cls,
        checkpoint_path: Path = CHECKPOINT_PATH,
        scaler_path: Path = SCALER_PATH,
        threshold_path: Path = THRESHOLD_PATH,
        device: torch.device = DEVICE,
    ) -> "AnomalyInferenceEngine":
        """Factory method loading serialized artifacts from disk."""
        model = load_model(checkpoint_path, device=device)
        scaler = ResidualScaler.load(scaler_path)

        with open(threshold_path, "r") as f:
            data = json.load(f)
        threshold = data.get("statistical_threshold_3sigma", 0.05)

        return cls(model=model, scaler=scaler, threshold=threshold, device=device)

    def predict(self, window: np.ndarray) -> AnomalyReport:
        """
        Evaluate a single window of shape (F, W) or (W, F).

        Args:
            window: Residual window array.
        """
        t0 = time.perf_counter()

        if window.ndim == 2:
            if window.shape[0] != NUM_SENSORS and window.shape[1] == NUM_SENSORS:
                window = window.T  # Convert (W, F) -> (F, W)
            x_tensor = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(self.device)
        elif window.ndim == 3:
            x_tensor = torch.tensor(window, dtype=torch.float32).to(self.device)
        else:
            raise ValueError(f"Unexpected window shape: {window.shape}")

        with torch.no_grad():
            x_hat = self.model(x_tensor)
            total_mse, sensor_mse = self.model.compute_reconstruction_error(x_tensor, x_hat)

        raw_mse = float(total_mse.item())
        sensor_errors = sensor_mse.squeeze(0).cpu().numpy()

        # Ratio relative to calibrated threshold
        anomaly_ratio = raw_mse / (self.threshold + 1e-8)
        # Normalized score mapped smoothly between [0, 1]
        anomaly_score = float(1.0 - np.exp(-0.693 * anomaly_ratio))

        is_anomaly = anomaly_ratio >= 1.0

        # Determine alert level
        alert_level = "NORMAL"
        for lvl, cutoff in sorted(ALERT_LEVELS.items(), key=lambda x: x[1], reverse=True):
            if anomaly_ratio >= cutoff:
                alert_level = lvl
                break

        # Calculate per-sensor contribution percentages
        total_sensor_err = np.sum(sensor_errors) + 1e-10
        sensor_pcts = {
            SENSOR_CHANNELS[i]: round(float(sensor_errors[i] / total_sensor_err * 100.0), 2)
            for i in range(NUM_SENSORS)
        }
        top_sensors = sorted(sensor_pcts.items(), key=lambda x: x[1], reverse=True)

        latency_ms = (time.perf_counter() - t0) * 1000.0

        return AnomalyReport(
            anomaly_score=anomaly_score,
            anomaly_ratio=anomaly_ratio,
            is_anomaly=is_anomaly,
            alert_level=alert_level,
            reconstruction_error=raw_mse,
            threshold_used=self.threshold,
            sensor_contributions_pct=sensor_pcts,
            top_anomalous_sensors=top_sensors,
            inference_latency_ms=latency_ms,
        )


# ═══════════════════════════════════════════════════════════════════════════
#  STREAMING MONITOR
# ═══════════════════════════════════════════════════════════════════════════

class StreamingAnomalyMonitor:
    """
    Online streaming buffer accepting real-time AUKF residual timesteps (F=9).
    Maintains a rolling window and fires inference every STRIDE timesteps.
    """

    def __init__(
        self,
        engine: AnomalyInferenceEngine,
        window_size: int = WINDOW_SIZE,
        stride: int = STRIDE,
        ewma_alpha: float = EWMA_ALPHA,
    ) -> None:
        self.engine = engine
        self.window_size = window_size
        self.stride = stride
        self.ewma_alpha = ewma_alpha

        self.buffer: Deque[np.ndarray] = deque(maxlen=window_size)
        self.steps_since_last_eval = 0
        self.smoothed_score: Optional[float] = None

    def push(self, residual_timestep: np.ndarray) -> Optional[AnomalyReport]:
        """
        Push a single raw AUKF residual reading of shape (F=9,).
        Returns an AnomalyReport whenever a full stride interval expires.
        """
        # Scale incoming timestep
        scaled = self.engine.scaler.transform(residual_timestep.reshape(1, -1)).squeeze(0)
        self.buffer.append(scaled)
        self.steps_since_last_eval += 1

        if len(self.buffer) == self.window_size and self.steps_since_last_eval >= self.stride:
            self.steps_since_last_eval = 0
            # Buffer content: (W, F) -> Transpose to (F, W)
            window = np.array(self.buffer, dtype=np.float32).T
            report = self.engine.predict(window)

            # Apply EWMA smoothing
            if self.smoothed_score is None:
                self.smoothed_score = report.anomaly_score
            else:
                self.smoothed_score = (
                    self.ewma_alpha * report.anomaly_score
                    + (1.0 - self.ewma_alpha) * self.smoothed_score
                )
            report.anomaly_score = round(self.smoothed_score, 4)
            return report

        return None


# ═══════════════════════════════════════════════════════════════════════════
#  DEMO RUNNER
# ═══════════════════════════════════════════════════════════════════════════

def run_inference_demo() -> None:
    """Demonstrate anomaly detection on healthy vs anomalous windows."""
    from preprocessing import generate_synthetic_telemetry

    print("\n--- Initializing Anomaly Detection Inference Engine ---")
    engine = AnomalyInferenceEngine.from_artifacts()

    # Generate synthetic telemetry containing both healthy and faulty intervals
    residuals, labels = generate_synthetic_telemetry(num_timesteps=3000, seed=101)
    norm_residuals = engine.scaler.transform(residuals)

    # 1. Healthy Window
    healthy_idx = np.where(labels == 0)[0][100]
    healthy_win = norm_residuals[healthy_idx : healthy_idx + WINDOW_SIZE].T
    print("\n--- Evaluating KNOWN HEALTHY Window ---")
    rep_healthy = engine.predict(healthy_win)
    rep_healthy.pretty_print()

    # 2. Anomalous Window
    anom_indices = np.where(labels == 1)[0]
    if len(anom_indices) > 0:
        anom_idx = anom_indices[len(anom_indices) // 2]
        anom_win = norm_residuals[anom_idx : anom_idx + WINDOW_SIZE].T
        print("--- Evaluating KNOWN ANOMALOUS Window ---")
        rep_anom = engine.predict(anom_win)
        rep_anom.pretty_print()


if __name__ == "__main__":
    run_inference_demo()
