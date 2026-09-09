"""
============================================================================
inference.py — Explainable Fault Inference Engine with DeepSHAP
============================================================================
Author  : Senior AI/ML Engineer — Aerospace PHM Division
Purpose : Production inference module for the MALE UAV fault detection
          Digital Twin.  Takes a single residual window, produces:
            (a) A fault class prediction with confidence
            (b) A per-sensor SHAP attribution breakdown
            (c) A human-readable audit report for GCS operators

Why SHAP / DeepSHAP?
  ─────────────────────
  IEC 61508 / MIL-STD-882 safety standards for airborne systems require
  that AI-driven fault predictions be explainable and auditable.  SHAP
  (SHapley Additive exPlanations) provides:

  • Game-theoretic guarantees: attributions satisfy consistency,
    dummy, efficiency, and symmetry axioms.
  • Sensor-level granularity: each SHAP value directly maps to one
    physical sensor residual — interpretable by engine engineers.
  • Signed values: distinguish sensors that push probability UP
    (fault-driving) from those pushing it DOWN (counter-evidence).

  DeepSHAP is chosen over LIME (too slow, stochastic) or GradCAM
  (activation-map only, not input-feature attribution) because:
    • It is an exact, fast approximation for differentiable PyTorch models.
    • No random sampling → deterministic outputs, critical for safety audits.
    • Latency is suitable for real-time GCS deployment (< 100 ms per window).

Streaming / Online Inference:
  The StreamingFaultMonitor class wraps ExplainableFaultInference and
  accepts a continuous residual feed, automatically extracting windows
  and running inference at the configured stride rate.

Usage:
    # One-shot inference
    engine = ExplainableFaultInference.from_checkpoint("best_fault_detector.pt",
                                                       background_windows)
    result = engine.predict_and_explain(window)

    # Streaming
    monitor = StreamingFaultMonitor(engine)
    for timestep_residual in aukf_stream:
        alert = monitor.push(timestep_residual)
        if alert:
            print(alert)
============================================================================
"""

from __future__ import annotations

import json
import logging
import time
from collections import deque
from datetime import datetime, timezone
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np
import shap
import torch

from config import (
    SENSOR_CHANNELS, SENSOR_UNITS,
    FAULT_CLASSES, FAULT_SHORT, FAULT_URGENCY, URGENCY_LABELS,
    NUM_SENSORS, NUM_CLASSES,
    WINDOW_SIZE, STRIDE,
    CONFIDENCE_THRESHOLD_PCT, SHAP_BACKGROUND_SIZE,
    DEVICE, CHECKPOINT_PATH,
)
from architecture import CNNLSTMFaultDetector, load_model
from preprocessing import ResidualScaler, SCALER_PATH

logger = logging.getLogger("UAV.PHM.Inference")


# ═══════════════════════════════════════════════════════════════════════════
#  FAULT REPORT DATA CLASS
# ═══════════════════════════════════════════════════════════════════════════

class FaultReport:
    """
    Structured container for one inference + SHAP attribution result.

    Designed to be JSON-serialisable for logging to the Digital Twin
    telemetry store and dashboard.

    Attributes:
        timestamp            : UTC ISO-8601 string of inference time.
        predicted_class_id   : Integer fault class (0–8).
        predicted_class_name : Human-readable fault name.
        confidence_pct       : Softmax probability of predicted class × 100.
        low_confidence_flag  : True if confidence < CONFIDENCE_THRESHOLD_PCT.
        urgency_level        : 0–4 maintenance urgency.
        urgency_label        : "NONE" / "WATCH" / "CAUTION" / "WARNING" / "CRITICAL".
        all_probabilities    : Dict {class_name: probability_pct}.
        shap_attributions_pct: Dict {sensor_name: |SHAP|% contribution}.
        shap_signed_mean     : Dict {sensor_name: signed mean SHAP value}.
        top_k_sensors        : List of (sensor_name, pct) sorted by importance.
        inference_latency_ms : Wall-clock time from input to output (ms).
    """

    def __init__(
        self,
        predicted_class_id:    int,
        predicted_class_name:  str,
        confidence_pct:        float,
        all_probabilities:     Dict[str, float],
        shap_attributions_pct: Dict[str, float],
        shap_signed_mean:      Dict[str, float],
        top_k_sensors:         List[Tuple[str, float]],
        inference_latency_ms:  float,
    ) -> None:
        self.timestamp            = datetime.now(timezone.utc).isoformat()
        self.predicted_class_id   = predicted_class_id
        self.predicted_class_name = predicted_class_name
        self.confidence_pct       = round(confidence_pct, 2)
        self.low_confidence_flag  = confidence_pct < CONFIDENCE_THRESHOLD_PCT
        self.urgency_level        = FAULT_URGENCY[predicted_class_id]
        self.urgency_label        = URGENCY_LABELS[self.urgency_level]
        self.all_probabilities    = all_probabilities
        self.shap_attributions_pct= shap_attributions_pct
        self.shap_signed_mean     = shap_signed_mean
        self.top_k_sensors        = top_k_sensors
        self.inference_latency_ms = round(inference_latency_ms, 2)

    def to_dict(self) -> Dict:
        """Serialise to a plain Python dict (JSON-safe)."""
        return {
            "timestamp":             self.timestamp,
            "predicted_class_id":    self.predicted_class_id,
            "predicted_class_name":  self.predicted_class_name,
            "confidence_pct":        self.confidence_pct,
            "low_confidence_flag":   self.low_confidence_flag,
            "urgency_level":         self.urgency_level,
            "urgency_label":         self.urgency_label,
            "all_probabilities":     self.all_probabilities,
            "shap_attributions_pct": self.shap_attributions_pct,
            "shap_signed_mean":      self.shap_signed_mean,
            "top_k_sensors":         self.top_k_sensors,
            "inference_latency_ms":  self.inference_latency_ms,
        }

    def to_json(self, indent: int = 2) -> str:
        """Return a formatted JSON string of this report."""
        return json.dumps(self.to_dict(), indent=indent)


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN INFERENCE ENGINE
# ═══════════════════════════════════════════════════════════════════════════

class ExplainableFaultInference:
    """
    Production inference engine combining:
      (1) CNN-LSTM fault classification
      (2) DeepSHAP sensor attribution

    Thread safety: This class is NOT thread-safe due to PyTorch's internal
    state.  For concurrent GCS deployments, instantiate one engine per thread
    or use a request queue pattern.

    Args:
        model            : Trained CNNLSTMFaultDetector in eval() mode.
        background_data  : (N, WINDOW_SIZE, NUM_SENSORS) array of HEALTHY
                           windows.  DeepSHAP uses these as the reference
                           distribution.  MUST be from the training/validation
                           set only — never from future test data.
        scaler           : Optional ResidualScaler.  If provided, raw
                           (unnormalised) residuals can be passed to
                           predict_and_explain().
        device           : Torch device.
        shap_bg_size     : Number of background samples to use (subset of N).
    """

    def __init__(
        self,
        model:           CNNLSTMFaultDetector,
        background_data: np.ndarray,
        scaler:          Optional[ResidualScaler] = None,
        device:          torch.device = DEVICE,
        shap_bg_size:    int = SHAP_BACKGROUND_SIZE,
    ) -> None:
        self.model   = model.to(device).eval()
        self.device  = device
        self.scaler  = scaler

        # ── Background tensor for DeepSHAP ───────────────────────────────
        # Select a representative subset of healthy windows as the baseline.
        # DeepSHAP computes attributions relative to the *expected* model
        # output over this background distribution.
        n_bg = min(shap_bg_size, len(background_data))
        bg   = background_data[:n_bg].astype(np.float32)   # (N, W, F)

        # Permute to model input shape (N, F, W) — Conv1d channels-first
        bg_tensor = torch.from_numpy(bg).permute(0, 2, 1).to(device)
        self.background = bg_tensor   # (N, F, W)

        # ── Instantiate DeepSHAP explainer ────────────────────────────────
        # shap.DeepExplainer requires a differentiable PyTorch model and a
        # reference tensor. It uses custom backward hooks that require CPU
        # for operators like max_unpool on Apple Silicon.
        logger.info(
            "Initialising DeepSHAP explainer with %d background windows...", n_bg
        )
        t0 = time.perf_counter()
        import copy
        self.shap_model = copy.deepcopy(self.model).to("cpu").eval()
        self.background_cpu = bg_tensor.to("cpu")
        self.explainer = shap.DeepExplainer(
            model=self.shap_model,
            data=self.background_cpu,
        )
        init_ms = (time.perf_counter() - t0) * 1000
        logger.info("DeepSHAP explainer ready (init=%.0f ms).", init_ms)

    # ── Class factory: load from checkpoint ───────────────────────────────
    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path:  str,
        background_data:  np.ndarray,
        scaler_path:      Optional[str] = None,
        device:           torch.device = DEVICE,
        shap_bg_size:     int = SHAP_BACKGROUND_SIZE,
    ) -> "ExplainableFaultInference":
        """
        Convenience constructor that loads a model from a .pt checkpoint
        and optionally loads a ResidualScaler for pre-processing.

        Args:
            checkpoint_path : Path to best_fault_detector.pt.
            background_data : (N, W, F) healthy windows (pre-normalised if
                              no scaler_path provided, raw otherwise).
            scaler_path     : Path to residual_scaler.pkl.  If provided,
                              the scaler is loaded and applied automatically.
            device          : Torch device.
            shap_bg_size    : Background subset size for SHAP.

        Returns:
            ExplainableFaultInference instance ready for inference.
        """
        model  = load_model(checkpoint_path, device=device)
        scaler = ResidualScaler.load(scaler_path) if scaler_path else None
        return cls(
            model=model,
            background_data=background_data,
            scaler=scaler,
            device=device,
            shap_bg_size=shap_bg_size,
        )

    # ── Core inference method ─────────────────────────────────────────────
    def predict_and_explain(
        self,
        window:     np.ndarray,
        top_k:      int  = 5,
        verbose:    bool = True,
        normalised: bool = True,
    ) -> FaultReport:
        """
        Run one residual window through the model and produce an auditable
        fault attribution report with DeepSHAP sensor contributions.

        Algorithm:
            1. Validate and prepare the input window.
            2. Forward pass → logits → softmax probabilities.
            3. DeepSHAP: compute per-input-feature SHAP values.
            4. Aggregate SHAP values over time → per-sensor importance.
            5. Normalise to percentage contributions.
            6. Package into a FaultReport (dict + printable).

        SHAP aggregation detail:
            shap_values[class_id] has shape (1, F, W) — one SHAP value per
            input feature (sensor channel) per timestep.

            We reduce over the temporal axis (W) by taking the mean of the
            absolute SHAP values:
                sensor_importance[f] = mean_t( |SHAP[f, t]| )

            Taking |SHAP| BEFORE averaging ensures that positive and negative
            SHAP contributions from different timesteps don't cancel each other.
            The resulting value correctly reflects HOW MUCH each sensor
            residual moved the prediction, direction-agnostic.

            We also provide the SIGNED mean (can be positive or negative) to
            show whether the sensor's residual was fault-driving or
            counter-evidence.

        Args:
            window     : (WINDOW_SIZE, NUM_SENSORS) normalised residual window.
                         If normalised=False and a scaler is available, the
                         window is normalised automatically.
            top_k      : Number of top sensors to highlight in the report.
            verbose    : If True, print the human-readable report to stdout.
            normalised : Whether the window is already normalised.
                         Set False if passing raw AUKF residuals.

        Returns:
            FaultReport instance with all fields populated.

        Raises:
            ValueError : If window shape does not match (WINDOW_SIZE, NUM_SENSORS).
        """
        # ── Input validation ──────────────────────────────────────────────
        if window.ndim != 2:
            raise ValueError(
                f"window must be 2-D (W, F), got {window.ndim}-D."
            )
        if window.shape != (WINDOW_SIZE, NUM_SENSORS):
            raise ValueError(
                f"Expected window shape ({WINDOW_SIZE}, {NUM_SENSORS}), "
                f"got {window.shape}.  "
                f"window_size={WINDOW_SIZE}, num_sensors={NUM_SENSORS}."
            )

        t0 = time.perf_counter()

        # ── Optional normalisation ────────────────────────────────────────
        if not normalised and self.scaler is not None:
            window = self.scaler.transform(window)
        window = window.astype(np.float32)

        # ── Step 1: Prepare model input tensor ────────────────────────────
        # (W, F) → (1, W, F) → permute → (1, F, W)   [Conv1d channels-first]
        x_tensor = (
            torch.from_numpy(window)
            .unsqueeze(0)           # (1, W, F)
            .permute(0, 2, 1)       # (1, F, W)
            .to(self.device)
        )

        # ── Step 2: Forward pass — class probabilities ────────────────────
        self.model.eval()
        with torch.no_grad():
            logits = self.model(x_tensor)                    # (1, NUM_CLASSES)
        probs = torch.softmax(logits, dim=-1).squeeze(0)     # (NUM_CLASSES,)

        pred_class_id   = int(probs.argmax().item())
        pred_class_name = FAULT_CLASSES[pred_class_id]
        confidence_pct  = float(probs[pred_class_id].item()) * 100.0

        all_probabilities = {
            FAULT_CLASSES[i]: round(float(probs[i].item()) * 100, 2)
            for i in range(NUM_CLASSES)
        }

        # ── Step 3: DeepSHAP attribution ──────────────────────────────────
        # shap_values is a list of length NUM_CLASSES.
        # Each element: np.ndarray of shape (1, F, W) — input-space gradients.
        # We use the element for the predicted class only.
        #
        # IMPORTANT: DeepExplainer.shap_values() temporarily sets the model
        # to train-ish mode internally (hooks).  It restores eval() after.
        # Run on CPU with check_additivity=False for Apple Silicon MPS compatibility.
        shap_values = self.explainer.shap_values(x_tensor.to("cpu"), check_additivity=False)

        # Extract SHAP attribution slice for the predicted class: (F, W)
        if isinstance(shap_values, list):
            shap_for_pred = shap_values[pred_class_id][0]   # (F, W)
        elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 4:
            shap_for_pred = shap_values[0, :, :, pred_class_id]  # (F, W)
        else:
            shap_for_pred = np.array(shap_values)[0]

        # ── Step 4: Aggregate over time → per-sensor importance ───────────
        # Mean absolute SHAP over the temporal axis W
        sensor_abs_importance = np.abs(shap_for_pred).mean(axis=1)   # (F,)

        # Signed mean SHAP (directional effect)
        sensor_signed_mean    = shap_for_pred.mean(axis=1)            # (F,)

        # ── Step 5: Percentage normalisation ─────────────────────────────
        total = sensor_abs_importance.sum()
        if total < 1e-12:
            total = 1.0   # Zero-residual edge case — flat attribution
        sensor_pct = (sensor_abs_importance / total) * 100.0

        shap_attributions_pct: Dict[str, float] = {
            SENSOR_CHANNELS[i]: round(float(sensor_pct[i]), 2)
            for i in range(NUM_SENSORS)
        }
        shap_signed_mean: Dict[str, float] = {
            SENSOR_CHANNELS[i]: round(float(sensor_signed_mean[i]), 6)
            for i in range(NUM_SENSORS)
        }

        # Top-K sensors sorted by absolute contribution (descending)
        sorted_sensors = sorted(
            shap_attributions_pct.items(), key=lambda kv: kv[1], reverse=True
        )
        top_k_sensors = sorted_sensors[:top_k]

        latency_ms = (time.perf_counter() - t0) * 1000

        # ── Step 6: Package result ────────────────────────────────────────
        report = FaultReport(
            predicted_class_id    = pred_class_id,
            predicted_class_name  = pred_class_name,
            confidence_pct        = confidence_pct,
            all_probabilities     = all_probabilities,
            shap_attributions_pct = shap_attributions_pct,
            shap_signed_mean      = shap_signed_mean,
            top_k_sensors         = top_k_sensors,
            inference_latency_ms  = latency_ms,
        )

        if verbose:
            self._print_report(report)

        return report

    # ── Batch inference (without SHAP — for fast screening) ───────────────
    def predict_batch(
        self,
        windows: np.ndarray,
    ) -> np.ndarray:
        """
        Run batch prediction WITHOUT SHAP (fast path for screening).

        Use this when you need to classify many windows quickly and only
        want SHAP explanations for the most suspicious ones.

        Args:
            windows : (N, WINDOW_SIZE, NUM_SENSORS) batch of windows.

        Returns:
            class_ids : (N,) predicted class ids.
        """
        self.model.eval()
        # Permute: (N, W, F) → (N, F, W) for Conv1d
        x = torch.from_numpy(
            windows.astype(np.float32)
        ).permute(0, 2, 1).to(self.device)

        with torch.no_grad():
            logits = self.model(x)     # (N, NUM_CLASSES)
        return logits.argmax(dim=-1).cpu().numpy()

    # ── Human-readable report printer ─────────────────────────────────────
    @staticmethod
    def _print_report(report: FaultReport) -> None:
        """
        Print a structured, GCS-operator-friendly fault audit report to stdout.

        Format:
            ════════════════════════════════════════════════
              UAV ENGINE FAULT DETECTION — AUDIT REPORT
            ════════════════════════════════════════════════
              Timestamp        : 2025-09-09T13:45:01.123Z
              Predicted Fault  : Misfire conditions
              Confidence       : 87.3%
              Urgency          : ⚠ WARNING
              Low Confidence   : No

              Class Probabilities:
                Normal operation          ░░░░░░░░░░░░░░░░░░░░   2.1%
                Misfire conditions        ████████████████░░░░  87.3%
                ...

              Sensor Residual Attributions (DeepSHAP):
              (% = contribution to this prediction; ▲=fault-driving, ▼=suppressing)
                egt_residual              ████████░░░░░░░░░░░░  38.4%  ▲ fault-driving
                rpm_residual              █████░░░░░░░░░░░░░░░  24.1%  ▲ fault-driving
                vibration_residual        ████░░░░░░░░░░░░░░░░  18.2%  ▲ fault-driving
                ...

              Top 3 Causal Sensors:
                1. egt_residual (38.4%)
                2. rpm_residual (24.1%)
                3. vibration_residual (18.2%)
            ════════════════════════════════════════════════
        """
        SEP = "═" * 62
        # Urgency emoji map
        urgency_emoji = {0: "✓", 1: "👁", 2: "⚡", 3: "⚠", 4: "🚨"}

        print(f"\n{SEP}")
        print("  UAV ENGINE FAULT DETECTION — AUDIT REPORT")
        print(SEP)
        print(f"  Timestamp        : {report.timestamp}")
        print(f"  Predicted Fault  : {report.predicted_class_name}")
        print(f"  Confidence       : {report.confidence_pct:.1f}%")
        u_emoji = urgency_emoji.get(report.urgency_level, "?")
        print(f"  Urgency          : {u_emoji} {report.urgency_label}")
        if report.low_confidence_flag:
            print("  ⚠ LOW CONFIDENCE — prediction uncertain, review sensor data.")
        print()

        # ── Class probability bars ────────────────────────────────────────
        print("  Class Probabilities:")
        for cls_name, pct in report.all_probabilities.items():
            filled = int(pct / 5)
            bar    = "█" * filled + "░" * (20 - filled)
            marker = " ◄" if cls_name == report.predicted_class_name else ""
            print(f"    {cls_name:<36s} {bar}  {pct:5.1f}%{marker}")
        print()

        # ── SHAP attribution bars ─────────────────────────────────────────
        print("  Sensor Residual Attributions (DeepSHAP):")
        print("  (% = contribution to this prediction;")
        print("   ▲ = fault-driving residual, ▼ = counter-evidence)")
        sorted_attrs = sorted(
            report.shap_attributions_pct.items(),
            key=lambda kv: kv[1], reverse=True,
        )
        for sensor, pct in sorted_attrs:
            signed     = report.shap_signed_mean[sensor]
            direction  = "▲ fault-driving  " if signed > 0 else "▼ counter-evidence"
            units      = SENSOR_UNITS.get(sensor, "")
            filled     = int(pct / 5)
            bar        = "█" * filled + "░" * (20 - filled)
            print(
                f"    {sensor:<30s} {bar}  {pct:5.1f}%  [{direction}]  [{units}]"
            )
        print()

        # ── Top-K causal sensors ──────────────────────────────────────────
        print(f"  Top {len(report.top_k_sensors)} Causal Sensors:")
        for rank, (sensor, pct) in enumerate(report.top_k_sensors, 1):
            print(f"    {rank}. {sensor}  ({pct:.1f}%)")
        print()
        print(f"  Inference latency: {report.inference_latency_ms:.0f} ms")
        print(SEP)


# ═══════════════════════════════════════════════════════════════════════════
#  STREAMING FAULT MONITOR
# ═══════════════════════════════════════════════════════════════════════════

class StreamingFaultMonitor:
    """
    Online / real-time fault monitor that wraps ExplainableFaultInference.

    Maintains a circular buffer of incoming residual timesteps.  Each time
    the buffer accumulates enough samples for a new stride-aligned window,
    it automatically triggers inference.

    Intended for integration with the AUKF output stream in the GCS pipeline:

        monitor = StreamingFaultMonitor(engine, alert_urgency_threshold=3)
        for residual_vector in aukf_output:     # shape (NUM_SENSORS,)
            alert = monitor.push(residual_vector)
            if alert:
                dashboard.send_alert(alert)

    Args:
        engine               : Initialised ExplainableFaultInference.
        window_size          : Timesteps per inference window.
        stride               : Timesteps between successive inferences.
        alert_urgency_threshold : Minimum urgency level to return an alert
                               (0=always, 3=WARNING+, 4=CRITICAL only).
        shap_on_alert        : If True, run DeepSHAP only when urgency ≥ threshold.
                               If False, always run SHAP (higher latency).
    """

    def __init__(
        self,
        engine:                   ExplainableFaultInference,
        window_size:              int = WINDOW_SIZE,
        stride:                   int = STRIDE,
        alert_urgency_threshold:  int = 2,
        shap_on_alert:            bool = True,
    ) -> None:
        self.engine    = engine
        self.W         = window_size
        self.stride    = stride
        self.threshold = alert_urgency_threshold
        self.shap_on_alert = shap_on_alert

        # Circular buffer: accumulates incoming residual vectors
        # Each element is shape (NUM_SENSORS,)
        self._buffer: Deque[np.ndarray] = deque(maxlen=window_size)
        self._steps_since_inference: int = 0

    def push(
        self,
        residual_vector: np.ndarray,
    ) -> Optional[FaultReport]:
        """
        Push one timestep of residual data into the monitor.

        Args:
            residual_vector : (NUM_SENSORS,) float32 residuals at current timestep.

        Returns:
            FaultReport if an inference was triggered and urgency ≥ threshold.
            None if the buffer is not full, or urgency below threshold.
        """
        if residual_vector.shape != (NUM_SENSORS,):
            raise ValueError(
                f"Expected residual_vector shape ({NUM_SENSORS},), "
                f"got {residual_vector.shape}."
            )

        self._buffer.append(residual_vector.astype(np.float32))
        self._steps_since_inference += 1

        # Only infer when buffer is full AND stride elapsed
        if (
            len(self._buffer) < self.W
            or self._steps_since_inference < self.stride
        ):
            return None

        self._steps_since_inference = 0

        # Build window: (W, F)
        window = np.stack(list(self._buffer), axis=0)   # (W, F)

        # Fast batch predict first (no SHAP) to check urgency
        pred_class = int(
            self.engine.predict_batch(window[np.newaxis])[0]
        )
        urgency = FAULT_URGENCY[pred_class]

        if urgency < self.threshold:
            return None    # Below alert threshold — no action needed

        # Urgency threshold met — run full SHAP explanation
        if self.shap_on_alert:
            report = self.engine.predict_and_explain(
                window, verbose=True
            )
        else:
            report = self.engine.predict_and_explain(
                window, verbose=False
            )

        return report


# ═══════════════════════════════════════════════════════════════════════════
#  DEMO / SELF-TEST
# ═══════════════════════════════════════════════════════════════════════════

def run_inference_demo(
    checkpoint_path: str = CHECKPOINT_PATH,
    scaler_path:     str = SCALER_PATH,
) -> FaultReport:
    """
    Load a saved model checkpoint and run explainable inference on a
    synthetic test window.  Useful for validating deployment.

    This function:
      1. Loads the trained model from checkpoint.
      2. Loads the fitted scaler.
      3. Generates a synthetic overheating fault window.
      4. Runs full predict_and_explain().
      5. Prints the audit report and returns the FaultReport.

    Args:
        checkpoint_path : Path to best_fault_detector.pt.
        scaler_path     : Path to residual_scaler.pkl.

    Returns:
        FaultReport for the synthetic test window.
    """
    import os
    from preprocessing import generate_synthetic_residuals, create_sliding_windows

    logger.info("=== Inference Demo: MALE UAV Fault Detection ===")

    # ── Load model ────────────────────────────────────────────────────────
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found at '{checkpoint_path}'.  "
            "Run training.py first."
        )
    model = load_model(checkpoint_path, device=DEVICE)

    # ── Load or create scaler ─────────────────────────────────────────────
    scaler: Optional[ResidualScaler] = None
    if os.path.exists(scaler_path):
        scaler = ResidualScaler.load(scaler_path)

    # ── Generate synthetic residuals for background and test ──────────────
    raw_residuals, labels = generate_synthetic_residuals(n_timesteps=50_000, seed=1)

    # Normalise (use scaler if available, else no-op)
    if scaler is not None:
        residuals = scaler.transform(raw_residuals)
    else:
        # Fallback: simple z-score normalisation on the full dataset
        mu = raw_residuals.mean(axis=0)
        sg = raw_residuals.std(axis=0) + 1e-8
        residuals = ((raw_residuals - mu) / sg).astype(np.float32)

    # Sliding windows
    X, y = create_sliding_windows(residuals, labels)

    # Background: first 100 healthy windows
    healthy_mask = (y == 0)
    background   = X[healthy_mask][:SHAP_BACKGROUND_SIZE]

    # ── Initialise inference engine ────────────────────────────────────────
    engine = ExplainableFaultInference(
        model=model,
        background_data=background,
        scaler=scaler,
        device=DEVICE,
    )

    # ── Test window: pick an overheating fault window (class 7) ───────────
    overheat_windows = X[y == 7]
    if len(overheat_windows) == 0:
        logger.warning("No overheating windows found — using first window.")
        test_window = X[0]
    else:
        test_window = overheat_windows[0]   # (W, F)

    # ── Run inference + SHAP ──────────────────────────────────────────────
    report = engine.predict_and_explain(test_window, top_k=5, verbose=True)

    # ── Also demonstrate JSON output (for dashboard / telemetry store) ────
    print("\n--- JSON Output (for dashboard / telemetry store) ---")
    print(report.to_json())

    return report


# ── Streaming demo ─────────────────────────────────────────────────────────
def run_streaming_demo() -> None:
    """
    Demonstrate StreamingFaultMonitor with a synthetic residual feed.

    Simulates a ~600-step mission segment containing a lubrication fault.
    """
    import os
    from preprocessing import generate_synthetic_residuals

    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(
            f"Checkpoint not found at '{CHECKPOINT_PATH}'.  Run training.py first."
        )

    model = load_model(CHECKPOINT_PATH)
    raw, labels = generate_synthetic_residuals(n_timesteps=50_000, seed=2)

    # Normalise
    mu = raw.mean(axis=0); sg = raw.std(axis=0) + 1e-8
    residuals = ((raw - mu) / sg).astype(np.float32)

    from preprocessing import create_sliding_windows
    X, y = create_sliding_windows(residuals, labels)
    bg   = X[y == 0][:SHAP_BACKGROUND_SIZE]

    engine  = ExplainableFaultInference(model=model, background_data=bg)
    monitor = StreamingFaultMonitor(engine, alert_urgency_threshold=2)

    logger.info("Streaming demo: feeding 600 timesteps...")
    n_alerts = 0
    # Feed timesteps from the lubrication fault region (t=25000..25600)
    for t in range(25_000, 25_600):
        alert = monitor.push(residuals[t])
        if alert:
            n_alerts += 1
            logger.info(
                "[t=%d] ALERT: %s (urgency=%s)",
                t, alert.predicted_class_name, alert.urgency_label,
            )
    logger.info("Streaming demo complete.  %d alerts raised.", n_alerts)


# ── Entrypoint ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    mode = sys.argv[1] if len(sys.argv) > 1 else "single"
    if mode == "stream":
        run_streaming_demo()
    else:
        run_inference_demo()
