"""
AeroTwin-4 Unified Anomaly Scorer Interface.

Provides standardized inference wrapper for streaming telemetry & window predictions.
OUTPUT CONTRACT: (run_id, window_id, anomaly_score, anomaly_flag).
Phase 5 outputs ONLY NORMAL vs ANOMALOUS.
"""

from typing import Dict, Any, Tuple, Union, Optional
import numpy as np
import pandas as pd


class AnomalyScorer:
    """
    Standardized inference wrapper for Phase 5 Anomaly Detectors.
    """

    def __init__(self, detector: Any, scaler: Any = None):
        self.detector = detector
        self.scaler = scaler

    def predict_window(self, X_window: pd.DataFrame) -> Tuple[float, bool]:
        """
        Predict anomaly score and flag for a single window.
        """
        X_proc = self.scaler.transform(X_window) if self.scaler is not None else X_window
        if isinstance(X_proc, np.ndarray):
            X_proc = pd.DataFrame(X_proc, columns=self.scaler.feature_names if self.scaler else X_window.columns)

        scores, flags = self.detector.predict(X_proc)
        return float(scores[0]), bool(flags[0])

    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict anomaly scores and flags for feature matrix X.
        """
        X_proc = self.scaler.transform(X) if self.scaler is not None else X
        if isinstance(X_proc, np.ndarray):
            X_proc = pd.DataFrame(X_proc, columns=self.scaler.feature_names if self.scaler else X.columns)

        scores, flags = self.detector.predict(X_proc)
        return scores, flags
