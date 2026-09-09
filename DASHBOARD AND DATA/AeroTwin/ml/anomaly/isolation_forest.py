"""
AeroTwin-4 Model 2: Isolation Forest Anomaly Detector.

Uses Scikit-Learn IsolationForest trained strictly on healthy feature vectors.
Normalizes score direction so HIGHER score = MORE ANOMALOUS.
"""

from typing import Optional, Dict, Any, Tuple
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


class IsolationForestAnomalyDetector:
    """
    Unsupervised Isolation Forest Anomaly Detector.
    """

    def __init__(self, n_estimators: int = 100, contamination: float = 0.05, random_state: int = 42):
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.random_state = random_state
        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.random_state,
        )
        self.feature_names: Optional[list] = None
        self.threshold: float = 0.5
        self.is_fitted: bool = False

    def fit(self, X_train: pd.DataFrame) -> "IsolationForestAnomalyDetector":
        """
        Fit IsolationForest strictly on healthy training feature vectors.
        """
        if isinstance(X_train, pd.DataFrame):
            self.feature_names = list(X_train.columns)
            X_mat = X_train.values
        else:
            X_mat = np.array(X_train)

        self.model.fit(X_mat)
        self.is_fitted = True
        return self

    def compute_anomaly_score(self, X: pd.DataFrame) -> np.ndarray:
        """
        Compute normalized anomaly score.
        IsolationForest.score_samples returns negative anomaly score (lower is more anomalous).
        We negate score_samples so HIGHER = MORE ANOMALOUS.
        """
        if not self.is_fitted:
            raise RuntimeError("IsolationForestAnomalyDetector is not fitted!")

        X_mat = X[self.feature_names].values if isinstance(X, pd.DataFrame) else np.array(X)
        # Negate score_samples so higher score = more anomalous
        scores = -self.model.score_samples(X_mat)
        return scores

    def fit_threshold(self, X_val_healthy: pd.DataFrame, target_fpr: float = 0.05) -> float:
        """
        Derive decision threshold on healthy validation data to target maximum FPR.
        """
        val_scores = self.compute_anomaly_score(X_val_healthy)
        percentile = (1.0 - target_fpr) * 100.0
        self.threshold = float(np.percentile(val_scores, percentile))
        return self.threshold

    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict (anomaly_score, anomaly_flag) for feature matrix X.
        """
        scores = self.compute_anomaly_score(X)
        flags = scores > self.threshold
        return scores, flags

    def save(self, filepath: str):
        """
        Save model artifact using joblib.
        """
        dirpath = os.path.dirname(filepath)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
            
        data = {
            "model": self.model,
            "feature_names": self.feature_names,
            "threshold": self.threshold,
            "is_fitted": self.is_fitted,
        }
        joblib.dump(data, filepath)

    def load(self, filepath: str) -> "IsolationForestAnomalyDetector":
        """
        Load model artifact using joblib.
        """
        data = joblib.load(filepath)
        self.model = data["model"]
        self.feature_names = data["feature_names"]
        self.threshold = data["threshold"]
        self.is_fitted = data["is_fitted"]
        return self
