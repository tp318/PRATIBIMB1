"""
AeroTwin-4 Model 1: Robust Statistical Baseline Anomaly Detector.

Calculates standardized feature distance from healthy training feature distribution.
"""

from typing import Optional, Dict, Any, Tuple
import os
import json
import numpy as np
import pandas as pd


class StatisticalAnomalyDetector:
    """
    Standardized Euclidean/Z-score feature distance baseline model.
    """

    def __init__(self, eps: float = 1e-6, min_std: float = 1e-2, z_clip: float = 20.0):
        self.eps = eps
        # Variance floor. Features arrive already standardised, so a channel whose
        # training std collapses to ~0 is a CONSTANT feature, not an informative
        # one. Dividing by eps=1e-6 there turns any deviation into a Z-score of
        # ~1e6 and one dead feature then dominates the RMS across all the others -
        # the same failure mode that drove the Phase 5 autoencoder loss to 1e33.
        self.min_std = min_std
        # Per-feature Z clip, so a single outlying channel cannot swamp the score.
        self.z_clip = z_clip
        self.mean_vector: Optional[np.ndarray] = None
        self.std_vector: Optional[np.ndarray] = None
        self.feature_names: Optional[list] = None
        self.threshold: float = 3.0  # default initial Z-score threshold
        self.is_fitted: bool = False

    def fit(self, X_train: pd.DataFrame) -> "StatisticalAnomalyDetector":
        """
        Fit mean and standard deviation vectors on healthy training features.
        """
        if isinstance(X_train, pd.DataFrame):
            self.feature_names = list(X_train.columns)
            X_mat = X_train.values
        else:
            X_mat = np.array(X_train)

        self.mean_vector = np.mean(X_mat, axis=0)
        self.std_vector = np.maximum(np.std(X_mat, axis=0), self.min_std) + self.eps
        self.is_fitted = True
        return self

    def compute_anomaly_score(self, X: pd.DataFrame) -> np.ndarray:
        """
        Compute normalized Z-score RMS distance per feature vector.
        Higher score = more anomalous.
        """
        if not self.is_fitted:
            raise RuntimeError("StatisticalAnomalyDetector is not fitted!")

        X_mat = X[self.feature_names].values if isinstance(X, pd.DataFrame) else np.array(X)
        z_scores = (X_mat - self.mean_vector) / self.std_vector
        z_scores = np.clip(z_scores, -self.z_clip, self.z_clip)
        # RMS Z-score across features
        scores = np.sqrt(np.mean(z_scores ** 2, axis=1))
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
        Save model parameters to JSON.
        """
        dirpath = os.path.dirname(filepath)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
            
        data = {
            "mean": self.mean_vector.tolist(),
            "std": self.std_vector.tolist(),
            "feature_names": self.feature_names,
            "threshold": self.threshold,
            "min_std": self.min_std,
            "z_clip": self.z_clip,
        }
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    def load(self, filepath: str) -> "StatisticalAnomalyDetector":
        """
        Load model parameters from JSON.
        """
        with open(filepath, "r") as f:
            data = json.load(f)

        self.mean_vector = np.array(data["mean"])
        self.std_vector = np.array(data["std"])
        self.feature_names = data["feature_names"]
        self.threshold = float(data["threshold"])
        self.min_std = float(data.get("min_std", 1e-2))
        self.z_clip = float(data.get("z_clip", 20.0))
        self.is_fitted = True
        return self
