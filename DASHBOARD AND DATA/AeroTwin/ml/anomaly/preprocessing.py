"""
AeroTwin-4 Feature Scaler.

Scales feature matrices using StandardScaler fitted STRICTLY on healthy training data.
"""

from typing import List, Optional
import os
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class FeatureScaler:
    """
    StandardScaler wrapper ensuring fit ONLY occurs on Healthy Training features.
    Includes min_std floor and clip_limit to prevent numerical explosion.
    """

    def __init__(self, min_std: float = 1e-2, clip_limit: float = 20.0):
        self.scaler = StandardScaler()
        self.min_std = min_std
        self.clip_limit = clip_limit
        self.feature_names: List[str] = []
        self.is_fitted: bool = False

    def fit(self, X_train: pd.DataFrame) -> "FeatureScaler":
        """
        Fit scaler parameters strictly on Healthy Training features.
        """
        self.feature_names = list(X_train.columns)
        self.scaler.fit(X_train[self.feature_names])
        # Enforce minimum standard deviation floor to prevent division by near-zero healthy residual std
        self.scaler.scale_ = np.maximum(self.scaler.scale_, self.min_std)
        self.is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transform feature matrix using healthy training fitted parameters and clip outliers.
        """
        if not self.is_fitted:
            raise RuntimeError("Scaler is not fitted yet! Call fit(X_train) first.")
        
        # Ensure column alignment
        X_aligned = X[self.feature_names] if isinstance(X, pd.DataFrame) else X
        scaled = self.scaler.transform(X_aligned)
        # Clip scaled feature values to [-clip_limit, +clip_limit]
        scaled_clipped = np.clip(scaled, -self.clip_limit, self.clip_limit)
        return scaled_clipped

    def fit_transform(self, X_train: pd.DataFrame) -> np.ndarray:
        """
        Fit on X_train and return transformed numpy array.
        """
        self.fit(X_train)
        return self.transform(X_train)

    def save(self, filepath: str):
        """
        Save scaler parameters and feature names to JSON file.
        """
        if not self.is_fitted:
            raise RuntimeError("Cannot save unfitted scaler.")
        
        dirpath = os.path.dirname(filepath)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)

        data = {
            "mean": self.scaler.mean_.tolist(),
            "scale": self.scaler.scale_.tolist(),
            "var": self.scaler.var_.tolist(),
            "min_std": self.min_std,
            "clip_limit": self.clip_limit,
            "feature_names": self.feature_names,
        }
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    def load(self, filepath: str) -> "FeatureScaler":
        """
        Load scaler parameters from JSON file.
        """
        with open(filepath, "r") as f:
            data = json.load(f)

        self.feature_names = data["feature_names"]
        self.min_std = float(data.get("min_std", 1e-2))
        self.clip_limit = float(data.get("clip_limit", 20.0))
        self.scaler.mean_ = np.array(data["mean"])
        self.scaler.scale_ = np.maximum(np.array(data["scale"]), self.min_std)
        self.scaler.var_ = np.array(data["var"])
        self.scaler.feature_names_in_ = np.array(data["feature_names"], dtype=object)
        self.scaler.n_features_in_ = len(data["feature_names"])
        self.is_fitted = True
        return self
