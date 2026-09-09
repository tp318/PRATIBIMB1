"""
AeroTwin-4 Health Index Estimator.

Stage 1 of the two-stage RUL pipeline.

Maps a window of Digital-Twin residual features to the engine health index
H in [0, 1], where H = 1 - S(t) and S is injected degradation severity. Health is
regressed rather than classified because RUL needs a continuous trend to
extrapolate, not a discrete fault label.
"""

import json
import os
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor


class HealthEstimator:
    """Regresses the continuous health index from residual window features."""

    def __init__(
        self,
        n_estimators: int = 300,
        learning_rate: float = 0.05,
        max_depth: int = 3,
        subsample: float = 0.9,
        random_state: int = 42,
    ):
        self.random_state = random_state
        self.model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=subsample,
            random_state=random_state,
        )
        self.feature_names: List[str] = []
        self.is_fitted = False
        # 1-sigma spread of training residuals: the estimator's own noise floor,
        # propagated into the RUL confidence interval downstream.
        self.residual_std: float = 0.0

    def fit(self, X: pd.DataFrame, health: pd.Series) -> "HealthEstimator":
        self.feature_names = list(X.columns)
        y = np.clip(np.asarray(health, dtype=float), 0.0, 1.0)
        self.model.fit(X.values, y)
        self.residual_std = float(np.std(y - self.model.predict(X.values)))
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("HealthEstimator must be fitted before use.")
        missing = [c for c in self.feature_names if c not in X.columns]
        if missing:
            raise ValueError(f"Missing {len(missing)} feature(s), e.g. {missing[:5]}")
        # Health is a physical index: clip rather than emit values outside [0, 1].
        return np.clip(self.model.predict(X[self.feature_names].values), 0.0, 1.0)

    def save(self, dirpath: str):
        import joblib

        os.makedirs(dirpath, exist_ok=True)
        joblib.dump(self.model, os.path.join(dirpath, "health_model.joblib"))
        with open(os.path.join(dirpath, "health_config.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "feature_names": self.feature_names,
                    "residual_std": self.residual_std,
                    "random_state": self.random_state,
                },
                f,
                indent=2,
            )

    def load(self, dirpath: str) -> "HealthEstimator":
        import joblib

        self.model = joblib.load(os.path.join(dirpath, "health_model.joblib"))
        with open(os.path.join(dirpath, "health_config.json"), encoding="utf-8") as f:
            cfg = json.load(f)
        self.feature_names = cfg["feature_names"]
        self.residual_std = cfg["residual_std"]
        self.is_fitted = True
        return self
