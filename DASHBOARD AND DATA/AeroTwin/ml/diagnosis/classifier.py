"""
AeroTwin-4 Fault Diagnosis Classifier.

Anomaly detection answers "is something wrong?". Diagnosis answers "what is wrong?"
- which of the four modelled subsystems is responsible, and how confident is that
call. Both are needed: an anomaly flag with no attribution gives a UAV operator
nothing to act on.

The classifier is deliberately a calibrated Random Forest rather than a deep model:
the dataset is a few hundred windows wide, the features are physically meaningful
residuals, and per-class probability plus feature attribution matter more here than
raw capacity.
"""

import json
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from .splits import FAULT_CLASSES


class FaultDiagnosisClassifier:
    """
    Multi-class fault-family classifier over Digital-Twin residual features.
    """

    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: Optional[int] = None,
        min_samples_leaf: int = 2,
        random_state: int = 42,
        class_weight: str = "balanced_subsample",
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.class_weight = class_weight

        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            class_weight=class_weight,
            n_jobs=-1,
        )
        self.feature_names: List[str] = []
        self.classes_: List[str] = []
        self.is_fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "FaultDiagnosisClassifier":
        self.feature_names = list(X.columns)
        self.model.fit(X.values, y.values)
        self.classes_ = list(self.model.classes_)
        self.is_fitted = True
        return self

    def _align(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("FaultDiagnosisClassifier must be fitted before use.")
        missing = [c for c in self.feature_names if c not in X.columns]
        if missing:
            raise ValueError(f"Missing {len(missing)} feature(s) at predict time, e.g. {missing[:5]}")
        return X[self.feature_names].values

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(self._align(X))

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        """Per-class probabilities, columns ordered by self.classes_."""
        proba = self.model.predict_proba(self._align(X))
        return pd.DataFrame(proba, columns=self.classes_)

    def diagnose(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Operator-facing output: predicted family, its confidence, and the runner-up.

        The margin between top-1 and top-2 is what tells a crew whether the call is
        firm or a coin-flip between two subsystems - a bare argmax hides that.
        """
        proba = self.predict_proba(X)
        values = proba.values
        order = np.argsort(-values, axis=1)

        top1 = order[:, 0]
        top2 = order[:, 1] if values.shape[1] > 1 else top1

        cls = np.array(self.classes_)
        return pd.DataFrame(
            {
                "predicted_fault": cls[top1],
                "confidence": values[np.arange(len(values)), top1],
                "runner_up_fault": cls[top2],
                "runner_up_confidence": values[np.arange(len(values)), top2],
                "margin": values[np.arange(len(values)), top1] - values[np.arange(len(values)), top2],
            }
        )

    def feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """Which residual channels drove the diagnosis - the explainability the PS asks for."""
        if not self.is_fitted:
            raise RuntimeError("FaultDiagnosisClassifier must be fitted before use.")
        imp = pd.DataFrame(
            {"feature": self.feature_names, "importance": self.model.feature_importances_}
        )
        return imp.sort_values("importance", ascending=False).head(top_n).reset_index(drop=True)

    def save(self, dirpath: str):
        import joblib

        os.makedirs(dirpath, exist_ok=True)
        joblib.dump(self.model, os.path.join(dirpath, "model.joblib"))
        with open(os.path.join(dirpath, "config.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "n_estimators": self.n_estimators,
                    "max_depth": self.max_depth,
                    "min_samples_leaf": self.min_samples_leaf,
                    "random_state": self.random_state,
                    "class_weight": self.class_weight,
                    "feature_names": self.feature_names,
                    "classes": self.classes_,
                },
                f,
                indent=2,
            )

    def load(self, dirpath: str) -> "FaultDiagnosisClassifier":
        import joblib

        self.model = joblib.load(os.path.join(dirpath, "model.joblib"))
        with open(os.path.join(dirpath, "config.json"), encoding="utf-8") as f:
            cfg = json.load(f)
        self.feature_names = cfg["feature_names"]
        self.classes_ = cfg["classes"]
        self.is_fitted = True
        return self
