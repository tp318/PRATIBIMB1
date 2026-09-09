"""
AeroTwin-4 Fault Diagnosis Evaluation.

Reports the confusion matrix and per-class precision / recall / F1 the problem
statement calls for. Accuracy alone is not reported as a headline: with an
imbalanced fault mix it hides exactly the failure that matters, namely one fault
family being systematically mistaken for another.
"""

from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from .splits import FAULT_CLASSES


class DiagnosisEvaluator:
    """Scores multi-class fault diagnosis predictions."""

    def __init__(self, classes: List[str] = None):
        self.classes = list(classes or FAULT_CLASSES)

    def confusion(self, y_true, y_pred) -> pd.DataFrame:
        cm = confusion_matrix(y_true, y_pred, labels=self.classes)
        return pd.DataFrame(
            cm,
            index=[f"true_{c}" for c in self.classes],
            columns=[f"pred_{c}" for c in self.classes],
        )

    def per_class(self, y_true, y_pred) -> pd.DataFrame:
        p = precision_score(y_true, y_pred, labels=self.classes, average=None, zero_division=0)
        r = recall_score(y_true, y_pred, labels=self.classes, average=None, zero_division=0)
        f = f1_score(y_true, y_pred, labels=self.classes, average=None, zero_division=0)
        support = [int((np.asarray(y_true) == c).sum()) for c in self.classes]
        return pd.DataFrame(
            {
                "fault_class": self.classes,
                "precision": p,
                "recall": r,
                "f1": f,
                "support": support,
            }
        )

    def summary(self, y_true, y_pred) -> Dict[str, float]:
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            # Balanced accuracy is the honest headline when class support is uneven.
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "macro_precision": float(
                precision_score(y_true, y_pred, labels=self.classes, average="macro", zero_division=0)
            ),
            "macro_recall": float(
                recall_score(y_true, y_pred, labels=self.classes, average="macro", zero_division=0)
            ),
            "macro_f1": float(
                f1_score(y_true, y_pred, labels=self.classes, average="macro", zero_division=0)
            ),
            "n_samples": int(len(y_true)),
        }

    def text_report(self, y_true, y_pred) -> str:
        return classification_report(
            y_true, y_pred, labels=self.classes, zero_division=0, digits=4
        )

    def evaluate(self, y_true, y_pred) -> Dict:
        """Full evaluation bundle: summary metrics, per-class table, confusion matrix."""
        return {
            "summary": self.summary(y_true, y_pred),
            "per_class": self.per_class(y_true, y_pred),
            "confusion_matrix": self.confusion(y_true, y_pred),
        }
