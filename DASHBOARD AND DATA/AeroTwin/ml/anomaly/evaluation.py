"""
AeroTwin-4 Evaluation Metrics Engine.

Calculates scientific evaluation metrics for Anomaly Detectors on held-out test runs.
Ground-truth labels are used ONLY AFTER model inference.
"""

from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    precision_recall_curve,
    auc,
    confusion_matrix,
)


class Evaluator:
    """
    Evaluator for Phase 5 Anomaly Detection models.
    """

    @staticmethod
    def evaluate_predictions(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_scores: np.ndarray,
        meta_df: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """
        Compute window-level classification and ranking metrics.
        """
        y_true = np.array(y_true, dtype=bool)
        y_pred = np.array(y_pred, dtype=bool)
        y_scores = np.array(y_scores, dtype=float)

        prec = float(precision_score(y_true, y_pred, zero_division=0))
        rec = float(recall_score(y_true, y_pred, zero_division=0))
        f1 = float(f1_score(y_true, y_pred, zero_division=0))

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[False, True]).ravel()
        fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
        tpr = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0

        try:
            roc_auc = float(roc_auc_score(y_true, y_scores))
        except Exception:
            roc_auc = 0.5

        try:
            p_curve, r_curve, _ = precision_recall_curve(y_true, y_scores)
            pr_auc = float(auc(r_curve, p_curve))
        except Exception:
            pr_auc = 0.5

        results = {
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "fpr": fpr,
            "tpr": tpr,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "tp": int(tp),
            "fp": int(fp),
            "tn": int(tn),
            "fn": int(fn),
        }

        # Breakdown by degradation family if metadata available
        if meta_df is not None and "gt_degradation_type" in meta_df.columns:
            family_results = {}
            for fam in ["CYLINDER", "BEARING", "COOLING", "LUBRICATION"]:
                fam_mask = meta_df["gt_degradation_type"].astype(str).str.upper() == fam
                if fam_mask.sum() > 0:
                    fam_true = y_true[fam_mask]
                    fam_pred = y_pred[fam_mask]
                    fam_rec = float(recall_score(fam_true, fam_pred, zero_division=0))
                    family_results[fam] = {
                        "count": int(fam_mask.sum()),
                        "detection_rate": fam_rec,
                    }
            results["by_family"] = family_results

        return results
