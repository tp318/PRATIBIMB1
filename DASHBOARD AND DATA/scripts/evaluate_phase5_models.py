"""
AeroTwin-4 Phase 5 Ablation Study & Held-Out Test Evaluation CLI.

Evaluates 3 Models x 3 Feature Configurations against held-out Test runs.
Outputs scientific ablation comparison matrix JSON and predictions CSV.
"""

import os
import sys
import json
import numpy as np
import pandas as pd

_script_dir = os.path.dirname(os.path.abspath(__file__))
_root_dir = os.path.dirname(_script_dir)

if _root_dir not in sys.path:
    sys.path.insert(0, _root_dir)

from AeroTwin.ml.anomaly.preprocessing import FeatureScaler
from AeroTwin.ml.anomaly.statistical import StatisticalAnomalyDetector
from AeroTwin.ml.anomaly.isolation_forest import IsolationForestAnomalyDetector
from AeroTwin.ml.anomaly.autoencoder import AutoencoderAnomalyDetector
from AeroTwin.ml.anomaly.scoring import AnomalyScorer
from AeroTwin.ml.anomaly.evaluation import Evaluator


def main():
    print("=" * 60)
    print("AeroTwin-4 Phase 5 Scientific Ablation Evaluator")
    print("=" * 60)

    data_dir = os.path.join(_root_dir, "data", "generated", "phase5", "features")
    models_dir = os.path.join(_root_dir, "models", "phase5")
    out_dir = os.path.join(_root_dir, "data", "generated", "phase5")
    os.makedirs(os.path.join(out_dir, "evaluation"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "predictions"), exist_ok=True)

    configs = ["raw", "residual", "hybrid"]
    ablation_results = {}

    for cfg in configs:
        print(f"\n==================== Evaluating Config: {cfg.upper()} ====================")
        cfg_path = os.path.join(data_dir, cfg)

        X_test_raw = pd.read_csv(os.path.join(cfg_path, "test", "X_test.csv"))
        meta_test = pd.read_csv(os.path.join(cfg_path, "test", "meta_test.csv"))
        y_test_true = meta_test["gt_is_degraded"].values.astype(bool)

        scaler = FeatureScaler().load(os.path.join(models_dir, "preprocessing", cfg, "scaler.json"))
        X_test_scaled = pd.DataFrame(scaler.transform(X_test_raw), columns=scaler.feature_names)

        # ----------------------------------------------------
        # 1. Model 1: Statistical Baseline
        # ----------------------------------------------------
        stat_model = StatisticalAnomalyDetector().load(os.path.join(models_dir, "statistical", cfg, "model.json"))
        stat_scorer = AnomalyScorer(detector=stat_model)
        scores_stat, flags_stat = stat_scorer.predict(X_test_scaled)
        metrics_stat = Evaluator.evaluate_predictions(y_test_true, flags_stat, scores_stat, meta_df=meta_test)
        print(f"  [Statistical Baseline] F1: {metrics_stat['f1_score']:.4f}, FPR: {metrics_stat['fpr']:.4f}, ROC-AUC: {metrics_stat['roc_auc']:.4f}")

        # ----------------------------------------------------
        # 2. Model 2: Isolation Forest
        # ----------------------------------------------------
        if_model = IsolationForestAnomalyDetector().load(os.path.join(models_dir, "isolation_forest", cfg, "model.joblib"))
        if_scorer = AnomalyScorer(detector=if_model)
        scores_if, flags_if = if_scorer.predict(X_test_scaled)
        metrics_if = Evaluator.evaluate_predictions(y_test_true, flags_if, scores_if, meta_df=meta_test)
        print(f"  [Isolation Forest]   F1: {metrics_if['f1_score']:.4f}, FPR: {metrics_if['fpr']:.4f}, ROC-AUC: {metrics_if['roc_auc']:.4f}")

        # ----------------------------------------------------
        # 3. Model 3: PyTorch Autoencoder
        # ----------------------------------------------------
        ae_model = AutoencoderAnomalyDetector().load(os.path.join(models_dir, "autoencoder", cfg))
        ae_scorer = AnomalyScorer(detector=ae_model)
        scores_ae, flags_ae = ae_scorer.predict(X_test_scaled)
        metrics_ae = Evaluator.evaluate_predictions(y_test_true, flags_ae, scores_ae, meta_df=meta_test)
        print(f"  [PyTorch Autoencoder] F1: {metrics_ae['f1_score']:.4f}, FPR: {metrics_ae['fpr']:.4f}, ROC-AUC: {metrics_ae['roc_auc']:.4f}")

        ablation_results[cfg] = {
            "statistical": metrics_stat,
            "isolation_forest": metrics_if,
            "autoencoder": metrics_ae,
        }

        # Save test predictions CSV for plotting
        preds_df = meta_test.copy()
        preds_df["stat_score"] = scores_stat
        preds_df["stat_flag"] = flags_stat
        preds_df["if_score"] = scores_if
        preds_df["if_flag"] = flags_if
        preds_df["ae_score"] = scores_ae
        preds_df["ae_flag"] = flags_ae
        preds_df.to_csv(os.path.join(out_dir, "predictions", f"predictions_{cfg}.csv"), index=False)

    # Save ablation summary JSON
    ablation_json_path = os.path.join(out_dir, "evaluation", "phase5_ablation_matrix.json")
    with open(ablation_json_path, "w") as f:
        json.dump(ablation_results, f, indent=2)

    # Print Summary Table
    print("\n" + "=" * 80)
    print("PHASE 5 ABLATION STUDY COMPARISON MATRIX (HELD-OUT TEST SET)")
    print("=" * 80)
    print(f"{'Feature Config':<15} | {'Model':<20} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'FPR':<10} | {'ROC-AUC':<10}")
    print("-" * 90)

    for cfg in configs:
        for m_name in ["statistical", "isolation_forest", "autoencoder"]:
            m = ablation_results[cfg][m_name]
            print(f"{cfg.upper():<15} | {m_name:<20} | {m['precision']:<10.4f} | {m['recall']:<10.4f} | {m['f1_score']:<10.4f} | {m['fpr']:<10.4f} | {m['roc_auc']:<10.4f}")

    print("=" * 80)


if __name__ == "__main__":
    main()
