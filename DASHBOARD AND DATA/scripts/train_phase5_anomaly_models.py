"""
AeroTwin-4 Phase 5 Model Trainer CLI.

Trains 3 Unsupervised Anomaly Detection Models across 3 Feature Configurations:
- Model 1: Statistical Baseline
- Model 2: Isolation Forest
- Model 3: PyTorch Feed-Forward Autoencoder

STRICT RULES:
1. Training is fitted STRICTLY on Healthy Training runs (HEALTHY_001, HEALTHY_002).
2. FeatureScalers are fitted ONLY on Healthy Training features.
3. Decision thresholds are derived on Healthy Validation data (HEALTHY_003) to target FPR <= 5%.
"""

import os
import sys
import json
import pandas as pd

_script_dir = os.path.dirname(os.path.abspath(__file__))
_root_dir = os.path.dirname(_script_dir)

if _root_dir not in sys.path:
    sys.path.insert(0, _root_dir)

from AeroTwin.ml.anomaly.preprocessing import FeatureScaler
from AeroTwin.ml.anomaly.statistical import StatisticalAnomalyDetector
from AeroTwin.ml.anomaly.isolation_forest import IsolationForestAnomalyDetector
from AeroTwin.ml.anomaly.autoencoder import AutoencoderAnomalyDetector


def main():
    print("=" * 60)
    print("AeroTwin-4 Phase 5 Anomaly Model Trainer")
    print("=" * 60)

    data_dir = os.path.join(_root_dir, "data", "generated", "phase5", "features")
    models_dir = os.path.join(_root_dir, "models", "phase5")
    os.makedirs(models_dir, exist_ok=True)

    configs = ["raw", "residual", "hybrid"]

    for cfg in configs:
        print(f"\n==================== Training Models for Config: {cfg.upper()} ====================")
        cfg_path = os.path.join(data_dir, cfg)

        X_train_raw = pd.read_csv(os.path.join(cfg_path, "train", "X_train.csv"))
        meta_train = pd.read_csv(os.path.join(cfg_path, "train", "meta_train.csv"))

        X_val_raw = pd.read_csv(os.path.join(cfg_path, "validation", "X_val.csv"))
        meta_val = pd.read_csv(os.path.join(cfg_path, "validation", "meta_val.csv"))

        # Filter Healthy Validation features ONLY for threshold derivation.
        # Derive from the ground-truth label, not a hardcoded run id, so the split
        # can grow healthy runs without silently selecting an empty frame.
        healthy_val_mask = ~meta_val["gt_is_degraded"].astype(bool)
        X_val_healthy_raw = X_val_raw[healthy_val_mask].reset_index(drop=True)
        if len(X_val_healthy_raw) == 0:
            raise ValueError(
                f"[{cfg}] Validation partition contains no healthy windows; "
                "the FPR-calibrated threshold cannot be derived."
            )
        print(f"  Healthy validation windows for threshold calibration: {len(X_val_healthy_raw)}")

        # 1. Fit Feature Scaler STRICTLY on Healthy Training features
        scaler = FeatureScaler()
        X_train_scaled = scaler.fit_transform(X_train_raw)
        X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=scaler.feature_names)

        X_val_healthy_scaled = scaler.transform(X_val_healthy_raw)
        X_val_healthy_scaled_df = pd.DataFrame(X_val_healthy_scaled, columns=scaler.feature_names)

        scaler_dir = os.path.join(models_dir, "preprocessing", cfg)
        os.makedirs(scaler_dir, exist_ok=True)
        scaler.save(os.path.join(scaler_dir, "scaler.json"))

        # ----------------------------------------------------
        # Model 1: Statistical Baseline Detector
        # ----------------------------------------------------
        print(f"  [Model 1: Statistical Baseline] Training ({cfg.upper()})...")
        stat_model = StatisticalAnomalyDetector()
        stat_model.fit(X_train_scaled_df)
        tau_stat = stat_model.fit_threshold(X_val_healthy_scaled_df, target_fpr=0.05)
        print(f"    -> Threshold derived on Healthy Val (FPR=5%): {tau_stat:.4f}")

        stat_save_dir = os.path.join(models_dir, "statistical", cfg)
        os.makedirs(stat_save_dir, exist_ok=True)
        stat_model.save(os.path.join(stat_save_dir, "model.json"))

        # ----------------------------------------------------
        # Model 2: Isolation Forest Detector
        # ----------------------------------------------------
        print(f"  [Model 2: Isolation Forest] Training ({cfg.upper()})...")
        if_model = IsolationForestAnomalyDetector(n_estimators=100, random_state=42)
        if_model.fit(X_train_scaled_df)
        tau_if = if_model.fit_threshold(X_val_healthy_scaled_df, target_fpr=0.05)
        print(f"    -> Threshold derived on Healthy Val (FPR=5%): {tau_if:.4f}")

        if_save_dir = os.path.join(models_dir, "isolation_forest", cfg)
        os.makedirs(if_save_dir, exist_ok=True)
        if_model.save(os.path.join(if_save_dir, "model.joblib"))

        # ----------------------------------------------------
        # Model 3: PyTorch Autoencoder Detector
        # ----------------------------------------------------
        print(f"  [Model 3: PyTorch Autoencoder] Training ({cfg.upper()})...")
        ae_model = AutoencoderAnomalyDetector(hidden_dim1=32, latent_dim=16, epochs=50, batch_size=32, random_seed=42)
        ae_model.fit(X_train_scaled_df, X_val_healthy=X_val_healthy_scaled_df)
        tau_ae = ae_model.fit_threshold(X_val_healthy_scaled_df, target_fpr=0.05)
        print(f"    -> Threshold derived on Healthy Val (FPR=5%): {tau_ae:.6f}")

        ae_save_dir = os.path.join(models_dir, "autoencoder", cfg)
        ae_model.save(ae_save_dir)

    print("\n" + "-" * 40)
    print("Phase 5 Model Training & Artifact Saving Complete!")
    print("-" * 40)


if __name__ == "__main__":
    main()
