# AeroTwin-4 Phase 5 Specification: Physics-Informed Anomaly Detection

## 1. Overview
Phase 5 converts Phase 3 raw telemetry and Phase 4 Digital Twin counterfactual residuals into ML-ready feature vectors and implements an **unsupervised physics-informed anomaly detection pipeline**.

Phase 5 answers strictly:
> **"Is the current engine behavior normal or anomalous?"**

Phase 5 does **NOT** perform fault classification, RUL prediction, SHAP, or final health scoring.

---

## 2. Unsupervised Healthy-Only Training Paradigm
- **Training Set (`HEALTHY_001`, `HEALTHY_002`)**: Models and scalers are fitted **strictly on healthy training runs**.
- **Threshold Derivation (`HEALTHY_003`)**: Decision thresholds $\tau$ are derived on healthy validation score distributions targeting maximum FPR $\le 5\%$.
- **Zero Ground-Truth Feature Leakage**: Ground-truth labels are strictly isolated for evaluation and are **NEVER** passed into ML feature matrices.
- **Run-Based Partitioning**: Splitting is performed by run IDs to prevent sliding-window overlap leakage across train/validation/test partitions.

---

## 3. Three Feature Configurations (Ablation Study)
1. **Configuration A (Raw Telemetry)**: 136 features derived from raw sensor channels + operating context.
2. **Configuration B (Digital Twin Residuals)**: 178 features derived from Phase 4 counterfactual residuals + indicators.
3. **Configuration C (Hybrid Physics-Informed)**: 314 features combining raw telemetry + Digital Twin residuals + operating context.

---

## 4. Models Implemented
- **Model 1 (Statistical Baseline)**: Standardized Z-score RMS distance from healthy feature mean.
- **Model 2 (Isolation Forest)**: Scikit-learn Isolation Forest trained on healthy features, with negated score orientation (higher score = more anomalous).
- **Model 3 (PyTorch Autoencoder)**: 4-layer feed-forward neural network trained via MSE loss to reconstruct healthy feature vectors.
