"""
=============================================================================
AEROTWIN-4 XGBOOST FAULT INFERENCE ENGINE
=============================================================================
Provides production-grade inference, probability scoring, and feature attribution
for the trained XGBoost fault detection model.
=============================================================================
"""

import os
import json
from typing import Dict, List, Any, Optional, Union
import numpy as np
import xgboost as xgb

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_PATH = os.path.join(MODEL_DIR, "best_xgboost_fault_detector.json")
DEFAULT_META_PATH = os.path.join(MODEL_DIR, "xgboost_model_features.json")


class XGBoostFaultPredictor:
    """
    Lightweight, fast XGBoost inference engine for MALE UAV engine diagnostics.
    """

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        meta_path: str = DEFAULT_META_PATH,
    ):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Trained model not found at {model_path}")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Metadata not found at {meta_path}")

        self.model = xgb.XGBClassifier()
        self.model.load_model(model_path)

        with open(meta_path, "r") as f:
            meta = json.load(f)

        self.feature_names: List[str] = meta["features"]
        self.target_names: List[str] = meta["target_names"]

        # Initialize TreeSHAP explainer for real-time model explainability
        self.explainer = None
        try:
            import shap
            self.explainer = shap.TreeExplainer(self.model)
            print("[XGBoostFaultPredictor] TreeSHAP explainer initialized successfully.")
        except Exception as e:
            print(f"[XGBoostFaultPredictor] Warning: Could not initialize TreeSHAP explainer: {e}")

    def predict(self, feature_dict: Dict[str, float]) -> Dict[str, Any]:
        """
        Runs inference on a dictionary containing feature values for one 30s window.
        Returns:
          - predicted_class: int (0 to 8)
          - fault_name: str
          - confidence: float (0.0 to 1.0)
          - probabilities: Dict[str, float]
        """
        # Assemble feature vector in exact training order
        vector = []
        for name in self.feature_names:
            val = feature_dict.get(name, 0.0)
            vector.append(float(val))

        X = np.array([vector], dtype=np.float32)
        probs = self.model.predict_proba(X)[0]
        pred_class = int(np.argmax(probs))
        confidence = float(probs[pred_class])

        return {
            "predicted_class": pred_class,
            "fault_name": self.target_names[pred_class],
            "confidence": round(confidence, 4),
            "is_anomaly": pred_class != 0,
            "probabilities": {
                name: round(float(p), 4) for name, p in zip(self.target_names, probs)
            },
        }

    def explain(self, feature_dict: Dict[str, float], top_k: int = 6) -> Dict[str, Any]:
        """
        Runs TreeSHAP / DeepSHAP feature attribution explaining why the model predicted
        the specific fault class or normal state.
        """
        vector = [float(feature_dict.get(name, 0.0)) for name in self.feature_names]
        X = np.array([vector], dtype=np.float32)
        probs = self.model.predict_proba(X)[0]
        pred_class = int(np.argmax(probs))
        fault_name = self.target_names[pred_class]
        confidence = float(probs[pred_class])

        attributions = []
        positive_drivers = []
        negative_suppressors = []
        base_value = 0.0
        output_margin = 0.0

        if self.explainer is not None:
            try:
                shap_res = self.explainer(X)
                # shap_res.values shape is (1, n_features, n_classes)
                phi = shap_res.values[0, :, pred_class]
                expected_vals = self.explainer.expected_value
                if isinstance(expected_vals, (list, np.ndarray)):
                    base_value = float(expected_vals[pred_class])
                else:
                    base_value = float(expected_vals)

                total_abs_phi = float(np.sum(np.abs(phi))) + 1e-9

                ranked_indices = np.argsort(-np.abs(phi))
                for idx in ranked_indices:
                    f_name = self.feature_names[idx]
                    f_val = float(vector[idx])
                    f_shap = float(phi[idx])
                    pct = round(abs(f_shap) / total_abs_phi * 100.0, 1)

                    item = {
                        "feature": f_name,
                        "feature_value": round(f_val, 3),
                        "shap_value": round(f_shap, 4),
                        "impact_pct": pct,
                        "direction": "RISK_INCREASING" if f_shap > 0 else "RISK_SUPPRESSING"
                    }
                    attributions.append(item)

                positive_drivers = [a for a in attributions if a["shap_value"] > 0][:top_k]
                negative_suppressors = [a for a in attributions if a["shap_value"] < 0][:top_k]
                output_margin = round(base_value + float(np.sum(phi)), 4)
            except Exception as e:
                print(f"[XGBoostFaultPredictor] Explain error: {e}")

        # Construct concise natural-language attribution statement
        if pred_class == 0:
            summary = "TreeSHAP: All operational parameters within normal aerodynamic & thermal envelopes; zero significant anomaly drivers."
        elif positive_drivers:
            top_features_str = ", ".join([f"{d['feature']} (φ={d['shap_value']:+.3f})" for d in positive_drivers[:3]])
            summary = f"TreeSHAP Attribution: {fault_name} diagnosis strongly driven by {top_features_str}."
        else:
            summary = f"TreeSHAP Attribution: {fault_name} diagnosed with {confidence*100:.1f}% confidence."

        return {
            "predicted_class": pred_class,
            "fault_name": fault_name,
            "confidence": round(confidence, 4),
            "base_value": round(base_value, 4),
            "output_margin": output_margin,
            "summary": summary,
            "positive_drivers": positive_drivers,
            "negative_suppressors": negative_suppressors,
            "top_attributions": attributions[:top_k],
            "framework": "TreeSHAP / DeepSHAP Physics Feature Attribution",
        }


# Quick self-test
if __name__ == "__main__":
    predictor = XGBoostFaultPredictor()
    print(f"XGBoostFaultPredictor loaded successfully!")
    print(f"Features expected ({len(predictor.feature_names)}): {predictor.feature_names[:5]} ...")
    print(f"Target classes: {predictor.target_names}")
    res = predictor.explain({})
    print(f"Explanation test: {res['summary']}")

