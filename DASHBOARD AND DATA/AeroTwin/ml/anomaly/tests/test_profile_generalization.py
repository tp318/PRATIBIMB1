"""
Unit test for Model Generalization across Unseen Operating Profiles.

Tests whether the unsupervised anomaly detector learns ENGINE HEALTH
rather than memorizing specific flight operating profiles.
"""

import os
import sys
import unittest
import numpy as np
import pandas as pd

_test_dir = os.path.dirname(os.path.abspath(__file__))
_anomaly_dir = os.path.dirname(_test_dir)
_ml_dir = os.path.dirname(_anomaly_dir)
_aerotwin_dir = os.path.dirname(_ml_dir)
_root_dir = os.path.dirname(_aerotwin_dir)

for _p in [_anomaly_dir, _ml_dir, _aerotwin_dir, _root_dir]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ml.anomaly.features import FeatureExtractor
from ml.anomaly.preprocessing import FeatureScaler
from ml.anomaly.autoencoder import AutoencoderAnomalyDetector


class TestProfileGeneralization(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        # Create Healthy CRUISE training data (1000 samples = 10s)
        n = 1000
        cruise_data = {
            "timestamp": np.linspace(0, 10.0, n),
            "simulation_time": np.linspace(0, 10.0, n),
            "run_id": ["HEALTHY_CRUISE"] * n,
            "operating_mode": ["CRUISE"] * n,
            "throttle": [0.60] * n,
            "ambient_temperature": [25.0] * n,
            "obs_rpm": 2500.0 + np.random.randn(n) * 5.0,
            "res_signed_rpm": np.random.randn(n) * 0.1,
            "res_signed_cht": np.random.randn(n) * 0.1,
            "ind_cylinder_balance_dev": [1.8] * n,
            "gt_is_degraded": [False] * n,
        }
        self.df_cruise = pd.DataFrame(cruise_data)

        # Create Healthy TAKEOFF data (unseen operating profile)
        takeoff_healthy = {
            "timestamp": np.linspace(0, 10.0, n),
            "simulation_time": np.linspace(0, 10.0, n),
            "run_id": ["HEALTHY_TAKEOFF"] * n,
            "operating_mode": ["TAKEOFF"] * n,
            "throttle": [1.00] * n,
            "ambient_temperature": [25.0] * n,
            "obs_rpm": 3200.0 + np.random.randn(n) * 5.0,
            "res_signed_rpm": np.random.randn(n) * 0.1,
            "res_signed_cht": np.random.randn(n) * 0.1,
            "ind_cylinder_balance_dev": [1.8] * n,
            "gt_is_degraded": [False] * n,
        }
        self.df_takeoff_healthy = pd.DataFrame(takeoff_healthy)

        # Create Degraded TAKEOFF data (unseen operating profile + degradation)
        takeoff_degraded = {
            "timestamp": np.linspace(0, 10.0, n),
            "simulation_time": np.linspace(0, 10.0, n),
            "run_id": ["DEGRADED_TAKEOFF"] * n,
            "operating_mode": ["TAKEOFF"] * n,
            "throttle": [1.00] * n,
            "ambient_temperature": [25.0] * n,
            "obs_rpm": 3100.0 + np.random.randn(n) * 5.0,
            "res_signed_rpm": -100.0 + np.random.randn(n) * 5.0,
            "res_signed_cht": 25.0 + np.random.randn(n) * 1.0,
            "ind_cylinder_balance_dev": [5.5] * n,
            "gt_is_degraded": [True] * n,
        }
        self.df_takeoff_degraded = pd.DataFrame(takeoff_degraded)

    def test_generalization_to_unseen_takeoff_profile(self):
        ext = FeatureExtractor(config_type="HYBRID")
        X_train, _ = ext.extract_dataset(self.df_cruise, window_size_sec=5.0, stride_sec=1.0)
        X_val_h, _ = ext.extract_dataset(self.df_takeoff_healthy, window_size_sec=5.0, stride_sec=1.0)
        X_test_d, _ = ext.extract_dataset(self.df_takeoff_degraded, window_size_sec=5.0, stride_sec=1.0)

        scaler = FeatureScaler().fit(X_train)
        X_tr_s = pd.DataFrame(scaler.transform(X_train), columns=scaler.feature_names)
        X_val_s = pd.DataFrame(scaler.transform(X_val_h), columns=scaler.feature_names)
        X_test_s = pd.DataFrame(scaler.transform(X_test_d), columns=scaler.feature_names)

        ae_model = AutoencoderAnomalyDetector(epochs=20, random_seed=42)
        ae_model.fit(X_tr_s)
        tau = ae_model.fit_threshold(X_val_s, target_fpr=0.05)

        scores_val, flags_val = ae_model.predict(X_val_s)
        scores_test, flags_test = ae_model.predict(X_test_s)

        # Unseen Healthy TAKEOFF score must be low (near threshold)
        # Unseen Degraded TAKEOFF score must be significantly higher
        self.assertGreater(float(np.mean(scores_test)), float(np.mean(scores_val)))
        self.assertGreater(float(np.mean(flags_test)), 0.8)


if __name__ == "__main__":
    unittest.main()
