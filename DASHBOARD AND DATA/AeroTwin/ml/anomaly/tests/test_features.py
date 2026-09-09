"""
Unit tests for FeatureExtractor & Zero-Leakage Enforcement.
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

from ml.anomaly.features import FeatureExtractor, FORBIDDEN_GROUND_TRUTH_FIELDS


class TestFeatures(unittest.TestCase):

    def setUp(self):
        # Create dummy 500-sample (5.0s at 100Hz) telemetry DataFrame
        n = 500
        data = {
            "timestamp": np.linspace(0, 5.0, n),
            "simulation_time": np.linspace(0, 5.0, n),
            "run_id": ["TEST_RUN_001"] * n,
            "operating_mode": ["CRUISE"] * n,
            "throttle": [0.60] * n,
            "ambient_temperature": [25.0] * n,
            "obs_rpm": 2500.0 + np.random.randn(n) * 10.0,
            "obs_mean_torque": 140.0 + np.random.randn(n) * 2.0,
            "obs_cht": 165.0 + np.random.randn(n) * 1.0,
            "res_signed_rpm": np.random.randn(n) * 2.0,
            "res_signed_cht": np.random.randn(n) * 0.5,
            "ind_cylinder_balance_dev": [1.8] * n,
            # Ground truth fields (MUST NOT LEAK into X)
            "gt_degradation_type": ["CYLINDER"] * n,
            "gt_active_severity": [0.40] * n,
            "gt_is_degraded": [True] * n,
        }
        self.df_run = pd.DataFrame(data)

    def test_feature_extraction_configurations(self):
        ext_raw = FeatureExtractor(config_type="RAW")
        ext_res = FeatureExtractor(config_type="RESIDUAL")
        ext_hyb = FeatureExtractor(config_type="HYBRID")

        X_raw, meta_raw = ext_raw.extract_dataset(self.df_run, window_size_sec=5.0, stride_sec=1.0)
        X_res, meta_res = ext_res.extract_dataset(self.df_run, window_size_sec=5.0, stride_sec=1.0)
        X_hyb, meta_hyb = ext_hyb.extract_dataset(self.df_run, window_size_sec=5.0, stride_sec=1.0)

        self.assertGreater(len(X_hyb.columns), len(X_raw.columns))
        self.assertGreater(len(X_hyb.columns), len(X_res.columns))

    def test_zero_ground_truth_leakage_assertion(self):
        ext = FeatureExtractor(config_type="HYBRID")
        X, meta = ext.extract_dataset(self.df_run, window_size_sec=5.0, stride_sec=1.0)

        for col in X.columns:
            self.assertNotIn(col, FORBIDDEN_GROUND_TRUTH_FIELDS, f"Forbidden ground truth field '{col}' leaked into X!")
            self.assertFalse(col.startswith("gt_"), f"Ground truth field '{col}' leaked into X!")


if __name__ == "__main__":
    unittest.main()
