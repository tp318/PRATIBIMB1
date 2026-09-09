"""
Unit tests for FeatureScaler (Fit-On-Train-Only).
"""

import os
import sys
import tempfile
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

from ml.anomaly.preprocessing import FeatureScaler


class TestPreprocessing(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        self.X_train = pd.DataFrame({
            "f1": np.random.normal(10.0, 2.0, 100),
            "f2": np.random.normal(50.0, 5.0, 100),
        })
        self.X_val = pd.DataFrame({
            "f1": np.random.normal(10.0, 2.0, 20),
            "f2": np.random.normal(50.0, 5.0, 20),
        })

    def test_fit_on_train_only_scaling(self):
        scaler = FeatureScaler()
        X_tr_scaled = scaler.fit_transform(self.X_train)

        # Transformed train features must have mean ~ 0 and std ~ 1
        self.assertAlmostEqual(float(np.mean(X_tr_scaled[:, 0])), 0.0, places=4)
        self.assertAlmostEqual(float(np.std(X_tr_scaled[:, 0])), 1.0, places=4)

        # Validation transform uses train parameters
        X_val_scaled = scaler.transform(self.X_val)
        self.assertEqual(X_val_scaled.shape, (20, 2))

    def test_save_and_load_scaler(self):
        scaler1 = FeatureScaler()
        scaler1.fit(self.X_train)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            scaler1.save(tmp_path)
            scaler2 = FeatureScaler().load(tmp_path)

            X1 = scaler1.transform(self.X_val)
            X2 = scaler2.transform(self.X_val)

            np.testing.assert_allclose(X1, X2)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


if __name__ == "__main__":
    unittest.main()
