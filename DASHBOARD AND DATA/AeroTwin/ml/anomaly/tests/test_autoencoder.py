"""
Unit tests for AutoencoderAnomalyDetector (Model 3 PyTorch Net).
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

from ml.anomaly.autoencoder import AutoencoderAnomalyDetector


class TestAutoencoder(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        self.X_train = pd.DataFrame({
            "f1": np.random.normal(0.0, 0.1, 100),
            "f2": np.random.normal(0.0, 0.1, 100),
        })
        self.X_anom = pd.DataFrame({
            "f1": np.random.normal(5.0, 0.5, 20),
            "f2": np.random.normal(5.0, 0.5, 20),
        })

    def test_autoencoder_training_and_reconstruction(self):
        detector = AutoencoderAnomalyDetector(epochs=20, random_seed=42)
        detector.fit(self.X_train)
        tau = detector.fit_threshold(self.X_train, target_fpr=0.05)

        scores_train, flags_train = detector.predict(self.X_train)
        scores_anom, flags_anom = detector.predict(self.X_anom)

        # Higher reconstruction error = MORE ANOMALOUS
        self.assertGreater(float(np.mean(scores_anom)), float(np.mean(scores_train)))
        self.assertGreater(float(np.mean(flags_anom)), 0.8)


if __name__ == "__main__":
    unittest.main()
