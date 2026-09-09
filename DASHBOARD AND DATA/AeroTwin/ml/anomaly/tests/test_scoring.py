"""
Unit tests for AnomalyScorer Interface.
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

from ml.anomaly.statistical import StatisticalAnomalyDetector
from ml.anomaly.scoring import AnomalyScorer


class TestScoring(unittest.TestCase):

    def test_anomaly_scorer_interface(self):
        detector = StatisticalAnomalyDetector()
        X_df = pd.DataFrame({"f1": [1.0, 2.0, 10.0], "f2": [0.5, 0.6, 5.0]})
        detector.fit(X_df.iloc[:2])

        scorer = AnomalyScorer(detector=detector)
        score, flag = scorer.predict_window(X_df.iloc[[2]])

        self.assertIsInstance(score, float)
        self.assertIsInstance(flag, bool)


if __name__ == "__main__":
    unittest.main()
