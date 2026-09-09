"""
Unit tests for Evaluator Engine.
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

from ml.anomaly.evaluation import Evaluator


class TestEvaluation(unittest.TestCase):

    def test_evaluator_metrics_calculation(self):
        y_true = np.array([False, False, True, True, True])
        y_pred = np.array([False, False, True, True, False])
        y_scores = np.array([0.1, 0.2, 0.8, 0.9, 0.4])

        res = Evaluator.evaluate_predictions(y_true, y_pred, y_scores)

        self.assertEqual(res["tp"], 2)
        self.assertEqual(res["fn"], 1)
        self.assertEqual(res["tn"], 2)
        self.assertEqual(res["fp"], 0)
        self.assertAlmostEqual(res["precision"], 1.0)
        self.assertAlmostEqual(res["recall"], 2 / 3)


if __name__ == "__main__":
    unittest.main()
