"""
Unit tests for the RUL pipeline: health regression and trend projection.
"""

import os
import sys
import unittest

import numpy as np
import pandas as pd

_test_dir = os.path.dirname(os.path.abspath(__file__))
_rul_dir = os.path.dirname(_test_dir)
_ml_dir = os.path.dirname(_rul_dir)
_aerotwin_dir = os.path.dirname(_ml_dir)
_root_dir = os.path.dirname(_aerotwin_dir)

for _p in [_rul_dir, _ml_dir, _aerotwin_dir, _root_dir]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ml.rul.health_estimator import HealthEstimator
from ml.rul.projector import DEFAULT_FAILURE_THRESHOLD, RULProjector


class TestHealthEstimator(unittest.TestCase):

    def setUp(self):
        rng = np.random.default_rng(0)
        n = 300
        health = rng.uniform(0.0, 1.0, n)
        # Two features that carry the health signal, plus noise.
        self.X = pd.DataFrame(
            {
                "f_sig": (1.0 - health) * 5.0 + rng.normal(0, 0.05, n),
                "f_sig2": (1.0 - health) ** 2 * 3.0 + rng.normal(0, 0.05, n),
                "f_noise": rng.normal(0, 1.0, n),
            }
        )
        self.health = pd.Series(health)

    def test_predict_before_fit_raises(self):
        with self.assertRaises(RuntimeError):
            HealthEstimator().predict(self.X)

    def test_missing_feature_raises(self):
        est = HealthEstimator().fit(self.X, self.health)
        with self.assertRaises(ValueError):
            est.predict(self.X[["f_noise"]])

    def test_predictions_stay_inside_physical_range(self):
        est = HealthEstimator().fit(self.X, self.health)
        pred = est.predict(self.X)
        self.assertGreaterEqual(pred.min(), 0.0)
        self.assertLessEqual(pred.max(), 1.0)

    def test_learns_the_health_signal(self):
        est = HealthEstimator().fit(self.X, self.health)
        mae = float(np.mean(np.abs(est.predict(self.X) - self.health.values)))
        self.assertLess(mae, 0.1, f"health MAE too high: {mae:.3f}")

    def test_residual_std_is_recorded(self):
        est = HealthEstimator().fit(self.X, self.health)
        self.assertGreater(est.residual_std, 0.0)

    def test_save_load_roundtrip(self):
        import tempfile

        est = HealthEstimator().fit(self.X, self.health)
        expected = est.predict(self.X)
        with tempfile.TemporaryDirectory() as d:
            est.save(d)
            back = HealthEstimator().load(d)
            np.testing.assert_allclose(back.predict(self.X), expected)
            self.assertAlmostEqual(back.residual_std, est.residual_std)


class TestRULProjector(unittest.TestCase):

    def setUp(self):
        self.proj = RULProjector(failure_threshold=0.30, health_noise_std=0.01)

    def test_too_few_points_returns_no_estimate(self):
        est = self.proj.estimate([0.0, 1.0], [1.0, 0.99])
        self.assertIsNone(est.rul_seconds)
        self.assertFalse(est.is_decaying)

    def test_mismatched_lengths_raise(self):
        with self.assertRaises(ValueError):
            self.proj.estimate([0.0, 1.0, 2.0], [1.0, 0.9])

    def test_flat_health_reports_no_decay(self):
        """A healthy engine must never be handed a finite remaining life."""
        t = np.arange(0.0, 60.0, 1.0)
        h = np.full_like(t, 0.98)
        est = self.proj.estimate(t, h)
        self.assertFalse(est.is_decaying)
        self.assertIsNone(est.rul_seconds)

    def test_noise_only_is_not_mistaken_for_a_trend(self):
        rng = np.random.default_rng(3)
        t = np.arange(0.0, 60.0, 1.0)
        h = 0.95 + rng.normal(0, 0.01, len(t))
        est = self.proj.estimate(t, h)
        self.assertFalse(est.is_decaying, "estimator jitter was read as real decay")

    def test_linear_decay_rul_is_accurate(self):
        # Health falls 0.01/s from 1.0; threshold 0.30 reached at t=70s.
        t = np.arange(0.0, 50.0, 1.0)
        h = 1.0 - 0.01 * t
        est = self.proj.estimate(t, h)
        self.assertTrue(est.is_decaying)
        # At t=49, health=0.51, so RUL = (0.51-0.30)/0.01 = 21s.
        self.assertAlmostEqual(est.rul_seconds, 21.0, delta=2.0)

    def test_already_failed_reports_zero(self):
        t = np.arange(0.0, 30.0, 1.0)
        h = np.linspace(0.5, 0.10, len(t))
        est = self.proj.estimate(t, h)
        self.assertEqual(est.rul_seconds, 0.0)

    def test_interval_brackets_the_point_estimate(self):
        t = np.arange(0.0, 50.0, 1.0)
        h = 1.0 - 0.01 * t
        est = self.proj.estimate(t, h)
        self.assertLessEqual(est.rul_lower_seconds, est.rul_seconds)
        if est.rul_upper_seconds is not None:
            self.assertGreaterEqual(est.rul_upper_seconds, est.rul_seconds)

    def test_interval_widens_with_longer_extrapolation(self):
        """Reaching further past the observed history must cost confidence."""
        t = np.arange(0.0, 40.0, 1.0)
        near = self.proj.estimate(t, 1.0 - 0.02 * t)   # crosses sooner
        far = self.proj.estimate(t, 1.0 - 0.004 * t)   # crosses much later
        near_w = near.rul_upper_seconds - near.rul_lower_seconds
        far_w = far.rul_upper_seconds - far.rul_lower_seconds
        self.assertGreater(far_w, near_w)

    def test_rul_never_negative(self):
        t = np.arange(0.0, 40.0, 1.0)
        est = self.proj.estimate(t, 1.0 - 0.015 * t)
        self.assertGreaterEqual(est.rul_seconds, 0.0)
        self.assertGreaterEqual(est.rul_lower_seconds, 0.0)

    def test_accelerating_decay_selects_the_curved_model(self):
        """A straight line through accelerating decay over-predicts life."""
        t = np.arange(0.0, 60.0, 1.0)
        h = 1.0 - 0.0002 * t ** 2
        est = self.proj.estimate(t, h)
        self.assertTrue(est.is_decaying)
        self.assertEqual(est.extra.get("trend_model"), "quadratic")

    def test_to_dict_is_serialisable(self):
        import json

        t = np.arange(0.0, 40.0, 1.0)
        est = self.proj.estimate(t, 1.0 - 0.01 * t)
        json.dumps(est.to_dict())

    def test_default_threshold_is_documented_value(self):
        self.assertAlmostEqual(DEFAULT_FAILURE_THRESHOLD, 0.30)


if __name__ == "__main__":
    unittest.main()
