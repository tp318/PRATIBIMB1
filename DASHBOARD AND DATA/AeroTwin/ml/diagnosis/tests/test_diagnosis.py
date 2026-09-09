"""
Unit tests for fault diagnosis: run-based splits, classifier contract, evaluation.
"""

import os
import sys
import unittest

import numpy as np
import pandas as pd

_test_dir = os.path.dirname(os.path.abspath(__file__))
_diag_dir = os.path.dirname(_test_dir)
_ml_dir = os.path.dirname(_diag_dir)
_aerotwin_dir = os.path.dirname(_ml_dir)
_root_dir = os.path.dirname(_aerotwin_dir)

for _p in [_diag_dir, _ml_dir, _aerotwin_dir, _root_dir]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ml.diagnosis.classifier import FaultDiagnosisClassifier
from ml.diagnosis.evaluation import DiagnosisEvaluator
from ml.diagnosis.splits import (
    FAULT_CLASSES,
    DiagnosisSplitter,
    label_from_run_id,
    severity_tag,
)


def _synthetic_dataset(n_windows=12, seed=0):
    """Separable synthetic faults: each family shifts a different feature."""
    rng = np.random.default_rng(seed)
    runs = (
        [f"HEALTHY_{i:03d}" for i in range(1, 10)]
        + [f"CYL1_SEV{s:03d}" for s in (20, 40, 60, 80)]
        + [f"CYL3_SEV{s:03d}" for s in (20, 40, 60, 80)]
        + [f"BEARING_SEV{s:03d}" for s in (20, 40, 60, 80)]
        + [f"COOLING_SEV{s:03d}" for s in (20, 40, 60, 80)]
        + [f"LUBRICATION_SEV{s:03d}" for s in (20, 40, 60, 80)]
    )
    offsets = {"HEALTHY": 0, "CYLINDER": 1, "BEARING": 2, "COOLING": 3, "LUBRICATION": 4}

    X_rows, meta_rows = [], []
    for rid in runs:
        fam = label_from_run_id(rid)
        for w in range(n_windows):
            vec = rng.normal(0.0, 0.05, size=5)
            if fam != "HEALTHY":
                vec[offsets[fam]] += 6.0
            X_rows.append({f"f{i}": vec[i] for i in range(5)})
            meta_rows.append(
                {"run_id": rid, "window_id": f"{rid}_W{w:04d}", "gt_is_degraded": fam != "HEALTHY"}
            )
    return pd.DataFrame(X_rows), pd.DataFrame(meta_rows)


class TestDiagnosisSplits(unittest.TestCase):

    def test_label_from_run_id(self):
        self.assertEqual(label_from_run_id("HEALTHY_001"), "HEALTHY")
        self.assertEqual(label_from_run_id("CYL1_SEV040"), "CYLINDER")
        self.assertEqual(label_from_run_id("CYL3_SEV080"), "CYLINDER")
        self.assertEqual(label_from_run_id("BEARING_SEV020"), "BEARING")
        self.assertEqual(label_from_run_id("COOLING_SEV060"), "COOLING")
        self.assertEqual(label_from_run_id("LUBRICATION_SEV080"), "LUBRICATION")

    def test_unknown_run_id_is_rejected(self):
        with self.assertRaises(ValueError):
            label_from_run_id("MYSTERY_RUN")

    def test_severity_tag(self):
        self.assertEqual(severity_tag("CYL1_SEV040"), "SEV040")
        self.assertEqual(severity_tag("HEALTHY_001"), "")

    def test_no_run_leaks_across_partitions(self):
        X, meta = _synthetic_dataset()
        parts = DiagnosisSplitter().split(X, meta)
        tr = set(parts["train"][1]["run_id"].unique())
        va = set(parts["val"][1]["run_id"].unique())
        te = set(parts["test"][1]["run_id"].unique())
        self.assertEqual(tr & va, set())
        self.assertEqual(tr & te, set())
        self.assertEqual(va & te, set())

    def test_test_partition_holds_out_unseen_severity(self):
        """SEV080 must never be trained on - that is the generalisation claim."""
        X, meta = _synthetic_dataset()
        parts = DiagnosisSplitter().split(X, meta)
        train_runs = parts["train"][1]["run_id"].unique()
        self.assertFalse(any("SEV080" in r for r in train_runs))
        test_runs = parts["test"][1]["run_id"].unique()
        self.assertTrue(any("SEV080" in r for r in test_runs))

    def test_every_class_present_in_training(self):
        X, meta = _synthetic_dataset()
        parts = DiagnosisSplitter().split(X, meta)
        self.assertEqual(set(parts["train"][2].unique()), set(FAULT_CLASSES))

    def test_healthy_present_in_all_partitions(self):
        X, meta = _synthetic_dataset()
        parts = DiagnosisSplitter().split(X, meta)
        for name in ("train", "val", "test"):
            self.assertIn("HEALTHY", set(parts[name][2].unique()), f"{name} has no healthy runs")


class TestDiagnosisClassifier(unittest.TestCase):

    def setUp(self):
        X, meta = _synthetic_dataset()
        self.parts = DiagnosisSplitter().split(X, meta)
        X_tr, _, y_tr = self.parts["train"]
        self.clf = FaultDiagnosisClassifier(n_estimators=60).fit(X_tr, y_tr)

    def test_predict_before_fit_raises(self):
        with self.assertRaises(RuntimeError):
            FaultDiagnosisClassifier().predict(pd.DataFrame({"f0": [0.0]}))

    def test_missing_feature_at_predict_time_raises(self):
        with self.assertRaises(ValueError):
            self.clf.predict(pd.DataFrame({"f0": [0.0]}))

    def test_probabilities_sum_to_one(self):
        X_te, _, _ = self.parts["test"]
        proba = self.clf.predict_proba(X_te)
        np.testing.assert_allclose(proba.sum(axis=1).values, 1.0, rtol=1e-9)

    def test_separable_faults_are_classified_on_unseen_severity(self):
        X_te, _, y_te = self.parts["test"]
        pred = self.clf.predict(X_te)
        acc = float((pred == y_te.values).mean())
        self.assertGreater(acc, 0.9, f"Separable synthetic faults should be easy; got {acc:.3f}")

    def test_diagnose_reports_confidence_and_margin(self):
        X_te, _, _ = self.parts["test"]
        out = self.clf.diagnose(X_te)
        for col in ("predicted_fault", "confidence", "runner_up_fault", "margin"):
            self.assertIn(col, out.columns)
        self.assertTrue((out["confidence"] >= out["runner_up_confidence"]).all())
        self.assertTrue((out["margin"] >= -1e-12).all())

    def test_feature_importance_is_ranked(self):
        imp = self.clf.feature_importance(top_n=5)
        self.assertLessEqual(len(imp), 5)
        self.assertTrue(imp["importance"].is_monotonic_decreasing)

    def test_save_and_load_roundtrip(self):
        import tempfile

        X_te, _, _ = self.parts["test"]
        expected = self.clf.predict(X_te)
        with tempfile.TemporaryDirectory() as d:
            self.clf.save(d)
            reloaded = FaultDiagnosisClassifier().load(d)
            np.testing.assert_array_equal(reloaded.predict(X_te), expected)


class TestDiagnosisEvaluator(unittest.TestCase):

    def test_confusion_matrix_shape_and_totals(self):
        ev = DiagnosisEvaluator()
        y_true = ["HEALTHY"] * 3 + ["BEARING"] * 2
        y_pred = ["HEALTHY"] * 3 + ["BEARING"] * 2
        cm = ev.confusion(y_true, y_pred)
        self.assertEqual(cm.shape, (len(FAULT_CLASSES), len(FAULT_CLASSES)))
        self.assertEqual(int(cm.values.sum()), 5)

    def test_perfect_prediction_scores_one(self):
        ev = DiagnosisEvaluator()
        y = ["HEALTHY", "BEARING", "COOLING", "CYLINDER", "LUBRICATION"]
        s = ev.summary(y, y)
        self.assertAlmostEqual(s["accuracy"], 1.0)
        self.assertAlmostEqual(s["macro_f1"], 1.0)

    def test_per_class_support_matches_truth(self):
        ev = DiagnosisEvaluator()
        y_true = ["HEALTHY"] * 4 + ["COOLING"] * 2
        y_pred = ["HEALTHY"] * 6
        pc = ev.per_class(y_true, y_pred).set_index("fault_class")
        self.assertEqual(int(pc.loc["HEALTHY", "support"]), 4)
        self.assertEqual(int(pc.loc["COOLING", "support"]), 2)
        self.assertAlmostEqual(float(pc.loc["COOLING", "recall"]), 0.0)


if __name__ == "__main__":
    unittest.main()
