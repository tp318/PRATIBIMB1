"""
Unit tests for RunSplitter (Run-Based Partitioning & Non-Leakage).
"""

import os
import sys
import unittest
import pandas as pd

_test_dir = os.path.dirname(os.path.abspath(__file__))
_anomaly_dir = os.path.dirname(_test_dir)
_ml_dir = os.path.dirname(_anomaly_dir)
_aerotwin_dir = os.path.dirname(_ml_dir)
_root_dir = os.path.dirname(_aerotwin_dir)

for _p in [_anomaly_dir, _ml_dir, _aerotwin_dir, _root_dir]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ml.anomaly.splits import RunSplitter


class TestSplits(unittest.TestCase):

    def setUp(self):
        rids = [
            "HEALTHY_001", "HEALTHY_002", "HEALTHY_003", "HEALTHY_004", "HEALTHY_005",
            "CYL1_SEV020", "CYL3_SEV020", "BEARING_SEV020", "COOLING_SEV020", "LUBRICATION_SEV020",
            "CYL1_SEV040", "BEARING_SEV040", "COOLING_SEV040",
        ]
        X_rows = []
        meta_rows = []
        for rid in rids:
            for w in range(5):
                X_rows.append({"feat1": 1.0, "feat2": 2.0})
                meta_rows.append({"run_id": rid, "window_id": f"{rid}_W{w:04d}", "gt_is_degraded": "SEV" in rid})

        self.X = pd.DataFrame(X_rows)
        self.meta_df = pd.DataFrame(meta_rows)

    def test_run_based_partitioning_non_leakage(self):
        splitter = RunSplitter(
            train_runs=["HEALTHY_001", "HEALTHY_002"],
            val_runs=["HEALTHY_003", "CYL1_SEV020"],
        )
        # HEALTHY_004/005 fall through to test, so the test partition carries both
        # classes - see test_test_partition_must_contain_both_classes below.

        (X_tr, m_tr), (X_va, m_va), (X_te, m_te) = splitter.split_dataset(self.X, self.meta_df)

        r_tr = set(m_tr["run_id"].unique())
        r_va = set(m_va["run_id"].unique())
        r_te = set(m_te["run_id"].unique())

        self.assertEqual(len(r_tr.intersection(r_va)), 0)
        self.assertEqual(len(r_tr.intersection(r_te)), 0)
        self.assertEqual(len(r_va.intersection(r_te)), 0)

        self.assertIn("HEALTHY_001", r_tr)
        self.assertIn("HEALTHY_003", r_va)
        self.assertIn("CYL1_SEV040", r_te)

    def test_test_partition_contains_both_classes(self):
        """A held-out test set needs negatives, or precision/FPR/ROC-AUC are undefined."""
        splitter = RunSplitter(
            train_runs=["HEALTHY_001", "HEALTHY_002"],
            val_runs=["HEALTHY_003", "CYL1_SEV020"],
        )
        _, _, (_, m_te) = splitter.split_dataset(self.X, self.meta_df)

        classes = set(m_te["gt_is_degraded"].astype(bool).unique())
        self.assertEqual(classes, {True, False})
        self.assertGreater((~m_te["gt_is_degraded"].astype(bool)).sum(), 0)

    def test_all_degraded_test_partition_is_rejected(self):
        """Guard against the regression that made every reported metric meaningless."""
        splitter = RunSplitter(
            train_runs=["HEALTHY_001", "HEALTHY_002"],
            val_runs=["HEALTHY_003", "HEALTHY_004", "HEALTHY_005"],
        )
        with self.assertRaises(AssertionError):
            splitter.split_dataset(self.X, self.meta_df)


if __name__ == "__main__":
    unittest.main()
