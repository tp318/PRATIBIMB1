"""
AeroTwin-4 Run-Based Splitter.

Partition dataset by RUN IDs to guarantee ZERO window-level overlap leakage across partitions.
"""

from typing import Dict, List, Set, Tuple
import pandas as pd


# Training is unsupervised and healthy-only: the detector learns what "normal"
# looks like across several engine units and ambient conditions.
DEFAULT_TRAIN_RUNS = [
    "HEALTHY_001",
    "HEALTHY_002",
    "HEALTHY_003",
    "HEALTHY_004",
]

# Validation carries healthy runs (to place the threshold at a target FPR) and the
# mildest fault severity of every family (to confirm the threshold still detects).
DEFAULT_VALIDATION_RUNS = [
    "HEALTHY_005",
    "HEALTHY_006",
    "CYL1_SEV020",
    "CYL3_SEV020",
    "BEARING_SEV020",
    "COOLING_SEV020",
    "LUBRICATION_SEV020",
]

# Everything else is test: HEALTHY_007..009 plus SEV040/060/080 of every family.
# The healthy test runs are what make precision, FPR and ROC-AUC meaningful -
# with an all-degraded test set there are no negatives, so precision is 1.0 and
# FPR is 0.0 by construction rather than by performance.


class RunSplitter:
    """
    Splits feature and metadata matrices strictly by run IDs.
    """

    def __init__(self, train_runs: List[str] = None, val_runs: List[str] = None):
        self.train_runs = list(train_runs) if train_runs is not None else list(DEFAULT_TRAIN_RUNS)
        self.val_runs = list(val_runs) if val_runs is not None else list(DEFAULT_VALIDATION_RUNS)
        
        # Verify no intersection between train and validation run IDs
        train_set = set(self.train_runs)
        val_set = set(self.val_runs)
        overlap = train_set.intersection(val_set)
        if overlap:
            raise ValueError(f"Train and Validation run sets overlap on: {overlap}")

    def split_dataset(
        self, X: pd.DataFrame, meta_df: pd.DataFrame
    ) -> Tuple[
        Tuple[pd.DataFrame, pd.DataFrame],
        Tuple[pd.DataFrame, pd.DataFrame],
        Tuple[pd.DataFrame, pd.DataFrame],
    ]:
        """
        Split (X, meta_df) into Train, Validation, and Test partitions.
        All runs not in Train or Validation are assigned to Test.
        """
        train_mask = meta_df["run_id"].isin(self.train_runs)
        val_mask = meta_df["run_id"].isin(self.val_runs)
        test_mask = ~(train_mask | val_mask)

        X_train, meta_train = X[train_mask].reset_index(drop=True), meta_df[train_mask].reset_index(drop=True)
        X_val, meta_val = X[val_mask].reset_index(drop=True), meta_df[val_mask].reset_index(drop=True)
        X_test, meta_test = X[test_mask].reset_index(drop=True), meta_df[test_mask].reset_index(drop=True)

        # Automated assertion check: Prove no run appears in multiple partitions
        train_rids = set(meta_train["run_id"].unique())
        val_rids = set(meta_val["run_id"].unique())
        test_rids = set(meta_test["run_id"].unique())

        assert len(train_rids.intersection(val_rids)) == 0, "Train and Validation run IDs overlap!"
        assert len(train_rids.intersection(test_rids)) == 0, "Train and Test run IDs overlap!"
        assert len(val_rids.intersection(test_rids)) == 0, "Validation and Test run IDs overlap!"

        # A single-class test set makes precision/FPR/ROC-AUC undefined. Fail loudly
        # here rather than silently reporting 1.0000 precision and 0.0000 FPR.
        if "gt_is_degraded" in meta_test.columns and len(meta_test) > 0:
            classes = set(meta_test["gt_is_degraded"].astype(bool).unique())
            assert classes == {True, False}, (
                "Test partition must contain BOTH healthy and degraded runs; "
                f"found only {classes}. Precision, FPR and ROC-AUC are undefined otherwise."
            )

        return (X_train, meta_train), (X_val, meta_val), (X_test, meta_test)
