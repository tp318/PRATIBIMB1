"""
AeroTwin-4 Fault Diagnosis Splits.

Anomaly detection is unsupervised and trains on healthy data only. Diagnosis is
supervised and must see faults during training, so it needs a different split
discipline.

The split here holds out the HIGHEST severity of every fault family. A classifier
that only works when it has already seen that exact severity is useless in flight;
holding out SEV080 asks the harder and more honest question - does the fault
SIGNATURE generalise across severity?
"""

from typing import Dict, List, Tuple

import pandas as pd

# Fault families the diagnoser must discriminate between.
FAULT_CLASSES = ["HEALTHY", "CYLINDER", "BEARING", "COOLING", "LUBRICATION"]

# Healthy runs are split so every partition carries the negative class.
DIAG_TRAIN_HEALTHY = ["HEALTHY_001", "HEALTHY_002", "HEALTHY_003", "HEALTHY_004"]
DIAG_VAL_HEALTHY = ["HEALTHY_005", "HEALTHY_006"]
DIAG_TEST_HEALTHY = ["HEALTHY_007", "HEALTHY_008", "HEALTHY_009"]

# Severity-based fault split: train on the mild/moderate band, validate on SEV060,
# and hold out SEV080 entirely as an unseen-severity generalisation test.
DIAG_TRAIN_SEVERITIES = ["SEV020", "SEV040"]
DIAG_VAL_SEVERITIES = ["SEV060"]
DIAG_TEST_SEVERITIES = ["SEV080"]


def label_from_run_id(run_id: str) -> str:
    """Map a run id to its fault family label."""
    rid = run_id.upper()
    if rid.startswith("HEALTHY"):
        return "HEALTHY"
    if rid.startswith("CYL"):
        return "CYLINDER"
    for fam in ("BEARING", "COOLING", "LUBRICATION"):
        if rid.startswith(fam):
            return fam
    raise ValueError(f"Cannot derive fault family from run id: {run_id!r}")


def severity_tag(run_id: str) -> str:
    """Extract the SEVxxx tag from a run id, or '' for healthy runs."""
    for part in run_id.upper().split("_"):
        if part.startswith("SEV"):
            return part
    return ""


class DiagnosisSplitter:
    """
    Partitions a labelled dataset into train / validation / test by RUN, never by
    window, so no window of a run can appear in two partitions.
    """

    def __init__(
        self,
        train_healthy: List[str] = None,
        val_healthy: List[str] = None,
        test_healthy: List[str] = None,
        train_severities: List[str] = None,
        val_severities: List[str] = None,
        test_severities: List[str] = None,
    ):
        self.train_healthy = list(train_healthy or DIAG_TRAIN_HEALTHY)
        self.val_healthy = list(val_healthy or DIAG_VAL_HEALTHY)
        self.test_healthy = list(test_healthy or DIAG_TEST_HEALTHY)
        self.train_severities = list(train_severities or DIAG_TRAIN_SEVERITIES)
        self.val_severities = list(val_severities or DIAG_VAL_SEVERITIES)
        self.test_severities = list(test_severities or DIAG_TEST_SEVERITIES)

        for a, b, what in [
            (self.train_healthy, self.val_healthy, "healthy train/val"),
            (self.train_healthy, self.test_healthy, "healthy train/test"),
            (self.val_healthy, self.test_healthy, "healthy val/test"),
            (self.train_severities, self.val_severities, "severity train/val"),
            (self.train_severities, self.test_severities, "severity train/test"),
            (self.val_severities, self.test_severities, "severity val/test"),
        ]:
            overlap = set(a) & set(b)
            if overlap:
                raise ValueError(f"{what} sets overlap on: {overlap}")

    def _partition_of(self, run_id: str) -> str:
        if label_from_run_id(run_id) == "HEALTHY":
            if run_id in self.train_healthy:
                return "train"
            if run_id in self.val_healthy:
                return "val"
            if run_id in self.test_healthy:
                return "test"
            return "unused"

        sev = severity_tag(run_id)
        if sev in self.train_severities:
            return "train"
        if sev in self.val_severities:
            return "val"
        if sev in self.test_severities:
            return "test"
        return "unused"

    def split(
        self, X: pd.DataFrame, meta_df: pd.DataFrame
    ) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame, pd.Series]]:
        """
        Returns {partition: (X_part, meta_part, y_part)} where y is the fault family.
        """
        parts = meta_df["run_id"].map(self._partition_of)
        labels = meta_df["run_id"].map(label_from_run_id)

        out = {}
        for name in ("train", "val", "test"):
            mask = (parts == name).to_numpy()
            out[name] = (
                X[mask].reset_index(drop=True),
                meta_df[mask].reset_index(drop=True),
                labels[mask].reset_index(drop=True),
            )

        run_sets = {n: set(out[n][1]["run_id"].unique()) for n in out}
        assert not (run_sets["train"] & run_sets["val"]), "Diagnosis train/val run leakage"
        assert not (run_sets["train"] & run_sets["test"]), "Diagnosis train/test run leakage"
        assert not (run_sets["val"] & run_sets["test"]), "Diagnosis val/test run leakage"

        # A diagnoser that never saw a class cannot be asked to predict it.
        train_classes = set(out["train"][2].unique())
        missing = set(FAULT_CLASSES) - train_classes
        assert not missing, f"Training partition is missing fault classes: {missing}"

        return out
