"""
windows.py
==========
Sequence construction for the temporal models (Part 16).

SAMPLING AND WINDOW GEOMETRY
----------------------------
    snapshot cadence   0.25 operating hours (15 min)  - the monitoring rate
    window length      60 snapshots = 15 operating hours (default)
    stride (train)     6 snapshots  = 1.5 operating hours
    stride (val/test) 12 snapshots  = 3.0 operating hours

WHY 60
------
The window has to be long enough to see a degradation trend through the noise
and short enough that the operating point has not changed character completely.
The relevant timescales are:

  * a sortie is 4-18 h, so a 15 h window typically spans one to three sorties and
    therefore several different mission phases. That matters directly: the
    identifiability analysis showed a single operating point is badly
    conditioned (descent alone: condition number 1107) while a mix of conditions
    is well conditioned (16.2). A window that spans several phases is the
    temporal equivalent of that excitation.

  * the residual slope features are computed over 60 snapshots, so a shorter
    window would carry slope features that see further back than the window
    itself - the model would be reading information it cannot justify.

  * the fastest engines in the fleet live ~90 h. A 30 h window would be a third
    of their entire life and could not resolve a trend within them.

30 and 120 are both evaluated in scripts/train_gru.py rather than assumed.

OVERLAP
-------
Stride 6 on a 60-snapshot window is 90% overlap, which sounds excessive, but the
overlap concern is about correlated samples spanning a TRAIN/TEST boundary, and
that cannot happen here because splits are by whole run. Within the training set
overlap is a form of augmentation, not leakage. It does inflate the apparent
sample count, so the effective sample size is reported as the number of RUNS
everywhere in the evaluation, and val/test use a coarser stride so their metrics
are not dominated by near-duplicate windows.

Every window lies entirely inside ONE run. Windows never span a run boundary.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from ..config import HEALTH_PARAMS, SEQ_LEN_DEFAULT, STRIDE_BY_SPLIT

# --------------------------------------------------------------------------- #
# Feature groups. Every one of these is REAL_ENGINE or DIGITAL_TWIN in the
# feature manifest - no GROUND_TRUTH column ever appears here.
# --------------------------------------------------------------------------- #

SENSOR_FEATURES = [
    "rpm", "cht", "egt", "oil_pressure", "oil_temperature", "fuel_flow",
    "battery_voltage", "alternator_current", "manifold_pressure",
    "vibration_rms", "vibration_kurtosis", "vibration_crest_factor",
    "vibration_1x", "vibration_2x", "vibration_3x", "vibration_half_x",
    "vibration_dominant_freq",
]

CONTEXT_FEATURES = [
    "altitude_ft", "ambient_temperature_c", "ambient_pressure_kpa", "humidity",
    "throttle", "engine_load", "injection_command", "injection_timing_deg",
    "load_factor", "airspeed_factor", "operating_hours",
]

WINDOW_STAT_FEATURES = [
    "rpm_window_std", "cht_window_std", "egt_window_std",
    "oil_pressure_window_std", "fuel_flow_window_std",
]

Z_FEATURES = [
    "rpm_z", "cht_z", "egt_z", "oil_pressure_z", "oil_temperature_z",
    "fuel_flow_z", "vibration_rms_z", "manifold_pressure_z",
    "injection_command_z",
]

Z_TREND_FEATURES = [f"{c}_slope" for c in Z_FEATURES] + \
                   [f"{c}_roll160" for c in Z_FEATURES] + \
                   ["z_absmax", "z_sq_sum_roll"]

TWIN_FEATURES = ["rpm_pred", "cht_pred", "egt_pred", "oil_pressure_pred",
                 "oil_temperature_pred", "fuel_flow_pred", "vibration_pred",
                 "torque_pred", "power_pred_w", "bsfc_pred"]

HEALTH_EST_FEATURES = [f"est_{k}" for k in HEALTH_PARAMS] + \
                      [f"est_{k}_std" for k in HEALTH_PARAMS]

DEGRADATION_FEATURES = [f"phi_{k}" for k in HEALTH_PARAMS] + \
                       [f"stress_{k}" for k in HEALTH_PARAMS] + \
                       [f"rate_phys_{k}" for k in HEALTH_PARAMS] + \
                       [f"a_{k}" for k in HEALTH_PARAMS]

# Feature sets used by the different architectures being compared.
FEATURE_SETS: Dict[str, List[str]] = {
    # Baseline 1 and 2: raw telemetry only, no twin, no filter.
    "telemetry": SENSOR_FEATURES + CONTEXT_FEATURES + WINDOW_STAT_FEATURES,
    # Baseline 3: estimated health only.
    "health_only": HEALTH_EST_FEATURES,
    # Proposed hybrid: everything the physics pipeline produces.
    "hybrid": (SENSOR_FEATURES + CONTEXT_FEATURES + WINDOW_STAT_FEATURES
               + Z_FEATURES + Z_TREND_FEATURES + TWIN_FEATURES
               + HEALTH_EST_FEATURES + DEGRADATION_FEATURES),
    # Residual-centric view used by the degradation-rate GRU.
    "rate_correction": (Z_FEATURES + Z_TREND_FEATURES + HEALTH_EST_FEATURES
                        + DEGRADATION_FEATURES + CONTEXT_FEATURES
                        + ["cht", "egt", "oil_temperature", "oil_pressure",
                           "rpm", "fuel_flow", "vibration_rms",
                           "power_pred_w", "bsfc_pred"]),
}


def available_features(df: pd.DataFrame, names: Sequence[str]) -> List[str]:
    return [c for c in names if c in df.columns]


# --------------------------------------------------------------------------- #
# Window index construction
# --------------------------------------------------------------------------- #

def window_index(df: pd.DataFrame, seq_len: int = SEQ_LEN_DEFAULT,
                 stride: int = 6, min_start: int | None = None) -> np.ndarray:
    """End-positions of every valid window, as absolute row offsets in `df`.

    `df` may contain many runs; windows are built per run and never cross a run
    boundary. A window ending at row e covers rows [e - seq_len + 1, e].

    `min_start` skips the first N snapshots of each run. The rolling residual
    features need a warm-up before they mean anything, and the UKF needs to
    settle; including windows built from unconverged features would teach the
    model to trust garbage.
    """
    min_start = seq_len if min_start is None else max(min_start, seq_len)
    ends: List[int] = []
    offset = 0
    for _, grp in df.groupby("run_id", sort=False):
        n = len(grp)
        if n >= min_start:
            e = np.arange(min_start - 1, n, stride, dtype=np.int64)
            ends.append(e + offset)
        offset += n
    return np.concatenate(ends) if ends else np.zeros(0, dtype=np.int64)


def build_sequences(df: pd.DataFrame, feature_cols: Sequence[str],
                    target_cols: Sequence[str],
                    seq_len: int = SEQ_LEN_DEFAULT,
                    stride: int = 6,
                    min_start: int | None = None
                    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Materialise (X, y, run_ids, end_rows).

    X is (N, seq_len, F) float32. Sequences are cut with a strided view over the
    contiguous feature matrix, which keeps memory sane: at 200k rows and 100
    features the underlying matrix is 80 MB and the windows are views into it
    until the final copy.
    """
    df = df.reset_index(drop=True)
    feats = np.ascontiguousarray(df[list(feature_cols)].to_numpy(dtype=np.float32))
    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
    targ = df[list(target_cols)].to_numpy(dtype=np.float32)
    runs = df["run_id"].to_numpy()

    ends = window_index(df, seq_len=seq_len, stride=stride, min_start=min_start)
    if len(ends) == 0:
        F = len(feature_cols)
        return (np.zeros((0, seq_len, F), np.float32),
                np.zeros((0, len(target_cols)), np.float32),
                np.zeros(0, dtype=object), np.zeros(0, dtype=np.int64))

    starts = ends - seq_len + 1
    idx = starts[:, None] + np.arange(seq_len)[None, :]
    X = feats[idx]
    y = targ[ends]
    return X, y, runs[ends], ends


class StandardScaler3D:
    """Per-feature standardisation fitted on TRAINING windows only."""

    def __init__(self):
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "StandardScaler3D":
        flat = X.reshape(-1, X.shape[-1])
        self.mean_ = flat.mean(axis=0)
        s = flat.std(axis=0)
        # A constant feature gets scale 1 rather than exploding to infinity.
        self.scale_ = np.where(s < 1e-8, 1.0, s)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return ((X - self.mean_) / self.scale_).astype(np.float32)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def to_dict(self) -> Dict:
        return {"mean": self.mean_.tolist(), "scale": self.scale_.tolist()}

    @classmethod
    def from_dict(cls, d: Dict) -> "StandardScaler3D":
        s = cls()
        s.mean_ = np.asarray(d["mean"], dtype=np.float32)
        s.scale_ = np.asarray(d["scale"], dtype=np.float32)
        return s
