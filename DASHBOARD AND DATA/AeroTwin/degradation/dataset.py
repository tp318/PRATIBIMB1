"""
AeroTwin-4 Dataset Builder & Sliding Window Pipeline.

Generates sliding time windows (5.0s window size, 1.0s stride default) from
simulation runs, aggregates raw telemetry features, attaches sample & run
ground-truth labels, and exports structured CSV/JSON datasets.

Strict Anti-Leakage Rule:
Ground truth parameters (combustion_efficiency, bearing_friction_multiplier, etc.)
are stored separately as ground-truth labels and are NEVER included in telemetry feature inputs.
"""

import os
import json
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple

from simulator.telemetry.schema import EngineTelemetry
from .ground_truth import RunGroundTruth, SampleGroundTruth


class SlidingWindowGenerator:
    """
    Divides continuous simulation runs into overlapping sliding time windows.
    """

    def __init__(self, window_size_seconds: float = 5.0, stride_seconds: float = 1.0, sample_rate_hz: float = 100.0):
        self.window_size_seconds = window_size_seconds
        self.stride_seconds = stride_seconds
        self.sample_rate_hz = sample_rate_hz
        self.window_samples = int(round(window_size_seconds * sample_rate_hz))
        self.stride_samples = int(round(stride_seconds * sample_rate_hz))

    def create_windows(
        self,
        telemetry_list: List[EngineTelemetry],
        gt_list: List[SampleGroundTruth],
        run_gt: RunGroundTruth
    ) -> List[Dict[str, Any]]:
        """
        Extract sliding time windows from run telemetry and ground truth lists.
        """
        n_samples = len(telemetry_list)
        if n_samples < self.window_samples:
            return []

        df_telem = pd.DataFrame([t.to_dict() for t in telemetry_list])
        df_gt = pd.DataFrame([g.to_dict() for g in gt_list])

        windows = []
        start_idx = 0
        window_id = 0

        # Numeric channels for statistical aggregation
        numeric_cols = [
            "throttle", "rpm", "crank_angle", "mean_torque", "instant_torque",
            "load_torque", "friction_torque", "net_torque",
            "cylinder_1_torque", "cylinder_2_torque", "cylinder_3_torque", "cylinder_4_torque",
            "cht", "egt", "oil_temperature", "oil_pressure", "oil_pressure_psi",
            "fuel_flow", "fuel_flow_lph", "fuel_pressure", "vibration"
        ]

        while start_idx + self.window_samples <= n_samples:
            end_idx = start_idx + self.window_samples
            w_telem = df_telem.iloc[start_idx:end_idx]
            w_gt = df_gt.iloc[start_idx:end_idx]

            # Aggregate window telemetry features
            feature_dict = {
                "window_id": f"{run_gt.run_id}_W{window_id:04d}",
                "run_id": run_gt.run_id,
                "window_start_sim_time": float(w_telem["simulation_time"].iloc[0]),
                "window_end_sim_time": float(w_telem["simulation_time"].iloc[-1]),
                "operating_mode": str(w_telem["operating_mode"].iloc[-1]),
            }

            for col in numeric_cols:
                if col in w_telem.columns:
                    vals = w_telem[col].values
                    feature_dict[f"{col}_mean"] = float(np.mean(vals))
                    feature_dict[f"{col}_std"] = float(np.std(vals))
                    feature_dict[f"{col}_min"] = float(np.min(vals))
                    feature_dict[f"{col}_max"] = float(np.max(vals))
                    if col in ["instant_torque", "vibration"]:
                        feature_dict[f"{col}_rms"] = float(np.sqrt(np.mean(vals ** 2)))
                        feature_dict[f"{col}_peak"] = float(np.max(np.abs(vals)))

            # Attach ground truth label corresponding to the window end state
            feature_dict["degradation_type"] = run_gt.degradation_type
            feature_dict["target_component"] = run_gt.target_component
            feature_dict["active_severity"] = float(w_gt["active_severity"].iloc[-1])
            feature_dict["current_health"] = float(w_gt["current_health"].iloc[-1])
            feature_dict["is_degraded"] = bool(w_gt["is_degraded"].iloc[-1])

            windows.append(feature_dict)
            start_idx += self.stride_samples
            window_id += 1

        return windows


class DatasetBuilder:
    """
    Manages dataset structure creation, windowing, and CSV/JSON metadata exporting.
    """

    def __init__(self, output_dir: str):
        self.output_dir = os.path.abspath(output_dir)
        self.window_gen = SlidingWindowGenerator()

    def export_run_dataset(
        self,
        telemetry_list: List[EngineTelemetry],
        gt_list: List[SampleGroundTruth],
        run_gt: RunGroundTruth,
        subfolder: str = "healthy"
    ) -> Tuple[str, str]:
        """
        Export raw telemetry CSV and sliding window CSV for a single run.
        """
        target_dir = os.path.join(self.output_dir, subfolder)
        os.makedirs(target_dir, exist_ok=True)

        # 1. Export raw sample-level telemetry with sample ground truth
        raw_rows = []
        for t, g in zip(telemetry_list, gt_list):
            r = t.to_dict()
            r["gt_degradation_type"] = g.degradation_type
            r["gt_target_component"] = g.target_component
            r["gt_active_severity"] = g.active_severity
            r["gt_current_health"] = g.current_health
            r["gt_is_degraded"] = g.is_degraded
            raw_rows.append(r)

        df_raw = pd.DataFrame(raw_rows)
        raw_csv_path = os.path.join(target_dir, f"{run_gt.run_id}_raw.csv")
        df_raw.to_csv(raw_csv_path, index=False)

        # 2. Export sliding window dataset
        windows = self.window_gen.create_windows(telemetry_list, gt_list, run_gt)
        window_dir = os.path.join(self.output_dir, "windows")
        os.makedirs(window_dir, exist_ok=True)

        df_win = pd.DataFrame(windows)
        win_csv_path = os.path.join(window_dir, f"{run_gt.run_id}_windows.csv")
        df_win.to_csv(win_csv_path, index=False)

        # 3. Export run metadata JSON
        meta_path = os.path.join(target_dir, f"{run_gt.run_id}_metadata.json")
        meta_dict = run_gt.to_dict()
        meta_dict["total_raw_samples"] = len(telemetry_list)
        meta_dict["total_window_samples"] = len(windows)
        with open(meta_path, "w") as f:
            json.dump(meta_dict, f, indent=2)

        return raw_csv_path, win_csv_path
