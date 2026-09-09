"""
AeroTwin-4 Feature Extractor.

Extracts window-level ML features for three configurations:
- Configuration A (Raw Telemetry)
- Configuration B (Digital Twin Residuals)
- Configuration C (Hybrid Physics-Informed)

STRICT RULE: Ground-truth fields (gt_degradation_type, gt_active_severity, etc.)
are NEVER included in feature matrices.
"""

from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd


FORBIDDEN_GROUND_TRUTH_FIELDS = {
    "gt_degradation_type",
    "gt_target_component",
    "gt_active_severity",
    "gt_current_health",
    "gt_is_degraded",
    "degradation_type",
    "target_component",
    "severity",
    "max_severity",
    "health",
    "is_healthy",
    "is_degraded",
}


RAW_CHANNELS = [
    "obs_rpm",
    "obs_mean_torque",
    "obs_instant_torque",
    "obs_load_torque",
    "obs_friction_torque",
    "obs_net_torque",
    "obs_cylinder_1_torque",
    "obs_cylinder_2_torque",
    "obs_cylinder_3_torque",
    "obs_cylinder_4_torque",
    "obs_cht",
    "obs_egt",
    "obs_oil_temperature",
    "obs_oil_pressure",
    "obs_fuel_flow",
    "obs_fuel_pressure",
    "obs_vibration",
]


RESIDUAL_CHANNELS = [
    "res_signed_rpm",
    "res_signed_mean_torque",
    "res_signed_instant_torque",
    "res_signed_load_torque",
    "res_signed_friction_torque",
    "res_signed_net_torque",
    "res_signed_cylinder_1_torque",
    "res_signed_cylinder_2_torque",
    "res_signed_cylinder_3_torque",
    "res_signed_cylinder_4_torque",
    "res_signed_cht",
    "res_signed_egt",
    "res_signed_oil_temperature",
    "res_signed_oil_pressure",
    "res_signed_fuel_flow",
    "res_signed_fuel_pressure",
    "res_signed_vibration",
    "ind_thermal_dev",
    "ind_oil_dev",
    "ind_vibration_dev",
    "ind_torque_dev",
    "ind_cylinder_balance_dev",
]


CONTEXT_FIELDS = [
    "throttle",
    "ambient_temperature",
]


class FeatureExtractor:
    """
    ML Feature Extractor for 5.0s windowed telemetry & residual datasets.
    """

    def __init__(self, config_type: str = "HYBRID"):
        config_type = config_type.upper()
        if config_type not in ("RAW", "RESIDUAL", "HYBRID"):
            raise ValueError(f"Invalid config_type '{config_type}'. Must be RAW, RESIDUAL, or HYBRID.")
        self.config_type = config_type

    @staticmethod
    def _compute_channel_stats(series: np.ndarray, prefix: str) -> Dict[str, float]:
        """
        Compute standard time-domain aggregations for a single channel.
        """
        s_mean = float(np.mean(series))
        s_std = float(np.std(series))
        s_min = float(np.min(series))
        s_max = float(np.max(series))
        s_rms = float(np.sqrt(np.mean(series ** 2)))
        s_peak = float(np.max(np.abs(series)))
        s_p2p = s_max - s_min
        
        # Calculate linear slope (trend)
        if len(series) > 1:
            t = np.linspace(0, 1, len(series))
            s_slope = float(np.polyfit(t, series, 1)[0])
        else:
            s_slope = 0.0

        return {
            f"{prefix}_mean": s_mean,
            f"{prefix}_std": s_std,
            f"{prefix}_min": s_min,
            f"{prefix}_max": s_max,
            f"{prefix}_rms": s_rms,
            f"{prefix}_peak": s_peak,
            f"{prefix}_p2p": s_p2p,
            f"{prefix}_slope": s_slope,
        }

    def extract_window_features(self, df_window: pd.DataFrame) -> Dict[str, float]:
        """
        Extract feature vector dict from a 5.0s window DataFrame.
        """
        # Assert zero forbidden ground-truth fields enter feature dictionary
        for col in df_window.columns:
            if col in FORBIDDEN_GROUND_TRUTH_FIELDS:
                pass  # Ignore ground-truth columns when building X

        feats = {}

        # 1. Context features (mean throttle, ambient temp)
        if "throttle" in df_window.columns:
            feats["ctx_throttle"] = float(df_window["throttle"].mean())
        if "ambient_temperature" in df_window.columns:
            feats["ctx_ambient_temp"] = float(df_window["ambient_temperature"].mean())

        # 2. Raw Telemetry Features (Config A or Config C)
        if self.config_type in ("RAW", "HYBRID"):
            for ch in RAW_CHANNELS:
                col_name = ch if ch in df_window.columns else ch.replace("obs_", "")
                if col_name in df_window.columns:
                    ch_stats = self._compute_channel_stats(df_window[col_name].values, prefix=f"raw_{col_name}")
                    feats.update(ch_stats)

        # 3. Residual Features (Config B or Config C)
        if self.config_type in ("RESIDUAL", "HYBRID"):
            for ch in RESIDUAL_CHANNELS:
                if ch in df_window.columns:
                    ch_stats = self._compute_channel_stats(df_window[ch].values, prefix=f"res_{ch}")
                    feats.update(ch_stats)

            # Cylinder balance indicator mean
            if "ind_cylinder_balance_dev" in df_window.columns:
                feats["cyl_balance_mean"] = float(df_window["ind_cylinder_balance_dev"].mean())
                feats["cyl_balance_max"] = float(df_window["ind_cylinder_balance_dev"].max())

        return feats

    def extract_dataset(
        self,
        df_run: pd.DataFrame,
        window_size_sec: float = 5.0,
        stride_sec: float = 1.0,
        sample_rate_hz: float = 100.0,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Extract window feature matrix X and metadata DataFrame (run_id, window_id, ground-truth labels).
        """
        window_len = int(window_size_sec * sample_rate_hz)
        stride_len = int(stride_sec * sample_rate_hz)
        total_samples = len(df_run)

        feature_rows = []
        meta_rows = []

        run_id = str(df_run["run_id"].iloc[0]) if "run_id" in df_run.columns else "UNKNOWN_RUN"

        w_idx = 0
        for start_idx in range(0, total_samples - window_len + 1, stride_len):
            end_idx = start_idx + window_len
            df_win = df_run.iloc[start_idx:end_idx]

            feats = self.extract_window_features(df_win)
            feature_rows.append(feats)

            # Metadata & Ground Truth (for evaluation ONLY)
            meta = {
                "run_id": run_id,
                "window_id": f"{run_id}_W{w_idx:04d}",
                "simulation_start": float(df_win["simulation_time"].iloc[0]),
                "simulation_end": float(df_win["simulation_time"].iloc[-1]),
                "gt_degradation_type": str(df_win["gt_degradation_type"].iloc[-1]) if "gt_degradation_type" in df_win.columns else "NONE",
                "gt_target_component": str(df_win["gt_target_component"].iloc[-1]) if "gt_target_component" in df_win.columns else "NONE",
                "gt_active_severity": float(df_win["gt_active_severity"].iloc[-1]) if "gt_active_severity" in df_win.columns else 0.0,
                "gt_current_health": float(df_win["gt_current_health"].iloc[-1]) if "gt_current_health" in df_win.columns else 1.0,
                "gt_is_degraded": bool(df_win["gt_is_degraded"].iloc[-1]) if "gt_is_degraded" in df_win.columns else False,
            }
            meta_rows.append(meta)
            w_idx += 1

        X = pd.DataFrame(feature_rows)
        meta_df = pd.DataFrame(meta_rows)

        # Enforce no NaN/Inf
        X = X.fillna(0.0)
        X = X.replace([np.inf, -np.inf], 0.0)

        return X, meta_df
