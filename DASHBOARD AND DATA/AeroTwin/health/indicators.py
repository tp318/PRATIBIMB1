"""
AeroTwin-4 Residual Indicator Engine.

Derives interpretable, physical health-related indicators from residual states.
Note: These are residual-derived indicator signals, NOT a final health score or fault diagnosis.
"""

import numpy as np
from typing import Dict, Any, List, Optional
from .models import ResidualState, ResidualIndicators, ExpectedState


class ResidualIndicatorEngine:
    """
    Computes health-related indicator signals and 4-cylinder torque balance metrics.
    """

    @staticmethod
    def calculate_cylinder_balance(torques: List[float]) -> float:
        """
        Calculate 4-cylinder torque balance ratio:
        std(cylinder_torques) / max(1.0, mean(cylinder_torques))
        Handles zero or near-zero mean safely.
        """
        if not torques or len(torques) == 0:
            return 0.0

        arr = np.array(torques, dtype=float)
        mean_val = float(np.mean(arr))
        std_val = float(np.std(arr))

        denom = max(1.0, abs(mean_val))
        return float(std_val / denom)

    def process_frame(self, residual_state: ResidualState, observed_torques: Optional[List[float]] = None) -> ResidualIndicators:
        """
        Derive point-in-time residual indicators from ResidualState.
        """
        norm = residual_state.normalized

        # 1. Thermal Deviation (CHT & EGT residual magnitude)
        cht_norm = norm.get("cht", 0.0)
        egt_norm = norm.get("egt", 0.0)
        thermal_dev = float(max(abs(cht_norm), abs(egt_norm)))

        # 2. Oil System Deviation (Oil pressure drop & Oil temp elevation)
        oil_p_norm = norm.get("oil_pressure", 0.0)
        oil_t_norm = norm.get("oil_temperature", 0.0)
        oil_dev = float(max(abs(oil_p_norm), abs(oil_t_norm)))

        # 3. Vibration Deviation
        vib_dev = float(abs(norm.get("vibration", 0.0)))

        # 4. Torque / Friction Deviation
        fric_norm = norm.get("friction_torque", 0.0)
        mean_t_norm = norm.get("mean_torque", 0.0)
        torque_dev = float(max(abs(fric_norm), abs(mean_t_norm)))

        # 5. Cylinder Balance Indicator
        if observed_torques:
            cyl_bal = self.calculate_cylinder_balance(observed_torques)
        else:
            c_torques = [
                norm.get("cylinder_1_torque", 0.0),
                norm.get("cylinder_2_torque", 0.0),
                norm.get("cylinder_3_torque", 0.0),
                norm.get("cylinder_4_torque", 0.0),
            ]
            cyl_bal = float(np.std(c_torques))

        return ResidualIndicators(
            thermal_deviation=thermal_dev,
            oil_deviation=oil_dev,
            vibration_deviation=vib_dev,
            torque_deviation=torque_dev,
            cylinder_balance_deviation=cyl_bal,
        )

    def process_window(self, residual_states: List[ResidualState]) -> Dict[str, Dict[str, float]]:
        """
        Compute sliding-window residual statistics (mean, std, RMS, peak, trend).
        """
        if not residual_states:
            return {}

        metrics = {}
        all_channels = residual_states[0].raw_signed.keys()

        for ch in all_channels:
            vals = np.array([r.raw_signed.get(ch, 0.0) for r in residual_states], dtype=float)
            mean_v = float(np.mean(vals))
            std_v = float(np.std(vals))
            rms_v = float(np.sqrt(np.mean(vals ** 2)))
            peak_v = float(np.max(np.abs(vals)))

            # Trend estimation via simple linear regression slope over sample indices
            if len(vals) > 1:
                x = np.arange(len(vals), dtype=float)
                slope = float(np.polyfit(x, vals, 1)[0])
            else:
                slope = 0.0

            metrics[ch] = {
                "mean": mean_v,
                "std": std_v,
                "rms": rms_v,
                "peak": peak_v,
                "trend": slope,
            }

        return metrics
