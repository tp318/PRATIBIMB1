"""
AeroTwin-4 Physics Causality & Dataset Validator.

Validates dataset integrity, absence of NaN/Inf, non-leakage partitioning,
and verifies that physical degradation causes expected causal sensor trends.
"""

import math
import numpy as np
import pandas as pd
from typing import Dict, Any, List


class DatasetValidator:
    """
    Validates generated Phase 3 datasets for numerical integrity, non-leakage partitioning,
    and physical causality relationships.
    """

    @staticmethod
    def validate_numerical_integrity(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Check dataframe for NaN, Infinity, or missing values.
        """
        nan_count = int(df.isna().sum().sum())
        inf_count = int(np.isinf(df.select_dtypes(include=[np.number])).sum().sum())

        # Check non-negative constraints ONLY for physically non-negative channels
        non_neg_cols = ["rpm", "cht", "egt", "oil_pressure", "oil_pressure_psi", "fuel_flow", "fuel_flow_lph", "vibration"]
        neg_violations = {}
        for col in non_neg_cols:
            if col in df.columns:
                viol = int((df[col] < -1e-6).sum())
                if viol > 0:
                    neg_violations[col] = viol

        is_valid = (nan_count == 0) and (inf_count == 0) and (len(neg_violations) == 0)

        return {
            "is_valid": is_valid,
            "nan_count": nan_count,
            "inf_count": inf_count,
            "negative_violations": neg_violations,
            "total_rows": len(df),
        }

    @staticmethod
    def validate_physical_causality(
        healthy_df: pd.DataFrame,
        degraded_df: pd.DataFrame,
        degradation_type: str,
        target_component: str = "NONE"
    ) -> Dict[str, Any]:
        """
        Verify physical causality trends between healthy and degraded simulation runs.

        Physical Expected Causality:
        - Cylinder (D1): Degradation reduces target cylinder torque pulse and increases vibration.
        - Bearing (D2): Degradation increases mechanical friction torque.
        - Cooling (D3): Degradation increases steady-state CHT under power.
        - Lubrication (D4): Degradation reduces oil pressure and increases friction.
        """
        results = {"degradation_type": degradation_type, "passed": True, "causal_effects": {}}

        if degradation_type == "CYLINDER":
            cyl_map = {
                "CYLINDER_1": "cylinder_1_torque",
                "CYLINDER_2": "cylinder_2_torque",
                "CYLINDER_3": "cylinder_3_torque",
                "CYLINDER_4": "cylinder_4_torque",
            }
            col = cyl_map.get(target_component, "cylinder_3_torque")
            if col in healthy_df.columns and col in degraded_df.columns:
                h_peak = np.max(healthy_df[col])
                d_peak = np.max(degraded_df[col])
                torq_drop = bool(d_peak < h_peak * 0.95)
                results["causal_effects"]["cylinder_torque_reduced"] = torq_drop
                results["passed"] = torq_drop

        elif degradation_type == "BEARING":
            h_fric = np.mean(healthy_df["friction_torque"])
            d_fric = np.mean(degraded_df["friction_torque"])
            fric_increase = bool(d_fric > h_fric * 1.05)
            results["causal_effects"]["friction_increased"] = fric_increase
            results["passed"] = fric_increase

        elif degradation_type == "COOLING":
            h_cht = np.mean(healthy_df["cht"])
            d_cht = np.mean(degraded_df["cht"])
            cht_increase = bool(d_cht > h_cht + 2.0)
            results["causal_effects"]["cht_increased"] = cht_increase
            results["passed"] = cht_increase

        elif degradation_type == "LUBRICATION":
            h_press = np.mean(healthy_df["oil_pressure"])
            d_press = np.mean(degraded_df["oil_pressure"])
            press_drop = bool(d_press < h_press * 0.95)
            results["causal_effects"]["oil_pressure_dropped"] = press_drop
            results["passed"] = press_drop

        return results

    @staticmethod
    def validate_non_leakage_partitioning(train_runs: List[str], val_runs: List[str], test_runs: List[str]) -> bool:
        """
        Ensure zero overlap of run IDs between train, validation, and test splits.
        """
        s_train = set(train_runs)
        s_val = set(val_runs)
        s_test = set(test_runs)

        overlap_tv = s_train.intersection(s_val)
        overlap_tt = s_train.intersection(s_test)
        overlap_vt = s_val.intersection(s_test)

        return len(overlap_tv) == 0 and len(overlap_tt) == 0 and len(overlap_vt) == 0
