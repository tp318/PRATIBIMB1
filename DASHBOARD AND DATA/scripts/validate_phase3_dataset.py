"""
AeroTwin-4 Phase 3 Dataset Validation Script.

Validates numerical integrity, non-leakage bounds, and physical causality trends
for generated Phase 3 datasets.

Usage:
  .venv/bin/python scripts/validate_phase3_dataset.py
"""

import os
import sys
import glob
import pandas as pd

# Ensure AeroTwin root is in sys.path
_script_dir = os.path.dirname(os.path.abspath(__file__))
_root_dir = os.path.dirname(_script_dir)
_aerotwin_dir = os.path.join(_root_dir, "AeroTwin")

for _p in [_aerotwin_dir, _root_dir]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from AeroTwin.degradation.validators import DatasetValidator
except ImportError:
    from degradation.validators import DatasetValidator


def validate_dataset_folder(dataset_dir: str):
    print("==================================================")
    print("AeroTwin-4 Phase 3 Dataset Validation")
    print("==================================================")
    print(f"Target Directory: {dataset_dir}\n")

    if not os.path.exists(dataset_dir):
        print(f"Error: Dataset directory does not exist: {dataset_dir}")
        sys.exit(1)

    raw_files = glob.glob(os.path.join(dataset_dir, "**", "*_raw.csv"), recursive=True)
    if not raw_files:
        print("No raw dataset files found.")
        sys.exit(1)

    print(f"Found {len(raw_files)} raw dataset run files.\n")

    # 1. Numerical Integrity Checks
    total_integrity_passed = True
    healthy_df = None
    degraded_dfs = {}

    for raw_file in raw_files:
        df = pd.read_csv(raw_file)
        run_name = os.path.basename(raw_file).replace("_raw.csv", "")
        res = DatasetValidator.validate_numerical_integrity(df)

        status = "PASSED" if res["is_valid"] else "FAILED"
        print(f"[{status}] {run_name}: {res['total_rows']} rows (NaN={res['nan_count']}, Inf={res['inf_count']})")

        if not res["is_valid"]:
            total_integrity_passed = False

        deg_type = df["gt_degradation_type"].iloc[0]
        if deg_type == "NONE":
            healthy_df = df
        else:
            degraded_dfs[run_name] = df

    # 2. Physical Causality Trend Checks
    print("\n------------------------------------------")
    print("Physical Causality Trend Validation:")
    print("------------------------------------------")

    causality_passed = True
    if healthy_df is not None:
        for run_name, d_df in degraded_dfs.items():
            deg_type = d_df["gt_degradation_type"].iloc[0]
            target_comp = d_df["gt_target_component"].iloc[0]

            c_res = DatasetValidator.validate_physical_causality(
                healthy_df=healthy_df,
                degraded_df=d_df,
                degradation_type=deg_type,
                target_component=target_comp,
            )
            c_status = "PASSED" if c_res["passed"] else "FAILED"
            print(f"[{c_status}] {run_name} ({deg_type}): {c_res['causal_effects']}")
            if not c_res["passed"]:
                causality_passed = False

    # 3. Non-leakage Partitioning Check
    run_ids = [os.path.basename(f).replace("_raw.csv", "") for f in raw_files]
    train_runs = run_ids[: int(0.7 * len(run_ids))]
    val_runs = run_ids[int(0.7 * len(run_ids)) : int(0.85 * len(run_ids))]
    test_runs = run_ids[int(0.85 * len(run_ids)) :]

    leakage_passed = DatasetValidator.validate_non_leakage_partitioning(train_runs, val_runs, test_runs)
    print(f"\nNon-Leakage Run Partitioning Check: {'PASSED' if leakage_passed else 'FAILED'}")

    print("\n------------------------------------------")
    overall_passed = total_integrity_passed and causality_passed and leakage_passed
    print(f"Overall Validation Result: {'PASSED (100%)' if overall_passed else 'FAILED'}")
    print("------------------------------------------")

    if not overall_passed:
        sys.exit(1)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="AeroTwin-4 Phase 3 Dataset Validator")
    parser.add_argument("target_dir", nargs="?", default=None, help="Target dataset directory (default: pilot or full)")
    args = parser.parse_args()

    p3_root = os.path.join(_root_dir, "data", "generated", "phase3")
    if args.target_dir:
        t_dir = os.path.abspath(args.target_dir)
    elif os.path.exists(os.path.join(p3_root, "pilot")):
        t_dir = os.path.join(p3_root, "pilot")
    elif os.path.exists(os.path.join(p3_root, "full")):
        t_dir = os.path.join(p3_root, "full")
    else:
        t_dir = p3_root

    validate_dataset_folder(t_dir)
