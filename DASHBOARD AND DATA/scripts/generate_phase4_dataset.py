"""
AeroTwin-4 Phase 4 Derived Residual Dataset Generation Script.

Processes Phase 3 simulation telemetry through DigitalTwinStateEngine,
computes counterfactual expected states, physical residuals, and indicators,
and exports derived Phase 4 datasets to data/generated/phase4/.

Usage:
  .venv/bin/python scripts/generate_phase4_dataset.py --pilot
  .venv/bin/python scripts/generate_phase4_dataset.py --full
"""

import os
import sys
import json
import glob
import argparse
import pandas as pd

# Ensure AeroTwin root is in sys.path
_script_dir = os.path.dirname(os.path.abspath(__file__))
_root_dir = os.path.dirname(_script_dir)
_aerotwin_dir = os.path.join(_root_dir, "AeroTwin")

for _p in [_aerotwin_dir, _root_dir]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from AeroTwin.health.engine import DigitalTwinStateEngine
    from AeroTwin.degradation.conditions import RunConditionSampler
except ImportError:
    from health.engine import DigitalTwinStateEngine
    from degradation.conditions import RunConditionSampler


def generate_phase4_residuals(sub_folder: str = "pilot"):
    p3_dir = os.path.join(_root_dir, "data", "generated", "phase3", sub_folder)
    p4_dir = os.path.join(_root_dir, "data", "generated", "phase4", sub_folder)
    os.makedirs(p4_dir, exist_ok=True)

    print("==================================================")
    print(f"AeroTwin-4 Phase 4 Derived Residual Generator ({sub_folder.upper()})")
    print("==================================================")
    print(f"Source Directory: {p3_dir}")
    print(f"Output Directory: {p4_dir}\n")

    raw_files = glob.glob(os.path.join(p3_dir, "**", "*_raw.csv"), recursive=True)
    if not raw_files:
        print(f"Error: No Phase 3 raw files found in {p3_dir}")
        sys.exit(1)

    print(f"Found {len(raw_files)} raw run files to process.\n")

    for idx, raw_file in enumerate(raw_files, 1):
        run_name = os.path.basename(raw_file).replace("_raw.csv", "")
        rel_sub = os.path.relpath(os.path.dirname(raw_file), p3_dir)
        target_sub = os.path.join(p4_dir, rel_sub)
        os.makedirs(target_sub, exist_ok=True)

        print(f"[{idx}/{len(raw_files)}] Processing {run_name}...")
        df_raw = pd.read_csv(raw_file)

        # Rebuild the twin for THIS engine unit in THIS sortie's ambient conditions.
        # Ambient temperature is a directly measured quantity on a real UAV (OAT
        # probe); build tolerance is what a twin is calibrated to from acceptance
        # test data. A twin left on library defaults produces residuals dominated by
        # ambient mismatch, which masks thermal faults completely.
        meta_path = raw_file.replace("_raw.csv", "_metadata.json")
        run_seed = 42
        if os.path.exists(meta_path):
            with open(meta_path, encoding="utf-8") as _mf:
                run_seed = int(json.load(_mf).get("seed", 42))

        conditions = RunConditionSampler().sample(run_name, run_seed)
        twin_parameters = RunConditionSampler().build_engine_parameters(conditions)

        dt_engine = DigitalTwinStateEngine(
            dt=0.01, seed=run_seed, mode="COUNTERFACTUAL", engine_parameters=twin_parameters
        )

        derived_rows = []
        for _, row in df_raw.iterrows():
            telem_dict = row.to_dict()
            frame = dt_engine.process_telemetry(telem_dict)

            # Flatten frame into flat row dict
            r_dict = {
                "timestamp": frame.timestamp,
                "simulation_time": frame.simulation_time,
                "engine_id": frame.engine_id,
                "operating_mode": frame.operating_mode,
            }

            # Add observed outputs
            for k, v in frame.observed_outputs.items():
                r_dict[f"obs_{k}"] = v

            # Add expected outputs
            for k, v in frame.expected_outputs.items():
                r_dict[f"exp_{k}"] = v

            # Add raw signed residuals
            for k, v in frame.residuals.raw_signed.items():
                r_dict[f"res_signed_{k}"] = v

            # Add normalized residuals
            for k, v in frame.residuals.normalized.items():
                r_dict[f"res_norm_{k}"] = v

            # Add indicators
            r_dict["ind_thermal_dev"] = frame.indicators.thermal_deviation
            r_dict["ind_oil_dev"] = frame.indicators.oil_deviation
            r_dict["ind_vibration_dev"] = frame.indicators.vibration_deviation
            r_dict["ind_torque_dev"] = frame.indicators.torque_deviation
            r_dict["ind_cylinder_balance_dev"] = frame.indicators.cylinder_balance_deviation

            # Preserve evaluation ground truth metadata
            for gt_col in ["gt_degradation_type", "gt_target_component", "gt_active_severity", "gt_current_health", "gt_is_degraded"]:
                if gt_col in telem_dict:
                    r_dict[gt_col] = telem_dict[gt_col]

            derived_rows.append(r_dict)

        df_p4 = pd.DataFrame(derived_rows)
        out_csv = os.path.join(target_sub, f"{run_name}_derived_residuals.csv")
        df_p4.to_csv(out_csv, index=False)

    print("\n------------------------------------------")
    print(f"Phase 4 Residual Generation Complete!")
    print(f"Processed Files: {len(raw_files)}")
    print("------------------------------------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AeroTwin-4 Phase 4 Dataset Generator")
    parser.add_argument("--pilot", action="store_true", help="Process pilot dataset (default)")
    parser.add_argument("--full", action="store_true", help="Process full dataset")
    args = parser.parse_args()

    sub = "full" if args.full else "pilot"
    generate_phase4_residuals(sub_folder=sub)
