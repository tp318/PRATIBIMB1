"""
AeroTwin-4 Phase 3 Dataset Generation Script.

Generates ground-truth dataset runs across healthy baselines and single-fault
degradation profiles (Cylinder, Bearing, Cooling, Lubrication).

Usage:
  .venv/bin/python scripts/generate_phase3_dataset.py --pilot
  .venv/bin/python scripts/generate_phase3_dataset.py --full
"""

import os
import sys
import argparse

# Ensure AeroTwin root is in sys.path
_script_dir = os.path.dirname(os.path.abspath(__file__))
_root_dir = os.path.dirname(_script_dir)
_aerotwin_dir = os.path.join(_root_dir, "AeroTwin")

for _p in [_aerotwin_dir, _root_dir]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from AeroTwin.simulator.runner import EngineRunner
    from AeroTwin.degradation.config import DegradationConfig, DegradationType, ComponentID, TrajectoryType
    from AeroTwin.degradation.injector import DegradationInjector
    from AeroTwin.degradation.dataset import DatasetBuilder
    from AeroTwin.degradation.conditions import RunConditionSampler
except ImportError:
    from simulator.runner import EngineRunner
    from degradation.config import DegradationConfig, DegradationType, ComponentID, TrajectoryType
    from degradation.injector import DegradationInjector
    from degradation.dataset import DatasetBuilder
    from degradation.conditions import RunConditionSampler


def generate_dataset(output_dir: str, is_pilot: bool = True):
    print("==================================================")
    print(f"AeroTwin-4 Phase 3 Dataset Generator ({'PILOT' if is_pilot else 'FULL'})")
    print("==================================================")

    builder = DatasetBuilder(output_dir=output_dir)
    print(f"Output Directory: {builder.output_dir}\n")

    # Define run configurations
    if is_pilot:
        # Pilot Dataset: 1 Healthy + 1 for each degradation family
        run_configs = [
            ("HEALTHY_001", DegradationConfig.healthy(), "healthy"),
            (
                "CYLINDER_3_SEV040",
                DegradationConfig.single_fault(DegradationType.CYLINDER, ComponentID.CYLINDER_3, 0.40),
                "cylinder",
            ),
            (
                "BEARING_SEV040",
                DegradationConfig.single_fault(DegradationType.BEARING, ComponentID.BEARING, 0.40),
                "bearing",
            ),
            (
                "COOLING_SEV040",
                DegradationConfig.single_fault(DegradationType.COOLING, ComponentID.COOLING_SYSTEM, 0.40),
                "cooling",
            ),
            (
                "LUBRICATION_SEV040",
                DegradationConfig.single_fault(DegradationType.LUBRICATION, ComponentID.LUBRICATION_SYSTEM, 0.40),
                "lubrication",
            ),
        ]
        duration = 15.0  # seconds per pilot run
    else:
        # Full Dataset
        run_configs = []
        # Healthy runs. Nine, not three: an anomaly threshold calibrated on healthy
        # data needs healthy runs in train AND validation AND the held-out test set,
        # otherwise test precision/FPR/ROC-AUC are undefined rather than perfect.
        for i in range(1, 10):
            run_configs.append((f"HEALTHY_{i:03d}", DegradationConfig.healthy(), "healthy"))
        # Cylinder runs
        for cyl, cid in [("CYL1", ComponentID.CYLINDER_1), ("CYL3", ComponentID.CYLINDER_3)]:
            for sev in [0.20, 0.40, 0.60, 0.80]:
                run_configs.append(
                    (f"{cyl}_SEV{int(sev*100):03d}", DegradationConfig.single_fault(DegradationType.CYLINDER, cid, sev), "cylinder")
                )
        # Bearing runs
        for sev in [0.20, 0.40, 0.60, 0.80]:
            run_configs.append(
                (f"BEARING_SEV{int(sev*100):03d}", DegradationConfig.single_fault(DegradationType.BEARING, ComponentID.BEARING, sev), "bearing")
            )
        # Cooling runs
        for sev in [0.20, 0.40, 0.60, 0.80]:
            run_configs.append(
                (f"COOLING_SEV{int(sev*100):03d}", DegradationConfig.single_fault(DegradationType.COOLING, ComponentID.COOLING_SYSTEM, sev), "cooling")
            )
        # Lubrication runs
        for sev in [0.20, 0.40, 0.60, 0.80]:
            run_configs.append(
                (f"LUBRICATION_SEV{int(sev*100):03d}", DegradationConfig.single_fault(DegradationType.LUBRICATION, ComponentID.LUBRICATION_SYSTEM, sev), "lubrication")
            )
        duration = 60.0  # seconds per full run

    total_runs = len(run_configs)
    generated_raw_files = []
    generated_win_files = []

    sampler = RunConditionSampler()

    for idx, (run_id, config, subfolder) in enumerate(run_configs, 1):
        print(f"[{idx}/{total_runs}] Generating {run_id} ({config.degradation_list[0].degradation_type.value})...")
        seed = 42 + idx
        # Each run is a different engine unit flying a different sortie: build
        # tolerance + ambient conditions + its own sensor-noise stream. Without this
        # every healthy run is bit-identical and healthy variance collapses to zero.
        conditions = sampler.sample(run_id, seed)
        engine_parameters = sampler.build_engine_parameters(conditions)
        runner = EngineRunner(dt=0.01, seed=seed, engine_parameters=engine_parameters)
        inj = DegradationInjector(
            config=config,
            runner=runner,
            run_id=run_id,
            noise_enabled=True,
            conditions=conditions,
        )
        telemetry_list, gt_list = inj.run_simulation(duration_seconds=duration)

        raw_csv, win_csv = builder.export_run_dataset(
            telemetry_list, gt_list, inj.run_ground_truth, subfolder=subfolder
        )
        generated_raw_files.append(raw_csv)
        generated_win_files.append(win_csv)

    print("\n------------------------------------------")
    print(f"Dataset Generation Complete!")
    print(f"Total Simulation Runs: {total_runs}")
    print(f"Raw CSV Files:         {len(generated_raw_files)}")
    print(f"Window CSV Files:      {len(generated_win_files)}")
    print("------------------------------------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AeroTwin-4 Phase 3 Dataset Generator")
    parser.add_argument("--pilot", action="store_true", help="Generate deterministic pilot dataset (default)")
    parser.add_argument("--full", action="store_true", help="Generate full Phase 3 dataset")
    args = parser.parse_args()

    is_pilot = not args.full
    sub_folder = "pilot" if is_pilot else "full"
    out_dir = os.path.join(_root_dir, "data", "generated", "phase3", sub_folder)

    generate_dataset(output_dir=out_dir, is_pilot=is_pilot)
