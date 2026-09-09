"""
AeroTwin-4 Progressive Degradation (RUL) Dataset Generator.

The Phase 3 dataset injects CONSTANT severity: every run sits at a fixed health
level for its whole duration. Remaining Useful Life cannot be learned from that -
with no degradation trend there is nothing to extrapolate.

This script generates runs where severity PROGRESSES over the sortie (LINEAR and
EXPONENTIAL trajectories), which is what makes a health trend, and therefore an
RUL estimate, meaningful.

Usage:
  .venv/Scripts/python.exe scripts/generate_rul_dataset.py
  .venv/Scripts/python.exe scripts/generate_rul_dataset.py --duration 240
"""

import argparse
import os
import sys

_script_dir = os.path.dirname(os.path.abspath(__file__))
_root_dir = os.path.dirname(_script_dir)
_aerotwin_dir = os.path.join(_root_dir, "AeroTwin")

for _p in [_aerotwin_dir, _root_dir]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from AeroTwin.degradation.conditions import RunConditionSampler
from AeroTwin.degradation.config import (
    ComponentID,
    DegradationConfig,
    DegradationType,
    TrajectoryType,
)
from AeroTwin.degradation.dataset import DatasetBuilder
from AeroTwin.degradation.injector import DegradationInjector
from AeroTwin.simulator.runner import EngineRunner

# Fault families and the component each one attacks.
FAMILIES = [
    ("CYL1", DegradationType.CYLINDER, ComponentID.CYLINDER_1, "cylinder"),
    ("CYL3", DegradationType.CYLINDER, ComponentID.CYLINDER_3, "cylinder"),
    ("BEARING", DegradationType.BEARING, ComponentID.BEARING, "bearing"),
    ("COOLING", DegradationType.COOLING, ComponentID.COOLING_SYSTEM, "cooling"),
    ("LUBRICATION", DegradationType.LUBRICATION, ComponentID.LUBRICATION_SYSTEM, "lubrication"),
]

TRAJECTORIES = [
    ("LIN", TrajectoryType.LINEAR),
    ("EXP", TrajectoryType.EXPONENTIAL),
]

# Two engine units per (family, trajectory) so the split can hold whole units out.
UNITS = [1, 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=180.0, help="seconds per run")
    ap.add_argument("--target-severity", type=float, default=0.95)
    ap.add_argument("--healthy-runs", type=int, default=4)
    args = ap.parse_args()

    out_dir = os.path.join(_root_dir, "data", "generated", "rul")
    builder = DatasetBuilder(output_dir=out_dir)
    sampler = RunConditionSampler()

    print("=" * 68)
    print("AeroTwin-4 Progressive Degradation (RUL) Dataset Generator")
    print("=" * 68)
    print(f"Duration per run : {args.duration:.0f} s")
    print(f"Target severity  : {args.target_severity:.2f}")
    print(f"Output           : {out_dir}\n")

    run_configs = []

    # Healthy reference runs: health stays at 1.0, RUL is unbounded. These teach the
    # health estimator what "no trend" looks like so it does not invent decay.
    for u in range(1, args.healthy_runs + 1):
        run_configs.append((f"RUL_HEALTHY_U{u:02d}", DegradationConfig.healthy(), "healthy"))

    for fam_tag, deg_type, comp_id, subfolder in FAMILIES:
        for traj_tag, traj_type in TRAJECTORIES:
            for u in UNITS:
                run_id = f"RUL_{fam_tag}_{traj_tag}_U{u:02d}"
                cfg = DegradationConfig.single_fault(
                    degradation_type=deg_type,
                    component_id=comp_id,
                    severity=args.target_severity,
                    trajectory_type=traj_type,
                    start_time=0.0,
                    # Ramp across the whole sortie so health decays for the entire run.
                    ramp_duration=args.duration,
                )
                run_configs.append((run_id, cfg, subfolder))

    total = len(run_configs)
    for idx, (run_id, cfg, subfolder) in enumerate(run_configs, 1):
        traj = cfg.trajectory_type.value
        print(f"[{idx}/{total}] {run_id}  ({cfg.degradation_list[0].degradation_type.value}, {traj})")

        seed = 9000 + idx
        conditions = sampler.sample(run_id, seed)
        engine_parameters = sampler.build_engine_parameters(conditions)

        runner = EngineRunner(dt=0.01, seed=seed, engine_parameters=engine_parameters)
        inj = DegradationInjector(
            config=cfg, runner=runner, run_id=run_id, noise_enabled=True, conditions=conditions
        )
        telemetry_list, gt_list = inj.run_simulation(duration_seconds=args.duration)
        builder.export_run_dataset(telemetry_list, gt_list, inj.run_ground_truth, subfolder=subfolder)

    print("\n" + "-" * 42)
    print(f"RUL Dataset Generation Complete: {total} runs")
    print("-" * 42)


if __name__ == "__main__":
    main()
