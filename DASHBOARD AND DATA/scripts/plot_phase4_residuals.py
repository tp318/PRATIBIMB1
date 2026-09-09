"""
AeroTwin-4 Phase 4 Residual Plot Generator.

Generates mandatory visual validation plots for Phase 4 Digital Twin State Engine:
1. Observed RPM vs Expected Healthy RPM
2. Observed CHT vs Expected Healthy CHT
3. CHT residual vs Time
4. Vibration residual vs Time
5. 4-Cylinder Torque Residuals (C1, C2, C3, C4) demonstrating Cylinder-3 degradation isolation.

Usage:
  .venv/bin/python scripts/plot_phase4_residuals.py
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt

# Ensure AeroTwin root is in sys.path
_script_dir = os.path.dirname(os.path.abspath(__file__))
_root_dir = os.path.dirname(_script_dir)
_aerotwin_dir = os.path.join(_root_dir, "AeroTwin")

for _p in [_aerotwin_dir, _root_dir]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from AeroTwin.simulator.runner import EngineRunner
    from AeroTwin.degradation.config import DegradationConfig, DegradationType, ComponentID
    from AeroTwin.degradation.injector import DegradationInjector
    from AeroTwin.health.engine import DigitalTwinStateEngine
except ImportError:
    from simulator.runner import EngineRunner
    from degradation.config import DegradationConfig, DegradationType, ComponentID
    from degradation.injector import DegradationInjector
    from health.engine import DigitalTwinStateEngine


def generate_phase4_plots(output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    print("==================================================")
    print("AeroTwin-4 Phase 4 Residual Visual Plot Generator")
    print("==================================================")
    print(f"Target Plot Directory: {output_dir}\n")

    # Set up dark modern aesthetic style for plots
    plt.style.use("dark_background")
    plt.rcParams["font.sans-serif"] = "DejaVu Sans"
    plt.rcParams["axes.edgecolor"] = "#444444"
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.color"] = "#222222"

    # 1. Run Cylinder 3 Degradation (severity = 0.50)
    c3_config = DegradationConfig.single_fault(DegradationType.CYLINDER, ComponentID.CYLINDER_3, 0.50)
    runner_c3 = EngineRunner(dt=0.01, seed=42)
    inj_c3 = DegradationInjector(config=c3_config, runner=runner_c3, run_id="CYL3_SEV050")
    t_c3, _ = inj_c3.run_simulation(duration_seconds=10.0)

    dt_c3 = DigitalTwinStateEngine(dt=0.01, seed=42, mode="COUNTERFACTUAL")
    frames_c3 = [dt_c3.process_telemetry(t) for t in t_c3]

    times = [f.simulation_time for f in frames_c3]

    # --- Plot 1: Observed RPM vs Expected Healthy RPM ---
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(times, [f.observed_outputs["rpm"] for f in frames_c3], label="Observed RPM (Degraded)", color="#ff4444", lw=2)
    ax.plot(times, [f.expected_outputs["rpm"] for f in frames_c3], label="Expected Healthy RPM", color="#00ccff", lw=2, linestyle="--")
    ax.set_title("Digital Twin State Engine: Observed vs Expected Healthy RPM", fontsize=14, pad=12)
    ax.set_xlabel("Simulation Time (s)", fontsize=12)
    ax.set_ylabel("Engine Speed (RPM)", fontsize=12)
    ax.legend(fontsize=11)
    fig.tight_layout()
    p1_path = os.path.join(output_dir, "phase4_rpm_observed_vs_expected.png")
    fig.savefig(p1_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {p1_path}")

    # 2. Run Cooling Degradation (severity = 0.50) for CHT plots
    cool_config = DegradationConfig.single_fault(DegradationType.COOLING, ComponentID.COOLING_SYSTEM, 0.50)
    runner_cool = EngineRunner(dt=0.01, seed=42)
    inj_cool = DegradationInjector(config=cool_config, runner=runner_cool, run_id="COOLING_SEV050")
    t_cool, _ = inj_cool.run_simulation(duration_seconds=15.0)

    dt_cool = DigitalTwinStateEngine(dt=0.01, seed=42, mode="COUNTERFACTUAL")
    frames_cool = [dt_cool.process_telemetry(t) for t in t_cool]
    times_cool = [f.simulation_time for f in frames_cool]

    # --- Plot 2: Observed CHT vs Expected Healthy CHT ---
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(times_cool, [f.observed_outputs["cht"] for f in frames_cool], label="Observed CHT (Cooling Fault)", color="#ff8800", lw=2)
    ax.plot(times_cool, [f.expected_outputs["cht"] for f in frames_cool], label="Expected Healthy CHT", color="#00ffcc", lw=2, linestyle="--")
    ax.set_title("Digital Twin State Engine: Observed CHT vs Expected Healthy CHT", fontsize=14, pad=12)
    ax.set_xlabel("Simulation Time (s)", fontsize=12)
    ax.set_ylabel("Cylinder Head Temp (°C)", fontsize=12)
    ax.legend(fontsize=11)
    fig.tight_layout()
    p2_path = os.path.join(output_dir, "phase4_cht_observed_vs_expected.png")
    fig.savefig(p2_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {p2_path}")

    # --- Plot 3: CHT Residual vs Time ---
    fig, ax = plt.subplots(figsize=(10, 5))
    cht_residuals = [f.residuals.raw_signed["cht"] for f in frames_cool]
    ax.plot(times_cool, cht_residuals, label="Raw Signed CHT Residual (°C)", color="#ffbb00", lw=2)
    ax.axhline(0.0, color="#666666", linestyle=":", lw=1.5)
    ax.set_title("Thermal Residual Trajectory under Cooling Degradation", fontsize=14, pad=12)
    ax.set_xlabel("Simulation Time (s)", fontsize=12)
    ax.set_ylabel("CHT Residual (°C)", fontsize=12)
    ax.legend(fontsize=11)
    fig.tight_layout()
    p3_path = os.path.join(output_dir, "phase4_cht_residual_time.png")
    fig.savefig(p3_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {p3_path}")

    # --- Plot 4: Vibration Residual vs Time ---
    fig, ax = plt.subplots(figsize=(10, 5))
    vib_residuals = [f.residuals.raw_signed["vibration"] for f in frames_c3]
    ax.plot(times, vib_residuals, label="Signed Vibration Residual (g)", color="#cc44ff", lw=2)
    ax.axhline(0.0, color="#666666", linestyle=":", lw=1.5)
    ax.set_title("Vibration Residual Trajectory under Cylinder-3 Degradation", fontsize=14, pad=12)
    ax.set_xlabel("Simulation Time (s)", fontsize=12)
    ax.set_ylabel("Vibration Residual (g)", fontsize=12)
    ax.legend(fontsize=11)
    fig.tight_layout()
    p4_path = os.path.join(output_dir, "phase4_vibration_residual_time.png")
    fig.savefig(p4_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {p4_path}")

    # --- Plot 5: 4-Cylinder Torque Residuals (C1, C2, C3, C4 Isolation) ---
    fig, ax = plt.subplots(figsize=(12, 6))
    # Slice to a 2-second window to visualize cylinder torque pulse residuals clearly
    slice_idx = 300  # 3.0s window
    slice_times = times[:slice_idx]

    c1_res = [f.residuals.raw_signed["cylinder_1_torque"] for f in frames_c3[:slice_idx]]
    c2_res = [f.residuals.raw_signed["cylinder_2_torque"] for f in frames_c3[:slice_idx]]
    c3_res = [f.residuals.raw_signed["cylinder_3_torque"] for f in frames_c3[:slice_idx]]
    c4_res = [f.residuals.raw_signed["cylinder_4_torque"] for f in frames_c3[:slice_idx]]

    ax.plot(slice_times, c1_res, label="Cylinder 1 Torque Residual", color="#44aaff", lw=1.5, alpha=0.7)
    ax.plot(slice_times, c2_res, label="Cylinder 2 Torque Residual", color="#44ffaa", lw=1.5, alpha=0.7)
    ax.plot(slice_times, c3_res, label="Cylinder 3 Torque Residual (FAULTED)", color="#ff3366", lw=2.5)
    ax.plot(slice_times, c4_res, label="Cylinder 4 Torque Residual", color="#ffaa44", lw=1.5, alpha=0.7)

    ax.axhline(0.0, color="#666666", linestyle=":", lw=1.5)
    ax.set_title("Phase 4 Cylinder-Level Residual Isolation (Cylinder-3 Fault)", fontsize=14, pad=12)
    ax.set_xlabel("Simulation Time (s)", fontsize=12)
    ax.set_ylabel("Torque Residual (N*m)", fontsize=12)
    ax.legend(fontsize=11)
    fig.tight_layout()
    p5_path = os.path.join(output_dir, "phase4_cylinder_torque_residuals.png")
    fig.savefig(p5_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {p5_path}")

    print("\n------------------------------------------")
    print("All Phase 4 Visual Validation Plots Generated Successfully!")
    print("------------------------------------------")


if __name__ == "__main__":
    plot_dir = os.path.join(_root_dir, "docs", "plots")
    generate_phase4_plots(plot_dir)
