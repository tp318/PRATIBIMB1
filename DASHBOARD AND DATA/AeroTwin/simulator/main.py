"""
AeroTwin-4 Phase 2 Real-Time Engine Simulator.

Main entry point for running continuous flight scenarios, measuring update latency,
generating multi-channel telemetry streams, and exporting telemetry data.
"""

import os
import sys

# Ensure AeroTwin and phase1 directories are in sys.path
_current_dir = os.path.dirname(os.path.abspath(__file__))
_aerotwin_dir = os.path.dirname(_current_dir)
_phase1_dir = os.path.join(_aerotwin_dir, "phase1")
_root_dir = os.path.dirname(_aerotwin_dir)

for _p in [_aerotwin_dir, _phase1_dir, _root_dir]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from simulator import EngineRunner, FlightProfile, OperatingMode, TelemetryExporter, benchmark_runner
except ImportError:
    from AeroTwin.simulator import EngineRunner, FlightProfile, OperatingMode, TelemetryExporter, benchmark_runner

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def run_phase2_simulation(duration=60.0, dt=0.01):
    print("\n==================================================")
    print("AeroTwin-4 Phase 2.1 Engine Simulator & Runtime")
    print("==================================================")

    # 1. Instantiate runner with flight profile
    runner = EngineRunner(dt=dt, seed=42)
    print(f"Engine ID:        {runner.engine_id}")
    print(f"Simulation dt:    {runner.clock.dt * 1000.0:.1f} ms (100 Hz telemetry rate)")
    print(f"Mission Duration: {duration:.1f} s")

    # 2. Run simulation over flight profile scenario (Fast Computation Batch Mode)
    print("\n[Mode 1] Running fast batch computation flight profile scenario...")
    history = runner.run_for(duration)
    last_frame = history[-1]

    print(f"Simulation completed: {len(history)} telemetry frames generated.")
    print("\nCanonical Telemetry Sample (Last Frame):")
    print("------------------------------------------")
    print(f"  Mode:           {last_frame.operating_mode}")
    print(f"  RPM:            {last_frame.rpm:.1f} RPM")
    print(f"  Total Torque:   {last_frame.instant_torque:.2f} Nm")
    print(f"  Cylinder 1:     {last_frame.cylinder_1_torque:.2f} Nm")
    print(f"  Cylinder 2:     {last_frame.cylinder_2_torque:.2f} Nm")
    print(f"  Cylinder 3:     {last_frame.cylinder_3_torque:.2f} Nm")
    print(f"  Cylinder 4:     {last_frame.cylinder_4_torque:.2f} Nm")
    print(f"  CHT / EGT:      {last_frame.cht:.1f}°C / {last_frame.egt:.1f}°C")
    print(f"  Oil Press/Temp: {last_frame.oil_pressure_psi:.1f} PSI / {last_frame.oil_temperature:.1f}°C")
    print(f"  Fuel Flow:      {last_frame.fuel_flow_lph:.2f} L/h")
    print(f"  Vibration:      {last_frame.vibration:.3f} g")
    print("------------------------------------------")

    # 3. Demonstrate 1.0x Real-Time Wall-Clock Pacing Mode
    print("\n[Mode 2] Demonstrating 1.0x Real-Time Wall-Clock Pacing Mode (1.0s demo)...")
    rt_runner = EngineRunner(dt=0.01, seed=42)
    rt_history = rt_runner.run_realtime(duration_seconds=1.0, playback_speed=1.0)
    print(f"Real-Time Pacing Demo Complete: {len(rt_history)} frames streamed at 1.0x playback speed.")

    # 4. Export telemetry to CSV and Parquet
    exporter = TelemetryExporter(history)
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "generated")
    os.makedirs(output_dir, exist_ok=True)
    csv_file = exporter.to_csv(os.path.join(output_dir, "telemetry_run.csv"))
    parquet_file = exporter.to_parquet(os.path.join(output_dir, "telemetry_run.parquet"))
    print(f"\nExported CSV:     {csv_file}")
    print(f"Exported Columnar: {parquet_file}")

    # 5. Performance Benchmarking
    print("\nRunning performance throughput benchmark (5,000 steps)...")
    bench = benchmark_runner(runner=runner, steps=5000)
    print("------------------------------------------")
    print(f"Benchmark Steps:            {bench['steps_evaluated']}")
    print(f"Average Latency:            {bench['avg_latency_ms']:.4f} ms")
    print(f"P95 Latency:                {bench['p95_latency_ms']:.4f} ms")
    print(f"Max Latency:                {bench['max_latency_ms']:.4f} ms")
    print(f"Max Computation Frequency:  {bench['execution_rate_hz']:.1f} Hz")
    print("------------------------------------------")

    # 5. Save multi-channel scenario plot
    if HAS_MATPLOTLIB:
        plots_dir = os.path.join(os.path.dirname(__file__), "plots")
        os.makedirs(plots_dir, exist_ok=True)

        df = exporter.to_dataframe()
        fig, axs = plt.subplots(3, 2, figsize=(14, 10))

        # Throttle & RPM
        axs[0, 0].plot(df["simulation_time"], df["throttle"], color="gray", label="Throttle", linestyle="--")
        axs[0, 0].set_ylabel("Throttle")
        ax_rpm = axs[0, 0].twinx()
        ax_rpm.plot(df["simulation_time"], df["rpm"], color="blue", label="RPM")
        ax_rpm.set_ylabel("RPM", color="blue")
        axs[0, 0].set_title("Flight Scenario: Throttle & RPM Trajectory")
        axs[0, 0].grid(True)

        # Torques
        axs[0, 1].plot(df["simulation_time"], df["mean_torque"], label="Mean Torque", color="red")
        axs[0, 1].plot(df["simulation_time"], df["load_torque"], label="Propeller Load", color="green", linestyle=":")
        axs[0, 1].set_title("Torque Dynamics (Nm)")
        axs[0, 1].set_ylabel("Torque (Nm)")
        axs[0, 1].legend()
        axs[0, 1].grid(True)

        # Thermal
        axs[1, 0].plot(df["simulation_time"], df["cht"], label="CHT", color="darkred")
        axs[1, 0].plot(df["simulation_time"], df["egt"], label="EGT", color="orange")
        axs[1, 0].plot(df["simulation_time"], df["oil_temperature"], label="Oil Temp", color="brown")
        axs[1, 0].set_title("Thermal Dynamics (°C)")
        axs[1, 0].set_ylabel("Temperature (°C)")
        axs[1, 0].legend()
        axs[1, 0].grid(True)

        # Lubrication & Fuel
        axs[1, 1].plot(df["simulation_time"], df["oil_pressure_psi"], label="Oil Pressure (PSI)", color="green")
        axs[1, 1].plot(df["simulation_time"], df["fuel_flow_lph"], label="Fuel Flow (L/h)", color="purple")
        axs[1, 1].set_title("Fluids: Oil Pressure (PSI) & Fuel Flow (L/h)")
        axs[1, 1].legend()
        axs[1, 1].grid(True)

        # Vibration
        axs[2, 0].plot(df["simulation_time"], df["vibration"], color="crimson")
        axs[2, 0].set_title("Mechanical Vibration Amplitude (g)")
        axs[2, 0].set_ylabel("Vibration (g)")
        axs[2, 0].set_xlabel("Simulation Time (s)")
        axs[2, 0].grid(True)

        # Operating Mode timeline
        modes = df["operating_mode"].unique()
        mode_indices = [list(modes).index(m) for m in df["operating_mode"]]
        axs[2, 1].step(df["simulation_time"], mode_indices, color="black", where="post")
        axs[2, 1].set_yticks(range(len(modes)))
        axs[2, 1].set_yticklabels(modes)
        axs[2, 1].set_title("Flight Scenario Operating Modes")
        axs[2, 1].set_xlabel("Simulation Time (s)")
        axs[2, 1].grid(True)

        plt.tight_layout()
        plot_path = os.path.join(plots_dir, "phase2_flight_scenario.png")
        plt.savefig(plot_path)
        plt.close(fig)
        print(f"\nSaved Flight Scenario Plot: {plot_path}")

    return history, bench


if __name__ == "__main__":
    dur = 60.0
    if len(sys.argv) > 1:
        dur = float(sys.argv[1])
    run_phase2_simulation(duration=dur)
