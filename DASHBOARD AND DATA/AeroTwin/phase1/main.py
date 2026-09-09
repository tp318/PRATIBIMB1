"""
AeroTwin-4
Phase 1 - Step 1

Basic engine rotational dynamics simulation.
"""

import os
import sys

_current_dir = os.path.dirname(os.path.abspath(__file__))
_aerotwin_dir = os.path.dirname(_current_dir)
_root_dir = os.path.dirname(_aerotwin_dir)

for _p in [_current_dir, _aerotwin_dir, _root_dir]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from engine.dynamics import EngineDynamics
except ImportError:
    from AeroTwin.phase1.engine.dynamics import EngineDynamics

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def print_ascii_plot(title, values, times, width=50, height=10):
    """
    Generate an ASCII plot when matplotlib is not available.
    """
    print(f"\n--- {title} (ASCII Representation) ---")
    min_v = min(values)
    max_v = max(values)
    span = (max_v - min_v) if max_v != min_v else 1.0

    step_idx = len(values) // width if len(values) > width else 1
    sample_values = values[::step_idx]

    for h in range(height, -1, -1):
        threshold = min_v + (h / height) * span
        line = ""
        for v in sample_values:
            if v >= threshold:
                line += "*"
            else:
                line += " "
        if h == height:
            print(f"{max_v:7.1f} |{line}")
        elif h == 0:
            print(f"{min_v:7.1f} |{line}")
        else:
            print(f"        |{line}")
    print("        +" + "-" * len(sample_values))
    print(f"Time:    0.0s{' ' * (len(sample_values) - 10)}{times[-1]:.1f}s\n")


def run_simulation(throttle=0.6, show_plots=False, save_plots=True):
    # ---------------------------------------------------------
    # Simulation configuration
    # ---------------------------------------------------------

    simulation_time = 10.0
    dt = 0.001

    # ---------------------------------------------------------
    # Create engine
    # ---------------------------------------------------------

    engine = EngineDynamics()

    # ---------------------------------------------------------
    # Storage
    # ---------------------------------------------------------

    times = []
    rpm_values = []
    times = []
    rpm_values = []
    crank_angles = []
    instant_torques = []
    mean_torques = []
    load_values = []
    cht_values = []
    egt_values = []
    oil_temp_values = []
    oil_press_values = []
    fuel_flow_values = []
    vibration_values = []
    cylinder_torques_hist = {1: [], 2: [], 3: [], 4: []}

    # ---------------------------------------------------------
    # Simulation loop
    # ---------------------------------------------------------

    current_time = 0.0

    while current_time <= simulation_time:

        state = engine.update(
            throttle=throttle,
            dt=dt
        )

        times.append(current_time)
        rpm_values.append(state["rpm"])
        crank_angles.append(state["crank_angle"])
        instant_torques.append(state["engine_torque"])
        mean_torques.append(state["mean_engine_torque"])
        load_values.append(state["load_torque"])
        cht_values.append(state["cht"])
        egt_values.append(state["egt"])
        oil_temp_values.append(state["oil_temperature"])
        oil_press_values.append(state["oil_pressure_psi"])
        fuel_flow_values.append(state["fuel_flow_lph"])
        vibration_values.append(state["vibration"])
        for cyl_id, t_val in state["cylinder_torques"].items():
            cylinder_torques_hist[cyl_id].append(t_val)

        current_time += dt

    # ---------------------------------------------------------
    # Print final state (Canonical Telemetry Payload)
    # ---------------------------------------------------------

    final_state = state

    print("\n==========================================")
    print("AeroTwin-4 Canonical Telemetry Payload")
    print("==========================================")

    print(f"Throttle:              {throttle:.2f}")
    print(f"Final RPM:             {final_state['rpm']:.2f}")
    print(f"Crank Angle:           {final_state['crank_angle']:.1f}°")
    print(f"Mean Engine Torque:    {final_state['mean_engine_torque']:.2f} Nm")
    print(f"Instant Engine Torque:   {final_state['engine_torque']:.2f} Nm")
    print(f"Load Torque:           {final_state['load_torque']:.2f} Nm")
    print(f"Friction Torque:       {final_state['friction_torque']:.2f} Nm")
    print(f"Net Instant Torque:    {final_state['net_torque']:.2f} Nm")
    print(f"CHT (Cylinder Head):   {final_state['cht']:.1f} °C")
    print(f"EGT (Exhaust Gas):     {final_state['egt']:.1f} °C")
    print(f"Oil Temperature:       {final_state['oil_temperature']:.1f} °C")
    print(f"Oil Pressure:          {final_state['oil_pressure_psi']:.1f} PSI ({final_state['oil_pressure']:.1f} kPa)")
    print(f"Fuel Flow Rate:        {final_state['fuel_flow_lph']:.2f} L/h ({final_state['fuel_flow']:.5f} kg/s)")
    print(f"Fuel Pressure:         {final_state['fuel_pressure']:.1f} kPa")
    print(f"Vibration Level:       {final_state['vibration']:.3f} g")

    if HAS_MATPLOTLIB:
        plots_dir = os.path.join(os.path.dirname(__file__), "plots")
        os.makedirs(plots_dir, exist_ok=True)

        # 1. RPM & Torque vs Time
        fig1 = plt.figure(figsize=(8, 5))
        plt.plot(times, rpm_values, color='b', label='Instantaneous RPM')
        plt.xlabel("Time (s)")
        plt.ylabel("RPM")
        plt.title(f"AeroTwin-4 Engine RPM (Throttle={throttle:.2f})")
        plt.grid(True)
        if save_plots:
            plt.savefig(os.path.join(plots_dir, f"rpm_throttle_{throttle:.2f}.png"))
        plt.close(fig1)

        # 2. Torques vs Time
        fig2 = plt.figure(figsize=(8, 5))
        plt.plot(times[-1000:], instant_torques[-1000:], label="Instantaneous Engine Torque", alpha=0.7)
        plt.plot(times[-1000:], mean_torques[-1000:], label="Mean Engine Torque", linestyle='--')
        plt.plot(times[-1000:], load_values[-1000:], label="Propeller Load Torque", linestyle=':')
        plt.xlabel("Time (s)")
        plt.ylabel("Torque (Nm)")
        plt.title(f"Engine vs Load Torque (Throttle={throttle:.2f})")
        plt.legend()
        plt.grid(True)
        if save_plots:
            plt.savefig(os.path.join(plots_dir, f"torque_throttle_{throttle:.2f}.png"))
        plt.close(fig2)

        # 3. Instantaneous Torque & Cylinder Contributions vs Crank Angle (0 - 720°)
        cycle_steps = min(500, len(crank_angles))
        sample_angles = crank_angles[-cycle_steps:]
        sample_instant = instant_torques[-cycle_steps:]

        sorted_indices = sorted(range(len(sample_angles)), key=lambda i: sample_angles[i])
        sorted_angles = [sample_angles[i] for i in sorted_indices]
        sorted_instant = [sample_instant[i] for i in sorted_indices]

        fig3 = plt.figure(figsize=(10, 6))
        plt.plot(sorted_angles, sorted_instant, label="Total Instantaneous Torque", color='red', linewidth=2)
        for cyl_id in [1, 3, 4, 2]:
            sorted_cyl = [cylinder_torques_hist[cyl_id][-cycle_steps:][i] for i in sorted_indices]
            plt.plot(sorted_angles, sorted_cyl, label=f"Cylinder {cyl_id}", linestyle='--', alpha=0.7)

        plt.xlabel("Crank Angle (Degrees 0° - 720°)")
        plt.ylabel("Torque (Nm)")
        plt.title(f"AeroTwin-4 4-Cylinder Torque Pulsation over 720° Cycle (Throttle={throttle:.2f})")
        plt.axvline(x=180, color='gray', linestyle=':', alpha=0.5)
        plt.axvline(x=360, color='gray', linestyle=':', alpha=0.5)
        plt.axvline(x=540, color='gray', linestyle=':', alpha=0.5)
        plt.legend(loc='upper right')
        plt.grid(True)
        if save_plots:
            plt.savefig(os.path.join(plots_dir, f"crank_torque_throttle_{throttle:.2f}.png"))
        plt.close(fig3)

        # 4. Multi-Subsystem Telemetry Dashboard (Thermal, Oil, Fuel, Vibration)
        fig4, axs = plt.subplots(2, 2, figsize=(12, 9))

        # Thermal response
        axs[0, 0].plot(times, cht_values, label="CHT (°C)", color="darkred")
        axs[0, 0].plot(times, egt_values, label="EGT (°C)", color="orange")
        axs[0, 0].plot(times, oil_temp_values, label="Oil Temp (°C)", color="brown")
        axs[0, 0].set_title("Thermal Dynamics (CHT, EGT, Oil Temp)")
        axs[0, 0].set_ylabel("Temperature (°C)")
        axs[0, 0].set_xlabel("Time (s)")
        axs[0, 0].legend()
        axs[0, 0].grid(True)

        # Lubrication
        axs[0, 1].plot(times, oil_press_values, color="green")
        axs[0, 1].set_title("Lubrication System (Oil Pressure PSI)")
        axs[0, 1].set_ylabel("Oil Pressure (PSI)")
        axs[0, 1].set_xlabel("Time (s)")
        axs[0, 1].grid(True)

        # Fuel Flow
        axs[1, 0].plot(times, fuel_flow_values, color="purple")
        axs[1, 0].set_title("Fuel Consumption (L/h)")
        axs[1, 0].set_ylabel("Fuel Flow Rate (L/h)")
        axs[1, 0].set_xlabel("Time (s)")
        axs[1, 0].grid(True)

        # Vibration Signature
        axs[1, 1].plot(times[-1000:], vibration_values[-1000:], color="crimson")
        axs[1, 1].set_title("Mechanical Vibration (g) vs Combustion Pulse")
        axs[1, 1].set_ylabel("Vibration (g)")
        axs[1, 1].set_xlabel("Time (s)")
        axs[1, 1].grid(True)

        plt.tight_layout()
        if save_plots:
            plt.savefig(os.path.join(plots_dir, f"subsystems_throttle_{throttle:.2f}.png"))
        plt.close(fig4)
    else:
        print_ascii_plot(f"Engine RPM vs Time (Throttle={throttle:.2f})", rpm_values, times)
        print_ascii_plot(f"Load Torque vs Time (Throttle={throttle:.2f})", load_values, times)

    return final_state


if __name__ == "__main__":
    throttle_val = 0.6
    if len(sys.argv) > 1:
        throttle_val = float(sys.argv[1])
    run_simulation(throttle=throttle_val)
