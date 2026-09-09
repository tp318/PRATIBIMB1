"""
AeroTwin-4 Engine Parameters

These parameters describe our representative 4-cylinder
aero piston-engine simulation.

They are prototype/model assumptions, NOT specifications
of any particular real UAV engine.
"""

ENGINE = {
    # -----------------------------
    # Mechanical properties
    # -----------------------------

    "cylinders": 4,

    # Crankshaft rotational inertia
    # kg*m^2
    "inertia": 0.20,

    # Maximum modeled engine torque
    # N*m
    "max_torque": 180.0,

    # Maximum modeled RPM
    "max_rpm": 3500.0,

    # Minimum useful RPM
    "idle_rpm": 900.0,

    # Peak torque RPM for efficiency curve
    "peak_torque_rpm": 2500.0,

    # -----------------------------
    # Propeller/load model
    # -----------------------------

    # T_load = K * omega^2
    "load_coefficient": 0.001386,

    # -----------------------------
    # Mechanical friction
    # -----------------------------

    # Simplified friction coefficient (N*m*s/rad)
    "friction_coefficient": 0.02,

    # -----------------------------
    # Thermal Subsystem Parameters
    # -----------------------------

    "ambient_temperature": 20.0,       # °C
    "cht_thermal_mass": 45.0,          # J/°C
    "oil_thermal_mass": 120.0,         # J/°C
    "egt_response_rate": 2.5,          # s^-1
    "cooling_coefficient": 0.35,       # W/°C per (rad/s)

    # -----------------------------
    # Lubrication Subsystem Parameters
    # -----------------------------

    "oil_pressure_idle": 280.0,        # kPa (~40 PSI)
    "oil_pressure_max": 410.0,         # kPa (~60 PSI)
    "nominal_oil_temp": 95.0,          # °C
    "lubrication_health": 1.0,         # 0.0 to 1.0

    # -----------------------------
    # Fuel Subsystem Parameters
    # -----------------------------

    "bsfc": 0.28,                       # kg / (kW * h)
    "fuel_density": 0.72,              # kg/L (gasoline/avgas)
    "nominal_fuel_pressure": 320.0,    # kPa (~46 PSI)
    "fuel_system_health": 1.0,         # 0.0 to 1.0

    # -----------------------------
    # Mechanical Vibration Parameters
    # -----------------------------

    "vibration_torque_gain": 0.0035,   # g / (N*m)
    "vibration_rotational_gain": 0.15, # g

    # -----------------------------
    # Simulation
    # -----------------------------

    # Initial RPM
    "initial_rpm": 900.0,

    # Simulation time step
    "dt": 0.001,
}
