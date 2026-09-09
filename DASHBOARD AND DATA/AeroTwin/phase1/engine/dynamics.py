"""
Basic rotational dynamics for AeroTwin-4.
"""

import math

from .parameters import ENGINE
from .cylinders import FourCylinderModel
from .thermal import ThermalModel
from .lubrication import LubricationModel
from .fuel import FuelModel
from .vibration import VibrationModel


class EngineDynamics:
    """
    Rotational dynamics & 4-cylinder combustion pulse simulation with fully coupled
    thermal, lubrication, fuel, and vibration physical subsystems.
    """

    def __init__(self, parameters=None):
        if parameters is None:
            parameters = ENGINE

        self.params = parameters

        # Angular velocity in rad/s
        self.omega = self.rpm_to_omega(
            self.params["initial_rpm"]
        )

        # Crank angle in degrees [0.0, 720.0)
        self.crank_angle = 0.0

        # Subsystems
        self.cylinder_model = FourCylinderModel()
        self.thermal_model = ThermalModel(
            ambient_temp=self.params.get("ambient_temperature", 20.0)
        )
        self.lubrication_model = LubricationModel(
            p_idle=self.params.get("oil_pressure_idle", 280.0),
            p_max=self.params.get("oil_pressure_max", 410.0),
            nominal_temp=self.params.get("nominal_oil_temp", 95.0),
        )
        self.fuel_model = FuelModel(
            bsfc=self.params.get("bsfc", 0.28),
            fuel_density=self.params.get("fuel_density", 0.72),
            nominal_pressure=self.params.get("nominal_fuel_pressure", 320.0),
        )
        self.vibration_model = VibrationModel(
            inertia=self.params.get("inertia", 0.20),
            torque_gain=self.params.get("vibration_torque_gain", 0.0035),
            rot_gain=self.params.get("vibration_rotational_gain", 0.15),
        )

    # ---------------------------------------------------------
    # Unit conversions
    # ---------------------------------------------------------

    @staticmethod
    def rpm_to_omega(rpm):
        """
        Convert RPM to angular velocity in rad/s.
        """
        return rpm * 2.0 * math.pi / 60.0

    @staticmethod
    def omega_to_rpm(omega):
        """
        Convert angular velocity in rad/s to RPM.
        """
        return omega * 60.0 / (2.0 * math.pi)

    # ---------------------------------------------------------
    # Engine torque
    # ---------------------------------------------------------

    def calculate_torque_efficiency(self, rpm):
        """
        Normalized torque efficiency curve depending on RPM.

        Piston engine torque peaks at mid-high RPM (~2500 RPM)
        and drops off at low RPM (near idle) and over-speed RPM.
        """
        peak_rpm = self.params.get("peak_torque_rpm", 2500.0)
        max_rpm = self.params["max_rpm"]

        normalized_diff = (rpm - peak_rpm) / max_rpm
        efficiency = 1.0 - 0.8 * (normalized_diff ** 2)

        # Bounded between 0.10 and 1.00
        return max(0.10, min(1.00, efficiency))

    def calculate_engine_torque(self, throttle):
        """
        Engine torque model depending on throttle and current RPM.

        throttle:
            0.0 = closed
            1.0 = maximum
        """
        throttle = max(0.0, min(1.0, throttle))
        max_torque = self.params["max_torque"]
        current_rpm = self.get_rpm()
        efficiency = self.calculate_torque_efficiency(current_rpm)

        return throttle * max_torque * efficiency

    # ---------------------------------------------------------
    # Load torque
    # ---------------------------------------------------------

    def calculate_load_torque(self):
        """
        Simplified propeller/load torque.

        T_load = K * omega^2
        """

        k = self.params["load_coefficient"]

        return k * self.omega ** 2

    # ---------------------------------------------------------
    # Mechanical friction
    # ---------------------------------------------------------

    def calculate_friction_torque(self, friction_multiplier=1.0, bearing_friction_multiplier=1.0):
        """
        Rotational mechanical friction model, combining oil fluid viscosity friction factor
        and mechanical bearing dry/boundary friction factor.
        """

        coefficient = self.params["friction_coefficient"] * friction_multiplier * bearing_friction_multiplier

        return coefficient * self.omega

    # ---------------------------------------------------------
    # Current RPM
    # ---------------------------------------------------------

    def get_rpm(self):
        """
        Return current engine speed in RPM.
        """

        return self.omega_to_rpm(self.omega)

    # ---------------------------------------------------------
    # Simulation update
    # ---------------------------------------------------------

    def update(
        self,
        throttle,
        dt=None,
        combustion_efficiencies=None,
        bearing_friction_multiplier=1.0,
        cooling_efficiency=1.0,
        lubrication_efficiency=1.0,
    ):
        """
        Advance engine state by one simulation time step across all physical subsystems.

        Parameters
        ----------
        throttle : float
            Value between 0 and 1.
        dt : float
            Simulation time step in seconds.
        combustion_efficiencies : Dict[int, float], optional
            Per-cylinder combustion efficiency dict. Default is healthy (1.0 for all cylinders).
        bearing_friction_multiplier : float, optional
            Bearing mechanical friction multiplier (>= 1.0). Default is 1.0 (healthy).
        cooling_efficiency : float, optional
            Cooling system heat dissipation efficiency in [0.5, 1.0]. Default is 1.0 (healthy).
        lubrication_efficiency : float, optional
            Oil subsystem efficiency in [0.5, 1.0]. Default is 1.0 (healthy).

        Returns
        -------
        dict
            Canonical multi-channel telemetry payload.
        """

        if dt is None:
            dt = self.params["dt"]

        # Current RPM
        rpm = self.get_rpm()

        # 1. Advance crank angle: d(theta)/dt = omega (converted to deg/s)
        self.crank_angle = (self.crank_angle + self.omega * dt * 180.0 / math.pi) % 720.0

        # 2. Mean engine torque capability (throttle & RPM dependent)
        mean_engine_torque = self.calculate_engine_torque(throttle)

        # 3. Four-cylinder instantaneous torque calculation
        cylinder_torques, engine_torque = self.cylinder_model.calculate_torques(
            self.crank_angle,
            mean_engine_torque,
            combustion_efficiencies=combustion_efficiencies,
        )

        # 4. Load torque
        load_torque = self.calculate_load_torque()

        # 5. Thermal subsystem update
        thermal_state = self.thermal_model.update(
            throttle=throttle,
            rpm=rpm,
            mean_torque=mean_engine_torque,
            dt=dt,
            cooling_efficiency=cooling_efficiency,
        )

        # 6. Lubrication subsystem update (coupled to oil_temperature & lubrication_efficiency)
        oil_state = self.lubrication_model.update(
            rpm=rpm,
            oil_temperature=thermal_state["oil_temperature"],
            dt=dt,
            lubrication_efficiency=lubrication_efficiency,
        )

        # 7. Friction torque (scaled by oil viscosity / temp and bearing mechanical degradation)
        friction_torque = self.calculate_friction_torque(
            friction_multiplier=oil_state["friction_multiplier"],
            bearing_friction_multiplier=bearing_friction_multiplier,
        )

        # 8. Net torque (instantaneous engine torque - load - friction)
        net_torque = engine_torque - load_torque - friction_torque

        # 9. Angular acceleration
        inertia = self.params["inertia"]
        angular_acceleration = net_torque / inertia

        # 10. Integrate angular velocity
        self.omega += angular_acceleration * dt

        # Prevent physically meaningless negative RPM
        self.omega = max(0.0, self.omega)

        # 11. Fuel subsystem update
        fuel_state = self.fuel_model.update(
            throttle=throttle,
            rpm=rpm,
            mean_torque=mean_engine_torque,
            dt=dt
        )

        # 12. Vibration subsystem update (coupled to torque fluctuation & RPM)
        vib_state = self.vibration_model.update(
            instant_torque=engine_torque,
            mean_torque=mean_engine_torque,
            rpm=rpm,
            dt=dt
        )

        # Updated RPM
        rpm = self.get_rpm()

        # 13. Return canonical telemetry payload
        return {
            "throttle": throttle,
            "crank_angle": self.crank_angle,
            "engine_torque": engine_torque,
            "mean_engine_torque": mean_engine_torque,
            "cylinder_torques": cylinder_torques,
            "load_torque": load_torque,
            "friction_torque": friction_torque,
            "net_torque": net_torque,
            "angular_acceleration": angular_acceleration,
            "angular_velocity": self.omega,
            "rpm": rpm,
            "cht": thermal_state["cht"],
            "egt": thermal_state["egt"],
            "oil_temperature": thermal_state["oil_temperature"],
            "oil_pressure": oil_state["oil_pressure"],
            "oil_pressure_psi": oil_state["oil_pressure_psi"],
            "fuel_flow": fuel_state["fuel_flow"],
            "fuel_flow_lph": fuel_state["fuel_flow_lph"],
            "fuel_pressure": fuel_state["fuel_pressure"],
            "vibration": vib_state["vibration"],
        }
