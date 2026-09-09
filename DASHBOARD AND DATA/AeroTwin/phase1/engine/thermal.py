"""
AeroTwin-4 Thermal Subsystem Model.

Energy-balance thermodynamic model for Cylinder Head Temperature (CHT),
Exhaust Gas Temperature (EGT), and Oil Temperature.

Governing Equation:
    C * (dT/dt) = Q_generated - Q_cooling
"""

import math


class ThermalModel:
    """
    Simulates engine thermal state (CHT, EGT, Oil Temperature)
    based on combustion power, cooling airflow, and ambient conditions.
    """

    def __init__(self, ambient_temp=20.0, cht_init=25.0, oil_init=25.0):
        self.ambient_temp = ambient_temp
        self.cht = cht_init
        self.egt = ambient_temp + 600.0  # Initial target EGT
        self.oil_temp = oil_init

    def update(self, throttle, rpm, mean_torque, dt=0.001, cooling_efficiency=1.0):
        """
        Advance thermal dynamics by time step dt.

        Parameters
        ----------
        throttle : float
            Throttle position [0.0, 1.0].
        rpm : float
            Current engine speed in RPM.
        mean_torque : float
            Mean engine torque in N*m.
        dt : float
            Time step in seconds.
        cooling_efficiency : float
            Cooling system heat dissipation efficiency in [0.5, 1.0]. Default is 1.0 (healthy).

        Returns
        -------
        dict
            Thermal state dictionary containing cht, egt, oil_temperature (°C).
        """
        cooling_efficiency = max(0.1, min(1.0, float(cooling_efficiency)))
        omega = rpm * 2.0 * math.pi / 60.0

        # Mechanical power generated (kW)
        power_kw = max(0.0, mean_torque * omega / 1000.0)

        # Heat generation rate Q_gen (kW = kJ/s)
        # Piston engines reject ~30% of energy to cylinder walls/cooling
        q_gen_cht = 0.85 * power_kw + 2.5 * throttle

        # Cooling heat dissipation Q_cool (kW)
        # Cooling airflow increases with RPM / airspeed, scaled by cooling_efficiency
        cooling_factor = (0.04 + 0.00015 * rpm) * cooling_efficiency
        q_cool_cht = cooling_factor * (self.cht - self.ambient_temp)

        # CHT energy balance: C_cht * d(CHT)/dt = Q_gen - Q_cool
        c_cht = 180.0  # Thermal mass J/°C
        d_cht = (q_gen_cht - q_cool_cht) / c_cht
        self.cht += d_cht * dt * 1000.0  # Scale kW to W for J/s

        # EGT responds rapidly to combustion temperature / throttle
        # Baseline idle EGT ~450°C, peak EGT ~750°C at high load
        target_egt = self.ambient_temp + 430.0 + 260.0 * throttle + 60.0 * (rpm / 3500.0)
        alpha_egt = 2.5  # Response speed rad/s
        self.egt += alpha_egt * (target_egt - self.egt) * dt

        # Oil temperature: slower thermal mass heat exchange with cylinder block & oil cooler
        q_gen_oil = 0.08 * power_kw + 0.02 * max(0.0, self.cht - 80.0)
        cooling_oil = (0.008 + 0.00004 * rpm) * (self.oil_temp - self.ambient_temp)
        c_oil = 1200.0  # Oil thermal mass J/°C
        d_oil = (q_gen_oil - cooling_oil) / c_oil
        self.oil_temp += d_oil * dt * 1000.0

        return {
            "cht": self.cht,
            "egt": self.egt,
            "oil_temperature": self.oil_temp,
        }
