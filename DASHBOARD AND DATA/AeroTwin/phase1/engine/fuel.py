"""
AeroTwin-4 Fuel Subsystem Model.

Calculates fuel demand, mass flow rate (kg/s and L/h), and fuel delivery line pressure
based on shaft power output and Brake Specific Fuel Consumption (BSFC).
"""

import math


class FuelModel:
    """
    Simulates engine fuel consumption and fuel delivery line pressure.
    """

    def __init__(self, bsfc=0.28, fuel_density=0.72, nominal_pressure=320.0):
        self.bsfc = bsfc                # kg / (kW * h)
        self.fuel_density = fuel_density  # kg / L
        self.nominal_pressure = nominal_pressure  # kPa
        self.fuel_system_health = 1.0   # 1.0 = healthy

    def update(self, throttle, rpm, mean_torque, dt=0.001):
        """
        Update fuel flow rate and fuel pressure.

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

        Returns
        -------
        dict
            Fuel state containing fuel_flow (kg/s), fuel_flow_lph (L/h), fuel_pressure (kPa).
        """
        omega = rpm * 2.0 * math.pi / 60.0

        # Mechanical shaft power output in kW
        power_kw = max(0.0, mean_torque * omega / 1000.0)

        # Idle baseline fuel flow (kg/s)
        idle_flow_kg_s = 0.0008  # ~4 L/h at idle

        # Fuel mass flow rate: m_dot = BSFC * Power_kW / 3600 (kg/s)
        power_flow_kg_s = (self.bsfc * power_kw) / 3600.0
        fuel_flow_kg_s = idle_flow_kg_s + power_flow_kg_s * self.fuel_system_health

        # Convert to Liters per hour (L/h)
        # (kg/s) * 3600 / (kg/L) = L/h
        fuel_flow_lph = (fuel_flow_kg_s * 3600.0) / self.fuel_density

        # Regulated fuel pressure (kPa)
        # Small fluctuations under sudden throttle changes
        pressure_fluctuation = 3.0 * math.sin(throttle * math.pi * 2.0)
        fuel_pressure_kpa = (self.nominal_pressure + pressure_fluctuation) * self.fuel_system_health

        return {
            "fuel_flow": fuel_flow_kg_s,
            "fuel_flow_lph": fuel_flow_lph,
            "fuel_pressure": fuel_pressure_kpa,
        }
