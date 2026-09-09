"""
AeroTwin-4 Lubrication & Oil Subsystem Model.

Models crankshaft-driven positive displacement oil pump, pressure relief valve,
oil temperature viscosity effects, and friction factor coupling.
"""

import math


class LubricationModel:
    """
    Simulates oil pressure and oil viscosity coupling to engine friction.
    """

    def __init__(self, p_idle=280.0, p_max=410.0, nominal_temp=95.0):
        self.p_idle = p_idle          # kPa at idle
        self.p_max = p_max            # kPa maximum relief pressure
        self.nominal_temp = nominal_temp
        self.lubrication_health = 1.0  # 1.0 = healthy

    def update(self, rpm, oil_temperature, dt=0.001, lubrication_efficiency=1.0):
        """
        Update oil pressure and friction coefficient multiplier.

        Parameters
        ----------
        rpm : float
            Current engine speed in RPM.
        oil_temperature : float
            Current oil temperature in °C.
        dt : float
            Time step in seconds.
        lubrication_efficiency : float
            Oil subsystem efficiency in [0.5, 1.0]. Default is 1.0 (healthy).

        Returns
        -------
        dict
            Oil subsystem state containing oil_pressure (kPa), oil_pressure_psi,
            and friction_multiplier.
        """
        lub_eff = max(0.1, min(1.0, float(lubrication_efficiency)))

        # Base oil pressure builds with RPM up to relief valve threshold, scaled by oil system efficiency
        rpm_ratio = min(1.0, max(0.0, rpm / 2800.0))
        base_pressure = (self.p_idle + (self.p_max - self.p_idle) * (rpm_ratio ** 0.8)) * lub_eff

        # Oil viscosity factor: higher oil temp lowers viscosity; oil degradation degrades film strength
        temp_delta = max(-30.0, oil_temperature - self.nominal_temp)
        viscosity_factor = (1.0 - 0.0035 * temp_delta)

        # Oil pressure in kPa
        oil_pressure_kpa = base_pressure * viscosity_factor * self.lubrication_health
        oil_pressure_kpa = max(20.0, min(self.p_max * 1.1, oil_pressure_kpa))

        # Convert to PSI (1 kPa ≈ 0.145038 PSI)
        oil_pressure_psi = oil_pressure_kpa * 0.145038

        # Friction multiplier: low oil pressure or high temp increases friction
        friction_mult = 1.0
        if oil_temperature > self.nominal_temp:
            friction_mult += 0.004 * (oil_temperature - self.nominal_temp)
        if oil_pressure_kpa < 300.0:
            friction_mult += 0.002 * (300.0 - oil_pressure_kpa)

        return {
            "oil_pressure": oil_pressure_kpa,
            "oil_pressure_psi": oil_pressure_psi,
            "friction_multiplier": max(1.0, friction_mult),
        }
