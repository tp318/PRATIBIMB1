"""
============================================================================
env_simulator.py — Atmospheric Environmental Simulator
============================================================================
Simulates ambient atmospheric conditions across MALE UAV flight envelope (0 to 7,600 m).
Conforms to International Standard Atmosphere (ISA).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass
class AtmosphericState:
    altitude_m: float
    ambient_temp_c: float
    pressure_hpa: float
    density_kg_m3: float
    density_ratio: float
    turbulence_intensity_g: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "altitude_m": round(self.altitude_m, 1),
            "ambient_temp_c": round(self.ambient_temp_c, 2),
            "pressure_hpa": round(self.pressure_hpa, 2),
            "density_kg_m3": round(self.density_kg_m3, 4),
            "density_ratio": round(self.density_ratio, 4),
            "turbulence_intensity_g": round(self.turbulence_intensity_g, 3),
        }


class EnvironmentalSimulator:
    """
    Simulates ambient atmospheric conditions across MALE UAV flight envelope (0 to 7,600 m).
    Conforms to International Standard Atmosphere (ISA).
    """

    P0_HPA = 1013.25
    T0_K = 288.15        # 15°C
    RHO0_KG_M3 = 1.225
    LAPSE_RATE_K_M = 0.0065
    R_AIR = 287.05

    def __init__(self, sea_level_temp_c: float = 15.0) -> None:
        self.t0_k = sea_level_temp_c + 273.15

    def get_atmosphere(self, altitude_m: float, turbulence_level: float = 0.1) -> AtmosphericState:
        """
        Calculates atmospheric properties at specified altitude.
        """
        alt = max(0.0, min(7600.0, altitude_m))

        # 1. Temperature Lapse: T(h) = T0 - L * h
        temp_k = max(216.65, self.t0_k - self.LAPSE_RATE_K_M * alt)
        temp_c = temp_k - 273.15

        # 2. Barometric Pressure
        p_ratio = (1.0 - (self.LAPSE_RATE_K_M * alt) / self.t0_k) ** 5.25588
        pressure_hpa = self.P0_HPA * p_ratio

        # 3. Density
        density = (pressure_hpa * 100.0) / (self.R_AIR * temp_k)
        density_ratio = density / self.RHO0_KG_M3

        # 4. Turbulence perturbation (Dryden wind turbulence proxy)
        turb_g = random.gauss(0.0, turbulence_level * (1.0 + alt / 4000.0))

        return AtmosphericState(
            altitude_m=alt,
            ambient_temp_c=temp_c,
            pressure_hpa=pressure_hpa,
            density_kg_m3=density,
            density_ratio=density_ratio,
            turbulence_intensity_g=abs(turb_g),
        )
