"""
environment.py
==============
DT CORE — Environment Model

Computes atmospheric properties from altitude and ambient temperature.
These feed directly into the air path model to determine air density,
which drives volumetric efficiency and air mass flow.

Physics basis (ISA approximation):
  Pa = P0 * (1 - L*h/T0)^(g*M/(R*L))   (troposphere, h < 11000 m)
  rho = Pa / (R_air * Ta)

References: ICAO Standard Atmosphere, Doc 7488
"""

from __future__ import annotations
import math
from dataclasses import dataclass

# ISA constants
_P0   = 101325.0   # Pa  — sea-level pressure
_T0   = 288.15     # K   — sea-level temperature
_L    = 0.0065     # K/m — lapse rate
_R    = 8.314      # J/(mol·K) — universal gas constant
_M    = 0.029      # kg/mol — molar mass of air
_g    = 9.807      # m/s²
_Rair = 287.058    # J/(kg·K) — specific gas constant for air


@dataclass(frozen=True)
class AtmosphericState:
    """Computed atmospheric properties at a given altitude."""
    altitude_m:   float   # m above sea level
    pressure_pa:  float   # Pa
    temperature_k: float  # K
    density_kgm3: float   # kg/m³
    rho_ratio:    float   # ρ/ρ₀ — density ratio relative to sea level
    altitude_ft:  float   # ft (input echo)
    ambient_c:    float   # °C (input echo)

    def to_dict(self) -> dict:
        return {
            "altitude_ft":   round(self.altitude_ft,   1),
            "ambient_c":     round(self.ambient_c,     2),
            "pressure_pa":   round(self.pressure_pa,   1),
            "temperature_k": round(self.temperature_k, 2),
            "density_kgm3":  round(self.density_kgm3,  5),
            "rho_ratio":     round(self.rho_ratio,      5),
        }


def compute_atmosphere(altitude_ft: float, ambient_c: float) -> AtmosphericState:
    """
    Compute atmospheric state at the given altitude.

    Parameters
    ----------
    altitude_ft : float
        Pressure altitude in feet (0 – 25 000 ft).
    ambient_c : float
        Ambient (OAT) in °C. Used to set sea-level temperature offset,
        modelling non-standard day conditions.

    Returns
    -------
    AtmosphericState
    """
    altitude_m = altitude_ft * 0.3048
    altitude_m = max(0.0, min(altitude_m, 11000.0))  # clamp to troposphere

    # Non-standard day: adjust T0 by OAT deviation from ISA at sea level
    isa_sl = 15.0  # ISA sea-level temperature in °C
    delta_t = ambient_c - isa_sl
    T0_adj  = _T0 + delta_t  # adjusted sea-level temperature

    # Temperature at altitude
    Ta_k = T0_adj - _L * altitude_m
    Ta_k = max(Ta_k, 216.65)  # tropopause floor

    # Pressure via hypsometric formula
    exponent = (_g * _M) / (_R * _L)
    Pa = _P0 * (Ta_k / T0_adj) ** exponent

    # Density
    rho = Pa / (_Rair * Ta_k)
    rho0 = _P0 / (_Rair * _T0)

    return AtmosphericState(
        altitude_m    = altitude_m,
        pressure_pa   = Pa,
        temperature_k = Ta_k,
        density_kgm3  = rho,
        rho_ratio     = rho / rho0,
        altitude_ft   = altitude_ft,
        ambient_c     = ambient_c,
    )
