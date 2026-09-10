"""
thermal.py
==========
DT CORE — Thermal Model

Computes Exhaust Gas Temperature (EGT) and Cylinder Head Temperature (CHT)
using first-order lag dynamics driven by the heat release from the combustion
model and the cooling provided by airflow.

Physics basis:
  - The cylinder head is a lumped thermal mass (mass × Cp).
  - Heat input  = combustion heat to CHT  (from combustion model)
  - Heat output = convective cooling by airflow (≈ proportional to air mass flow)

    m_cht × Cp_cht × dT_cht/dt = Q̇_in_cht − Q̇_cool_cht

  This reduces to a first-order lag toward an operating-point-dependent
  target, with a time constant τ_cht ≈ 25 s.

  EGT is similarly modelled as the exhaust gas temperature set by the
  enthalpy of the exhaust stream, with a shorter time constant τ_egt ≈ 3 s
  because the exhaust gas mass is small.
"""

from __future__ import annotations
from dataclasses import dataclass
from combustion import CombustionState
from air_path   import AirPathState

# First-order lag time constants (seconds)
TAU_CHT = 25.0    # large thermal mass of cylinder head
TAU_EGT =  3.0    # light exhaust gas thermal inertia

# Representative steady-state gains
# CHT rises with heat input and falls with cooling airflow
CHT_BASE         = 60.0    # °C at idle sea level
CHT_HEAT_GAIN    = 3.5e-3  # °C per W of heat-to-CHT
CHT_COOL_COEFF   = 0.30    # cooling effectiveness coefficient

# EGT steady state: driven by exhaust enthalpy
EGT_BASE         = 200.0   # °C at idle
EGT_HEAT_GAIN    = 3.0e-3  # °C per W of heat-to-exhaust
EGT_AMBIENT_COEFF = 0.15   # fraction of ambient temp added to EGT base

# Physical limits
CHT_MIN, CHT_MAX = 50.0,  280.0
EGT_MIN, EGT_MAX = 100.0, 950.0


@dataclass
class ThermalState:
    """
    Mutable thermal state — persists across timesteps for the integrator.
    """
    cht_c: float = 70.0    # °C — Cylinder Head Temperature
    egt_c: float = 300.0   # °C — Exhaust Gas Temperature

    def to_dict(self) -> dict:
        return {
            "cht_c": round(self.cht_c, 2),
            "egt_c": round(self.egt_c, 2),
        }


def step_thermal(
    state:    ThermalState,
    comb:     CombustionState,
    air:      AirPathState,
    ambient_c: float,
    dt:       float = 0.1,
) -> ThermalState:
    """
    Advance the thermal model by one timestep.

    Modifies `state` in place and returns it.

    Parameters
    ----------
    state : ThermalState
        Mutable thermal state (updated in place).
    comb : CombustionState
        Combustion model output (heat rates).
    air : AirPathState
        Air path state (air mass flow drives cooling).
    ambient_c : float
        Outside air temperature in °C.
    dt : float
        Integration timestep (s).
    """
    # ---------- CHT target ----------
    # Heat-driven rise
    cht_heat_term = CHT_HEAT_GAIN * comb.heat_to_cht_js

    # Cooling: proportional to air mass flow (ram-air cooling over cylinder fins)
    # Normalise to a representative cruise flow ~0.025 kg/s
    cool_factor = air.air_mass_flow_kgs / 0.025
    cool_factor = max(0.05, min(2.0, cool_factor))
    cht_cool_term = CHT_COOL_COEFF * cool_factor * (state.cht_c - ambient_c)

    cht_target = CHT_BASE + cht_heat_term - cht_cool_term * 0.0  # incorporated via lag
    # The steady-state target is driven by heat balance; we model it as:
    cht_ss = CHT_BASE + cht_heat_term + ambient_c * 0.3
    cht_ss = max(CHT_MIN, min(CHT_MAX, cht_ss))

    # Effective cooling reduces the target
    cht_target_eff = cht_ss - CHT_COOL_COEFF * (cool_factor - 1.0) * 20.0
    cht_target_eff = max(CHT_MIN, min(CHT_MAX, cht_target_eff))

    # First-order lag integration
    state.cht_c += dt * (cht_target_eff - state.cht_c) / TAU_CHT
    state.cht_c  = max(CHT_MIN, min(CHT_MAX, state.cht_c))

    # ---------- EGT target ----------
    egt_ss = EGT_BASE + EGT_HEAT_GAIN * comb.heat_to_egt_js + ambient_c * EGT_AMBIENT_COEFF
    egt_ss = max(EGT_MIN, min(EGT_MAX, egt_ss))

    state.egt_c += dt * (egt_ss - state.egt_c) / TAU_EGT
    state.egt_c  = max(EGT_MIN, min(EGT_MAX, state.egt_c))

    return state
