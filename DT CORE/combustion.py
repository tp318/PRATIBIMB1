"""
combustion.py
=============
DT CORE — Combustion Model

Converts the air-fuel mixture into torque and heat release.

Physics chain (simplified Wiebe/mean-value approach):
  1. Heat released per cycle:
       Q_comb = m_fuel_per_cycle × LHV × η_comb
  2. Mean effective pressure:
       BMEP = (Q_comb × η_mech) / Vd
  3. Torque:
       T_engine = BMEP × Vd / (4π)   [4-stroke relation]
  4. Heat rejection (to cylinder head / exhaust):
       Q_cht  = Q_comb × (1 - η_therm) × f_cht   → CHT target
       Q_egt  = Q_comb × f_egt                     → EGT target

Combustion efficiency η_comb depends on λ (air/fuel ratio):
  - Near stoichiometric → peak
  - Rich or lean → reduced

Units: SI internally.
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from fuel_model import FuelState
from air_path   import AirPathState

# Fuel heating value and thermal constants
LHV_AVGAS        = 43.5e6    # J/kg — lower heating value of AVGAS 100LL
STROKES_PER_CYCLE = 2         # 4-stroke

# Representative thermal efficiency and heat split
ETA_THERM_PEAK   = 0.28       # peak brake thermal efficiency (~28% for NA piston)
ETA_MECH         = 0.92       # mechanical efficiency (friction losses)
F_CHT_HEAT       = 0.25       # fraction of rejected heat that goes to cylinder head
F_EGT_HEAT       = 0.70       # fraction that exits via exhaust

# Engine geometry (matching air_path)
ENGINE_DISPLACEMENT_M3 = 1.352e-3

# Combustion efficiency vs. lambda polynomial (peak at λ ≈ 0.9)
def _eta_combustion(lambda_: float) -> float:
    """Combustion efficiency as a function of λ (normalised AFR)."""
    # Peak at λ = 0.9, falls off symmetrically
    peak_lambda = 0.90
    d = lambda_ - peak_lambda
    eta = 0.97 - 3.0 * d**2 - 0.5 * abs(d)**3
    return max(0.60, min(0.98, eta))


@dataclass(frozen=True)
class CombustionState:
    """Outputs of the combustion model."""
    torque_nm:          float   # N·m  — mean engine (brake) torque
    bmep_pa:            float   # Pa   — brake mean effective pressure
    heat_released_j:    float   # J/s  — heat released per second (power)
    heat_to_cht_js:     float   # J/s  — heat load to cylinder head
    heat_to_egt_js:     float   # J/s  — heat carried away in exhaust
    eta_combustion:     float   # combustion efficiency

    def to_dict(self) -> dict:
        return {
            "torque_nm":       round(self.torque_nm,       2),
            "bmep_kpa":        round(self.bmep_pa / 1000,  2),
            "heat_cht_watts":  round(self.heat_to_cht_js,  1),
            "heat_egt_watts":  round(self.heat_to_egt_js,  1),
            "eta_combustion":  round(self.eta_combustion,  4),
        }


def compute_combustion(
    rpm:  float,
    fuel: FuelState,
    air:  AirPathState,
) -> CombustionState:
    """
    Compute mean-value combustion torque and heat release.

    Parameters
    ----------
    rpm : float
        Current engine speed (rev/min).
    fuel : FuelState
        Fuel delivery state.
    air : AirPathState
        Air path state (used for volumetric efficiency).
    """
    rpm = max(500.0, rpm)
    rps = rpm / 60.0

    eta_comb = _eta_combustion(fuel.lambda_)

    # Heat release rate: ṁ_fuel × LHV × η_comb  [W]
    q_dot_released = fuel.fuel_mass_flow_kgs * LHV_AVGAS * eta_comb

    # Brake power: apply thermal and mechanical efficiency
    p_brake = q_dot_released * ETA_THERM_PEAK * ETA_MECH
    p_brake = max(0.0, p_brake)

    # Torque from power: P = T × ω = T × 2π × n
    omega   = 2.0 * math.pi * rps
    torque  = p_brake / omega if omega > 1.0 else 0.0
    torque  = max(0.0, torque)

    # BMEP: T = BMEP × Vd / (4π)  →  BMEP = T × 4π / Vd
    bmep = torque * 4.0 * math.pi / ENGINE_DISPLACEMENT_M3

    # Heat rejection split
    q_rejected = q_dot_released * (1.0 - ETA_THERM_PEAK)
    q_to_cht   = q_rejected * F_CHT_HEAT
    q_to_egt   = q_rejected * F_EGT_HEAT

    return CombustionState(
        torque_nm       = torque,
        bmep_pa         = bmep,
        heat_released_j = q_dot_released,
        heat_to_cht_js  = q_to_cht,
        heat_to_egt_js  = q_to_egt,
        eta_combustion  = eta_comb,
    )
