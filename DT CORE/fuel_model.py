"""
fuel_model.py
=============
DT CORE — Fuel Delivery Model

Computes fuel mass flow and the resulting Air/Fuel Ratio (AFR) from
the air mass flow delivered by the air path model.

For a stoichiometric petrol/AVGAS engine:
    AFR_stoich ≈ 14.7  (by mass)

The engine controller targets a slight rich mixture (~13.5) at high load
(for cooling and knock avoidance) and lean (~15–16) at cruise.

Fuel flow in L/hr is derived from mass flow via fuel density.
"""

from __future__ import annotations
from dataclasses import dataclass
from air_path import AirPathState

# Fuel properties (AVGAS 100LL)
FUEL_DENSITY_KGL   = 0.720    # kg/L  (AVGAS 100LL at 15°C)
AFR_STOICH         = 14.7     # stoichiometric air/fuel ratio by mass
AFR_RICH_WOT       = 12.5     # target AFR at wide-open throttle
AFR_LEAN_CRUISE    = 15.5     # target AFR at cruise/lean setting
AFR_IDLE           = 13.0     # target AFR at idle


@dataclass(frozen=True)
class FuelState:
    """Outputs of the fuel delivery model."""
    fuel_mass_flow_kgs: float   # kg/s
    fuel_flow_lph:      float   # L/hr (user-facing)
    afr:                float   # air/fuel ratio (mass)
    lambda_:            float   # λ = AFR / AFR_stoich (1.0 = stoich)

    def to_dict(self) -> dict:
        return {
            "fuel_flow_lph": round(self.fuel_flow_lph, 3),
            "afr":           round(self.afr,           3),
            "lambda":        round(self.lambda_,       4),
        }


def compute_fuel(
    throttle: float,
    air:      AirPathState,
) -> FuelState:
    """
    Compute fuel delivery state.

    The fuel controller targets a load-scheduled AFR:
    - At idle (throttle → 0): AFR_IDLE
    - At cruise (throttle ≈ 0.5–0.7): AFR_LEAN_CRUISE
    - At WOT (throttle → 1): AFR_RICH_WOT

    Parameters
    ----------
    throttle : float
        Normalised throttle position, 0 – 1.
    air : AirPathState
        Air path state from compute_air_path().
    """
    throttle = max(0.0, min(1.0, throttle))

    # Target AFR schedule: blend between idle, cruise, WOT
    if throttle < 0.15:
        # Idle enrichment
        afr_target = AFR_IDLE
    elif throttle < 0.6:
        # Lean cruise — linear interpolation from idle to cruise
        t = (throttle - 0.15) / 0.45
        afr_target = AFR_IDLE + t * (AFR_LEAN_CRUISE - AFR_IDLE)
    else:
        # Power enrichment — from cruise lean to WOT rich
        t = (throttle - 0.6) / 0.4
        afr_target = AFR_LEAN_CRUISE + t * (AFR_RICH_WOT - AFR_LEAN_CRUISE)

    afr_target = max(10.0, min(18.0, afr_target))

    # Fuel mass flow = air mass flow / AFR
    m_dot_fuel_kgs = air.air_mass_flow_kgs / afr_target
    fuel_flow_lph  = (m_dot_fuel_kgs / FUEL_DENSITY_KGL) * 3600.0   # kg/s → L/hr

    # Clamp to physical limits
    fuel_flow_lph  = max(2.0, min(45.0, fuel_flow_lph))
    m_dot_fuel_kgs = fuel_flow_lph * FUEL_DENSITY_KGL / 3600.0

    lambda_ = afr_target / AFR_STOICH

    return FuelState(
        fuel_mass_flow_kgs = m_dot_fuel_kgs,
        fuel_flow_lph      = fuel_flow_lph,
        afr                = afr_target,
        lambda_            = lambda_,
    )
