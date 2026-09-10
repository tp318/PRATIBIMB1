"""
air_path.py
===========
DT CORE — Air Path Model

Computes manifold pressure, volumetric efficiency, and air mass flow rate
from throttle position and atmospheric state.

The air path is the first link in the combustion chain:

    throttle → throttle body → manifold → cylinders

Simplified approach (appropriate for a MALE UAV piston engine):
  - Manifold pressure is a linear function of throttle and ambient pressure
    (no turbocharger; naturally aspirated or fixed-boost assumption)
  - Volumetric efficiency accounts for: throttle loss, altitude derating,
    and mild speed-dependence
  - Air mass flow = ρ_manifold × Vd × N / (2 × 60)  [4-stroke relation]

Units: all internal SI; user-facing outputs annotated.
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from environment import AtmosphericState


# Engine displacement and geometry constants (Rotax 914-class representative)
ENGINE_DISPLACEMENT_L   = 1.352   # litres  (Rotax 914 actual: 1352 cm³)
ENGINE_DISPLACEMENT_M3  = ENGINE_DISPLACEMENT_L * 1e-3
N_CYLINDERS             = 4
STROKES_PER_CYCLE       = 2        # 4-stroke: 2 crank revs per power stroke per cyl.

# Volumetric efficiency model coefficients
ETA_V_MAX   = 0.88    # peak VE at wide-open throttle, sea level
ETA_V_IDLE  = 0.30    # VE at closed throttle (pumping loss)
RPM_VE_PEAK = 5000.0  # RPM at which VE peaks


@dataclass(frozen=True)
class AirPathState:
    """Outputs of the air path model for one timestep."""
    manifold_pressure_pa:  float   # Pa
    volumetric_efficiency: float   # 0 – 1
    air_mass_flow_kgs:     float   # kg/s
    air_density_manifold:  float   # kg/m³

    def to_dict(self) -> dict:
        return {
            "manifold_pressure_kpa": round(self.manifold_pressure_pa / 1000, 3),
            "volumetric_efficiency": round(self.volumetric_efficiency, 4),
            "air_mass_flow_gs":      round(self.air_mass_flow_kgs * 1000, 4),
        }


def compute_air_path(
    throttle:   float,          # 0.0 – 1.0
    rpm:        float,          # rev/min (current engine speed)
    atm:        AtmosphericState,
) -> AirPathState:
    """
    Compute air path state.

    Parameters
    ----------
    throttle : float
        Normalised throttle position, 0 (closed) to 1 (WOT).
    rpm : float
        Current engine speed in rev/min.
    atm : AtmosphericState
        Atmospheric conditions from the environment model.
    """
    throttle = max(0.0, min(1.0, throttle))
    rpm      = max(500.0, rpm)

    # Manifold pressure: interpolate between idle and WOT
    # At WOT the manifold pressure equals ambient (no boost on NA engine).
    # At idle, manifold pressure is ~30–40 % of ambient due to throttle restriction.
    idle_frac   = 0.30
    p_manifold  = atm.pressure_pa * (idle_frac + throttle * (1.0 - idle_frac))

    # Volumetric efficiency: parabolic model peaking at RPM_VE_PEAK,
    # scaled by throttle opening.
    # η_v = η_v_max × throttle_factor × rpm_factor
    throttle_factor = ETA_V_IDLE + (ETA_V_MAX - ETA_V_IDLE) * throttle
    rpm_factor = 1.0 - 0.4 * ((rpm - RPM_VE_PEAK) / RPM_VE_PEAK) ** 2
    rpm_factor = max(0.3, min(1.0, rpm_factor))
    eta_v = throttle_factor * rpm_factor

    # Air density inside the manifold (ideal gas: ρ = P / (R_air × T))
    R_air       = 287.058
    rho_manifold = p_manifold / (R_air * atm.temperature_k)

    # Air mass flow (4-stroke):
    # ṁ_air = ρ_man × Vd × N_cyl × (rpm / 60) / (strokes_per_cycle) × η_v
    rps  = rpm / 60.0   # rev/s
    power_strokes_per_sec = rps / STROKES_PER_CYCLE
    m_dot_air = rho_manifold * ENGINE_DISPLACEMENT_M3 * power_strokes_per_sec * eta_v

    return AirPathState(
        manifold_pressure_pa  = p_manifold,
        volumetric_efficiency = eta_v,
        air_mass_flow_kgs     = m_dot_air,
        air_density_manifold  = rho_manifold,
    )
