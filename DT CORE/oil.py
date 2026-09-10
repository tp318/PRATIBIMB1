"""
oil.py
======
DT CORE — Lubrication System Model

Models oil pressure and oil temperature as functions of engine speed
and the thermal state.

Physics basis:
  - Oil pump is engine-driven (gear pump, speed-proportional output).
  - Oil pressure scales with pump speed and falls with oil viscosity
    (which decreases as oil temperature rises).
  - Oil temperature is thermally coupled to the cylinder head (CHT)
    and rises with engine load.

First-order lag dynamics are used for both, reflecting the thermal
inertia of the oil circuit volume.
"""

from __future__ import annotations
from dataclasses import dataclass
from thermal import ThermalState

# Oil system constants
TAU_OIL_PRESS = 2.0     # s — oil pressure time constant (fast — pump-driven)
TAU_OIL_TEMP  = 20.0    # s — oil temperature time constant (slow — large volume)

# Oil pressure model: P_oil = P_idle + (rpm - rpm_idle) * k_pump − k_visc*(T_oil − T_ref)
OIL_PRESS_IDLE_BAR = 1.8     # bar at idle RPM
OIL_PRESS_K_PUMP   = 7.0e-4  # bar per RPM above idle
RPM_IDLE            = 800.0   # RPM at which pump starts building pressure
OIL_PRESS_K_VISC   = 0.008   # bar lost per °C above oil reference temp
OIL_TEMP_REF        = 80.0    # °C — reference oil temp at which viscosity is nominal
OIL_PRESS_MIN       = 1.0     # bar
OIL_PRESS_MAX       = 8.0     # bar

# Oil temperature model
OIL_TEMP_BASE       = 50.0    # °C — minimum oil temp (cold start / sea level)
OIL_TEMP_CHT_COEFF  = 0.45    # fraction of CHT that couples to oil temp
OIL_TEMP_MIN        = 40.0    # °C
OIL_TEMP_MAX        = 130.0   # °C


@dataclass
class OilState:
    """Mutable oil system state (persists across timesteps)."""
    oil_press_bar: float = 4.5   # bar
    oil_temp_c:    float = 60.0  # °C

    def to_dict(self) -> dict:
        return {
            "oil_press_bar": round(self.oil_press_bar, 3),
            "oil_temp_c":    round(self.oil_temp_c,    2),
        }


def step_oil(
    state:   OilState,
    rpm:     float,
    thermal: ThermalState,
    dt:      float = 0.1,
) -> OilState:
    """
    Advance the oil system model by one timestep.

    Updates `state` in place and returns it.

    Parameters
    ----------
    state : OilState
        Mutable oil state (updated in place).
    rpm : float
        Current engine speed (RPM).
    thermal : ThermalState
        Thermal model state (CHT drives oil temp).
    dt : float
        Integration timestep (s).
    """
    # ---------- Oil Pressure ----------
    # Pump pressure rises with RPM above idle; viscosity loss at high T reduces it.
    pump_press  = OIL_PRESS_IDLE_BAR + max(0.0, rpm - RPM_IDLE) * OIL_PRESS_K_PUMP
    visc_loss   = max(0.0, state.oil_temp_c - OIL_TEMP_REF) * OIL_PRESS_K_VISC
    oil_press_ss = pump_press - visc_loss
    oil_press_ss = max(OIL_PRESS_MIN, min(OIL_PRESS_MAX, oil_press_ss))

    state.oil_press_bar += dt * (oil_press_ss - state.oil_press_bar) / TAU_OIL_PRESS
    state.oil_press_bar  = max(OIL_PRESS_MIN, min(OIL_PRESS_MAX, state.oil_press_bar))

    # ---------- Oil Temperature ----------
    # Oil temp is thermally coupled to CHT; tends toward a fraction of CHT.
    oil_temp_ss = OIL_TEMP_BASE + OIL_TEMP_CHT_COEFF * thermal.cht_c
    oil_temp_ss = max(OIL_TEMP_MIN, min(OIL_TEMP_MAX, oil_temp_ss))

    state.oil_temp_c += dt * (oil_temp_ss - state.oil_temp_c) / TAU_OIL_TEMP
    state.oil_temp_c  = max(OIL_TEMP_MIN, min(OIL_TEMP_MAX, state.oil_temp_c))

    return state
