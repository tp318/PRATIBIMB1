"""
crankshaft.py
=============
DT CORE — Crankshaft Dynamics Model

Integrates the rotational equation of motion to compute engine RPM.

    J · dω/dt = T_engine − T_load − T_friction

where:
    J            = effective polar moment of inertia of crankshaft + flywheel
    ω            = angular velocity (rad/s)
    T_engine     = mean torque from combustion model
    T_load       = propeller/load torque (speed-squared law)
    T_friction   = viscous friction torque (linear in ω)

The model is integrated with a first-order Euler step (dt = 0.1 s).
Stability is guaranteed because the load and friction terms are negative
feedback (restoring forces that increase with speed).

Units: SI.
"""

from __future__ import annotations
import math
from dataclasses import dataclass

# Crankshaft / flywheel inertia (representative Rotax 914 class)
J_KGM2 = 0.12         # kg·m²  — combined crank + flywheel inertia

# Propeller load coefficient: T_load = K_prop × ω²
# At max RPM (5800 RPM = 607 rad/s) with rated torque (~85 N·m):
#   K_prop = T_rated / ω_max² ≈ 85 / 607² ≈ 2.3e-4
K_PROP_NM_RAD2S2 = 2.3e-4  # N·m / (rad/s)²

# Viscous friction coefficient
B_FRICTION_NM_RADS = 0.02   # N·m / (rad/s)

# RPM limits
RPM_MIN = 500.0
RPM_MAX = 6500.0


@dataclass
class CrankshaftState:
    """
    Mutable crankshaft state — RPM and angular velocity.
    Must be carried across timesteps (first-order integrator).
    """
    rpm:   float = 800.0    # current engine speed (rev/min)
    omega: float = 800.0 * (2 * math.pi / 60)  # rad/s

    def to_dict(self) -> dict:
        return {
            "rpm":         round(self.rpm,          1),
            "omega_rads":  round(self.omega,         3),
        }


def step_crankshaft(
    state:    CrankshaftState,
    torque_engine: float,    # N·m from combustion model
    dt:       float = 0.1,   # timestep seconds
) -> float:
    """
    Advance crankshaft dynamics by one timestep.

    Updates `state` in place and returns the new RPM.

    Parameters
    ----------
    state : CrankshaftState
        Mutable crankshaft state (modified in place).
    torque_engine : float
        Mean engine (brake) torque from combustion model (N·m).
    dt : float
        Integration timestep (s).

    Returns
    -------
    float
        Updated engine speed (RPM).
    """
    omega = state.omega

    # Load torque: propeller (quadratic in ω)
    T_load = K_PROP_NM_RAD2S2 * omega * abs(omega)

    # Friction torque: viscous (linear in ω)
    T_friction = B_FRICTION_NM_RADS * omega

    # Net torque
    T_net = torque_engine - T_load - T_friction

    # Euler integration: dω = (T_net / J) * dt
    d_omega = (T_net / J_KGM2) * dt
    omega_new = omega + d_omega

    # Clamp to physical limits
    omega_min = RPM_MIN * (2 * math.pi / 60)
    omega_max = RPM_MAX * (2 * math.pi / 60)
    omega_new = max(omega_min, min(omega_max, omega_new))

    state.omega = omega_new
    state.rpm   = omega_new * 60.0 / (2 * math.pi)

    return state.rpm
