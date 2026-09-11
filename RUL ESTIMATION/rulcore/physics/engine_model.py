"""
engine_model.py
===============
Parameterised mean-value engine model (MVEM) for a Rotax 914-class aero piston
engine, written so that the SAME equations serve three roles:

    1. PLANT   - truth simulator, called with the engine's individual build
                 parameters and its true (degraded) health vector.
    2. TWIN    - Digital Twin predictor, called with the FLEET-NOMINAL build
                 parameters and a NOMINAL health vector (all ones).
    3. UKF     - measurement model, called with the fleet-nominal build
                 parameters and a CANDIDATE health vector (sigma points).

Sharing the equations is deliberate. What separates plant from twin is not a
different set of formulas but:
      (a) per-engine build scatter that the twin does not know about,
      (b) the true health vector that the twin assumes is nominal,
      (c) sensor bias/noise applied only on the plant side,
      (d) a genuine structural mismatch: the plant integrates transient
          dynamics while the twin evaluates the quasi-steady solution.
Those four differences are exactly what a real Digital Twin faces, and they are
what makes the healthy residual non-zero and non-trivial.

Everything is vectorised over numpy arrays. Scalars, (N,) arrays and broadcast
combinations all work, which is what lets the UKF evaluate 13 sigma points and
the particle filter evaluate 600 particles in a single call.

Causal chain (Part 7 requires degradation to propagate, not be pasted on):

    altitude, OAT ---> atmosphere (rho, Pa, Ta)
                              |
    throttle ------------> manifold pressure ---> air mass flow  <-- eta_vol
                              |                        |
                        ECU speed-density              |
                        (health-blind calibration)     |
                              |                        |
                     injection_command                 |
                              |                        |
                     delivered fuel  <-- eta_inj       |
                              |                        |
                              +--------> lambda -------+
                                             |
                                       burn completeness
                                             |
                     heat release ---> indicated power  <-- eta_comb
                                             |
                                 +-----------+-----------+
                                 |                       |
                          rejected heat            brake torque  <-- friction_mult
                                 |                       |
                      +----------+-------+          crankshaft <--> prop load
                      |                  |                |
                 head (CHT) <-- h_cool   exhaust (EGT)    RPM
                      |
                 oil temp / oil pressure <-- lub_health
                      |
                 vibration <-- friction_mult, combustion quality
"""

from __future__ import annotations

from typing import Dict, Mapping

import numpy as np

# --------------------------------------------------------------------------- #
# Physical constants
# --------------------------------------------------------------------------- #

R_AIR = 287.058          # J/(kg.K)
P0_PA = 101325.0         # Pa
T0_K = 288.15            # K
LAPSE_K_M = 0.0065       # K/m
G0 = 9.80665             # m/s^2
M_AIR = 0.0289644        # kg/mol
R_UNIV = 8.31447         # J/(mol.K)
RHO0 = P0_PA / (R_AIR * T0_K)

AFR_STOICH = 14.7
FUEL_DENSITY_KGL = 0.720
CP_EXHAUST = 1150.0      # J/(kg.K) exhaust gas specific heat

# Heat split of the rejected (non-work) energy.
F_HEAD = 0.26            # to cylinder head / cooling fins
F_EXHAUST = 0.47         # carried out with the exhaust stream
# remainder (0.27) leaves via oil, radiation and blow-by

RPM_MIN = 700.0
RPM_MAX = 6200.0
OMEGA_MIN = RPM_MIN * 2.0 * np.pi / 60.0
OMEGA_MAX = RPM_MAX * 2.0 * np.pi / 60.0


# --------------------------------------------------------------------------- #
# Atmosphere
# --------------------------------------------------------------------------- #

def atmosphere(altitude_ft, ambient_c):
    """ISA-with-temperature-offset atmosphere.

    `ambient_c` is the measured outside air temperature at ALTITUDE (not sea
    level), so a hot day at altitude is represented directly rather than being
    back-projected to sea level.

    Returns dict with pressure_pa, temperature_k, density, rho_ratio.
    """
    altitude_ft = np.asarray(altitude_ft, dtype=float)
    ambient_c = np.asarray(ambient_c, dtype=float)

    h_m = np.clip(altitude_ft * 0.3048, 0.0, 11000.0)

    # Standard pressure at altitude (pressure altitude is a pressure statement,
    # so it must not be modified by the temperature offset).
    t_std = T0_K - LAPSE_K_M * h_m
    exponent = (G0 * M_AIR) / (R_UNIV * LAPSE_K_M)
    p_a = P0_PA * (t_std / T0_K) ** exponent

    t_a = ambient_c + 273.15
    t_a = np.maximum(t_a, 200.0)

    rho = p_a / (R_AIR * t_a)
    return {
        "pressure_pa": p_a,
        "temperature_k": t_a,
        "density": rho,
        "rho_ratio": rho / RHO0,
        "t_std_k": t_std,
    }


# --------------------------------------------------------------------------- #
# Helper curves
# --------------------------------------------------------------------------- #

def _volumetric_efficiency(throttle, rpm, p):
    """Volumetric efficiency of the healthy engine (excludes eta_vol health)."""
    thr = np.clip(throttle, 0.0, 1.0)
    throttle_factor = p["eta_v_idle"] + (p["eta_v_max"] - p["eta_v_idle"]) * thr
    rpm_factor = 1.0 - 0.40 * ((rpm - p["rpm_ve_peak"]) / p["rpm_ve_peak"]) ** 2
    rpm_factor = np.clip(rpm_factor, 0.30, 1.0)
    return throttle_factor * rpm_factor


def _afr_target(throttle):
    """ECU target air/fuel ratio schedule (idle rich, cruise lean, WOT rich)."""
    thr = np.clip(throttle, 0.0, 1.0)
    afr_idle, afr_cruise, afr_wot = 13.0, 15.5, 12.5
    lo = afr_idle + (afr_cruise - afr_idle) * np.clip((thr - 0.15) / 0.45, 0.0, 1.0)
    hi = afr_cruise + (afr_wot - afr_cruise) * np.clip((thr - 0.60) / 0.40, 0.0, 1.0)
    return np.where(thr < 0.60, lo, hi)


def _burn_completeness(lam):
    """Fraction of injected fuel energy actually released, as a function of
    lambda. Peaks slightly rich, falls off when very lean or very rich."""
    d = lam - 0.95
    eta = 0.985 - 1.35 * d ** 2 - 0.55 * np.abs(d) ** 3
    return np.clip(eta, 0.55, 0.99)


def _exhaust_split_lambda(lam):
    """Multiplier on the exhaust heat split as a function of lambda.

    Leaning past best-power moves the burn later in the cycle and pushes more
    energy out of the exhaust valve, which is why EGT rises as a mixture leans
    toward peak-EGT. This is the mechanism that gives injector degradation an
    EGT signature distinct from combustion degradation.
    """
    return np.clip(1.0 + 0.30 * (lam - 0.95), 0.80, 1.28)


def _cooling_airflow_factor(rho_ratio, airspeed_factor):
    """Cooling mass flow over the fins relative to the reference condition."""
    return np.clip(rho_ratio * airspeed_factor, 0.18, 1.9)


# --------------------------------------------------------------------------- #
# Core algebraic evaluation at a given shaft speed
# --------------------------------------------------------------------------- #

def evaluate_at_speed(rpm, throttle, altitude_ft, ambient_c, theta, p,
                      airspeed_factor=1.0, load_factor=1.0):
    """Evaluate the full thermo-mechanical chain at a GIVEN shaft speed.

    This is the inner function used both by the steady-state solver (which
    searches for the rpm where brake torque equals load torque) and by the
    transient integrator (which uses the instantaneous rpm).

    Parameters
    ----------
    rpm : array_like            shaft speed, rev/min
    throttle : array_like       0..1
    altitude_ft, ambient_c      environment
    theta : mapping             health parameters (eta_inj, eta_comb, h_cool,
                                friction_mult, lub_health, eta_vol)
    p : mapping                 engine build parameters
    airspeed_factor : array_like  cooling airflow scaling (mission dependent)
    load_factor : array_like    propeller load scaling (pitch / airspeed)
    """
    rpm = np.clip(np.asarray(rpm, dtype=float), RPM_MIN, RPM_MAX)
    throttle = np.clip(np.asarray(throttle, dtype=float), 0.0, 1.0)
    omega = rpm * 2.0 * np.pi / 60.0
    cycles_per_s = rpm / 120.0            # 4-stroke: one cycle per 2 revs

    atm = atmosphere(altitude_ft, ambient_c)

    # ---- air path -------------------------------------------------------- #
    map_pa = atm["pressure_pa"] * (p["map_idle_frac"] + throttle * (1.0 - p["map_idle_frac"]))
    # Manifold charge is slightly heated by the head; small but keeps the
    # density calculation honest at high ambient temperature.
    t_man = atm["temperature_k"] + 12.0 + 8.0 * throttle
    rho_man = map_pa / (R_AIR * t_man)

    eta_v_healthy = _volumetric_efficiency(throttle, rpm, p)
    eta_v_actual = eta_v_healthy * theta["eta_vol"]

    displacement = p["displacement_m3"]
    m_dot_air = rho_man * displacement * cycles_per_s * eta_v_actual

    # ---- ECU fuel command (speed-density feedforward, health-blind) ------- #
    # The ECU estimates airflow from its calibration map. It does not know the
    # engine has lost volumetric efficiency, and it cannot see injector wear.
    m_dot_air_est = rho_man * displacement * cycles_per_s * eta_v_healthy
    afr_tgt = _afr_target(throttle)
    m_dot_fuel_cmd = m_dot_air_est / afr_tgt
    inj_cmd = m_dot_fuel_cmd / np.maximum(p["k_inj"] * cycles_per_s, 1e-12)
    inj_cmd = np.clip(inj_cmd, 0.0, 1.35)

    # ---- delivered fuel (this is where injector health bites) ------------- #
    m_dot_fuel = inj_cmd * p["k_inj"] * theta["eta_inj"] * cycles_per_s
    m_dot_fuel = np.maximum(m_dot_fuel, 1e-8)

    afr_actual = m_dot_air / m_dot_fuel
    lam = afr_actual / AFR_STOICH

    # ---- combustion ------------------------------------------------------- #
    eta_burn = _burn_completeness(lam)
    q_released = m_dot_fuel * p["lhv"] * eta_burn                    # W

    # eta_comb is the WORK-CONVERSION health: how much of the released heat is
    # turned into indicated work. Losing it pushes energy into the exhaust and
    # the head instead, which is the classic "power down, EGT up, CHT up,
    # fuel flow unchanged" signature.
    p_indicated = q_released * p["eta_therm"] * theta["eta_comb"]
    p_indicated = np.maximum(p_indicated, 0.0)

    t_indicated = p_indicated / np.maximum(omega, 1.0)
    t_friction = theta["friction_mult"] * (p["fric_b0"] + p["fric_b1"] * omega)
    t_brake = t_indicated - t_friction

    p_friction = t_friction * omega
    p_brake = np.maximum(t_brake, 0.0) * omega

    # ---- heat rejection --------------------------------------------------- #
    q_reject = np.maximum(q_released - p_indicated, 0.0)
    q_head = q_reject * F_HEAD
    q_exhaust = q_reject * F_EXHAUST * _exhaust_split_lambda(lam)

    # ---- steady-state thermal targets ------------------------------------- #
    # Cooling conductance has two multiplicative parts:
    #   (a) an ENGINE-DRIVEN part, because the same airflow that makes power also
    #       scales the fin-side convection (Nusselt ~ Re^0.8-ish). This is what
    #       compresses the idle-to-takeoff CHT range into the ~110-215 C band
    #       real air-cooled heads actually show.
    #   (b) a RAM part from flight speed and air density, which is independent of
    #       power setting. Keeping it separate is what stops the filter from
    #       confusing "high altitude / hot day" with "cooling degradation".
    cool_factor = _cooling_airflow_factor(atm["rho_ratio"], airspeed_factor)
    flow_ratio = m_dot_air / p["cool_mdot_ref"]
    engine_flow_term = 0.15 + 0.85 * np.clip(flow_ratio, 0.0, 6.0) ** 0.95
    ram_term = 0.45 + 0.55 * cool_factor
    hA_eff = p["cht_hA"] * theta["h_cool"] * engine_flow_term * ram_term
    cht_ss = ambient_c + p["cht_base_c"] + q_head / np.maximum(hA_eff, 1.0)

    # EGT: the adiabatic enthalpy rise of the exhaust stream, reduced by heat
    # lost to the exhaust system on the way to the probe. That loss fraction is
    # large at low mass flow, which is exactly why measured EGT is low at idle
    # and high at cruise even though the in-cylinder temperature is not.
    m_dot_exh = m_dot_air + m_dot_fuel
    egt_adiabatic = q_exhaust / np.maximum(m_dot_exh * CP_EXHAUST, 1e-6)
    probe_transfer = m_dot_exh / (m_dot_exh + p["egt_loss_mdot"])
    egt_ss = ambient_c + egt_adiabatic * probe_transfer * p["egt_gain_scale"]

    # ---- oil -------------------------------------------------------------- #
    # Oil temperature: coupled to the head, plus friction heating, minus the
    # cooler's ability to reject it (which falls with lubrication health).
    oil_temp_ss = (p["oil_temp_base_c"]
                   + p["oil_cht_coupling"] * cht_ss
                   + p["oil_fric_gain"] * p_friction / np.maximum(theta["lub_health"], 0.3))

    # Oil pressure: pump delivery scales with speed and with lubrication health
    # (pump/bearing clearance growth and viscosity loss), minus a viscosity term
    # that grows as the oil gets hot.
    pump = p["oil_press_idle_bar"] + np.maximum(rpm - 800.0, 0.0) * p["oil_k_pump"]
    visc_loss = np.maximum(oil_temp_ss - p["oil_temp_ref_c"], 0.0) * p["oil_k_visc"]
    oil_press_ss = pump * theta["lub_health"] - visc_loss / np.maximum(theta["lub_health"], 0.3)

    # ---- vibration -------------------------------------------------------- #
    # Mechanical wear grows broadband and 1X; combustion quality loss grows the
    # half-order and firing-order content.
    speed_term = (rpm / 5000.0) ** p["vib_rpm_exp"]
    load_term = 0.72 + 0.55 * np.clip(p_brake / 60000.0, 0.0, 1.6)
    mech_term = 1.0 + 2.35 * np.maximum(theta["friction_mult"] - 1.0, 0.0)
    comb_term = 1.0 + 1.55 * np.maximum(1.0 - theta["eta_comb"], 0.0) \
                    + 1.05 * np.maximum(1.0 - theta["eta_inj"], 0.0)
    lub_term = 1.0 + 1.25 * np.maximum(1.0 - theta["lub_health"], 0.0) ** 1.4
    vib_rms_ss = p["vib_base_g"] * speed_term * load_term * mech_term * comb_term * lub_term

    # ---- electrical ------------------------------------------------------- #
    alt_current = 8.5 + 5.0 * np.clip((rpm - 1200.0) / 4000.0, 0.0, 1.0)
    batt_v = 13.85 + 0.55 * np.clip((rpm - 1200.0) / 2500.0, 0.0, 1.0) \
             - 0.10 * np.maximum(1.0 - theta["lub_health"], 0.0)

    return {
        "rpm": rpm,
        "omega": omega,
        "manifold_pressure_kpa": map_pa / 1000.0,
        "air_mass_flow": m_dot_air,
        "eta_v_actual": eta_v_actual,
        "injection_command": inj_cmd,
        "fuel_mass_flow": m_dot_fuel,
        "fuel_flow_lph": m_dot_fuel / FUEL_DENSITY_KGL * 3600.0,
        "afr": afr_actual,
        "lambda": lam,
        "eta_burn": eta_burn,
        "heat_released_w": q_released,
        "power_indicated_w": p_indicated,
        "power_brake_w": p_brake,
        "power_friction_w": p_friction,
        "torque_indicated": t_indicated,
        "torque_friction": t_friction,
        "torque_brake": t_brake,
        "torque_load": p["k_prop"] * load_factor * omega ** 2,
        "q_head_w": q_head,
        "q_exhaust_w": q_exhaust,
        "cht_ss": cht_ss,
        "egt_ss": egt_ss,
        "oil_temp_ss": oil_temp_ss,
        "oil_press_ss": oil_press_ss,
        "vib_rms_ss": vib_rms_ss,
        "alternator_current": alt_current,
        "battery_voltage": batt_v,
        "bsfc_kg_per_kwh": m_dot_fuel * 3.6e6 / np.maximum(p_brake, 1.0),
    }


# --------------------------------------------------------------------------- #
# Steady-state solver
# --------------------------------------------------------------------------- #

def solve_steady_rpm(throttle, altitude_ft, ambient_c, theta, p,
                     airspeed_factor=1.0, load_factor=1.0, n_iter=22):
    """Find the shaft speed at which brake torque balances propeller load.

    Vectorised bisection. The torque balance g(rpm) = T_brake - T_load is
    monotonically decreasing over the usable speed range (brake torque falls and
    quadratic prop load rises), so bisection is unconditionally robust - no
    initial guess, no divergence, and it costs a fixed 22 evaluations regardless
    of how degraded the engine is. That determinism matters because this solver
    runs inside the UKF and inside the particle filter.

    22 bisections resolve the speed to 5500/2^22 = 1.3e-3 RPM, which is four
    orders of magnitude below the ~106 RPM healthy residual sigma. Anything
    finer is wasted work on a quantity that is then buried in noise.
    """
    shape = np.broadcast(np.asarray(throttle, dtype=float),
                         np.asarray(altitude_ft, dtype=float),
                         np.asarray(ambient_c, dtype=float),
                         np.asarray(theta["eta_inj"], dtype=float)).shape
    lo = np.full(shape, RPM_MIN, dtype=float)
    hi = np.full(shape, RPM_MAX, dtype=float)

    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        out = evaluate_at_speed(mid, throttle, altitude_ft, ambient_c, theta, p,
                                airspeed_factor=airspeed_factor,
                                load_factor=load_factor)
        g = out["torque_brake"] - out["torque_load"]
        too_slow = g > 0.0          # engine can still accelerate -> search higher
        lo = np.where(too_slow, mid, lo)
        hi = np.where(too_slow, hi, mid)

    return 0.5 * (lo + hi)


def steady_state(throttle, altitude_ft, ambient_c, theta, p,
                 airspeed_factor=1.0, load_factor=1.0):
    """Full quasi-steady operating point. This is what the Digital Twin predicts
    and what the UKF measurement model evaluates."""
    rpm = solve_steady_rpm(throttle, altitude_ft, ambient_c, theta, p,
                           airspeed_factor=airspeed_factor, load_factor=load_factor)
    out = evaluate_at_speed(rpm, throttle, altitude_ft, ambient_c, theta, p,
                            airspeed_factor=airspeed_factor, load_factor=load_factor)
    out["cht"] = out["cht_ss"]
    out["egt"] = out["egt_ss"]
    out["oil_temperature"] = out["oil_temp_ss"]
    out["oil_pressure"] = out["oil_press_ss"]
    out["vibration_rms"] = out["vib_rms_ss"]
    return out


# --------------------------------------------------------------------------- #
# Transient integration (fast scale)
# --------------------------------------------------------------------------- #

class FastState:
    """Mutable fast-timescale state, vectorised over a batch of snapshots."""

    __slots__ = ("rpm", "cht", "egt", "oil_t", "oil_p")

    def __init__(self, rpm, cht, egt, oil_t, oil_p):
        self.rpm = np.asarray(rpm, dtype=float)
        self.cht = np.asarray(cht, dtype=float)
        self.egt = np.asarray(egt, dtype=float)
        self.oil_t = np.asarray(oil_t, dtype=float)
        self.oil_p = np.asarray(oil_p, dtype=float)

    def copy(self) -> "FastState":
        return FastState(self.rpm.copy(), self.cht.copy(), self.egt.copy(),
                         self.oil_t.copy(), self.oil_p.copy())


def step_fast(state: FastState, throttle, altitude_ft, ambient_c, theta, p,
              dt=0.1, airspeed_factor=1.0, load_factor=1.0):
    """Advance the fast dynamics one step (default 0.1 s, i.e. 10 Hz).

    Crankshaft:  J dw/dt = T_brake - T_load
    Thermal/oil: first-order lags toward the operating-point targets.
    """
    out = evaluate_at_speed(state.rpm, throttle, altitude_ft, ambient_c, theta, p,
                            airspeed_factor=airspeed_factor, load_factor=load_factor)

    omega = state.rpm * 2.0 * np.pi / 60.0
    t_net = out["torque_brake"] - out["torque_load"]
    omega = omega + (t_net / p["j_rot"]) * dt
    omega = np.clip(omega, OMEGA_MIN, OMEGA_MAX)
    state.rpm = omega * 60.0 / (2.0 * np.pi)

    state.cht += dt * (out["cht_ss"] - state.cht) / p["tau_cht_s"]
    state.egt += dt * (out["egt_ss"] - state.egt) / p["tau_egt_s"]
    state.oil_t += dt * (out["oil_temp_ss"] - state.oil_t) / p["tau_oil_t_s"]
    state.oil_p += dt * (out["oil_press_ss"] - state.oil_p) / p["tau_oil_p_s"]

    state.cht = np.clip(state.cht, -20.0, 400.0)
    state.egt = np.clip(state.egt, 50.0, 1150.0)
    state.oil_t = np.clip(state.oil_t, 10.0, 190.0)
    state.oil_p = np.clip(state.oil_p, 0.2, 9.0)

    return out


# --------------------------------------------------------------------------- #
# Parameter helpers
# --------------------------------------------------------------------------- #

def nominal_params(overrides: Mapping[str, float] | None = None) -> Dict[str, float]:
    """Fleet-nominal engine build parameters (what the Digital Twin assumes)."""
    from ..config import ENGINE_NOMINAL

    p = dict(ENGINE_NOMINAL)
    p.setdefault("oil_fric_gain", 2.30e-3)
    p.setdefault("egt_gain_scale", 1.0)
    p.setdefault("vib_rpm_exp", 1.55)
    if overrides:
        p.update(overrides)
    return p


def nominal_health(shape=()) -> Dict[str, np.ndarray]:
    """A health vector of all-nominal values, broadcastable to `shape`."""
    from ..config import HEALTH_NOMINAL

    return {k: np.full(shape, v, dtype=float) if shape else np.array(v, dtype=float)
            for k, v in HEALTH_NOMINAL.items()}
