"""
engine_physics.py
==================
Mean Value Engine Model (MVEM) for a MALE UAV aero piston engine.

This module implements a lightweight, deterministic, first-order-lag based
physics model of a Rotax 914/915-class piston aero-engine. It produces two
parallel trajectories every time step:

    * `expected` -> the clean, fault-free, noise-free "digital twin ground
      truth" trajectory. This represents what the engine SHOULD be doing
      given the current pilot/control inputs.

    * `actual`   -> the "real" telemetry trajectory. It includes:
        - the same first-order dynamics as `expected`
        - fault-mode effects (if a fault is active)
        - Gaussian sensor noise (to mimic real telemetry)

Comparing `actual` vs `expected` (the residual) is the basis of the
model-based fault detection used by the frontend dashboard.

Physics approach
-----------------
Every physical channel (RPM, CHT, EGT, ...) is modeled as a first-order lag
system, i.e. a system that exponentially approaches a "target" value that is
itself a (static) function of the current control inputs:

    state += dt * (target - state) / tau

This is a standard reduced-order approximation used in MVEM-style engine
models: instead of resolving the fast, cylinder-by-cylinder combustion
events, we track the mean (time-averaged) evolution of thermodynamic and
mechanical states, each relaxing towards a quasi-steady operating point with
its own characteristic time constant (tau). Fast-changing quantities (RPM)
get a short tau; slow thermal masses (CHT, oil temperature) get a long tau.

Fault modes
-----------
Eight independent, severity-scaled fault modes can be injected (fault_id 1-8).
Each fault only perturbs the `actual` trajectory -- `expected` always
represents the healthy baseline -- which is what makes the residual
(actual - expected) a meaningful fault indicator.

Fault 1  – Misfire: intermittent RPM/EGT drops + vibration spikes
Fault 2  – Cooling Degradation: CHT gradually rises (accumulates cooling_gain)
Fault 3  – Lubrication Issue: oil pressure decays, oil temp rises, RPM droops
Fault 4  – Sensor Drift (EGT): random-walk bias on EGT sensor only
Fault 5  – Combustion Instability: high-frequency RPM oscillations + vibration
Fault 6  – Injector Abnormality: erratic fuel flow causing EGT swings
Fault 7  – Overheating Trend: rapid CHT + EGT rise (faster than cooling deg.)
Fault 8  – Alternator Failure: progressive voltage sag + electrical ripple

Determinism
------------
All randomness (sensor noise, fault dice-rolls, sensor drift random walk)
is drawn from a single `numpy.random.Generator` created with a fixed seed
(`np.random.default_rng(seed=42)`) so that, given identical control input
histories, the simulation is perfectly reproducible.

Units
------
RPM          : rev/min (RPM)
CHT          : degrees Celsius
EGT          : degrees Celsius
Oil Pressure : bar
Oil Temp     : degrees Celsius
Fuel Flow    : L/hr
Alt Voltage  : V
Vibration    : g RMS
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Simulation constants
# ----------------------------------------------------------------------

DT = 0.1  # simulation time step, seconds (10 Hz update rate)

# Physical clamp ranges (min, max) -- realistic bounds for a small aero
# piston engine (Rotax 914/915-class).
# Oil Pressure in BAR (not psi).
CLAMP_RANGES: Dict[str, Tuple[float, float]] = {
    "RPM":        (800.0,  6000.0),
    "CHT":        (50.0,   280.0),    # deg C
    "EGT":        (100.0,  950.0),    # deg C
    "OilPress":   (1.0,    8.0),      # bar
    "OilTemp":    (40.0,   130.0),    # deg C
    "FuelFlow":   (2.0,    45.0),     # L/hr
    "AltVoltage": (10.5,   15.0),     # V
    "Vibration":  (0.02,   2.5),      # g RMS
}

# First-order lag time constants (seconds).
# Fast mechanical states (RPM, fuel flow, oil pressure, alternator) settle
# quickly; large thermal masses (CHT, oil temperature) settle slowly.
TAU: Dict[str, float] = {
    "RPM":        1.5,
    "CHT":        25.0,
    "EGT":        3.0,
    "OilPress":   2.0,
    "OilTemp":    20.0,
    "FuelFlow":   0.8,
    "AltVoltage": 1.0,
    "Vibration":  0.3,
}

# Gaussian sensor-noise standard deviations applied ONLY to the `actual`
# (measured) trajectory, to mimic real telemetry noise.
NOISE_STD: Dict[str, float] = {
    "RPM":        15.0,    # RPM
    "CHT":        1.5,     # deg C
    "EGT":        4.0,     # deg C
    "OilPress":   0.05,    # bar
    "OilTemp":    0.8,     # deg C
    "FuelFlow":   0.3,     # L/hr
    "AltVoltage": 0.05,    # V
    "Vibration":  0.01,    # g RMS (baseline)
}

# Fault catalog: fault_id -> display name
FAULT_NAMES: Dict[int, str] = {
    0: "Healthy",
    1: "Misfire",
    2: "Cooling_Degradation",
    3: "Lubrication_Issue",
    4: "Sensor_Drift_EGT",
    5: "Combustion_Instability",
    6: "Injector_Abnormality",
    7: "Overheating_Trend",
    8: "Alternator_Failure",
}


def clamp(value: float, key: str) -> float:
    """Clamp a physical value into its realistic operating range."""
    lo, hi = CLAMP_RANGES[key]
    return min(max(value, lo), hi)


# ----------------------------------------------------------------------
# Data structures
# ----------------------------------------------------------------------

@dataclass
class Controls:
    """Pilot / environment control inputs, as received from the frontend."""
    throttle: float    = 0.5        # 0.0 – 1.0
    altitude_ft: float = 3000.0     # 0 – 20000 ft
    ambient_c: float   = 15.0       # -10 – 50 deg C
    fault_id: int      = 0          # 0 = healthy, 1-8 = fault mode
    severity: float    = 0.5        # 0.0 – 1.0


@dataclass
class EngineState:
    """
    Full mutable state of the digital twin: both the clean (expected) and
    faulted/noisy (actual) trajectories, plus internal fault accumulators
    that must persist across steps (e.g. cooling degradation accumulates
    over time; sensor drift is a random walk).
    """

    t: float = 0.0

    # --- Expected (clean, fault-free) trajectory -----------------------
    rpm_exp:        float = 800.0
    cht_exp:        float = 70.0
    egt_exp:        float = 300.0
    oil_press_exp:  float = 4.9     # bar  (at idle: 5.5 - 0.5*0.8 ≈ 4.9 ish)
    oil_temp_exp:   float = 60.0
    fuel_flow_exp:  float = 6.0
    alt_v_exp:      float = 13.3
    vib_exp:        float = 0.02

    # --- Actual (faulted + noisy) trajectory ----------------------------
    rpm_act:        float = 800.0
    cht_act:        float = 70.0
    egt_act:        float = 300.0
    oil_press_act:  float = 4.9
    oil_temp_act:   float = 60.0
    fuel_flow_act:  float = 6.0
    alt_v_act:      float = 13.3
    vib_act:        float = 0.02

    # --- Fault-internal persistent accumulators -------------------------
    cooling_gain:       float = 0.0    # Fault 2: accumulated CHT offset (deg C)
    overheat_gain:      float = 0.0    # Fault 7: accumulated CHT+EGT offset (deg C)
    lube_gain:          float = 0.0    # Fault 3: accumulated lube degradation
    alt_fail_gain:      float = 0.0    # Fault 8: accumulated voltage sag (V)
    egt_drift_bias:     float = 0.0    # Fault 4: random-walk sensor bias (deg C)
    misfire_cooldown:   float = 0.0    # Fault 1: seconds until next dice-roll
    misfire_active:     float = 0.0    # Fault 1: seconds remaining in event
    instability_phase:  float = 0.0    # Fault 5: oscillator phase (radians)

    running: bool = False


def create_initial_state() -> EngineState:
    """Factory for a fresh engine state, e.g. after a 'reset' command."""
    return EngineState()


# ----------------------------------------------------------------------
# Steady-state target functions
# ----------------------------------------------------------------------
# These functions describe the quasi-static operating point the engine
# would settle at for a *given, held-constant* set of control inputs. The
# first-order lag equations then describe *how fast* the engine actually
# gets there. All targets are computed identically for `expected`; the
# `actual` targets are then perturbed by whichever fault is active.


def _rho_ratio(altitude_ft: float) -> float:
    """Air density ratio relative to sea level using barometric formula."""
    return math.exp(-altitude_ft / 25000.0)


def _target_rpm(c: Controls) -> float:
    """RPM target: linear with throttle per spec."""
    return 800.0 + c.throttle * 5200.0


def _target_egt(c: Controls) -> float:
    """EGT target: throttle + altitude lean effect per spec."""
    return 300.0 + c.throttle * 550.0 + c.altitude_ft * 0.002


def _target_cht(c: Controls) -> float:
    """CHT target: throttle heat minus altitude cooling per spec."""
    return 70.0 + c.throttle * 90.0 - c.altitude_ft * 0.005


def _target_oil_press(c: Controls) -> float:
    """Oil pressure: decreases slightly at higher throttle (per spec)."""
    return 5.5 - c.throttle * 0.8


def _target_oil_temp(c: Controls) -> float:
    """Oil temp: thermally driven by throttle load per spec."""
    return 60.0 + c.throttle * 50.0


def _target_fuel_flow(c: Controls) -> float:
    """Fuel flow: throttle-scaled, with air-density correction per spec."""
    rho = _rho_ratio(c.altitude_ft)
    return 6.0 + c.throttle * 34.0 * rho


def _target_alt_voltage(rpm: float) -> float:
    """Alternator voltage: rises with RPM per spec."""
    return 13.2 + 1.2 * (rpm / 5800.0)


def _target_vibration(rpm: float) -> float:
    """Baseline mechanical vibration scales gently with RPM."""
    return 0.02 + (rpm / 6000.0) * 0.30


# ----------------------------------------------------------------------
# Main step function
# ----------------------------------------------------------------------


def step(
    state: EngineState,
    controls: Controls,
    rng: np.random.Generator,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Advance the engine model by one fixed timestep (DT = 0.1 s).

    Parameters
    ----------
    state : EngineState
        Mutable simulation state (updated in place).
    controls : Controls
        Current pilot/environment control inputs.
    rng : np.random.Generator
        Seeded random generator, used for all noise / fault randomness so
        that the whole simulation is reproducible run-to-run.

    Returns
    -------
    (expected, actual) : Tuple[Dict[str, float], Dict[str, float]]
        Clean baseline and faulted/noisy telemetry dictionaries, both
        keyed by channel name (RPM, CHT, EGT, OilPress, OilTemp, FuelFlow,
        AltVoltage, Vibration).
    """

    dt = DT
    state.t += dt

    # ----------------------------------------------------------------
    # Baseline (healthy) targets from control inputs
    # ----------------------------------------------------------------
    rpm_t      = _target_rpm(controls)
    egt_t      = _target_egt(controls)
    cht_t      = _target_cht(controls)
    oil_p_t    = _target_oil_press(controls)
    oil_temp_t = _target_oil_temp(controls)
    fuel_t     = _target_fuel_flow(controls)
    alt_v_t    = _target_alt_voltage(rpm_t)
    vib_t      = _target_vibration(rpm_t)

    # ----------------------------------------------------------------
    # Expected (clean) trajectory: first-order lag
    # These NEVER see faults or noise -- they represent the digital
    # twin's "healthy" reference used to compute residuals.
    # ----------------------------------------------------------------
    state.rpm_exp       += dt * (rpm_t      - state.rpm_exp)       / TAU["RPM"]
    state.cht_exp       += dt * (cht_t      - state.cht_exp)       / TAU["CHT"]
    state.egt_exp       += dt * (egt_t      - state.egt_exp)       / TAU["EGT"]
    state.oil_press_exp += dt * (oil_p_t    - state.oil_press_exp) / TAU["OilPress"]
    state.oil_temp_exp  += dt * (oil_temp_t - state.oil_temp_exp)  / TAU["OilTemp"]
    state.fuel_flow_exp += dt * (fuel_t     - state.fuel_flow_exp) / TAU["FuelFlow"]
    state.alt_v_exp     += dt * (alt_v_t    - state.alt_v_exp)     / TAU["AltVoltage"]
    state.vib_exp       += dt * (vib_t      - state.vib_exp)       / TAU["Vibration"]

    # Clamp expected trajectory to physical bounds.
    state.rpm_exp       = clamp(state.rpm_exp,       "RPM")
    state.cht_exp       = clamp(state.cht_exp,       "CHT")
    state.egt_exp       = clamp(state.egt_exp,       "EGT")
    state.oil_press_exp = clamp(state.oil_press_exp, "OilPress")
    state.oil_temp_exp  = clamp(state.oil_temp_exp,  "OilTemp")
    state.fuel_flow_exp = clamp(state.fuel_flow_exp, "FuelFlow")
    state.alt_v_exp     = clamp(state.alt_v_exp,     "AltVoltage")
    state.vib_exp       = clamp(state.vib_exp,       "Vibration")

    # ----------------------------------------------------------------
    # Actual (faulted) targets: start from healthy targets, then let
    # the active fault mode perturb them.
    # ----------------------------------------------------------------
    rpm_t_act      = rpm_t
    cht_t_act      = cht_t
    egt_t_act      = egt_t
    oil_p_t_act    = oil_p_t
    oil_temp_t_act = oil_temp_t
    fuel_t_act     = fuel_t
    vib_t_act      = vib_t
    alt_v_t_act    = alt_v_t

    fault_id = controls.fault_id
    severity = float(np.clip(controls.severity, 0.0, 1.0))

    # ---- Fault 1: Misfire --------------------------------------------
    # Intermittent RPM/EGT drops + vibration spikes (~3-8 Hz random pulses).
    # A stochastic dice-roll process: during "cooldown" we wait, then roll
    # a chance (scaled by severity) to trigger a short ~0.3s misfire event.
    if fault_id == 1 and severity > 0.0:
        state.misfire_cooldown -= dt
        if state.misfire_active > 0.0:
            rpm_t_act -= severity * 500.0
            egt_t_act -= severity * 200.0
            vib_t_act += severity * 2.0
            state.misfire_active -= dt
        elif state.misfire_cooldown <= 0.0:
            if rng.random() < severity * 0.35:
                state.misfire_active = 0.3
            state.misfire_cooldown = rng.uniform(0.15, 0.35)  # 3-8 Hz burst rate
    else:
        state.misfire_cooldown = 0.0
        state.misfire_active   = 0.0

    # ---- Fault 2: Cooling Degradation --------------------------------
    # Heat soak accumulates in `cooling_gain`. Clears gradually when fault
    # is removed, simulating gradual recovery of the cooling system.
    if fault_id == 2 and severity > 0.0:
        state.cooling_gain += dt * severity * 0.6
        state.cooling_gain  = min(state.cooling_gain, 80.0)
    else:
        state.cooling_gain = max(0.0, state.cooling_gain - dt * 2.0)
    cht_t_act      += state.cooling_gain
    oil_temp_t_act += state.cooling_gain * 0.3

    # ---- Fault 3: Lubrication Issue ----------------------------------
    # Oil pump degradation causes: oil pressure to fall, oil temp to rise
    # (friction heat), and mild RPM droops from increased mechanical drag.
    if fault_id == 3 and severity > 0.0:
        state.lube_gain += dt * severity * 0.05
        state.lube_gain  = min(state.lube_gain, 1.0)
        decay = 1.0 - math.exp(-state.lube_gain * 5.0)
        oil_p_t_act    -= decay * severity * 3.0
        oil_temp_t_act += decay * severity * 35.0
        rpm_t_act      -= decay * severity * 300.0
    else:
        state.lube_gain = max(0.0, state.lube_gain - dt * 0.1)

    # ---- Fault 5: Combustion Instability ----------------------------
    # Unstable combustion phasing produces high-frequency RPM oscillations
    # plus elevated vibration. Sinusoidal at ~7 Hz, superimposed on actual.
    instability_oscillation = 0.0
    if fault_id == 5 and severity > 0.0:
        state.instability_phase += dt * 2.0 * math.pi * 7.0
        instability_oscillation  = severity * 180.0 * math.sin(state.instability_phase)
        vib_t_act += severity * 1.5
    else:
        state.instability_phase = 0.0

    # ---- Fault 6: Injector Abnormality --------------------------------
    # Erratic injector behaviour causes swings in delivered fuel flow.
    # Rich mixture → cooler EGT; lean mixture → hotter EGT.
    injector_egt_swing = 0.0
    injector_fuel_swing = 0.0
    if fault_id == 6 and severity > 0.0:
        fuel_err = rng.normal(0.0, severity * 6.0)
        injector_fuel_swing = fuel_err
        injector_egt_swing  = -fuel_err * 8.0   # rich=cooler, lean=hotter

    # ---- Fault 7: Overheating Trend ----------------------------------
    # More rapid CHT + EGT accumulation than fault 2 (blocked cowl flap /
    # cooling duct). Separate accumulator `overheat_gain`.
    if fault_id == 7 and severity > 0.0:
        state.overheat_gain += dt * severity * 1.5
        state.overheat_gain  = min(state.overheat_gain, 120.0)
    else:
        state.overheat_gain = max(0.0, state.overheat_gain - dt * 1.0)
    cht_t_act += state.overheat_gain
    egt_t_act += state.overheat_gain * 0.7

    # ---- Fault 8: Alternator Failure ----------------------------------
    # Progressive voltage sag as alternator output degrades. Battery
    # supplies bus but cannot sustain full 14V. Ripple noise increases.
    if fault_id == 8 and severity > 0.0:
        state.alt_fail_gain += dt * severity * 0.08
        state.alt_fail_gain  = min(state.alt_fail_gain, 3.0)
        alt_v_t_act -= state.alt_fail_gain
    else:
        state.alt_fail_gain = max(0.0, state.alt_fail_gain - dt * 0.2)

    # ----------------------------------------------------------------
    # Actual trajectory: first-order lag towards (perturbed) targets
    # ----------------------------------------------------------------
    state.rpm_act       += dt * (rpm_t_act      - state.rpm_act)       / TAU["RPM"]
    state.rpm_act       += instability_oscillation * dt / TAU["RPM"] * 5.0
    state.cht_act       += dt * (cht_t_act      - state.cht_act)       / TAU["CHT"]
    state.egt_act       += dt * (egt_t_act      - state.egt_act)       / TAU["EGT"]
    state.oil_press_act += dt * (oil_p_t_act    - state.oil_press_act) / TAU["OilPress"]
    state.oil_temp_act  += dt * (oil_temp_t_act - state.oil_temp_act)  / TAU["OilTemp"]
    state.fuel_flow_act += dt * (fuel_t_act     - state.fuel_flow_act) / TAU["FuelFlow"]
    state.fuel_flow_act += injector_fuel_swing * dt / TAU["FuelFlow"]
    state.alt_v_act     += dt * (alt_v_t_act    - state.alt_v_act)     / TAU["AltVoltage"]
    state.vib_act       += dt * (vib_t_act      - state.vib_act)       / TAU["Vibration"]

    # ---- Fault 4: Sensor Drift (EGT only) ----------------------------
    # The TRUE exhaust temperature is unaffected; only the SENSOR reading
    # drifts via a random walk (thermocouple degradation). Added after the
    # physical lag update, never touches `expected`.
    if fault_id == 4 and severity > 0.0:
        state.egt_drift_bias += rng.normal(0.0, severity * 3.0) * dt
        state.egt_drift_bias  = float(np.clip(state.egt_drift_bias, -120.0, 120.0))
    else:
        state.egt_drift_bias *= max(0.0, 1.0 - dt / 5.0)

    egt_reported = state.egt_act + state.egt_drift_bias

    # Alternator failure ripple noise (post-lag, on reported value)
    alt_ripple = 0.0
    if fault_id == 8 and severity > 0.0:
        alt_ripple = rng.normal(0.0, severity * 0.15)

    # ----------------------------------------------------------------
    # Add Gaussian sensor noise to all actual readings
    # ----------------------------------------------------------------
    rpm_noisy      = state.rpm_act       + rng.normal(0.0, NOISE_STD["RPM"])
    cht_noisy      = state.cht_act       + rng.normal(0.0, NOISE_STD["CHT"])
    egt_noisy      = egt_reported        + rng.normal(0.0, NOISE_STD["EGT"])
    egt_noisy     += injector_egt_swing  * dt               # injector EGT swing
    oil_p_noisy    = state.oil_press_act + rng.normal(0.0, NOISE_STD["OilPress"])
    oil_temp_noisy = state.oil_temp_act  + rng.normal(0.0, NOISE_STD["OilTemp"])
    fuel_noisy     = state.fuel_flow_act + rng.normal(0.0, NOISE_STD["FuelFlow"])
    alt_v_noisy    = state.alt_v_act     + rng.normal(0.0, NOISE_STD["AltVoltage"]) + alt_ripple
    vib_noisy      = state.vib_act       + abs(rng.normal(0.0, NOISE_STD["Vibration"]))

    # ----------------------------------------------------------------
    # Clamp reported values to physical bounds
    # ----------------------------------------------------------------
    expected = {
        "RPM":        clamp(state.rpm_exp,       "RPM"),
        "CHT":        clamp(state.cht_exp,       "CHT"),
        "EGT":        clamp(state.egt_exp,       "EGT"),
        "OilPress":   clamp(state.oil_press_exp, "OilPress"),
        "OilTemp":    clamp(state.oil_temp_exp,  "OilTemp"),
        "FuelFlow":   clamp(state.fuel_flow_exp, "FuelFlow"),
        "AltVoltage": clamp(state.alt_v_exp,     "AltVoltage"),
        "Vibration":  clamp(state.vib_exp,       "Vibration"),
    }

    actual = {
        "RPM":        clamp(rpm_noisy,      "RPM"),
        "CHT":        clamp(cht_noisy,      "CHT"),
        "EGT":        clamp(egt_noisy,      "EGT"),
        "OilPress":   clamp(oil_p_noisy,    "OilPress"),
        "OilTemp":    clamp(oil_temp_noisy, "OilTemp"),
        "FuelFlow":   clamp(fuel_noisy,     "FuelFlow"),
        "AltVoltage": clamp(alt_v_noisy,    "AltVoltage"),
        "Vibration":  clamp(vib_noisy,      "Vibration"),
    }

    return expected, actual


def compute_residual(
    expected: Dict[str, float],
    actual: Dict[str, float],
) -> Dict[str, float]:
    """
    Compute the fault-detection residual for every channel.

    IMPORTANT: residual is defined strictly as `actual - expected`
    (never the reverse), so that a POSITIVE residual means the real
    engine is reading HIGHER than the healthy digital-twin baseline
    (e.g. CHT residual > 0 => engine running hotter than expected),
    and a NEGATIVE residual means it is reading LOWER than expected
    (e.g. RPM residual < 0 during a misfire event).
    """
    return {key: actual[key] - expected[key] for key in expected}
