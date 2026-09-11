"""
eol.py
======
End-of-life definition and health index construction (Parts 8 and 9).

DESIGN PRINCIPLE
----------------
EOL is defined by ENGINEERING LIMITS, not by a health index, and the health
index is then defined so that it is exactly consistent with those limits. The
causality runs:

    health parameters  ->  reference-condition performance  ->  limit margins
                                                                     |
                              EOL = first persistent margin >= 1      |
                              HI  = 1 - max(margin)  <-----------------

This is the opposite of the common shortcut `RUL = 200 * health_index`, where
the label is manufactured from the very quantity the model is asked to predict.
Here the health index is a diagnostic readout of the same margins that define
EOL, so HI = 0 at EOL by construction rather than by assumption.

THE REFERENCE CONDITION ("virtual power assurance check")
--------------------------------------------------------
Limits are evaluated at a fixed reference condition rather than at whatever the
aircraft happened to be doing. A CHT of 200 C means something entirely different
at 45 C sea level and at -20 C / 18000 ft, so a limit applied to raw flight data
would mostly measure the weather. At every snapshot the simulator therefore
evaluates the engine's TRUE physics at a standard condition, noise free. This is
ground truth only: it is never a model input.

Limits are relative to the SAME ENGINE's commissioning (green-run) values where
the quantity has meaningful build scatter, and absolute where airworthiness sets
a hard number (oil pressure, oil temperature, CHT red line). Using each engine's
own baseline is what real fleets do, and it removes build scatter from the limit
so the criterion measures degradation rather than birth strength.

CRITERIA
--------
  C1 power    max brake power at WOT falls below 90% of commissioning
  C2 bsfc     brake specific fuel consumption rises more than 10%
  C3a cht     CHT at reference cruise rises more than 40 C above commissioning
  C3b cht     CHT at WOT exceeds the 232 C absolute red line
  C4 oil_p    oil pressure at reference cruise falls below 2.6 bar
  C5 oil_t    oil temperature at reference cruise exceeds 118 C
  C6 vib      vibration RMS at reference cruise exceeds 1.70x commissioning

EOL is the first snapshot at which any margin reaches 1.0 and stays there for
EOL_PERSISTENCE_SNAPSHOTS consecutive snapshots (45 minutes), which prevents a
single noisy excursion from declaring the engine dead.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from ..config import (EOL_CRITERIA, EOL_REF_CONDITION, EOL_PERSISTENCE_SNAPSHOTS,
                      HEALTH_PARAMS, HEALTH_NOMINAL)
from ..physics.engine_model import steady_state, nominal_params, nominal_health

CRITERION_NAMES = ["power", "bsfc", "cht_rise", "cht_abs", "oil_press", "oil_temp", "vib"]


def reference_performance(theta: Dict[str, np.ndarray], p: Dict[str, float]) -> Dict[str, np.ndarray]:
    """Evaluate the engine at the standard reference condition and at WOT.

    Returns the handful of quantities the EOL criteria are built from. Runs the
    true physics with no sensor noise - this is a virtual test-cell check.
    """
    ref = EOL_REF_CONDITION
    cruise = steady_state(ref["throttle"], ref["altitude_ft"], ref["ambient_c"],
                          theta, p, airspeed_factor=1.0,
                          load_factor=ref["load_factor"])
    wot = steady_state(1.0, ref["altitude_ft"], ref["ambient_c"],
                       theta, p, airspeed_factor=1.0, load_factor=ref["load_factor"])
    return {
        "power_wot_w": wot["power_brake_w"],
        "cht_wot_c": wot["cht"],
        "bsfc": cruise["bsfc_kg_per_kwh"],
        "cht_c": cruise["cht"],
        "oil_press_bar": cruise["oil_pressure"],
        "oil_temp_c": cruise["oil_temperature"],
        "vib_rms_g": cruise["vibration_rms"],
        "power_cruise_w": cruise["power_brake_w"],
        "rpm_cruise": cruise["rpm"],
        "fuel_flow_lph": cruise["fuel_flow_lph"],
        "egt_c": cruise["egt"],
    }


def criterion_margins(perf: Dict[str, np.ndarray],
                      baseline: Dict[str, float]) -> Dict[str, np.ndarray]:
    """Normalised consumption of each limit: 0 at commissioning, 1 at the limit.

    Margins are allowed to exceed 1 (the engine can run past its limit in the
    simulation); they are clipped only where a negative value would be
    meaningless.
    """
    c = EOL_CRITERIA
    m = {}

    # C1 power loss at WOT, relative to this engine's own green run
    loss = 1.0 - perf["power_wot_w"] / np.maximum(baseline["power_wot_w"], 1e-6)
    m["power"] = np.maximum(loss, 0.0) / c["power_loss_frac"]

    # C2 BSFC rise
    rise = perf["bsfc"] / np.maximum(baseline["bsfc"], 1e-9) - 1.0
    m["bsfc"] = np.maximum(rise, 0.0) / c["bsfc_rise_frac"]

    # C3a CHT rise at reference cruise
    m["cht_rise"] = np.maximum(perf["cht_c"] - baseline["cht_c"], 0.0) / c["cht_rise_c"]

    # C3b CHT absolute red line at WOT. Expressed as consumed headroom from the
    # commissioning WOT temperature to the red line.
    head = np.maximum(c["cht_limit_c"] - baseline["cht_wot_c"], 5.0)
    m["cht_abs"] = np.maximum(perf["cht_wot_c"] - baseline["cht_wot_c"], 0.0) / head

    # C4 oil pressure: consumed headroom from commissioning down to the minimum
    drop_avail = np.maximum(baseline["oil_press_bar"] - c["oil_press_min_bar"], 0.05)
    m["oil_press"] = np.maximum(baseline["oil_press_bar"] - perf["oil_press_bar"], 0.0) / drop_avail

    # C5 oil temperature: consumed headroom up to the limit
    head_t = np.maximum(c["oil_temp_limit_c"] - baseline["oil_temp_c"], 3.0)
    m["oil_temp"] = np.maximum(perf["oil_temp_c"] - baseline["oil_temp_c"], 0.0) / head_t

    # C6 vibration growth
    ratio = perf["vib_rms_g"] / np.maximum(baseline["vib_rms_g"], 1e-9)
    m["vib"] = np.maximum(ratio - 1.0, 0.0) / max(c["vib_rms_ratio"] - 1.0, 1e-6)

    return m


def health_index(theta: Dict[str, np.ndarray], p: Dict[str, float],
                 baseline: Dict[str, float]) -> np.ndarray:
    """Interpretable health index: 1.0 at commissioning, 0.0 at EOL.

    WEAKEST-LINK formulation. HI = 1 - max_j margin_j.

    Why the maximum and not an average: EOL is a FIRST-CROSSING event over a set
    of limits, so the engine's life is governed by whichever limit is closest to
    being violated. Averaging the margins would let a healthy oil system mask a
    cylinder head that is about to exceed its red line, and would break the
    HI = 0 <=> EOL equivalence that makes the index meaningful.

    The index is a pure function of the health parameters (through the physics),
    so the deployed system can evaluate it from UKF estimates with no access to
    ground truth.
    """
    perf = reference_performance(theta, p)
    m = criterion_margins(perf, baseline)
    worst = np.max(np.stack([np.asarray(m[k], dtype=float) for k in CRITERION_NAMES], axis=0),
                   axis=0)
    return 1.0 - worst


def commissioning_baseline(theta0: Dict[str, float], p: Dict[str, float]) -> Dict[str, float]:
    """Green-run reference values for one specific engine build."""
    perf = reference_performance({k: np.asarray(v, dtype=float) for k, v in theta0.items()}, p)
    return {k: float(np.asarray(v).ravel()[0]) for k, v in perf.items()}


def find_eol_index(margins_over_time: np.ndarray,
                   persistence: int = EOL_PERSISTENCE_SNAPSHOTS) -> int:
    """First index at which the worst margin reaches 1.0 and stays there.

    `margins_over_time` is the per-snapshot worst margin (shape (T,)).
    Returns -1 if the engine never reaches EOL within the trajectory.
    """
    over = margins_over_time >= 1.0
    if over.sum() == 0:
        return -1
    n = len(over)
    run = 0
    for i in range(n):
        if over[i]:
            run += 1
            if run >= persistence:
                return i - persistence + 1
        else:
            run = 0
    # Trailing partial run at the very end of the trajectory still counts if the
    # trajectory simply stopped; otherwise no EOL.
    return -1


# --------------------------------------------------------------------------- #
# Derived per-parameter EOL values
# --------------------------------------------------------------------------- #

def calibrate_health_eol(p: Dict[str, float] | None = None,
                         n_iter: int = 48) -> Dict[str, float]:
    """Solve for the value of each health parameter at which, acting ALONE, it
    first trips one of the EOL criteria.

    These derived values are what the degradation model uses to express wear as
    a fraction of a parameter's usable range. Deriving them - rather than
    hand-picking them - guarantees the bookkeeping in degradation.py is
    consistent with the EOL definition in this file. If a limit is retuned, the
    ranges follow automatically.
    """
    p = p or nominal_params()
    theta0 = nominal_health()
    base = commissioning_baseline({k: float(v) for k, v in HEALTH_NOMINAL.items()}, p)

    search_range = {
        "eta_inj": (0.60, 1.0),
        "eta_comb": (0.60, 1.0),
        "h_cool": (0.35, 1.0),
        "friction_mult": (1.0, 3.0),
        "lub_health": (0.35, 1.0),
        "eta_vol": (0.60, 1.0),
    }

    out = {}
    for k in HEALTH_PARAMS:
        lo, hi = search_range[k]
        # `lo` is the degraded end for falling parameters, `hi` for rising ones.
        degraded_end, healthy_end = (lo, hi) if k != "friction_mult" else (hi, lo)

        a, b = healthy_end, degraded_end
        for _ in range(n_iter):
            mid = 0.5 * (a + b)
            th = {kk: np.array(float(v)) for kk, v in HEALTH_NOMINAL.items()}
            th[k] = np.array(mid)
            perf = reference_performance(th, p)
            m = criterion_margins(perf, base)
            worst = float(np.max([np.asarray(m[c]).ravel()[0] for c in CRITERION_NAMES]))
            if worst >= 1.0:
                b = mid       # already past the limit -> move toward healthy
            else:
                a = mid       # still healthy -> move toward degraded
        out[k] = float(0.5 * (a + b))

    return out


def binding_criterion(theta: Dict[str, np.ndarray], p: Dict[str, float],
                      baseline: Dict[str, float]) -> np.ndarray:
    """Index of the criterion currently closest to its limit (for explainability)."""
    perf = reference_performance(theta, p)
    m = criterion_margins(perf, baseline)
    stacked = np.stack([np.asarray(m[k], dtype=float) for k in CRITERION_NAMES], axis=0)
    return np.argmax(stacked, axis=0)
