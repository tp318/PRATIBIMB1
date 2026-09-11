"""
degradation.py
==============
Degradation mechanism models (Part 7).

Each mechanism drives one or more LATENT HEALTH PARAMETERS. Those parameters are
then fed into the engine model, so every observable consequence emerges from the
physics rather than being pasted onto the sensor values. Nothing in this file
touches a sensor reading directly.

Degradation is STRESS-DRIVEN, not clock-driven. The wear increment at each
snapshot depends on how hard the engine was worked in that snapshot:

    dtheta = -rate * f_stress(operating point, current health) * dt

This matters for the prognostics problem: two engines with identical initial
wear rates but different duty cycles must reach EOL at different times, or the
RUL problem degenerates into "read the clock".

Causal chains implemented
-------------------------
INJECTOR_FOULING
    deposit growth on the injector tip (thermally driven, Arrhenius-like)
    -> eta_inj falls -> delivered fuel falls for the same command
    -> mixture leans  -> burn completeness and exhaust split shift
    -> EGT and torque change -> RPM falls at fixed throttle
    -> combustion roughness raises vibration slightly

LUBRICATION_DEGRADATION
    oil oxidation + additive depletion (accelerated by oil temperature)
    -> lub_health falls -> pump delivery and film strength fall
    -> oil pressure falls, oil temperature rises (worse heat rejection)
    -> boundary friction rises (couples into friction_mult)
    -> bearing vibration rises

COOLING_DEGRADATION
    fin fouling / baffle leakage / cooling-path blockage
    -> h_cool falls -> head heat rejection falls
    -> CHT rises -> oil temperature rises
    -> higher metal temperature accelerates injector fouling and oil oxidation
       (cross-coupling, see MECHANISM_COUPLING)

COMBUSTION_DEGRADATION
    valve seat recession / ignition energy loss / chamber deposits
    -> eta_comb falls -> less of the released heat becomes work
    -> power falls, EGT rises, CHT rises, fuel flow ~unchanged

MECHANICAL_WEAR
    bearing and ring/bore wear
    -> friction_mult rises -> friction torque and friction heating rise
    -> RPM falls at fixed throttle, oil temperature rises, vibration rises

RING_BLOWBY
    ring/bore wear opening the combustion seal
    -> eta_vol falls -> trapped mass falls, mixture richens for the same command
    -> power falls, EGT rises; also loads the oil (blow-by contamination)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from ..config import HEALTH_PARAMS, HEALTH_NOMINAL, HEALTH_EOL, RATE_ARCHETYPES

# --------------------------------------------------------------------------- #
# Which health parameters each mechanism attacks, and how hard.
# The value is the share of that mechanism's wear budget directed at the
# parameter, expressed as a fraction of the parameter's full nominal->EOL range.
# --------------------------------------------------------------------------- #

MECHANISM_TARGETS: Dict[str, Dict[str, float]] = {
    "INJECTOR_FOULING": {
        "eta_inj": 1.00, "eta_comb": 0.16, "friction_mult": 0.05,
    },
    "LUBRICATION_DEGRADATION": {
        "lub_health": 1.00, "friction_mult": 0.42, "h_cool": 0.06,
    },
    "COOLING_DEGRADATION": {
        "h_cool": 1.00, "lub_health": 0.20, "eta_comb": 0.10,
    },
    "COMBUSTION_DEGRADATION": {
        "eta_comb": 1.00, "eta_inj": 0.14, "eta_vol": 0.18,
    },
    "MECHANICAL_WEAR": {
        "friction_mult": 1.00, "lub_health": 0.28, "eta_vol": 0.30,
    },
    "RING_BLOWBY": {
        "eta_vol": 1.00, "lub_health": 0.34, "eta_comb": 0.22,
    },
}

# Background wear: every parameter creeps a little on every engine, regardless
# of which mechanism dominates. Without this the "healthy" parameters would sit
# at exactly 1.000 forever, which is both unphysical and a give-away signal.
BACKGROUND_WEAR_FRAC = 0.11

# Cross-coupling: a hot engine fouls injectors and oxidises oil faster. Applied
# as a multiplier on the wear rate of the target parameter, driven by how far
# the source parameter has already degraded.
MECHANISM_COUPLING = [
    # (source parameter, target parameter, coupling gain)
    ("h_cool", "eta_inj", 0.85),
    ("h_cool", "lub_health", 0.70),
    ("lub_health", "friction_mult", 0.95),
    ("friction_mult", "lub_health", 0.55),
    ("eta_vol", "lub_health", 0.40),
]


@dataclass
class DegradationPlan:
    """Everything that defines how one engine wears out."""
    mechanism: str
    secondary_mechanism: str | None
    rate_archetype: str
    base_rates: Dict[str, float]      # fraction of range consumed per stress-hour
    shape_exponent: Dict[str, float]  # >1 accelerating, <1 decelerating
    stress_sensitivity: float         # how strongly duty cycle modulates wear
    onset_hours: Dict[str, float]     # incubation before a mechanism engages
    theta0: Dict[str, float]          # commissioning health (near but not at 1.0)
    meta: Dict[str, float] = field(default_factory=dict)


def sample_degradation_plan(rng: np.random.Generator,
                            mechanism: str,
                            rate_archetype: str,
                            target_life_h: float) -> DegradationPlan:
    """Build a degradation plan aimed at roughly `target_life_h` operating hours.

    The rate is calibrated so that the DOMINANT parameter would consume its full
    nominal->EOL range in about `target_life_h` hours at average stress. It is
    only approximate: coupling, stress history and the multi-criteria EOL test
    move the realised life around, which is exactly what we want - the label
    must not be a deterministic function of the plan.
    """
    secondary = None
    targets = dict(MECHANISM_TARGETS.get(mechanism, {}))

    if mechanism == "COMBINED":
        # Two genuine mechanisms running at once (harder generalisation case).
        pair = rng.choice(list(MECHANISM_TARGETS), size=2, replace=False)
        mechanism_primary, secondary = str(pair[0]), str(pair[1])
        targets = {}
        for m, w in ((mechanism_primary, 0.72), (secondary, 0.62)):
            for k, v in MECHANISM_TARGETS[m].items():
                targets[k] = targets.get(k, 0.0) + v * w
        meta_primary = mechanism_primary
    else:
        meta_primary = mechanism

    # Background creep on every parameter.
    for k in HEALTH_PARAMS:
        targets[k] = targets.get(k, 0.0) + BACKGROUND_WEAR_FRAC * float(rng.uniform(0.5, 1.5))

    rate_mult = RATE_ARCHETYPES[rate_archetype]

    # Per-parameter base rate: fraction of the nominal->EOL range consumed per
    # hour of average-stress operation.
    base_rates: Dict[str, float] = {}
    for k in HEALTH_PARAMS:
        share = targets.get(k, 0.0)
        # Parameter-level scatter so two engines with the same mechanism still
        # wear differently.
        jitter = float(np.exp(rng.normal(0.0, 0.22)))
        base_rates[k] = share * jitter * rate_mult / max(target_life_h, 1.0)

    # Shape: accelerating runs have a convex trajectory (wear begets wear).
    shape: Dict[str, float] = {}
    for k in HEALTH_PARAMS:
        if rate_archetype == "accelerating":
            shape[k] = float(rng.uniform(1.55, 2.60))
        else:
            shape[k] = float(rng.uniform(0.90, 1.30))

    # Incubation: many mechanisms show nothing at all for a while.
    onset: Dict[str, float] = {}
    for k in HEALTH_PARAMS:
        is_primary = targets.get(k, 0.0) > 0.4
        if is_primary:
            onset[k] = float(rng.uniform(0.0, 0.22) * target_life_h)
        else:
            onset[k] = float(rng.uniform(0.0, 0.45) * target_life_h)

    # Commissioning health: a new engine is not exactly 1.000 on every parameter.
    theta0 = {}
    for k in HEALTH_PARAMS:
        span = HEALTH_EOL[k] - HEALTH_NOMINAL[k]
        theta0[k] = float(HEALTH_NOMINAL[k] + span * rng.uniform(0.0, 0.045))

    return DegradationPlan(
        mechanism=mechanism,
        secondary_mechanism=secondary,
        rate_archetype=rate_archetype,
        base_rates=base_rates,
        shape_exponent=shape,
        stress_sensitivity=float(rng.uniform(0.55, 1.35)),
        onset_hours=onset,
        theta0=theta0,
        meta={"target_life_h": target_life_h, "primary": meta_primary},
    )


# --------------------------------------------------------------------------- #
# Stress model
# --------------------------------------------------------------------------- #

def stress_factors(cht_c, oil_temp_c, rpm, power_w, vib_rms) -> Dict[str, np.ndarray]:
    """Per-parameter stress multipliers for one snapshot.

    Each mechanism responds to the physical driver that actually accelerates it.
    Thermal mechanisms use an Arrhenius-style temperature factor; mechanical
    mechanisms use speed and load.
    """
    def arrhenius(temp_c, ref_c, ea_over_r=5200.0):
        t_k = np.asarray(temp_c, dtype=float) + 273.15
        t_ref = ref_c + 273.15
        return np.exp(ea_over_r * (1.0 / t_ref - 1.0 / t_k))

    # Injector deposits form on the hot tip: driven by head temperature.
    s_inj = arrhenius(cht_c, 165.0) * (0.55 + 0.45 * np.clip(power_w / 22000.0, 0.0, 2.2))
    # Oil oxidation: strongly driven by oil temperature.
    s_lub = arrhenius(oil_temp_c, 95.0, ea_over_r=6400.0)
    # Fin fouling / baffle wear: driven by run time and thermal cycling, weakly
    # by temperature. Deliberately the LEAST duty-sensitive mechanism.
    s_cool = 0.75 + 0.25 * np.clip(np.asarray(power_w) / 22000.0, 0.0, 2.0)
    # Valve/chamber: temperature and load.
    s_comb = arrhenius(cht_c, 168.0, ea_over_r=4300.0) * \
             (0.6 + 0.4 * np.clip(power_w / 22000.0, 0.0, 2.2))
    # Bearing/ring wear: speed and load dominate (a PV-type wear law).
    s_mech = (np.clip(np.asarray(rpm, dtype=float) / 4200.0, 0.25, 1.7) ** 1.35) * \
             (0.5 + 0.5 * np.clip(np.asarray(power_w) / 22000.0, 0.0, 2.2))
    # Ring/bore: load and speed, plus a vibration term.
    s_vol = s_mech * (0.85 + 0.15 * np.clip(np.asarray(vib_rms) / 0.8, 0.0, 3.0))

    return {
        "eta_inj": s_inj,
        "lub_health": s_lub,
        "h_cool": s_cool,
        "eta_comb": s_comb,
        "friction_mult": s_mech,
        "eta_vol": s_vol,
    }


def consumed_fraction(theta: Dict[str, float]) -> Dict[str, np.ndarray]:
    """Fraction of each parameter's nominal->EOL range that has been used up."""
    out = {}
    for k in HEALTH_PARAMS:
        span = HEALTH_EOL[k] - HEALTH_NOMINAL[k]
        out[k] = np.clip((np.asarray(theta[k], dtype=float) - HEALTH_NOMINAL[k]) / span,
                         0.0, 3.0)
    return out


def step_degradation(theta: Dict[str, float],
                     plan: DegradationPlan,
                     stress: Dict[str, np.ndarray],
                     operating_hours: float,
                     dt_h: float,
                     rng: np.random.Generator) -> Dict[str, float]:
    """Advance the latent health parameters by one snapshot.

    Returns the NEW theta dict. Wear is strictly monotone in the degradation
    direction: a physical wear process does not undo itself, and letting it
    would make the ground-truth RUL label non-monotone for reasons that have
    nothing to do with the engine.
    """
    frac = consumed_fraction(theta)
    new = {}

    # Cross-coupling multipliers
    couple = {k: 1.0 for k in HEALTH_PARAMS}
    for src, tgt, gain in MECHANISM_COUPLING:
        couple[tgt] = couple[tgt] * (1.0 + gain * float(np.clip(frac[src], 0.0, 1.4)))

    for k in HEALTH_PARAMS:
        span = HEALTH_EOL[k] - HEALTH_NOMINAL[k]      # signed
        rate = plan.base_rates[k]

        # Incubation
        if operating_hours < plan.onset_hours[k]:
            gate = 0.0
        else:
            # smooth engagement over the first few hours after onset
            gate = float(np.clip((operating_hours - plan.onset_hours[k]) / 8.0, 0.0, 1.0))

        # Shape term: convex (accelerating) or mildly concave trajectories.
        f = float(np.clip(frac[k], 0.0, 1.2))
        n = plan.shape_exponent[k]
        shape_term = max(n * (f ** (1.0 - 1.0 / n)) if n > 1.0 else 1.0, 0.35) if n > 1.0 \
            else (1.0 + 0.25 * (1.0 - f))
        if n > 1.0:
            # normalised so the average over the life is ~1
            shape_term = 0.45 + 1.35 * (f ** (n - 1.0))

        s = float(np.asarray(stress[k]).mean())
        s_eff = s ** plan.stress_sensitivity

        # Stochastic wear: lognormal jitter keeps the trajectory from being a
        # smooth analytic curve the GRU could simply memorise.
        noise = float(np.exp(rng.normal(0.0, 0.30)))

        d_frac = rate * gate * shape_term * s_eff * noise * dt_h
        d_frac = max(d_frac, 0.0)                      # monotone

        new[k] = float(theta[k] + span * d_frac)

    return new
