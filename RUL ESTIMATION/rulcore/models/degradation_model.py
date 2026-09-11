"""
degradation_model.py
====================
Hybrid degradation dynamics (Part 12).

    dD/dt  =  physics_based_rate  +  learned_correction  +  process_noise

WORKING VARIABLE
----------------
Everything is expressed in CONSUMED LIFE FRACTION per parameter,

    phi_k = (theta_k - theta_nominal_k) / (theta_eol_k - theta_nominal_k)

so phi = 0 at commissioning and phi = 1 when that parameter alone would end the
engine's life. Working in phi rather than raw parameter values makes the six
mechanisms commensurable (friction_mult spans 0.43 while eta_comb spans 0.062)
and makes a single set of rate units meaningful.

THE PHYSICS TERM
----------------
Wear is stress-driven, not clock-driven:

    dphi_k/dt = a_k * s_k(operating point) * g_k(phi_k)

  s_k   the physical driver of that mechanism - an Arrhenius factor in head or
        oil temperature for the thermal mechanisms, a speed-load product for the
        mechanical ones. Computed entirely from MEASURED telemetry.
  g_k   the shape term, convex for accelerating wear.
  a_k   the engine-specific severity. This is the one thing that cannot be known
        in advance: two engines in the same fleet, flown the same way, wear at
        different rates. It is identified ONLINE from the UKF history by robust
        regression of observed increments against the stress predictor.

Identifying a_k per engine is what makes the physics term useful rather than
decorative. A fleet-average rate would predict the fleet-average life for every
engine, which is exactly the failure mode a prognostic system exists to avoid.

THE LEARNED TERM
----------------
A GRU supplies a bounded multiplicative correction to the physics rate. Why a
correction to the RATE, and not a direct prediction of the next state or of RUL:

  * Predicting the next degradation STATE invites the network to learn the
    identity map. phi_{t+1} is approximately phi_t, so a model that copies its
    input scores superbly and has learned nothing. Errors then compound when the
    model is rolled forward over the hundreds of steps a real RUL horizon needs.

  * A rate correction keeps the physics term dominant and leaves the network a
    small, roughly stationary quantity to learn. If the network outputs zero the
    system degrades gracefully to the pure physics model rather than collapsing.

  * The rate is horizon-agnostic. The same learned quantity integrates over 10
    hours or 300 hours, which is what the Monte Carlo propagation needs.

  * Monotonicity is free: the correction is multiplicative and positive, so a
    non-negative physics rate stays non-negative and health cannot improve.

The correction is bounded to [exp(-1.6), exp(+1.6)] ~ [0.2, 5.0]. A model that
wants to move the rate by more than a factor of five is not correcting the
physics, it is overriding it, and that is a sign the physics term is wrong
rather than something to be permitted silently.
"""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
import pandas as pd

from ..config import HEALTH_EOL, HEALTH_NOMINAL, HEALTH_PARAMS, SNAPSHOT_HOURS

PHI_RANGE = {k: (HEALTH_EOL[k] - HEALTH_NOMINAL[k]) for k in HEALTH_PARAMS}


# --------------------------------------------------------------------------- #
# Coordinate transforms
# --------------------------------------------------------------------------- #

def theta_to_phi(theta: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    return {k: (np.asarray(theta[k], dtype=float) - HEALTH_NOMINAL[k]) / PHI_RANGE[k]
            for k in HEALTH_PARAMS if k in theta}


def phi_to_theta(phi: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    return {k: HEALTH_NOMINAL[k] + np.asarray(phi[k], dtype=float) * PHI_RANGE[k]
            for k in HEALTH_PARAMS if k in phi}


def shape_term(phi: np.ndarray, exponent: float = 1.35) -> np.ndarray:
    """Convex acceleration of wear with accumulated damage.

    A mildly convex form covers both the linear and the accelerating archetypes
    without needing to know which one this engine is on; the online rate
    identification absorbs the difference into a_k.
    """
    p = np.clip(np.asarray(phi, dtype=float), 0.0, 1.6)
    return 0.45 + 1.35 * p ** (exponent - 1.0) if exponent > 1.0 else np.ones_like(p)


# --------------------------------------------------------------------------- #
# Stress from OBSERVABLE telemetry only
# --------------------------------------------------------------------------- #

def stress_from_observables(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """Per-mechanism stress drivers computed from measured telemetry.

    Deliberately mirrors dataset/degradation.stress_factors(), but every input is
    something the aircraft actually measures: head temperature, oil temperature,
    shaft speed, vibration, and the twin's power estimate. No ground truth and no
    health parameter is used, so this function is valid in deployment.
    """
    cht = df["cht"].to_numpy()
    oil_t = df["oil_temperature"].to_numpy()
    rpm = df["rpm"].to_numpy()
    vib = df["vibration_rms"].to_numpy()
    # Twin power at the commanded operating point: the best available estimate of
    # how hard the engine is working without a torque sensor.
    power = df["power_pred_w"].to_numpy() if "power_pred_w" in df.columns \
        else np.full(len(df), 20000.0)

    def arrhenius(temp_c, ref_c, ea_over_r=5200.0):
        t_k = np.asarray(temp_c, dtype=float) + 273.15
        return np.exp(ea_over_r * (1.0 / (ref_c + 273.15) - 1.0 / t_k))

    load = np.clip(power / 22000.0, 0.0, 2.2)
    s_mech = (np.clip(rpm / 4200.0, 0.25, 1.7) ** 1.35) * (0.5 + 0.5 * load)

    return {
        "eta_inj": arrhenius(cht, 165.0) * (0.55 + 0.45 * load),
        "lub_health": arrhenius(oil_t, 95.0, 6400.0),
        "h_cool": 0.75 + 0.25 * np.clip(load, 0.0, 2.0),
        "eta_comb": arrhenius(cht, 168.0, 4300.0) * (0.6 + 0.4 * load),
        "friction_mult": s_mech,
        "eta_vol": s_mech * (0.85 + 0.15 * np.clip(vib / 0.8, 0.0, 3.0)),
    }


# --------------------------------------------------------------------------- #
# Online per-engine rate identification
# --------------------------------------------------------------------------- #

class PhysicsRateEstimator:
    """Identify the engine-specific severity a_k from the UKF history.

    Robust ratio regression of observed damage increments on the stress
    predictor, accumulated recursively so it costs nothing per snapshot:

        a_k  =  sum(w * dphi * x) / sum(w * x^2),      x = s_k * g_k(phi) * dt

    Two safeguards matter in practice:

      * EXPONENTIAL FORGETTING. A mechanism that only engages after an
        incubation period would otherwise be permanently diluted by the long
        quiet stretch before it started. Forgetting lets the estimate follow the
        engine's current behaviour.

      * NON-NEGATIVITY and a WARM-UP FLOOR. Early in life the observed
        increments are pure UKF noise and the ratio is meaningless, so the
        estimate is blended toward a fleet prior until enough damage has
        actually accumulated to identify anything.
    """

    def __init__(self, params: Sequence[str] = tuple(HEALTH_PARAMS),
                 forget: float = 0.9985,
                 fleet_prior: Dict[str, float] | None = None,
                 warmup_damage: float = 0.02):
        self.params = list(params)
        self.forget = forget
        self.warmup_damage = warmup_damage
        # Fleet prior: full life consumed over ~1500 snapshots at unit stress.
        self.fleet_prior = fleet_prior or {k: 1.0 / (1500 * SNAPSHOT_HOURS)
                                           for k in self.params}
        self.sxy = {k: 0.0 for k in self.params}
        self.sxx = {k: 0.0 for k in self.params}
        self.max_phi = {k: 0.0 for k in self.params}
        self.resid_sq = {k: 0.0 for k in self.params}
        self.n_obs = 0

    def update(self, dphi: Dict[str, float], x: Dict[str, float],
               phi: Dict[str, float]):
        self.n_obs += 1
        for k in self.params:
            self.sxy[k] = self.forget * self.sxy[k] + dphi[k] * x[k]
            self.sxx[k] = self.forget * self.sxx[k] + x[k] * x[k]
            self.max_phi[k] = max(self.max_phi[k], float(phi[k]))
            pred = self.rate_coefficient(k) * x[k]
            self.resid_sq[k] = self.forget * self.resid_sq[k] + (dphi[k] - pred) ** 2

    def rate_coefficient(self, k: str) -> float:
        if self.sxx[k] <= 1e-18:
            return self.fleet_prior[k]
        a = max(self.sxy[k] / self.sxx[k], 0.0)
        # Blend toward the fleet prior until real damage has accumulated.
        w = float(np.clip(self.max_phi[k] / self.warmup_damage, 0.0, 1.0))
        return w * a + (1.0 - w) * self.fleet_prior[k]

    def coefficients(self) -> Dict[str, float]:
        return {k: self.rate_coefficient(k) for k in self.params}

    def rate_sigma(self) -> Dict[str, float]:
        """Std of the unexplained part of the increment, per snapshot.

        Feeds the process-noise term of the Monte Carlo propagation, so the
        forecast spread reflects how well the physics term actually fitted THIS
        engine rather than a number chosen by hand.
        """
        out = {}
        for k in self.params:
            n = max(self.n_obs, 1)
            out[k] = float(np.sqrt(max(self.resid_sq[k], 0.0) / n))
        return out


def physics_rate(coeffs: Dict[str, float], stress: Dict[str, np.ndarray],
                 phi: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """dphi/dt from the identified physics model, per operating hour."""
    return {k: np.maximum(coeffs[k] * np.asarray(stress[k], dtype=float)
                          * shape_term(np.asarray(phi[k], dtype=float)), 0.0)
            for k in coeffs}


def fit_run_rates(df: pd.DataFrame, est_prefix: str = "est_") -> pd.DataFrame:
    """Run the online rate identification over one engine's UKF history.

    Returns a frame with, for every snapshot: the stress predictors, the
    identified coefficients, the physics rate and the observed rate. This is the
    input the GRU corrects and the basis of the pure-physics baseline.
    """
    n = len(df)
    stress = stress_from_observables(df)
    phi_est = {k: (df[f"{est_prefix}{k}"].to_numpy() - HEALTH_NOMINAL[k]) / PHI_RANGE[k]
               for k in HEALTH_PARAMS}

    est = PhysicsRateEstimator()
    dt = SNAPSHOT_HOURS

    cols = {}
    for k in HEALTH_PARAMS:
        cols[f"phi_{k}"] = phi_est[k]
        cols[f"stress_{k}"] = stress[k]
        cols[f"a_{k}"] = np.empty(n)
        cols[f"rate_phys_{k}"] = np.empty(n)
        cols[f"rate_obs_{k}"] = np.empty(n)

    for i in range(n):
        phi_i = {k: float(phi_est[k][i]) for k in HEALTH_PARAMS}
        x_i = {k: float(stress[k][i] * shape_term(np.array(phi_i[k])) * dt)
               for k in HEALTH_PARAMS}
        if i > 0:
            dphi = {k: float(phi_est[k][i] - phi_est[k][i - 1]) for k in HEALTH_PARAMS}
            est.update(dphi, x_i, phi_i)
        coeffs = est.coefficients()
        for k in HEALTH_PARAMS:
            cols[f"a_{k}"][i] = coeffs[k]
            cols[f"rate_phys_{k}"][i] = max(
                coeffs[k] * stress[k][i] * float(shape_term(np.array(phi_i[k]))), 0.0)
            cols[f"rate_obs_{k}"][i] = (
                (phi_est[k][i] - phi_est[k][i - 1]) / dt if i > 0 else 0.0)

    out = pd.DataFrame(cols, index=df.index)
    out["rate_sigma_mean"] = float(np.mean(list(est.rate_sigma().values())))
    return out


def true_rates(df: pd.DataFrame, smooth: int = 40) -> pd.DataFrame:
    """Ground-truth dphi/dt, smoothed. TRAINING LABEL ONLY.

    Smoothed because the per-snapshot truth carries the lognormal wear jitter
    injected by the simulator, which is genuinely unpredictable. Asking the
    network to fit that noise would waste capacity and inflate the apparent
    error of a model that is actually right about the trend.
    """
    out = {}
    for k in HEALTH_PARAMS:
        phi = (df[f"true_{k}"].to_numpy() - HEALTH_NOMINAL[k]) / PHI_RANGE[k]
        d = np.gradient(phi) / SNAPSHOT_HOURS
        s = pd.Series(d).rolling(smooth, center=True, min_periods=1).mean().to_numpy()
        out[f"rate_true_{k}"] = np.maximum(s, 0.0)
        out[f"phi_true_{k}"] = phi
    return pd.DataFrame(out, index=df.index)
