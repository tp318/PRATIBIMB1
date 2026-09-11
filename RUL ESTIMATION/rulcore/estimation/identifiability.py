"""
identifiability.py
==================
Structural identifiability analysis of the health parameters (Part 11).

The question this answers is not "does the UKF converge" - a badly posed filter
converges happily to a meaningless answer. The question is whether the chosen
measurement set can distinguish the chosen health parameters AT ALL, and if not,
which combinations are invisible.

METHOD
------
Build the sensitivity Jacobian of the NORMALISED residual vector with respect to
each health parameter:

        J[i, j]  =  (1 / sigma_i) * d y_i / d phi_j

where phi_j is the fraction of parameter j's nominal->EOL range that has been
consumed. Two scalings matter and both are deliberate:

  * dividing by sigma_i puts every measurement in units of "healthy residual
    standard deviations", so a Jacobian entry reads directly as detectability:
    an entry of 3.0 means consuming 100% of that parameter's life moves that
    measurement by 3 healthy sigmas.

  * differentiating with respect to consumed LIFE FRACTION rather than raw
    parameter value makes the columns comparable. Otherwise friction_mult (range
    ~0.43) and eta_inj (range ~0.074) would be compared on different scales and
    the condition number would be meaningless.

Then:
  singular values of J        -> how many independent directions are observable
  condition number            -> ratio of best to worst observable direction
  column norms                -> per-parameter detectability
  right singular vector of
    the smallest sigma        -> the parameter combination that is (nearly)
                                 invisible to this sensor set
  pairwise cosine similarity  -> which two parameters are confusable

MULTI-CONDITION EXCITATION
--------------------------
A single operating point is far less informative than a mission. Stacking the
Jacobians from several conditions is the identifiability equivalent of exciting
a system across its bandwidth, and it is the reason a UAV that climbs, cruises
and loiters is easier to diagnose than one that only cruises. Both the
single-point and pooled results are reported so the difference is visible.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

from ..config import HEALTH_EOL, HEALTH_NOMINAL, HEALTH_PARAMS, UKF_MEAS
from ..physics.engine_model import nominal_params, steady_state
from .ukf import MEAS_KEY

# A representative spread of operating points across the flight envelope.
DEFAULT_CONDITIONS: List[Dict[str, float]] = [
    {"name": "cruise_mid", "throttle": 0.75, "altitude_ft": 8000, "ambient_c": 5.0,
     "airspeed_factor": 1.00, "load_factor": 1.00},
    {"name": "climb", "throttle": 0.90, "altitude_ft": 5000, "ambient_c": 8.0,
     "airspeed_factor": 0.92, "load_factor": 0.97},
    {"name": "loiter_high", "throttle": 0.58, "altitude_ft": 15000, "ambient_c": -14.0,
     "airspeed_factor": 0.90, "load_factor": 0.94},
    {"name": "takeoff", "throttle": 0.97, "altitude_ft": 500, "ambient_c": 22.0,
     "airspeed_factor": 0.85, "load_factor": 1.02},
    {"name": "descent", "throttle": 0.28, "altitude_ft": 7000, "ambient_c": 6.0,
     "airspeed_factor": 1.10, "load_factor": 1.06},
    {"name": "hot_low", "throttle": 0.80, "altitude_ft": 1500, "ambient_c": 41.0,
     "airspeed_factor": 1.00, "load_factor": 1.00},
]


def _theta_at(fractions: Dict[str, float]) -> Dict[str, np.ndarray]:
    """Health vector at the given consumed-life fractions."""
    th = {}
    for k in HEALTH_PARAMS:
        span = HEALTH_EOL[k] - HEALTH_NOMINAL[k]
        th[k] = np.array(HEALTH_NOMINAL[k] + span * fractions.get(k, 0.0), dtype=float)
    return th


def sensitivity_jacobian(condition: Dict[str, float],
                         sigma: Dict[str, float],
                         params: Dict[str, float] | None = None,
                         meas_names: Sequence[str] = tuple(UKF_MEAS),
                         state_names: Sequence[str] = tuple(HEALTH_PARAMS),
                         base_fractions: Dict[str, float] | None = None,
                         delta: float = 0.05) -> np.ndarray:
    """Central-difference Jacobian d(z)/d(consumed life fraction). Shape (m, n)."""
    p = params or nominal_params()
    base_fractions = base_fractions or {}
    m, n = len(meas_names), len(state_names)
    J = np.zeros((m, n))

    def measure(fr: Dict[str, float]) -> np.ndarray:
        th = _theta_at(fr)
        ss = steady_state(condition["throttle"], condition["altitude_ft"],
                          condition["ambient_c"], th, p,
                          airspeed_factor=condition["airspeed_factor"],
                          load_factor=condition["load_factor"])
        return np.array([float(np.asarray(ss[MEAS_KEY[k]]).ravel()[0]) for k in meas_names])

    for j, name in enumerate(state_names):
        f_plus = dict(base_fractions)
        f_minus = dict(base_fractions)
        f_plus[name] = base_fractions.get(name, 0.0) + delta
        f_minus[name] = max(base_fractions.get(name, 0.0) - delta, 0.0)
        step = f_plus[name] - f_minus[name]
        dy = (measure(f_plus) - measure(f_minus)) / max(step, 1e-9)
        J[:, j] = dy / np.array([sigma[k] for k in meas_names])

    return J


def analyse(J: np.ndarray, state_names: Sequence[str]) -> Dict:
    """Singular-value analysis of a (stacked) Jacobian."""
    U, s, Vt = np.linalg.svd(J, full_matrices=False)
    s_safe = np.maximum(s, 1e-15)
    cond = float(s_safe[0] / s_safe[-1])

    col_norm = np.linalg.norm(J, axis=0)
    # cosine similarity between parameter columns
    norms = np.maximum(col_norm, 1e-15)
    cos = (J.T @ J) / np.outer(norms, norms)

    worst_dir = Vt[-1]
    return {
        "singular_values": s.tolist(),
        "condition_number": cond,
        "column_norms": {k: float(v) for k, v in zip(state_names, col_norm)},
        "cosine_similarity": cos,
        "least_observable_direction": {k: float(v) for k, v in zip(state_names, worst_dir)},
        "effective_rank_1pct": int(np.sum(s > 0.01 * s_safe[0])),
        "n_directions_above_1sigma": int(np.sum(s > 1.0)),
    }


def pooled_jacobian(sigma: Dict[str, float],
                    conditions: Sequence[Dict[str, float]] = tuple(DEFAULT_CONDITIONS),
                    state_names: Sequence[str] = tuple(HEALTH_PARAMS),
                    meas_names: Sequence[str] = tuple(UKF_MEAS),
                    params: Dict[str, float] | None = None,
                    base_fractions: Dict[str, float] | None = None) -> np.ndarray:
    """Stack the Jacobians from several operating points.

    Rows are scaled by 1/sqrt(n_conditions) so that the singular values keep the
    interpretation of "sigmas of measurement change per unit consumed life at a
    typical operating point", rather than growing simply because more conditions
    were added.
    """
    Js = [sensitivity_jacobian(c, sigma, params=params, meas_names=meas_names,
                               state_names=state_names,
                               base_fractions=base_fractions)
          for c in conditions]
    return np.vstack(Js) / np.sqrt(len(Js))


# --------------------------------------------------------------------------- #
# Specific confounding tests the brief asks for by name
# --------------------------------------------------------------------------- #

def _residual_change_ensemble(delta_env: Dict[str, float],
                              condition: Dict[str, float],
                              sigma: Dict[str, float],
                              twin_params: Dict[str, float],
                              meas_names: Sequence[str],
                              n_builds: int = 240,
                              seed: int = 7) -> np.ndarray:
    """Change in the NORMALISED RESIDUAL caused by an environment change, over
    an ensemble of healthy engine builds. Shape (n_builds, m).

    This is the crux of the "is a hot day mistaken for cooling degradation"
    question, and it only means something if the PLANT and the TWIN are
    different objects. The plant is a healthy engine drawn from the fleet build
    distribution; the twin is the fleet-nominal model, re-evaluated at the new
    environment exactly as it would be in service.

    Anything the twin can explain about the environment therefore cancels. What
    survives is the part of the environment change the twin gets wrong because
    this particular engine is not the average engine - which is precisely the
    quantity that could be mistaken for degradation.
    """
    from ..physics.engine_model import nominal_health

    from ..dataset.simulate_run import sample_engine_build

    def measure(cond, theta, p):
        ss = steady_state(cond["throttle"], cond["altitude_ft"], cond["ambient_c"],
                          theta, p,
                          airspeed_factor=cond["airspeed_factor"],
                          load_factor=cond["load_factor"])
        return np.array([float(np.asarray(ss[MEAS_KEY[k]]).ravel()[0]) for k in meas_names])

    healthy = nominal_health()
    cond2 = dict(condition)
    for k, v in delta_env.items():
        cond2[k] = condition[k] + v

    scale = np.array([sigma[k] for k in meas_names])
    y_twin_old = measure(condition, healthy, twin_params)
    y_twin_new = measure(cond2, healthy, twin_params)

    rng = np.random.default_rng(seed)
    out = np.empty((n_builds, len(meas_names)))
    for b in range(n_builds):
        p_plant = sample_engine_build(rng)
        r_old = measure(condition, healthy, p_plant) - y_twin_old
        r_new = measure(cond2, healthy, p_plant) - y_twin_new
        out[b] = (r_new - r_old) / scale
    return out


def confounding_report(sigma: Dict[str, float],
                       params: Dict[str, float] | None = None,
                       meas_names: Sequence[str] = tuple(UKF_MEAS)) -> Dict:
    """Answer the three confounding questions the brief poses explicitly."""
    p = params or nominal_params()
    cond = DEFAULT_CONDITIONS[0]

    def unit(v):
        n = np.linalg.norm(v)
        return v / n if n > 1e-12 else v

    J = sensitivity_jacobian(cond, sigma, params=p, meas_names=meas_names)
    Jp = pooled_jacobian(sigma, params=p, meas_names=meas_names)
    names = list(HEALTH_PARAMS)

    def cos_between(a: str, b: str, mat: np.ndarray) -> float:
        ia, ib = names.index(a), names.index(b)
        return float(np.dot(unit(mat[:, ia]), unit(mat[:, ib])))

    # 1. injector vs combustion efficiency
    inj_comb_single = cos_between("eta_inj", "eta_comb", J)
    inj_comb_pooled = cos_between("eta_inj", "eta_comb", Jp)

    # 2. cooling degradation vs a hot day / a climb, over a fleet of builds.
    # Reported as the MEAN ABSOLUTE cosine (how much the environment-induced
    # residual looks like cooling degradation for a typical engine) and the RMS
    # magnitude in sigmas (how large that spurious signal actually is). A large
    # cosine on a tiny vector is harmless; both numbers are needed.
    cool_col = unit(J[:, names.index("h_cool")])

    def env_vs_cooling(delta_env):
        E = _residual_change_ensemble(delta_env, cond, sigma, p, meas_names)
        norms = np.linalg.norm(E, axis=1)
        ok = norms > 1e-12
        cosines = (E[ok] @ cool_col) / norms[ok]
        return (float(np.mean(np.abs(cosines))) if ok.any() else 0.0,
                float(np.sqrt(np.mean(norms ** 2))),
                float(np.percentile(norms, 95)))

    hot_vs_cool, hot_norm, hot_p95 = env_vs_cooling({"ambient_c": +20.0})
    high_vs_cool, high_norm, high_p95 = env_vs_cooling({"altitude_ft": +9000.0})

    # For scale: how many sigmas does REAL cooling degradation produce?
    cool_detectability = float(np.linalg.norm(J[:, names.index("h_cool")]))

    # 3. lubrication degradation vs an oil-pressure sensor drift
    drift = np.zeros(len(meas_names))
    drift[list(meas_names).index("oil_pressure")] = 1.0     # a pure 1-sigma bias
    lub_col = J[:, names.index("lub_health")]
    drift_vs_lub_single = float(np.dot(unit(drift), unit(lub_col)))
    drift_vs_lub_pooled = float(np.dot(unit(np.tile(drift, len(DEFAULT_CONDITIONS))
                                            / np.sqrt(len(DEFAULT_CONDITIONS))),
                                       unit(Jp[:, names.index("lub_health")])))

    return {
        "eta_inj_vs_eta_comb_single_point": inj_comb_single,
        "eta_inj_vs_eta_comb_pooled": inj_comb_pooled,
        "hot_day_vs_cooling_degradation": hot_vs_cool,
        "high_altitude_vs_cooling_degradation": high_vs_cool,
        "hot_day_residual_norm_sigmas": hot_norm,
        "high_altitude_residual_norm_sigmas": high_norm,
        "hot_day_residual_p95_sigmas": hot_p95,
        "high_altitude_residual_p95_sigmas": high_p95,
        "cooling_degradation_detectability_sigmas": cool_detectability,
        "oilp_sensor_drift_vs_lubrication_single": drift_vs_lub_single,
        "oilp_sensor_drift_vs_lubrication_pooled": drift_vs_lub_pooled,
    }


def recommend_state_vector(sigma: Dict[str, float],
                           params: Dict[str, float] | None = None,
                           meas_names: Sequence[str] = tuple(UKF_MEAS),
                           min_detectability: float = 1.0,
                           max_condition: float = 40.0) -> Tuple[List[str], Dict]:
    """Greedily drop the least identifiable parameter until the pooled Jacobian
    is acceptably conditioned.

    A parameter is kept only if consuming its full life produces at least
    `min_detectability` healthy sigmas of measurement change; anything weaker
    cannot be estimated from this sensor set and including it just injects
    noise into the ones that can.
    """
    names = list(HEALTH_PARAMS)
    history = []
    while len(names) > 2:
        Jp = pooled_jacobian(sigma, state_names=names, meas_names=meas_names, params=params)
        res = analyse(Jp, names)
        history.append({"state": list(names),
                        "condition_number": res["condition_number"],
                        "column_norms": res["column_norms"],
                        "singular_values": res["singular_values"]})
        weakest = min(res["column_norms"], key=res["column_norms"].get)
        if (res["condition_number"] <= max_condition
                and res["column_norms"][weakest] >= min_detectability):
            break
        names.remove(weakest)
    return names, {"history": history}
