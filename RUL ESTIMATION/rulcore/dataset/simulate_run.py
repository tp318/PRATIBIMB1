"""
simulate_run.py
===============
Generation of one complete run-to-failure engine trajectory.

A run is a single physical engine flown from commissioning to end of life. It is
the atomic unit of this dataset: splits are made over runs, never over rows, and
the effective sample size of the whole exercise is the number of runs.

Pipeline for one run
--------------------
  1. draw an engine build (per-engine parameter scatter + sensor calibration)
  2. draw a mission plan (theatre, sortie mix) - INDEPENDENT of the fault
  3. draw a degradation plan (mechanism, rate archetype, target life)
  4. integrate the latent health parameters over the operating-hour axis,
     driven by the stress the engine actually experiences
  5. evaluate the EOL criteria at the reference condition and locate EOL
  6. truncate at EOL, then run a vectorised fast-time burst at every snapshot
     to produce realistic within-window statistics
  7. synthesise vibration and extract features
  8. apply the sensor model to obtain measured telemetry

PERFORMANCE NOTE
----------------
Step 4 is inherently sequential (health at t+1 depends on health at t) while
steps 6-8 are embarrassingly parallel across snapshots. Step 4 is therefore run
in CHUNKS: inside a chunk of `CHUNK` snapshots the health vector is held fixed
for the purpose of evaluating the operating point, the resulting stresses are
computed for the whole chunk in one vectorised call, and health is then
integrated snapshot by snapshot using those stresses. Over a 10-hour chunk the
health parameters move by well under 0.1%, so the splitting error is negligible
compared with the stochastic wear term.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Dict, Tuple

import numpy as np

from ..config import (ENGINE_BUILD_SCATTER, ENGINE_NOMINAL, FAST_DT_S,
                      HEALTH_PARAMS, LIFE_HOURS_RANGE, MECHANISMS,
                      MECHANISM_WEIGHTS, RATE_WEIGHTS, RUL_CAP_H,
                      SNAPSHOT_BURST_S, SNAPSHOT_HOURS)
from ..physics.engine_model import (FastState, evaluate_at_speed, nominal_params,
                                    solve_steady_rpm, steady_state, step_fast)
from ..physics.sensors import (apply_sensor_model, sample_drift_fault,
                               sample_sensor_calibration)
from ..physics.vibration import extract_features, synthesise_windows
from .degradation import (consumed_fraction, sample_degradation_plan,
                          step_degradation, stress_factors)
from .eol import (CRITERION_NAMES, commissioning_baseline, criterion_margins,
                  find_eol_index, reference_performance)
from .profiles import PHASE_NAMES, generate_operating_points, sample_mission_plan

CHUNK = 40                      # snapshots per degradation integration chunk
MAX_HORIZON_SNAPSHOTS = 6000    # 1500 operating hours - a hard safety stop


# --------------------------------------------------------------------------- #
# Engine build
# --------------------------------------------------------------------------- #

def sample_engine_build(rng: np.random.Generator,
                        scatter_mult: float = 1.0) -> Dict[str, float]:
    """Draw one physical engine. The Digital Twin never sees these values."""
    p = nominal_params()
    for key, rel in ENGINE_BUILD_SCATTER.items():
        p[key] = float(ENGINE_NOMINAL[key] * np.exp(rng.normal(0.0, rel * scatter_mult)))
    # Thermal time constants also vary a little between installations.
    p["tau_cht_s"] = float(ENGINE_NOMINAL["tau_cht_s"] * np.exp(rng.normal(0.0, 0.10 * scatter_mult)))
    p["tau_oil_t_s"] = float(ENGINE_NOMINAL["tau_oil_t_s"] * np.exp(rng.normal(0.0, 0.10 * scatter_mult)))
    return p


def _theta_arrays(theta: Dict[str, float], n: int) -> Dict[str, np.ndarray]:
    return {k: np.full(n, float(theta[k])) for k in HEALTH_PARAMS}


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #

def simulate_run(run_id: str,
                 seed: int,
                 envelope: str = "nominal",
                 scatter_mult: float = 1.0,
                 mechanism: str | None = None,
                 rate_archetype: str | None = None,
                 target_life_h: float | None = None,
                 sensor_drift_fault: bool = False,
                 noise_scale: float = 1.0) -> Tuple[Dict[str, np.ndarray], Dict]:
    """Simulate one run-to-failure trajectory.

    Returns (columns, metadata). `columns` is a dict of equal-length arrays ready
    to become a DataFrame; `metadata` records everything about how the run was
    generated (used for auditing and for building stratified splits).
    """
    rng = np.random.default_rng(seed)

    # ---- 1. engine build and instrumentation ----------------------------- #
    p = sample_engine_build(rng, scatter_mult)
    calibration = sample_sensor_calibration(rng, scatter_mult)

    # ---- 2. mission plan (drawn independently of the mechanism) ---------- #
    plan_mission = sample_mission_plan(rng, envelope=envelope)

    # ---- 3. degradation plan --------------------------------------------- #
    if mechanism is None:
        mechanism = str(rng.choice(MECHANISMS, p=np.array(MECHANISM_WEIGHTS) / sum(MECHANISM_WEIGHTS)))
    if rate_archetype is None:
        names = list(RATE_WEIGHTS)
        w = np.array([RATE_WEIGHTS[n] for n in names], dtype=float)
        rate_archetype = str(rng.choice(names, p=w / w.sum()))
    if target_life_h is None:
        target_life_h = float(np.exp(rng.uniform(np.log(LIFE_HOURS_RANGE[0]),
                                                 np.log(LIFE_HOURS_RANGE[1]))))

    deg = sample_degradation_plan(rng, mechanism, rate_archetype, target_life_h)

    # ---- baseline: this engine's own green-run reference ------------------ #
    baseline = commissioning_baseline(deg.theta0, p)

    # ---- 4. operating points and health integration ---------------------- #
    horizon = int(min(MAX_HORIZON_SNAPSHOTS,
                      max(600, 3.2 * target_life_h / SNAPSHOT_HOURS)))
    ops = generate_operating_points(plan_mission, horizon, SNAPSHOT_HOURS, rng)

    theta_hist = {k: np.empty(horizon) for k in HEALTH_PARAMS}
    theta = dict(deg.theta0)
    op_hours = 0.0
    eol_i = -1
    worst_margin = np.empty(horizon)
    margins_hist = {c: np.empty(horizon) for c in CRITERION_NAMES}

    for start in range(0, horizon, CHUNK):
        stop = min(start + CHUNK, horizon)
        n = stop - start
        th_arr = _theta_arrays(theta, n)

        ss = steady_state(ops["throttle"][start:stop],
                          ops["altitude_ft"][start:stop],
                          ops["ambient_temperature_c"][start:stop],
                          th_arr, p,
                          airspeed_factor=ops["airspeed_factor"][start:stop],
                          load_factor=ops["load_factor"][start:stop])

        stress = stress_factors(ss["cht"], ss["oil_temperature"], ss["rpm"],
                                ss["power_brake_w"], ss["vibration_rms"])

        for j in range(n):
            i = start + j
            for k in HEALTH_PARAMS:
                theta_hist[k][i] = theta[k]
            s_j = {k: stress[k][j] for k in stress}
            theta = step_degradation(theta, deg, s_j, op_hours, SNAPSHOT_HOURS, rng)
            op_hours += SNAPSHOT_HOURS

        # ---- 5. EOL check on this chunk (vectorised) --------------------- #
        th_chunk = {k: theta_hist[k][start:stop] for k in HEALTH_PARAMS}
        perf = reference_performance(th_chunk, p)
        marg = criterion_margins(perf, baseline)
        for c in CRITERION_NAMES:
            margins_hist[c][start:stop] = marg[c]
        worst_margin[start:stop] = np.max(
            np.stack([np.asarray(marg[c], dtype=float) for c in CRITERION_NAMES], axis=0), axis=0)

        idx = find_eol_index(worst_margin[:stop])
        if idx >= 0:
            eol_i = idx
            break

    if eol_i < 0:
        return {}, {"run_id": run_id, "status": "no_eol_within_horizon",
                    "target_life_h": target_life_h, "mechanism": mechanism,
                    "rate_archetype": rate_archetype}

    # Keep everything up to and including EOL.
    T = eol_i + 1
    for k in HEALTH_PARAMS:
        theta_hist[k] = theta_hist[k][:T]
    worst_margin = worst_margin[:T]
    for c in CRITERION_NAMES:
        margins_hist[c] = margins_hist[c][:T]
    ops = {k: v[:T] for k, v in ops.items()}

    theta_true = {k: theta_hist[k] for k in HEALTH_PARAMS}

    # ---- 6. fast-time burst at every snapshot (vectorised over snapshots) - #
    latent = _run_transient_burst(ops, theta_true, p, rng)

    # ---- 7. vibration ----------------------------------------------------- #
    windows = synthesise_windows(latent["rpm"], latent["vibration_rms"],
                                 theta_true, rng)
    vib = extract_features(windows, latent["rpm"])

    # ---- 8. sensor model -------------------------------------------------- #
    drift = sample_drift_fault(rng, T, sensor_drift_fault)

    measurable = {
        "rpm": latent["rpm"],
        "cht": latent["cht"],
        "egt": latent["egt"],
        "oil_pressure": latent["oil_pressure"],
        "oil_temperature": latent["oil_temperature"],
        "fuel_flow": latent["fuel_flow"],
        "battery_voltage": latent["battery_voltage"],
        "alternator_current": latent["alternator_current"],
        "manifold_pressure": latent["manifold_pressure"],
        "vibration_rms": vib["vibration_rms"],
    }
    measured = apply_sensor_model(measurable, calibration, rng,
                                 drift_series=drift["series"],
                                 noise_scale=noise_scale)

    # ---- 9. labels -------------------------------------------------------- #
    t_hours = np.arange(T, dtype=float) * SNAPSHOT_HOURS
    rul_true = (T - 1 - np.arange(T, dtype=float)) * SNAPSHOT_HOURS
    rul_capped = np.minimum(rul_true, RUL_CAP_H)
    health_index_true = 1.0 - worst_margin
    binding = np.argmax(np.stack([margins_hist[c] for c in CRITERION_NAMES], axis=0), axis=0)

    # ---- 10. assemble ----------------------------------------------------- #
    cols: Dict[str, np.ndarray] = {}
    cols["run_id"] = np.array([run_id] * T, dtype=object)
    cols["snapshot_index"] = np.arange(T, dtype=np.int32)
    cols["operating_hours"] = t_hours
    cols["sortie_index"] = ops["sortie_index"]
    cols["mission_phase"] = np.array([PHASE_NAMES[i] for i in ops["mission_phase"]], dtype=object)

    # environment (measured / known on a real aircraft)
    cols["altitude_ft"] = ops["altitude_ft"]
    cols["ambient_temperature_c"] = ops["ambient_temperature_c"]
    cols["ambient_pressure_kpa"] = latent["ambient_pressure_kpa"]
    cols["humidity"] = ops["humidity"]

    # commands
    cols["throttle"] = ops["throttle"]
    cols["engine_load"] = ops["throttle"] * ops["load_factor"]
    cols["injection_command"] = latent["injection_command"]
    cols["injection_timing_deg"] = latent["injection_timing_deg"]
    cols["load_factor"] = ops["load_factor"]
    cols["airspeed_factor"] = ops["airspeed_factor"]

    # measured sensors
    for k, v in measured.items():
        cols[k] = v

    # within-window statistics (available on a real engine from the FADEC log)
    for k in ("rpm", "cht", "egt", "oil_pressure", "fuel_flow"):
        cols[f"{k}_window_std"] = latent[f"{k}_std"]

    # vibration features
    for k, v in vib.items():
        if k == "vibration_rms":
            continue                       # already in `measured`
        cols[k] = v

    # ---- ground truth (TRAINING ONLY - never a model input) --------------- #
    for k in HEALTH_PARAMS:
        cols[f"true_{k}"] = theta_true[k]
    cols["true_health_index"] = health_index_true
    cols["true_worst_margin"] = worst_margin
    for c in CRITERION_NAMES:
        cols[f"true_margin_{c}"] = margins_hist[c]
    cols["true_binding_criterion"] = np.array([CRITERION_NAMES[i] for i in binding], dtype=object)
    cols["RUL_hours"] = rul_true
    cols["RUL_hours_capped"] = rul_capped
    cols["life_fraction"] = np.arange(T, dtype=float) / max(T - 1, 1)

    meta = {
        "run_id": run_id,
        "status": "ok",
        "seed": seed,
        "envelope": envelope,
        "scatter_mult": scatter_mult,
        "mechanism": mechanism,
        "secondary_mechanism": deg.secondary_mechanism,
        "rate_archetype": rate_archetype,
        "target_life_h": target_life_h,
        "realised_life_h": float(t_hours[-1]),
        "n_snapshots": int(T),
        "n_sorties": int(ops["sortie_index"][-1] + 1),
        "theatre": plan_mission.theatre,
        "sortie_mix": plan_mission.sortie_mix,
        "sl_temp_mean_c": plan_mission.sl_temp_mean_c,
        "humidity_mean": plan_mission.humidity_mean,
        "sensor_drift_fault": bool(sensor_drift_fault),
        "noise_scale": noise_scale,
        "binding_criterion_at_eol": CRITERION_NAMES[int(binding[-1])],
        "engine_params": {k: float(v) for k, v in p.items()},
        "sensor_calibration": calibration,
        "commissioning_baseline": baseline,
        "theta0": {k: float(v) for k, v in deg.theta0.items()},
        "theta_eol": {k: float(theta_true[k][-1]) for k in HEALTH_PARAMS},
        "mean_altitude_ft": float(np.mean(ops["altitude_ft"])),
        "mean_ambient_c": float(np.mean(ops["ambient_temperature_c"])),
        "max_ambient_c": float(np.max(ops["ambient_temperature_c"])),
        "max_altitude_ft": float(np.max(ops["altitude_ft"])),
    }
    meta.update(drift["meta"])
    return cols, meta


# --------------------------------------------------------------------------- #
# Fast-time burst
# --------------------------------------------------------------------------- #

def _run_transient_burst(ops: Dict[str, np.ndarray],
                         theta: Dict[str, np.ndarray],
                         p: Dict[str, float],
                         rng: np.random.Generator) -> Dict[str, np.ndarray]:
    """Run a short fast-time simulation at every snapshot, all snapshots at once.

    Each snapshot is initialised at its own quasi-steady operating point and then
    integrated for SNAPSHOT_BURST_S seconds while the throttle fluctuates with
    the volatility of the current mission phase. The mean over the burst becomes
    the reported value and the standard deviation becomes a window feature.

    This is what gives the dataset genuine transient content: during a
    `transition` phase the throttle moves faster than the cylinder head can
    follow, so CHT lags and the Digital Twin's quasi-steady prediction is
    genuinely wrong for a while. That mismatch is a real property of Digital
    Twins and the residual pipeline has to cope with it.
    """
    T = len(ops["throttle"])
    thr0 = ops["throttle"]
    alt = ops["altitude_ft"]
    oat = ops["ambient_temperature_c"]
    aspd = ops["airspeed_factor"]
    load = ops["load_factor"]

    ss = steady_state(thr0, alt, oat, theta, p, airspeed_factor=aspd, load_factor=load)
    state = FastState(ss["rpm"].copy(), ss["cht"].copy(), ss["egt"].copy(),
                      ss["oil_temperature"].copy(), ss["oil_pressure"].copy())

    n_steps = int(round(SNAPSHOT_BURST_S / FAST_DT_S))
    vol = ops["throttle_volatility"]

    # Accumulators for mean and variance (Welford would be overkill here).
    acc = {k: np.zeros(T) for k in ("rpm", "cht", "egt", "oil_p", "oil_t",
                                    "fuel", "inj", "map", "vib", "alt_i", "batt",
                                    "power", "torque")}
    acc2 = {k: np.zeros(T) for k in ("rpm", "cht", "egt", "oil_p", "fuel")}

    # Throttle activity: an AR(1) fluctuation around the phase mean.
    thr = thr0.copy()
    rho = 0.985
    for _ in range(n_steps):
        thr = thr0 + rho * (thr - thr0) + rng.normal(0.0, vol * 0.12, size=T)
        thr = np.clip(thr, 0.05, 1.0)

        out = step_fast(state, thr, alt, oat, theta, p, dt=FAST_DT_S,
                        airspeed_factor=aspd, load_factor=load)

        acc["rpm"] += state.rpm
        acc["cht"] += state.cht
        acc["egt"] += state.egt
        acc["oil_p"] += state.oil_p
        acc["oil_t"] += state.oil_t
        acc["fuel"] += out["fuel_flow_lph"]
        acc["inj"] += out["injection_command"]
        acc["map"] += out["manifold_pressure_kpa"]
        acc["vib"] += out["vib_rms_ss"]
        acc["alt_i"] += out["alternator_current"]
        acc["batt"] += out["battery_voltage"]
        acc["power"] += out["power_brake_w"]
        acc["torque"] += out["torque_brake"]

        acc2["rpm"] += state.rpm ** 2
        acc2["cht"] += state.cht ** 2
        acc2["egt"] += state.egt ** 2
        acc2["oil_p"] += state.oil_p ** 2
        acc2["fuel"] += out["fuel_flow_lph"] ** 2

    inv = 1.0 / n_steps
    mean = {k: v * inv for k, v in acc.items()}
    std = {k: np.sqrt(np.maximum(acc2[k] * inv - mean[{"rpm": "rpm", "cht": "cht",
                                                       "egt": "egt", "oil_p": "oil_p",
                                                       "fuel": "fuel"}[k]] ** 2, 0.0))
           for k in acc2}

    from ..physics.engine_model import atmosphere
    atm = atmosphere(alt, oat)

    # ECU injection/ignition timing schedule (a command, not a measurement).
    timing = (18.0 + 9.0 * np.clip((mean["rpm"] - 2000.0) / 3200.0, 0.0, 1.0)
              - 6.0 * np.clip(thr0 - 0.5, 0.0, 0.5) * 2.0
              + rng.normal(0.0, 0.25, size=T))

    return {
        "rpm": mean["rpm"],
        "cht": mean["cht"],
        "egt": mean["egt"],
        "oil_pressure": mean["oil_p"],
        "oil_temperature": mean["oil_t"],
        "fuel_flow": mean["fuel"],
        "injection_command": mean["inj"],
        "manifold_pressure": mean["map"],
        "vibration_rms": mean["vib"],
        "alternator_current": mean["alt_i"],
        "battery_voltage": mean["batt"],
        "power_brake_w": mean["power"],
        "torque_brake": mean["torque"],
        "injection_timing_deg": timing,
        "ambient_pressure_kpa": atm["pressure_pa"] / 1000.0,
        "rpm_std": std["rpm"],
        "cht_std": std["cht"],
        "egt_std": std["egt"],
        "oil_pressure_std": std["oil_p"],
        "fuel_flow_std": std["fuel"],
    }
