"""
sensors.py
==========
Sensor realism layer (Part 6).

The measured telemetry must never equal the latent physical value. Four separate
effects are applied, each of which breaks a different shortcut the ML model might
otherwise take:

  1. WHITE NOISE          per-sample, per-channel. Sets the noise floor.
  2. CALIBRATION BIAS     fixed per engine and per channel. This is the reason a
                          model cannot learn absolute thresholds - the same true
                          CHT reads differently on two airframes.
  3. LOW-FREQUENCY DRIFT  an Ornstein-Uhlenbeck process in operating hours. This
                          is the nastiest one for prognostics because slow drift
                          looks exactly like slow degradation. A system that
                          cannot tell them apart will raise false alarms on
                          healthy engines.
  4. OUTLIERS             occasional single-snapshot spikes, as produced by
                          connector noise, EMI and telemetry dropouts.

Optionally a SENSOR DRIFT FAULT can be injected on one channel: a genuine
instrument failure that is NOT engine degradation. Runs carrying it are used to
test whether the health estimator wrongly attributes the drift to the engine.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from ..config import (OUTLIER_PROB, OUTLIER_SIGMA_MULT, SENSOR_BIAS_SIGMA,
                      SENSOR_LF_NOISE_FRAC, SENSOR_LF_TAU_H, SENSOR_NOISE,
                      SNAPSHOT_HOURS)

CHANNELS: List[str] = list(SENSOR_NOISE.keys())


def sample_sensor_calibration(rng: np.random.Generator,
                              scatter_mult: float = 1.0) -> Dict[str, float]:
    """Fixed per-engine calibration bias for every channel."""
    return {c: float(rng.normal(0.0, SENSOR_BIAS_SIGMA[c] * scatter_mult))
            for c in CHANNELS}


def sample_drift_fault(rng: np.random.Generator, n_snapshots: int,
                       enabled: bool) -> Dict[str, np.ndarray]:
    """Optionally inject a slow instrument drift fault on one channel.

    Returns a dict with the additive drift series for each channel (mostly
    zeros), plus metadata describing what was injected.
    """
    drift = {c: np.zeros(n_snapshots) for c in CHANNELS}
    meta = {"drift_channel": "none", "drift_onset_frac": np.nan,
            "drift_magnitude_sigma": 0.0}
    if not enabled:
        return {"series": drift, "meta": meta}

    # Drift on a channel that plausibly drifts in service.
    ch = str(rng.choice(["egt", "cht", "oil_pressure", "fuel_flow", "oil_temperature"]))
    onset = float(rng.uniform(0.25, 0.70))
    onset_i = int(onset * n_snapshots)
    magnitude = float(rng.uniform(2.5, 7.0)) * float(rng.choice([-1.0, 1.0]))
    ramp = np.zeros(n_snapshots)
    if onset_i < n_snapshots:
        k = np.arange(n_snapshots - onset_i, dtype=float)
        ramp[onset_i:] = magnitude * SENSOR_NOISE[ch] * (k / max(len(k) - 1, 1)) ** 1.15
    drift[ch] = ramp
    meta = {"drift_channel": ch, "drift_onset_frac": onset,
            "drift_magnitude_sigma": magnitude}
    return {"series": drift, "meta": meta}


def ou_drift(rng: np.random.Generator, n: int, sigma: float,
             tau_h: float = SENSOR_LF_TAU_H,
             dt_h: float = SNAPSHOT_HOURS) -> np.ndarray:
    """Ornstein-Uhlenbeck low-frequency drift sampled at the snapshot cadence.

    Mean-reverting so it wanders without running away, with a correlation time
    of `tau_h` operating hours.
    """
    a = np.exp(-dt_h / max(tau_h, 1e-6))
    innov = sigma * np.sqrt(max(1.0 - a * a, 1e-12))
    x = np.empty(n)
    x[0] = rng.normal(0.0, sigma)
    e = rng.normal(0.0, innov, size=n)
    for i in range(1, n):
        x[i] = a * x[i - 1] + e[i]
    return x


def apply_sensor_model(truth: Dict[str, np.ndarray],
                       calibration: Dict[str, float],
                       rng: np.random.Generator,
                       drift_series: Dict[str, np.ndarray] | None = None,
                       noise_scale: float = 1.0) -> Dict[str, np.ndarray]:
    """Turn latent physical values into measured telemetry.

    `truth` maps channel name -> (N,) latent values. Channels absent from
    SENSOR_NOISE are passed through untouched (they are commands, not
    measurements).
    """
    out = {}
    n = len(next(iter(truth.values())))

    for name, values in truth.items():
        v = np.asarray(values, dtype=float).copy()
        if name not in SENSOR_NOISE:
            out[name] = v
            continue

        sigma = SENSOR_NOISE[name] * noise_scale

        # 1. white noise
        v = v + rng.normal(0.0, sigma, size=n)

        # 2. fixed calibration bias
        v = v + calibration.get(name, 0.0)

        # 3. low-frequency drift
        v = v + ou_drift(rng, n, sigma * SENSOR_LF_NOISE_FRAC)

        # 4. outliers
        hits = rng.random(n) < OUTLIER_PROB
        if hits.any():
            mult = rng.uniform(*OUTLIER_SIGMA_MULT, size=int(hits.sum()))
            sign = rng.choice([-1.0, 1.0], size=int(hits.sum()))
            v[hits] = v[hits] + sign * mult * sigma

        # 5. optional instrument drift fault
        if drift_series is not None and name in drift_series:
            v = v + drift_series[name]

        out[name] = v

    # Physical floors: a sensor cannot read a negative absolute pressure.
    if "oil_pressure" in out:
        out["oil_pressure"] = np.maximum(out["oil_pressure"], 0.05)
    if "vibration_rms" in out:
        out["vibration_rms"] = np.maximum(out["vibration_rms"], 0.01)
    if "fuel_flow" in out:
        out["fuel_flow"] = np.maximum(out["fuel_flow"], 0.0)
    if "rpm" in out:
        out["rpm"] = np.maximum(out["rpm"], 0.0)

    return out
