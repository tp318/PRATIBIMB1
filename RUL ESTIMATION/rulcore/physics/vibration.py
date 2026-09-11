"""
vibration.py
============
Synthesis of raw vibration windows and extraction of the statistical / spectral
features (Part 3).

The features are DERIVED from a synthesised time-domain signal rather than
written down directly. That matters: if kurtosis and crest factor were generated
as independent analytic functions of the health parameters, they would be
perfectly consistent with each other by construction and a model could exploit
relationships that no real accelerometer would ever produce. Building the signal
first and measuring it afterwards means the features carry the same correlations,
leakage between orders and estimator noise that real feature extraction has.

Signal model for a 4-cylinder 4-stroke engine at shaft frequency f = rpm/60:

  0.5X  half order      cylinder-to-cylinder imbalance (uneven fuelling)
  1X    shaft order     rotating imbalance, bearing wear
  2X    firing order    two power strokes per revolution on a 4-cyl 4-stroke
  3X                    higher harmonic content of the gas load
  impulses              metal-to-metal contact when the oil film degrades
  broadband             combustion and structural noise

Health coupling (all of it flows from the health parameters, never from a fault
label):
  friction_mult  -> 1X growth and impulsiveness (bearing / journal wear)
  eta_comb       -> 2X and 3X growth (rough, late burning)
  eta_inj        -> 0.5X growth (uneven cylinder fuelling)
  lub_health     -> impulsiveness and broadband floor (film breakdown)
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from ..config import VIB_FS_HZ, VIB_WINDOW_S


def synthesise_windows(rpm: np.ndarray,
                       target_rms: np.ndarray,
                       theta: Dict[str, np.ndarray],
                       rng: np.random.Generator,
                       fs: float = VIB_FS_HZ,
                       window_s: float = VIB_WINDOW_S) -> np.ndarray:
    """Synthesise one raw vibration window per snapshot.

    Parameters
    ----------
    rpm : (N,) shaft speed for each snapshot
    target_rms : (N,) overall RMS level from the MVEM
    theta : health parameters, each (N,)

    Returns
    -------
    (N, M) array of acceleration windows in g, M = fs * window_s
    """
    rpm = np.asarray(rpm, dtype=float)
    n = rpm.shape[0]
    m = int(round(fs * window_s))
    t = np.arange(m, dtype=float) / fs                     # (M,)

    f_shaft = (rpm / 60.0)[:, None]                        # (N,1)

    # --- health-driven order amplitudes (relative, unitless) --------------- #
    wear = np.maximum(np.asarray(theta["friction_mult"], dtype=float) - 1.0, 0.0)[:, None]
    comb_loss = np.maximum(1.0 - np.asarray(theta["eta_comb"], dtype=float), 0.0)[:, None]
    inj_loss = np.maximum(1.0 - np.asarray(theta["eta_inj"], dtype=float), 0.0)[:, None]
    lub_loss = np.maximum(1.0 - np.asarray(theta["lub_health"], dtype=float), 0.0)[:, None]

    a_half = 0.10 + 5.2 * inj_loss + 1.1 * comb_loss
    a_1x = 0.60 + 3.4 * wear + 0.7 * lub_loss
    a_2x = 0.85 + 6.5 * comb_loss + 0.8 * wear
    a_3x = 0.30 + 3.1 * comb_loss + 0.5 * wear
    broadband = 0.42 + 1.5 * lub_loss + 0.8 * wear

    # Random phases so windows are not synchronised across snapshots.
    ph = rng.uniform(0.0, 2.0 * np.pi, size=(n, 4))

    sig = (a_half * np.sin(2.0 * np.pi * 0.5 * f_shaft * t + ph[:, 0:1])
           + a_1x * np.sin(2.0 * np.pi * 1.0 * f_shaft * t + ph[:, 1:2])
           + a_2x * np.sin(2.0 * np.pi * 2.0 * f_shaft * t + ph[:, 2:3])
           + a_3x * np.sin(2.0 * np.pi * 3.0 * f_shaft * t + ph[:, 3:4]))

    # Structural resonance colouring of the broadband floor: a lightly damped
    # second-order response around 320 Hz, applied to white noise.
    white = rng.standard_normal((n, m))
    freqs = np.fft.rfftfreq(m, d=1.0 / fs)
    f0, zeta = 320.0, 0.16
    resp = 1.0 / np.sqrt((1.0 - (freqs / f0) ** 2) ** 2 + (2.0 * zeta * freqs / f0) ** 2)
    resp = resp / np.max(resp)
    coloured = np.fft.irfft(np.fft.rfft(white, axis=1) * resp[None, :], n=m, axis=1)
    coloured = coloured / np.maximum(coloured.std(axis=1, keepdims=True), 1e-9)
    sig = sig + broadband * coloured

    # --- impulsive content from oil-film breakdown / metal contact --------- #
    # Impulse rate scales with lubrication loss and mechanical wear. These are
    # what drive kurtosis and crest factor away from the Gaussian values.
    imp_strength = 7.0 * lub_loss + 3.2 * wear                # (N,1)
    active = np.asarray(imp_strength).ravel() > 0.02
    if active.any():
        idx = np.nonzero(active)[0]
        # number of impacts in the window scales with shaft speed
        n_imp = np.clip((f_shaft.ravel()[idx] * window_s * 0.9).astype(int), 1, 90)
        decay = np.exp(-t * 900.0)                             # sharp ring-down
        ring = decay * np.sin(2.0 * np.pi * 620.0 * t)
        for j, i in enumerate(idx):
            k = int(n_imp[j])
            starts = rng.integers(0, m, size=k)
            amps = np.abs(rng.normal(0.0, float(imp_strength[i, 0]), size=k))
            train = np.zeros(m)
            for s0, a0 in zip(starts, amps):
                seg = min(m - s0, 180)
                train[s0:s0 + seg] += a0 * ring[:seg]
            sig[i] += train

    # Scale each window to the RMS demanded by the mean-value model, then add a
    # small amount of sensor noise on top so the RMS is not exact.
    cur = np.sqrt(np.mean(sig ** 2, axis=1, keepdims=True))
    sig = sig / np.maximum(cur, 1e-9) * np.asarray(target_rms, dtype=float)[:, None]
    sig = sig + rng.normal(0.0, 0.012, size=sig.shape)
    return sig


def extract_features(windows: np.ndarray, rpm: np.ndarray,
                     fs: float = VIB_FS_HZ) -> Dict[str, np.ndarray]:
    """Statistical and order-domain features from raw vibration windows."""
    x = np.asarray(windows, dtype=float)
    n, m = x.shape
    mean = x.mean(axis=1, keepdims=True)
    xc = x - mean

    rms = np.sqrt(np.mean(x ** 2, axis=1))
    std = xc.std(axis=1)
    peak = np.max(np.abs(xc), axis=1)
    crest = peak / np.maximum(rms, 1e-9)
    var = np.maximum(np.mean(xc ** 2, axis=1), 1e-18)
    kurt = np.mean(xc ** 4, axis=1) / var ** 2                 # Fisher + 3
    skew = np.mean(xc ** 3, axis=1) / var ** 1.5

    # Single-sided amplitude spectrum with a Hann window
    win = np.hanning(m)
    coherent_gain = win.mean()
    spec = np.abs(np.fft.rfft(xc * win[None, :], axis=1)) * 2.0 / (m * coherent_gain)
    freqs = np.fft.rfftfreq(m, d=1.0 / fs)

    # Ignore the DC bin when looking for the dominant frequency
    spec_nodc = spec.copy()
    spec_nodc[:, 0] = 0.0
    dom_idx = np.argmax(spec_nodc, axis=1)
    dom_freq = freqs[dom_idx]

    f_shaft = np.asarray(rpm, dtype=float) / 60.0
    df = freqs[1] - freqs[0]

    def order_amplitude(order: float) -> np.ndarray:
        """Peak amplitude in a narrow band around the requested shaft order."""
        target = f_shaft * order
        centre = np.clip((target / df).round().astype(int), 0, spec.shape[1] - 1)
        half = max(1, int(round(2.5 / df)))                    # +/- 2.5 Hz band
        out = np.empty(n)
        for i in range(n):
            lo = max(0, centre[i] - half)
            hi = min(spec.shape[1], centre[i] + half + 1)
            out[i] = spec[i, lo:hi].max() if hi > lo else 0.0
        return out

    return {
        "vibration_rms": rms,
        "vibration_std": std,
        "vibration_peak": peak,
        "vibration_kurtosis": kurt,
        "vibration_skew": skew,
        "vibration_crest_factor": crest,
        "vibration_dominant_freq": dom_freq,
        "vibration_1x": order_amplitude(1.0),
        "vibration_2x": order_amplitude(2.0),
        "vibration_3x": order_amplitude(3.0),
        "vibration_half_x": order_amplitude(0.5),
    }
