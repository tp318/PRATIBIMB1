"""
twin.py
=======
Digital Twin prediction and residual generation (Parts 4 and 5).

The twin answers one question: given the environment and the commands the
aircraft is issuing RIGHT NOW, what would a HEALTHY, FLEET-NOMINAL engine be
doing? It is deliberately blind to three things:

  * the true health parameters of the engine (it assumes all-nominal),
  * the individual engine's build scatter (it uses the fleet mean),
  * anything derived from the label.

Consequently the twin's predictions are NOT a copy of the measurements. A
healthy engine still produces non-zero residuals, because of build scatter,
sensor calibration and the twin's quasi-steady approximation of a transient
engine. That healthy residual spread is what sigma is estimated from, and what
makes a normalised residual meaningful.

Inputs the twin is allowed to use, and why each is legitimately available on a
real MALE UAV:

  throttle            FADEC/ECU command, logged
  altitude_ft         air data computer
  ambient_temp_c      OAT probe
  load_factor         propeller pitch / airspeed - from prop governor + air data
  airspeed_factor     cooling airflow proxy - derived from indicated airspeed

Note what is NOT used: injection_command is an OUTPUT of the twin, not an input.
Feeding the measured injection command into the twin would hide injector
degradation, because the twin would then reproduce whatever the ECU was doing
rather than what a healthy engine ought to need.
"""

from __future__ import annotations

from typing import Dict, Iterable

import numpy as np
import pandas as pd

from ..physics.engine_model import nominal_health, nominal_params, steady_state

# measured channel -> twin prediction channel
RESIDUAL_CHANNELS: Dict[str, str] = {
    "rpm": "rpm_pred",
    "cht": "cht_pred",
    "egt": "egt_pred",
    "oil_pressure": "oil_pressure_pred",
    "oil_temperature": "oil_temperature_pred",
    "fuel_flow": "fuel_flow_pred",
    "vibration_rms": "vibration_pred",
    "manifold_pressure": "manifold_pressure_pred",
}

TWIN_INPUTS = ["throttle", "altitude_ft", "ambient_temperature_c",
               "load_factor", "airspeed_factor"]


class DigitalTwin:
    """Quasi-steady healthy-engine predictor built on the shared MVEM."""

    def __init__(self, params: Dict[str, float] | None = None):
        self.params = params or nominal_params()

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a frame of twin predictions aligned to `df`'s index."""
        n = len(df)
        theta = nominal_health((n,))
        ss = steady_state(df["throttle"].to_numpy(),
                          df["altitude_ft"].to_numpy(),
                          df["ambient_temperature_c"].to_numpy(),
                          theta, self.params,
                          airspeed_factor=df["airspeed_factor"].to_numpy(),
                          load_factor=df["load_factor"].to_numpy())

        return pd.DataFrame({
            "rpm_pred": ss["rpm"],
            "cht_pred": ss["cht"],
            "egt_pred": ss["egt"],
            "oil_pressure_pred": ss["oil_pressure"],
            "oil_temperature_pred": ss["oil_temperature"],
            "fuel_flow_pred": ss["fuel_flow_lph"],
            "vibration_pred": ss["vibration_rms"],
            "manifold_pressure_pred": ss["manifold_pressure_kpa"],
            "torque_pred": ss["torque_brake"],
            "power_pred_w": ss["power_brake_w"],
            "injection_command_pred": ss["injection_command"],
            "bsfc_pred": ss["bsfc_kg_per_kwh"],
            "afr_pred": ss["afr"],
        }, index=df.index)

    def residuals(self, df: pd.DataFrame, pred: pd.DataFrame | None = None) -> pd.DataFrame:
        """raw residual r_i = measured_i - predicted_i for every channel."""
        pred = self.predict(df) if pred is None else pred
        out = {}
        for meas, pcol in RESIDUAL_CHANNELS.items():
            out[f"{meas}_residual"] = df[meas].to_numpy() - pred[pcol].to_numpy()
        # The injection command residual is informative in its own right: it is
        # the gap between what the ECU is asking for and what a healthy engine
        # at this operating point should need.
        out["injection_command_residual"] = (df["injection_command"].to_numpy()
                                             - pred["injection_command_pred"].to_numpy())
        return pd.DataFrame(out, index=df.index)


class SigmaModel:
    """Healthy residual dispersion, estimated from data rather than assumed.

    Part 5 forbids picking sigma by hand. Sigma is fitted from EARLY-LIFE
    snapshots of TRAINING runs only, which is the closest thing the dataset has
    to "a fleet of known-good engines", and is exactly what an operator would
    have available before any engine had degraded.

    Two refinements matter:

    1. PER MISSION PHASE. Residual spread is much larger during throttle
       transitions than in steady cruise, because that is where the twin's
       quasi-steady assumption is weakest. A single global sigma would call
       every transition an anomaly and every cruise degradation invisible.

    2. ROBUST ESTIMATOR. Sigma is estimated from the median absolute deviation
       scaled by 1.4826, so the injected outliers do not inflate it.
    """

    def __init__(self, channels: Iterable[str] | None = None):
        self.channels = list(channels) if channels else \
            [f"{c}_residual" for c in RESIDUAL_CHANNELS] + ["injection_command_residual"]
        self.global_sigma: Dict[str, float] = {}
        self.phase_sigma: Dict[str, Dict[str, float]] = {}
        self.global_mu: Dict[str, float] = {}
        self.phase_mu: Dict[str, Dict[str, float]] = {}
        self.fitted = False

    @staticmethod
    def _robust_sigma(x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float)
        x = x[np.isfinite(x)]
        if x.size < 8:
            return float("nan")
        mad = np.median(np.abs(x - np.median(x)))
        return float(max(mad * 1.4826, 1e-9))

    def fit(self, healthy: pd.DataFrame) -> "SigmaModel":
        for ch in self.channels:
            if ch not in healthy.columns:
                continue
            v = healthy[ch].to_numpy()
            self.global_sigma[ch] = self._robust_sigma(v)
            self.global_mu[ch] = float(np.median(v[np.isfinite(v)]))
            self.phase_sigma[ch] = {}
            self.phase_mu[ch] = {}
            for phase, grp in healthy.groupby("mission_phase"):
                s = self._robust_sigma(grp[ch].to_numpy())
                if np.isfinite(s):
                    self.phase_sigma[ch][str(phase)] = s
                    self.phase_mu[ch][str(phase)] = float(np.median(grp[ch].to_numpy()))
        self.fitted = True
        return self

    def normalise(self, df: pd.DataFrame) -> pd.DataFrame:
        """z_i = (r_i - mu_i(phase)) / sigma_i(phase).

        The healthy median mu is subtracted as well as dividing by sigma. A twin
        with a small systematic offset in a phase would otherwise produce a
        permanently non-zero z on a perfectly healthy engine, which would eat the
        detection budget for no reason.
        """
        if not self.fitted:
            raise RuntimeError("SigmaModel.fit() must be called before normalise()")
        phases = df["mission_phase"].astype(str).to_numpy()
        out = {}
        for ch in self.channels:
            if ch not in df.columns:
                continue
            sig = np.array([self.phase_sigma[ch].get(p, self.global_sigma[ch]) for p in phases])
            mu = np.array([self.phase_mu[ch].get(p, self.global_mu[ch]) for p in phases])
            base = ch[:-9] if ch.endswith("_residual") else ch
            out[f"{base}_z"] = (df[ch].to_numpy() - mu) / np.maximum(sig, 1e-9)
        return pd.DataFrame(out, index=df.index)

    def to_dict(self) -> Dict:
        return {"global_sigma": self.global_sigma, "phase_sigma": self.phase_sigma,
                "global_mu": self.global_mu, "phase_mu": self.phase_mu}

    @classmethod
    def from_dict(cls, d: Dict) -> "SigmaModel":
        m = cls(channels=list(d["global_sigma"]))
        m.global_sigma = d["global_sigma"]
        m.phase_sigma = d["phase_sigma"]
        m.global_mu = d["global_mu"]
        m.phase_mu = d["phase_mu"]
        m.fitted = True
        return m
