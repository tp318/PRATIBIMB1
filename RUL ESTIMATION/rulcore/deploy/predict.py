"""
predict.py
==========
Deployment interface (Deliverable J).

Exposes the whole pipeline behind one call:

    predictor = RULPredictor.load()
    result = predictor.predict_rul(recent_telemetry)

`recent_telemetry` is a DataFrame of at least `seq_len` consecutive
condition-monitoring snapshots containing only quantities a real aircraft
produces: environment, ECU commands, sensors and vibration features. No health
parameter, no label, nothing from the simulator.

Two usage modes:

  BATCH     predict_rul(df) runs the filter over the supplied history from
            scratch. Simple, stateless, and what the ground station uses when
            replaying a mission.

  STREAMING update(snapshot) advances the filter one snapshot at a time and
            keeps the UKF state and the rate identification between calls. This
            is the edge mode: constant memory, one filter step per snapshot, and
            no need to re-process history.

The split between the two matters operationally. The filter is cheap and runs on
the aircraft; the Monte Carlo propagation is comparatively expensive and only
needs to run when someone asks for a prognosis, so it runs on the ground station
or on a slow cadence onboard.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..config import (ARTIFACT_DIR, FEATURE_DIR, HEALTH_EOL, HEALTH_NOMINAL,
                      HEALTH_PARAMS, RUL_N_PARTICLES, SEQ_LEN_DEFAULT,
                      SNAPSHOT_HOURS, UKF_MEAS)
from ..dataset.build_features import ROLL_LONG, ROLL_MED, ROLL_SHORT, _rolling_features
from ..estimation.twin import DigitalTwin, SigmaModel
from ..estimation.ukf import HealthUKF
from ..models.degradation_model import (PHI_RANGE, PhysicsRateEstimator,
                                        shape_term, stress_from_observables)
from ..models.rul_propagation import HealthIndexSurrogate, propagate_rul
from ..models.windows import FEATURE_SETS, StandardScaler3D, available_features

# The engine's own commissioning baseline is taken as the median health estimate
# over snapshots [SKIP, SNAPSHOTS) of its service life: skip the filter's
# start-up transient, then average long enough to beat the noise. 40 snapshots
# is 10 operating hours, comfortably inside the incubation period of every
# degradation mechanism in the fleet.
BASELINE_SKIP = 12
BASELINE_SNAPSHOTS = 40

Z_CHANNELS = ["rpm_z", "cht_z", "egt_z", "oil_pressure_z", "oil_temperature_z",
              "fuel_flow_z", "vibration_rms_z", "manifold_pressure_z",
              "injection_command_z"]


class RULPredictor:
    """End-to-end physics-informed prognostic estimator."""

    def __init__(self, sigma_model: SigmaModel, surrogate: HealthIndexSurrogate,
                 rate_model=None, rate_features: Optional[List[str]] = None,
                 rate_scaler: Optional[StandardScaler3D] = None,
                 seq_len: int = SEQ_LEN_DEFAULT,
                 gru_log_mult_std: float = 0.25,
                 q_scale: float = 1.0):
        self.twin = DigitalTwin()
        self.sigma_model = sigma_model
        self.surrogate = surrogate
        self.rate_model = rate_model
        self.rate_features = rate_features or []
        self.rate_scaler = rate_scaler
        self.seq_len = seq_len
        self.gru_log_mult_std = gru_log_mult_std
        self.q_scale = q_scale

        self.sigma_r = {c: sigma_model.global_sigma[f"{c}_residual"] for c in UKF_MEAS}
        self.reset()

    # ------------------------------------------------------------------ #
    @classmethod
    def load(cls, artifact_dir: str = ARTIFACT_DIR,
             feature_dir: str = FEATURE_DIR) -> "RULPredictor":
        sigma = SigmaModel.from_dict(
            json.load(open(os.path.join(feature_dir, "sigma_model.json"), encoding="utf-8")))

        sur_path = os.path.join(artifact_dir, "health_index_surrogate.json")
        if os.path.exists(sur_path):
            surrogate = HealthIndexSurrogate.from_dict(
                json.load(open(sur_path, encoding="utf-8")))
        else:
            surrogate = HealthIndexSurrogate().fit()

        rate_model = rate_features = rate_scaler = None
        gru_std, seq_len = 0.25, SEQ_LEN_DEFAULT
        rp = os.path.join(artifact_dir, "gru_rate_correction.pt")
        if os.path.exists(rp):
            import torch

            from ..models.gru_correction import DegradationRateGRU
            ck = torch.load(rp, map_location="cpu", weights_only=False)
            rate_features = ck["features"]
            seq_len = ck["seq_len"]
            rate_scaler = StandardScaler3D.from_dict(ck["scaler"])
            rate_model = DegradationRateGRU(n_features=len(rate_features))
            rate_model.load_state_dict(ck["state_dict"])
            rate_model.eval()

        tsum = os.path.join(os.path.dirname(feature_dir), "..", "outputs", "reports",
                            "training_summary.json")
        tsum = os.path.normpath(tsum)
        if os.path.exists(tsum):
            try:
                gru_std = float(json.load(open(tsum, encoding="utf-8"))
                                ["rate_gru"]["log_mult_std"])
            except Exception:
                pass

        qs = 1.0
        tune = os.path.normpath(os.path.join(os.path.dirname(tsum), "ukf_tuning_best.json"))
        if os.path.exists(tune):
            try:
                qs = float(json.load(open(tune, encoding="utf-8"))["q_scale"])
            except Exception:
                pass

        return cls(sigma, surrogate, rate_model, rate_features, rate_scaler,
                   seq_len, gru_std, qs)

    # ------------------------------------------------------------------ #
    def reset(self):
        """Clear the streaming state (new engine, or new deployment)."""
        self.ukf = HealthUKF(self.sigma_r, q_scale=self.q_scale)
        self.rate_est = PhysicsRateEstimator()
        self._stress_hist: List[np.ndarray] = []
        self._phi_prev: Optional[Dict[str, float]] = None
        self._n_seen = 0
        self._baseline: Optional[Dict[str, float]] = None
        self._phi_early: List[np.ndarray] = []

    # ------------------------------------------------------------------ #
    def _prepare(self, telemetry: pd.DataFrame) -> pd.DataFrame:
        """Twin predictions, residuals, z-scores and rolling features."""
        df = telemetry.reset_index(drop=True).copy()
        pred = self.twin.predict(df)
        resid = self.twin.residuals(df, pred)
        df = pd.concat([df, pred, resid], axis=1)
        df = pd.concat([df, self.sigma_model.normalise(df)], axis=1)
        df = pd.concat([df, _rolling_features(df)], axis=1)
        zc = [c for c in Z_CHANNELS if c in df.columns]
        zm = df[zc].to_numpy()
        df["z_absmax"] = np.nanmax(np.abs(zm), axis=1)
        df["z_sq_sum"] = np.nansum(zm ** 2, axis=1)
        df["z_sq_sum_roll"] = pd.Series(df["z_sq_sum"]).rolling(
            ROLL_MED, min_periods=8).median()
        return df

    def _run_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """Advance the UKF and the rate identification over `df`."""
        Y = df[list(UKF_MEAS)].to_numpy()
        U = df[["throttle", "altitude_ft", "ambient_temperature_c",
                "airspeed_factor", "load_factor"]].to_numpy()
        stress = stress_from_observables(df)

        n = len(df)
        est = {k: np.empty(n) for k in HEALTH_PARAMS}
        std = {k: np.empty(n) for k in HEALTH_PARAMS}

        for i in range(n):
            u = {"throttle": U[i, 0], "altitude_ft": U[i, 1],
                 "ambient_temperature_c": U[i, 2], "airspeed_factor": U[i, 3],
                 "load_factor": U[i, 4]}
            self.ukf.step(Y[i], u)
            s, sd = self.ukf.state, self.ukf.std
            for k in HEALTH_PARAMS:
                est[k][i] = s[k]
                std[k][i] = sd[k]

            phi_i = {k: (s[k] - HEALTH_NOMINAL[k]) / PHI_RANGE[k] for k in HEALTH_PARAMS}
            x_i = {k: float(stress[k][i] * shape_term(np.array(phi_i[k])) * SNAPSHOT_HOURS)
                   for k in HEALTH_PARAMS}
            if self._phi_prev is not None:
                dphi = {k: phi_i[k] - self._phi_prev[k] for k in HEALTH_PARAMS}
                self.rate_est.update(dphi, x_i, phi_i)
            self._phi_prev = phi_i
            self._stress_hist.append(np.array([stress[k][i] for k in HEALTH_PARAMS]))
            self._n_seen += 1

        for k in HEALTH_PARAMS:
            df[f"est_{k}"] = est[k]
            df[f"est_{k}_std"] = std[k]
            df[f"phi_{k}"] = (est[k] - HEALTH_NOMINAL[k]) / PHI_RANGE[k]
            df[f"stress_{k}"] = stress[k]

        # Freeze this engine's own commissioning baseline once the filter has
        # settled over its first hours in service. Scoring against the engine's
        # own green run rather than the fleet mean removes build scatter from
        # the health index; measured on the raw fleet, not doing this biased
        # EOL 18 h early on average.
        if self._baseline is None:
            self._phi_early.extend(
                np.stack([df[f'phi_{k}'].to_numpy() for k in HEALTH_PARAMS], axis=1))
            if len(self._phi_early) >= BASELINE_SNAPSHOTS:
                early = np.array(self._phi_early[BASELINE_SKIP:BASELINE_SNAPSHOTS])
                self._baseline = self.surrogate.baseline_from_phi(
                    np.median(early, axis=0))

        coeffs = self.rate_est.coefficients()
        for k in HEALTH_PARAMS:
            df[f"a_{k}"] = coeffs[k]
            df[f"rate_phys_{k}"] = np.maximum(
                coeffs[k] * stress[k] * shape_term(df[f"phi_{k}"].to_numpy()), 0.0)
        return df

    # ------------------------------------------------------------------ #
    def _gru_multiplier(self, df: pd.DataFrame) -> np.ndarray:
        """Learned correction to the physics rate for the latest snapshot."""
        if self.rate_model is None or len(df) < self.seq_len:
            return np.ones(len(HEALTH_PARAMS))
        import torch

        feats = available_features(df, self.rate_features)
        if len(feats) != len(self.rate_features):
            return np.ones(len(HEALTH_PARAMS))
        X = df[self.rate_features].to_numpy(dtype=np.float32)[-self.seq_len:]
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)[None, :, :]
        X = self.rate_scaler.transform(X)
        rp = df[[f"rate_phys_{k}" for k in HEALTH_PARAMS]].to_numpy(
            dtype=np.float32)[-1:][None, 0]
        with torch.no_grad():
            out = self.rate_model(torch.from_numpy(X), torch.from_numpy(rp))
        return out["mult"].numpy().ravel()

    # ------------------------------------------------------------------ #
    def predict_rul(self, recent_telemetry: pd.DataFrame,
                    n_particles: int = RUL_N_PARTICLES,
                    seed: int = 0, reset: bool = True) -> Dict:
        """Full prognosis from a window of recent telemetry.

        Returns the health index, the estimated health parameters, the RUL
        quantiles, the probability of falling below operational thresholds, and
        a confidence flag.
        """
        if reset:
            self.reset()
        df = self._prepare(recent_telemetry)
        df = self._run_filter(df)

        state = {k: float(df[f"est_{k}"].iloc[-1]) for k in HEALTH_PARAMS}
        state_std = {k: float(df[f"est_{k}_std"].iloc[-1]) for k in HEALTH_PARAMS}

        phi_mean = np.array([(state[k] - HEALTH_NOMINAL[k]) / PHI_RANGE[k]
                             for k in HEALTH_PARAMS])
        # Map the UKF covariance into phi units (diagonal is sufficient: the
        # propagation re-samples the correlated part through the shared stress
        # bootstrap anyway).
        phi_std = np.array([state_std[k] / abs(PHI_RANGE[k]) for k in HEALTH_PARAMS])
        phi_cov = np.diag(np.maximum(phi_std, 1e-4) ** 2)

        coeffs = np.array([self.rate_est.rate_coefficient(k) for k in HEALTH_PARAMS])
        stress_hist = np.array(self._stress_hist[-400:]) if self._stress_hist \
            else np.ones((1, len(HEALTH_PARAMS)))
        mult = self._gru_multiplier(df)

        res = propagate_rul(phi_mean, phi_cov, coeffs, stress_hist, self.surrogate,
                            gru_multiplier=mult, baseline=self._baseline,
                            gru_rel_sigma=self.gru_log_mult_std,
                            n_particles=n_particles, seed=seed)

        hi = float(self.surrogate.health_index(phi_mean[None, :], self._baseline)[0])
        binding = self.surrogate.binding(phi_mean[None, :], self._baseline)[0]
        from ..dataset.eol import CRITERION_NAMES

        # Confidence: driven by how much data the filter has seen, how wide the
        # RUL interval is relative to its median, and whether the particle
        # ensemble was censored by the horizon.
        rel_width = ((res["rul_p90_hours"] - res["rul_p10_hours"])
                     / max(res["rul_median_hours"], 1.0))
        if self._n_seen < self.seq_len:
            conf = "LOW"
        elif res["censored_fraction"] > 0.35 or rel_width > 1.5:
            conf = "LOW"
        elif rel_width > 0.8:
            conf = "MEDIUM"
        else:
            conf = "HIGH"

        return {
            "health_index": round(hi, 4),
            "estimated_health_parameters": {k: round(v, 5) for k, v in state.items()},
            "estimated_health_parameter_std": {k: round(v, 5) for k, v in state_std.items()},
            "consumed_life_fraction": {k: round(float(p), 4)
                                       for k, p in zip(HEALTH_PARAMS, phi_mean)},
            "rul_median_hours": round(res["rul_median_hours"], 1),
            "rul_p10_hours": round(res["rul_p10_hours"], 1),
            "rul_p90_hours": round(res["rul_p90_hours"], 1),
            "probability_rul_below_100h": round(res["probability_rul_below_100h"], 4),
            "probability_rul_below_50h": round(res["probability_rul_below_50h"], 4),
            "probability_rul_below_25h": round(res["probability_rul_below_25h"], 4),
            "confidence": conf,
            "limiting_criterion": CRITERION_NAMES[int(binding)],
            "expected_failure_mode": res["binding_criterion_at_eol"],
            "censored_fraction": round(res["censored_fraction"], 3),
            "snapshots_processed": int(self._n_seen),
            "dominant_degradation": max(
                zip(HEALTH_PARAMS, phi_mean), key=lambda x: x[1])[0],
        }

    # ------------------------------------------------------------------ #
    def update(self, snapshot: pd.DataFrame) -> Dict:
        """Streaming edge update: one snapshot, filter only, no propagation.

        Returns the current health estimate and residual health scores. Cheap
        enough to run at the monitoring cadence on an embedded computer.
        """
        df = self._prepare(snapshot)
        df = self._run_filter(df.tail(1).reset_index(drop=True))
        phi = np.array([float(df[f"phi_{k}"].iloc[-1]) for k in HEALTH_PARAMS])
        return {
            "health_index": float(self.surrogate.health_index(phi[None, :], self._baseline)[0]),
            "estimated_health_parameters": {k: float(df[f"est_{k}"].iloc[-1])
                                            for k in HEALTH_PARAMS},
            "z_absmax": float(df["z_absmax"].iloc[-1]),
            "snapshots_processed": self._n_seen,
        }


def predict_rul(recent_telemetry: pd.DataFrame, predictor: RULPredictor | None = None
                ) -> Dict:
    """Module-level convenience wrapper matching the requested signature."""
    predictor = predictor or RULPredictor.load()
    return predictor.predict_rul(recent_telemetry)
