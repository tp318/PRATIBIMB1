"""
ukf.py
======
Unscented Kalman Filter for online estimation of latent engine health (Part 10).

STATE VECTOR
------------
    x = [eta_inj, eta_comb, h_cool, friction_mult, lub_health, eta_vol]

This is a JOINT PARAMETER estimation problem, not a fast-state tracking problem,
and the state vector reflects that. The reason is timescale separation, and it is
worth being explicit because the brief suggested including Pman, Tman, Tengine,
Toil and omega as filter states:

    tau_egt  ~ 3 s        tau_oil_p ~ 2 s
    tau_cht  ~ 25 s       tau_oil_t ~ 20 s
    snapshot interval = 900 s

At a 15-minute condition-monitoring cadence every one of those dynamic states
has settled to within e^(-36) of its steady value. Carrying them as filter
states would add five dimensions whose propagation is exactly "go to steady
state", which is what the measurement model already computes. It would cost
sigma points, worsen conditioning and estimate nothing that is not already
determined. So the dynamic states are solved algebraically INSIDE the
measurement model and the filter estimates only what is genuinely unobserved and
genuinely slow: the health parameters.

    x_dot = 0 + w,          w ~ N(0, Q)      (random walk on health)
    y     = g(x, u, env)    + v,  v ~ N(0, R)

where g() is the quasi-steady MVEM evaluated with FLEET-NOMINAL build
parameters. The filter does not know the individual engine's build.

MEASUREMENT NOISE R
-------------------
R is NOT the sensor datasheet noise. From the filter's point of view the
"noise" on a measurement is everything that separates it from the nominal
model prediction on a healthy engine: sensor noise, per-engine calibration
bias, build scatter, and the twin's quasi-steady approximation error. That is
precisely the quantity SigmaModel estimates from healthy training data, so R is
built from those sigmas. Using datasheet noise instead would make the filter
wildly overconfident and it would chase build scatter as if it were degradation.

NO LEAKAGE
----------
The filter receives measurements, commands and environment. It never receives
true_* columns. The true health parameters are used only to score the estimates
afterwards.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

from ..config import (HEALTH_DIRECTION, UKF_ALPHA, UKF_BETA, UKF_BOUNDS,
                      UKF_KAPPA, UKF_MEAS, UKF_MONOTONE, UKF_MONOTONE_BURNIN,
                      UKF_P0_DIAG, UKF_Q_DIAG, UKF_RECOVERY_FRAC, UKF_STATE)
from ..physics.engine_model import nominal_params, steady_state

# measurement name -> key in the steady_state() output
MEAS_KEY = {
    "rpm": "rpm",
    "cht": "cht",
    "egt": "egt",
    "oil_pressure": "oil_pressure",
    "oil_temperature": "oil_temperature",
    "fuel_flow": "fuel_flow_lph",
    "vibration_rms": "vibration_rms",
    "manifold_pressure": "manifold_pressure_kpa",
}


class HealthUKF:
    """Unscented Kalman Filter over the engine health parameter vector."""

    def __init__(self,
                 sigma_r: Dict[str, float],
                 params: Dict[str, float] | None = None,
                 state_names: Sequence[str] = tuple(UKF_STATE),
                 meas_names: Sequence[str] = tuple(UKF_MEAS),
                 q_scale: float = 1.0,
                 r_scale: float = 1.0,
                 gate_chi2: float = 25.0,
                 monotone: bool = UKF_MONOTONE):
        self.state_names: List[str] = list(state_names)
        self.meas_names: List[str] = list(meas_names)
        self.params = params or nominal_params()
        self.n = len(self.state_names)
        self.m = len(self.meas_names)

        self.x = np.array([1.0 for _ in self.state_names], dtype=float)
        self.P = np.diag([UKF_P0_DIAG[k] ** 2 for k in self.state_names])
        self.Q = np.diag([(UKF_Q_DIAG[k] * q_scale) ** 2 for k in self.state_names])
        self.R = np.diag([(sigma_r[k] * r_scale) ** 2 for k in self.meas_names])

        self.gate_chi2 = gate_chi2
        self._build_weights()

        self.monotone = monotone
        self._direction = np.array([HEALTH_DIRECTION[k] for k in self.state_names], dtype=float)
        self._q_std = np.array([UKF_Q_DIAG[k] * q_scale for k in self.state_names])
        self._best = None            # most-healthy estimate seen so far

        self.n_gated = 0
        self.n_updates = 0

    def _apply_monotone(self):
        """Ratchet the estimate so health can only improve very slowly.

        Wear is irreversible, so an estimate that swings back toward "as new"
        is almost always the filter trading one parameter against a correlated
        one rather than a real recovery. The ratchet allows a small recovery
        allowance per snapshot (a fraction of the process-noise step) so that a
        genuinely unlucky start can still be corrected, but it stops a
        collinear pair from drifting apart indefinitely.
        """
        if not self.monotone:
            return
        if self.n_updates < UKF_MONOTONE_BURNIN or self._best is None:
            self._best = self.x.copy()
            return

        allow = UKF_RECOVERY_FRAC * self._q_std
        # For falling parameters (direction -1) "healthier" means larger.
        falling = self._direction < 0
        limit = np.where(falling, self._best + allow, self._best - allow)
        self.x = np.where(falling, np.minimum(self.x, limit), np.maximum(self.x, limit))
        self._best = np.where(falling,
                              np.minimum(self._best, self.x),
                              np.maximum(self._best, self.x))

    # ------------------------------------------------------------------ #
    def _build_weights(self):
        n = self.n
        lam = UKF_ALPHA ** 2 * (n + UKF_KAPPA) - n
        self.lam = lam
        self.gamma = np.sqrt(max(n + lam, 1e-12))
        wm = np.full(2 * n + 1, 1.0 / (2.0 * (n + lam)))
        wc = wm.copy()
        wm[0] = lam / (n + lam)
        wc[0] = lam / (n + lam) + (1.0 - UKF_ALPHA ** 2 + UKF_BETA)
        self.wm, self.wc = wm, wc

    def _sigma_points(self) -> np.ndarray:
        n = self.n
        P = 0.5 * (self.P + self.P.T)
        # Jitter until the Cholesky succeeds; the nonlinear update can nudge P
        # very slightly non-PSD and failing hard here would abort a whole run.
        jitter = 0.0
        for _ in range(6):
            try:
                S = np.linalg.cholesky(P + jitter * np.eye(n))
                break
            except np.linalg.LinAlgError:
                jitter = max(jitter * 10.0, 1e-14)
        else:
            S = np.diag(np.sqrt(np.maximum(np.diag(P), 1e-16)))

        pts = np.empty((2 * n + 1, n))
        pts[0] = self.x
        for i in range(n):
            pts[1 + i] = self.x + self.gamma * S[:, i]
            pts[1 + n + i] = self.x - self.gamma * S[:, i]
        return self._clip(pts)

    def _clip(self, pts: np.ndarray) -> np.ndarray:
        for i, k in enumerate(self.state_names):
            lo, hi = UKF_BOUNDS[k]
            pts[..., i] = np.clip(pts[..., i], lo, hi)
        return pts

    # ------------------------------------------------------------------ #
    def measurement_model(self, pts: np.ndarray, u: Dict[str, float]) -> np.ndarray:
        """Evaluate g() for every sigma point at once.

        `pts` is (K, n). Returns (K, m).
        """
        k = pts.shape[0]
        theta = {name: pts[:, i] for i, name in enumerate(self.state_names)}
        # Any health parameter not in the state vector is held at nominal.
        for name in ("eta_inj", "eta_comb", "h_cool", "friction_mult",
                     "lub_health", "eta_vol"):
            if name not in theta:
                theta[name] = np.ones(k)

        ss = steady_state(np.full(k, u["throttle"]),
                          np.full(k, u["altitude_ft"]),
                          np.full(k, u["ambient_temperature_c"]),
                          theta, self.params,
                          airspeed_factor=np.full(k, u["airspeed_factor"]),
                          load_factor=np.full(k, u["load_factor"]))
        return np.stack([ss[MEAS_KEY[nm]] for nm in self.meas_names], axis=1)

    # ------------------------------------------------------------------ #
    def predict(self):
        """Random-walk propagation: the mean is unchanged, covariance grows."""
        self.P = self.P + self.Q

    def update(self, y: np.ndarray, u: Dict[str, float]) -> Dict[str, float]:
        """One measurement update. Returns diagnostics for this step."""
        pts = self._sigma_points()
        Y = self.measurement_model(pts, u)

        y_hat = np.einsum("k,km->m", self.wm, Y)
        dY = Y - y_hat
        Pyy = np.einsum("k,km,kn->mn", self.wc, dY, dY) + self.R
        dX = pts - np.einsum("k,kn->n", self.wm, pts)
        Pxy = np.einsum("k,kn,km->nm", self.wc, dX, dY)

        innov = y - y_hat
        try:
            Pyy_inv = np.linalg.inv(Pyy)
        except np.linalg.LinAlgError:
            Pyy_inv = np.linalg.pinv(Pyy)

        # Innovation gating. The dataset deliberately contains outlier spikes;
        # without a gate a single 10-sigma spike drags the health estimate and,
        # because health is meant to be slow, that error persists for hours.
        d2 = float(innov @ Pyy_inv @ innov)
        gated = d2 > self.gate_chi2 * self.m
        if gated:
            self.n_gated += 1
            # Still allow a heavily damped correction rather than ignoring the
            # sample entirely, so a genuine step change is eventually tracked.
            scale = np.sqrt(self.gate_chi2 * self.m / max(d2, 1e-12))
            innov = innov * scale

        K = Pxy @ Pyy_inv
        self.x = self.x + K @ innov
        self.P = self.P - K @ Pyy @ K.T
        self.P = 0.5 * (self.P + self.P.T)

        # Keep the estimate physical.
        self.x = self._clip(self.x.reshape(1, -1)).ravel()
        self.n_updates += 1
        self._apply_monotone()
        # Guard against a numerically collapsed covariance.
        d = np.diag(self.P).copy()
        floor = np.array([(UKF_Q_DIAG[k] * 0.5) ** 2 for k in self.state_names])
        if np.any(d < floor):
            np.fill_diagonal(self.P, np.maximum(d, floor))

        return {"mahalanobis2": d2, "gated": bool(gated),
                "innovation_norm": float(np.linalg.norm(innov))}

    # ------------------------------------------------------------------ #
    def step(self, y: np.ndarray, u: Dict[str, float]) -> Dict[str, float]:
        self.predict()
        return self.update(y, u)

    @property
    def state(self) -> Dict[str, float]:
        return {k: float(v) for k, v in zip(self.state_names, self.x)}

    @property
    def std(self) -> Dict[str, float]:
        s = np.sqrt(np.maximum(np.diag(self.P), 0.0))
        return {k: float(v) for k, v in zip(self.state_names, s)}


def run_ukf_on_frame(df, sigma_r: Dict[str, float],
                     state_names: Sequence[str] = tuple(UKF_STATE),
                     meas_names: Sequence[str] = tuple(UKF_MEAS),
                     params: Dict[str, float] | None = None,
                     q_scale: float = 1.0,
                     r_scale: float = 1.0) -> Dict[str, np.ndarray]:
    """Run the filter over one complete engine trajectory.

    `df` must be a single run, ordered by snapshot_index. Returns per-snapshot
    estimates and standard deviations for every state.
    """
    ukf = HealthUKF(sigma_r, params=params, state_names=state_names,
                    meas_names=meas_names, q_scale=q_scale, r_scale=r_scale)

    n = len(df)
    est = {k: np.empty(n) for k in ukf.state_names}
    std = {f"{k}_std": np.empty(n) for k in ukf.state_names}
    diag = {"mahalanobis2": np.empty(n), "gated": np.zeros(n, dtype=bool)}

    Y = df[list(meas_names)].to_numpy()
    U = df[["throttle", "altitude_ft", "ambient_temperature_c",
            "airspeed_factor", "load_factor"]].to_numpy()

    for i in range(n):
        u = {"throttle": U[i, 0], "altitude_ft": U[i, 1],
             "ambient_temperature_c": U[i, 2], "airspeed_factor": U[i, 3],
             "load_factor": U[i, 4]}
        d = ukf.step(Y[i], u)
        s = ukf.state
        sd = ukf.std
        for k in ukf.state_names:
            est[k][i] = s[k]
            std[f"{k}_std"][i] = sd[k]
        diag["mahalanobis2"][i] = d["mahalanobis2"]
        diag["gated"][i] = d["gated"]

    out = {f"est_{k}": v for k, v in est.items()}
    out.update({f"est_{k}": v for k, v in std.items()})
    out["ukf_mahalanobis2"] = diag["mahalanobis2"]
    out["ukf_gated"] = diag["gated"]
    return out
