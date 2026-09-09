"""
AeroTwin-4 Remaining Useful Life Projector.

Stage 2 of the RUL pipeline: given the recent history of estimated health, fit the
decay trend and extrapolate to the failure threshold.

UNITS AND SCOPE - read before quoting any number from this module:
  RUL is returned in the SIMULATION's own time base (seconds of simulated
  operation). It is NOT engine flight hours. Converting one to the other requires
  a validated wear-rate mapping against real engine test-cell data, which this
  project does not have. Treat RUL here as a relative trend indicator.

The estimate is deliberately reported as an INTERVAL. A single-number RUL implies
a precision this model cannot support, and AGENTS.md rule 'avoid unsupported
precision' is a hard requirement, not a style preference.

INTERVAL SEMANTICS - do not call this a 95% confidence interval.
  The band is constructed from three real error sources (slope standard error,
  linear-vs-quadratic model disagreement, and an extrapolation-horizon penalty),
  but it is NOT calibrated to a nominal coverage level. Measured coverage on the
  held-out engine unit is ~71% of mid-flight predictions, not 95%. It is reported
  as a TREND UNCERTAINTY BAND. The parameters were not tuned to hit a coverage
  target, because tuning them against the held-out set would make the resulting
  coverage figure meaningless.
"""

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence

import numpy as np

# Health at which the engine is declared unfit to continue the mission.
DEFAULT_FAILURE_THRESHOLD = 0.30

# A trend flatter than this is treated as "no measurable decay": extrapolating a
# slope buried in estimator noise manufactures confident nonsense.
MIN_DECAY_RATE_PER_S = 1e-5

# A trailing fit this poor is noise, not a trend. Without this floor a healthy
# engine's estimator jitter fits a faint downward slope and the projector happily
# reports a finite RUL for an engine that is not degrading at all.
MIN_TREND_R2 = 0.40


@dataclass
class RULEstimate:
    """A remaining-useful-life estimate with its uncertainty and provenance."""

    rul_seconds: Optional[float]           # None => no measurable decay trend
    rul_lower_seconds: Optional[float]
    rul_upper_seconds: Optional[float]
    current_health: float
    decay_rate_per_s: float                # >0 means health is falling
    failure_threshold: float
    confidence: float                      # 0..1, how well the trend fits
    is_decaying: bool
    n_points: int
    notes: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rul_seconds": None if self.rul_seconds is None else round(self.rul_seconds, 2),
            "rul_lower_seconds": None if self.rul_lower_seconds is None else round(self.rul_lower_seconds, 2),
            "rul_upper_seconds": None if self.rul_upper_seconds is None else round(self.rul_upper_seconds, 2),
            "current_health": round(self.current_health, 4),
            "decay_rate_per_s": round(self.decay_rate_per_s, 8),
            "failure_threshold": self.failure_threshold,
            "confidence": round(self.confidence, 4),
            "is_decaying": self.is_decaying,
            "n_points": self.n_points,
            "notes": self.notes,
        }


class RULProjector:
    """
    Fits a linear decay trend to recent health history and projects to failure.

    A linear fit over a trailing window is used rather than a global curve fit
    because the trailing slope is what a crew actually needs: how fast is this
    engine losing health RIGHT NOW.
    """

    def __init__(
        self,
        failure_threshold: float = DEFAULT_FAILURE_THRESHOLD,
        window_points: int = 30,
        health_noise_std: float = 0.02,
        # Extrapolating a decay rate from a handful of noisy health estimates is not
        # evidence. Six points were enough to hand a perfectly healthy engine a 28 s
        # remaining life and an ABORT recommendation, so the bar is set higher.
        min_points: int = 15,
        min_trend_r2: float = MIN_TREND_R2,
        horizon_penalty: float = 1.0,
        significant_drop_sigma: float = 3.0,
    ):
        self.failure_threshold = failure_threshold
        self.window_points = window_points
        # Estimator noise floor, normally taken from HealthEstimator.residual_std.
        self.health_noise_std = max(1e-6, health_noise_std)
        self.min_points = min_points
        self.min_trend_r2 = min_trend_r2
        # Interval widening per unit of (projected horizon / observed history).
        self.horizon_penalty = horizon_penalty
        # Multiples of the health-estimator noise the observed drop must exceed
        # before it counts as a real trend rather than jitter.
        self.significant_drop_sigma = significant_drop_sigma

    def estimate(
        self, times: Sequence[float], healths: Sequence[float]
    ) -> RULEstimate:
        t = np.asarray(times, dtype=float)
        h = np.asarray(healths, dtype=float)

        if len(t) != len(h):
            raise ValueError("times and healths must be the same length")

        if len(t) < self.min_points:
            return RULEstimate(
                rul_seconds=None, rul_lower_seconds=None, rul_upper_seconds=None,
                current_health=float(h[-1]) if len(h) else 1.0,
                decay_rate_per_s=0.0, failure_threshold=self.failure_threshold,
                confidence=0.0, is_decaying=False, n_points=len(t),
                notes=f"Need at least {self.min_points} health points to fit a trend.",
            )

        # Trailing window only.
        t_w, h_w = t[-self.window_points:], h[-self.window_points:]
        current_health = float(h_w[-1])

        slope, intercept = np.polyfit(t_w, h_w, 1)
        decay_rate = float(-slope)  # positive == losing health

        fitted = slope * t_w + intercept
        ss_res = float(np.sum((h_w - fitted) ** 2))
        ss_tot = float(np.sum((h_w - np.mean(h_w)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
        confidence = float(max(0.0, min(1.0, r2)))

        if current_health <= self.failure_threshold:
            return RULEstimate(
                rul_seconds=0.0, rul_lower_seconds=0.0, rul_upper_seconds=0.0,
                current_health=current_health, decay_rate_per_s=decay_rate,
                failure_threshold=self.failure_threshold, confidence=confidence,
                is_decaying=decay_rate > MIN_DECAY_RATE_PER_S, n_points=len(t),
                notes="Health is already at or below the failure threshold.",
            )

        # The health drop actually observed across the window. If it does not clear
        # the health estimator's own noise band, the "trend" is estimator jitter -
        # a slope with a good R2 can still be a fit to pure noise of tiny amplitude.
        observed_drop = float(h_w[0] - h_w[-1])
        drop_is_significant = observed_drop > self.significant_drop_sigma * self.health_noise_std

        if (
            decay_rate <= MIN_DECAY_RATE_PER_S
            or confidence < self.min_trend_r2
            or not drop_is_significant
        ):
            if decay_rate <= MIN_DECAY_RATE_PER_S:
                reason = "No measurable health decay"
            elif not drop_is_significant:
                reason = (
                    "Observed health change (%.4f) is within estimator noise (%.1f sigma = %.4f)"
                    % (observed_drop, self.significant_drop_sigma,
                       self.significant_drop_sigma * self.health_noise_std)
                )
            else:
                reason = "Health trend too weak to extrapolate (R2=%.2f < %.2f)" % (
                    confidence, self.min_trend_r2
                )
            return RULEstimate(
                rul_seconds=None, rul_lower_seconds=None, rul_upper_seconds=None,
                current_health=current_health, decay_rate_per_s=decay_rate,
                failure_threshold=self.failure_threshold, confidence=confidence,
                is_decaying=False, n_points=len(t),
                notes=reason + "; RUL is unbounded on current evidence.",
            )

        margin = current_health - self.failure_threshold
        rul = margin / decay_rate
        rul_linear = rul

        # MODEL-FORM UNCERTAINTY.
        # A straight line through a decelerating/accelerating trend is the dominant
        # error source, not fit scatter: real degradation is frequently exponential,
        # and a linear extrapolation of accelerating decay over-predicts life badly.
        # Fit a quadratic as a second opinion and let the disagreement between the
        # two widen the interval. Reporting a 95% interval built only from slope
        # scatter yields intervals that look tight and are wrong far more than 5%
        # of the time.
        rul_quadratic = None
        r2_quadratic = 0.0
        if len(t_w) >= 4:
            try:
                qa, qb, qc = np.polyfit(t_w, h_w, 2)
                q_fit = qa * t_w * t_w + qb * t_w + qc
                q_res = float(np.sum((h_w - q_fit) ** 2))
                r2_quadratic = 1.0 - q_res / ss_tot if ss_tot > 1e-12 else 0.0

                t_last = float(t_w[-1])
                # Solve qa*t^2 + qb*t + qc = threshold for the next crossing.
                disc = qb * qb - 4.0 * qa * (qc - self.failure_threshold)
                if disc >= 0.0 and abs(qa) > 1e-15:
                    root = math.sqrt(disc)
                    crossings = [(-qb + root) / (2.0 * qa), (-qb - root) / (2.0 * qa)]
                    future = [c - t_last for c in crossings if c > t_last]
                    if future:
                        rul_quadratic = float(min(future))
            except (np.linalg.LinAlgError, ValueError):
                rul_quadratic = None

        # MODEL SELECTION.
        # Degradation is often accelerating. When the quadratic fit explains the
        # trailing health trend materially better than the straight line, the line
        # is the wrong model and extrapolating it over-predicts remaining life -
        # the dangerous direction of error for a flight-critical call.
        # Nested model selection by ADJUSTED R^2. The quadratic always fits the data
        # at least as well as the line simply because it has one more parameter, so
        # comparing raw R^2 would always pick the curve. Adjusted R^2 charges for
        # that parameter and is the standard answer here - no hand-tuned threshold,
        # and nothing fitted against the held-out set.
        used_model = "linear"
        if rul_quadratic is not None and rul_quadratic > 0.0:
            n_pts = len(t_w)
            if n_pts > 4:
                adj_linear = 1.0 - (1.0 - confidence) * (n_pts - 1) / (n_pts - 2)
                adj_quad = 1.0 - (1.0 - r2_quadratic) * (n_pts - 1) / (n_pts - 3)
                if adj_quad > adj_linear:
                    rul = rul_quadratic
                    used_model = "quadratic"

        # Uncertainty: propagate the standard error of the fitted slope, widened by
        # the health estimator's own noise floor. Both are real error sources.
        n = len(t_w)
        t_var = float(np.sum((t_w - np.mean(t_w)) ** 2))
        if n > 2 and t_var > 1e-12:
            resid_var = ss_res / (n - 2)
            slope_se = math.sqrt(max(resid_var, self.health_noise_std ** 2) / t_var)
        else:
            slope_se = decay_rate * 0.5

        # 95%-ish interval on the decay rate -> interval on RUL (inverted, so the
        # fast-decay bound gives the SHORT life).
        rate_hi = decay_rate + 1.96 * slope_se
        rate_lo = max(MIN_DECAY_RATE_PER_S, decay_rate - 1.96 * slope_se)

        rul_lower = margin / rate_hi
        rul_upper = margin / rate_lo if rate_lo > MIN_DECAY_RATE_PER_S else None

        # Absorb the linear-vs-quadratic disagreement into the interval.
        if rul_quadratic is not None and rul_quadratic > 0:
            rul_lower = min(rul_lower, rul_quadratic, rul_linear)
            if rul_upper is not None:
                rul_upper = max(rul_upper, rul_quadratic, rul_linear)

        # EXTRAPOLATION HORIZON PENALTY.
        # Predicting 150 s ahead from 30 s of history is a far weaker claim than
        # predicting 20 s ahead from the same history, yet slope scatter alone gives
        # both a similar band. Widen the interval in proportion to how far past the
        # observed history the projection reaches, so the reported uncertainty grows
        # with the reach of the extrapolation as it physically must.
        history_span = float(t_w[-1] - t_w[0]) if len(t_w) > 1 else 0.0
        if history_span > 1e-6:
            horizon_ratio = rul / history_span
            widen = 1.0 + self.horizon_penalty * max(0.0, horizon_ratio - 1.0)
            rul_lower = rul - (rul - rul_lower) * widen
            if rul_upper is not None:
                rul_upper = rul + (rul_upper - rul) * widen

        notes = "RUL is in SIMULATION seconds, not engine flight hours."
        if rul_quadratic is not None:
            spread = abs(rul_quadratic - rul_linear) / max(rul_linear, 1e-9)
            if spread > 0.25:
                notes += (
                    " Linear and quadratic trend fits disagree by %.0f%%; decay looks"
                    " non-linear, so the point estimate is weakly supported." % (100.0 * spread)
                )

        return RULEstimate(
            rul_seconds=float(rul),
            rul_lower_seconds=float(max(0.0, rul_lower)),
            rul_upper_seconds=None if rul_upper is None else float(rul_upper),
            current_health=current_health,
            decay_rate_per_s=decay_rate,
            failure_threshold=self.failure_threshold,
            confidence=confidence,
            is_decaying=True,
            n_points=len(t),
            notes=notes,
            extra={
                "rul_quadratic_seconds": rul_quadratic,
                "r2_quadratic": r2_quadratic,
                "trend_model": used_model,
            },
        )
