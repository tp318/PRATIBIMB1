"""
AeroTwin-4 Mission Reliability & Risk Assessment.

Turns engine health evidence into the decision a UAV operator actually has to make:
can this airframe be sent on THIS mission, and if not, what should be done.

Inputs are the outputs of the rest of the stack - anomaly score, fault diagnosis,
health index, RUL - plus the mission requirement. Output is a risk band, a
go/no-go recommendation, and the reasoning behind it, because an unexplained
"NO-GO" is not actionable and will simply be overridden.

SCOPE: this is a decision-support aid built on a reduced-order engine model. It is
not an airworthiness authority, and the thresholds below are engineering defaults
for demonstration, not certified limits.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class RiskBand(str, Enum):
    """Coarse risk bands. Coarse on purpose - finer grading is not supported."""

    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    SEVERE = "SEVERE"


class Recommendation(str, Enum):
    GO = "GO"
    GO_WITH_MONITORING = "GO_WITH_MONITORING"
    ABORT_OR_SHORTEN = "ABORT_OR_SHORTEN"
    NO_GO = "NO_GO"


# Health index below which the engine is not dispatched at all.
HEALTH_NO_GO = 0.35
# Health index below which dispatch requires active monitoring.
HEALTH_WATCH = 0.70

# RUL must exceed mission duration by this factor to count as covered.
# 1.5 is a reserve margin: finishing a sortie with exactly zero predicted life
# left is not an acceptable plan.
RUL_RESERVE_FACTOR = 1.5

# Faults whose failure mode is abrupt rather than gradual get a higher weighting:
# a seizing bearing or oil starvation ends a flight far faster than a slowly
# fouling cylinder.
FAULT_SEVERITY_WEIGHT = {
    "HEALTHY": 0.0,
    "CYLINDER": 0.55,
    "COOLING": 0.70,
    "BEARING": 0.90,
    "LUBRICATION": 0.95,
}


@dataclass
class MissionProfile:
    """The mission being assessed."""

    name: str = "SORTIE"
    required_duration_s: float = 3600.0
    # Reserve the crew wants left over on return.
    reserve_duration_s: float = 0.0

    @property
    def total_required_s(self) -> float:
        return self.required_duration_s + self.reserve_duration_s


@dataclass
class MissionRiskAssessment:
    """The assessment handed to the operator."""

    risk_score: float                  # 0.0 (nominal) .. 1.0 (severe)
    risk_band: RiskBand
    recommendation: Recommendation
    mission_coverable: Optional[bool]  # None when RUL is unbounded/unknown
    health_index: float
    predicted_fault: str
    fault_confidence: float
    rul_seconds: Optional[float]
    required_duration_s: float
    reasons: List[str] = field(default_factory=list)
    contributions: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "risk_score": round(self.risk_score, 4),
            "risk_band": self.risk_band.value,
            "recommendation": self.recommendation.value,
            "mission_coverable": self.mission_coverable,
            "health_index": round(self.health_index, 4),
            "predicted_fault": self.predicted_fault,
            "fault_confidence": round(self.fault_confidence, 4),
            "rul_seconds": None if self.rul_seconds is None else round(self.rul_seconds, 2),
            "required_duration_s": self.required_duration_s,
            "reasons": self.reasons,
            "contributions": {k: round(v, 4) for k, v in self.contributions.items()},
        }


class MissionRiskAssessor:
    """
    Combines health, RUL, diagnosis and anomaly evidence into a dispatch decision.

    The score is a transparent weighted sum rather than a learned model: an operator
    has to be able to see WHY the system said no, and a black box that outputs 0.83
    with no breakdown will not be trusted or used.
    """

    def __init__(
        self,
        w_health: float = 0.35,
        w_rul: float = 0.30,
        w_fault: float = 0.25,
        w_anomaly: float = 0.10,
        reserve_factor: float = RUL_RESERVE_FACTOR,
        min_rul_confidence: float = 0.60,
    ):
        total = w_health + w_rul + w_fault + w_anomaly
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"Risk weights must sum to 1.0, got {total}")
        self.w_health = w_health
        self.w_rul = w_rul
        self.w_fault = w_fault
        self.w_anomaly = w_anomaly
        self.reserve_factor = reserve_factor
        # Trend fit quality (R^2) below which an RUL is ignored for dispatch.
        self.min_rul_confidence = min_rul_confidence

    @staticmethod
    def _band(score: float) -> RiskBand:
        if score < 0.25:
            return RiskBand.LOW
        if score < 0.50:
            return RiskBand.MODERATE
        if score < 0.75:
            return RiskBand.HIGH
        return RiskBand.SEVERE

    def assess(
        self,
        mission: MissionProfile,
        health_index: float,
        rul_seconds: Optional[float] = None,
        rul_lower_seconds: Optional[float] = None,
        predicted_fault: str = "HEALTHY",
        fault_confidence: float = 0.0,
        anomaly_flagged: bool = False,
        anomaly_score_ratio: float = 0.0,
        rul_confidence: float = 1.0,
    ) -> MissionRiskAssessment:
        """
        Parameters
        ----------
        health_index : current health H in [0, 1]
        rul_seconds : point RUL estimate, or None if no decay trend was measurable
        rul_lower_seconds : pessimistic bound; the decision uses this, not the point
        anomaly_score_ratio : anomaly score divided by its threshold (1.0 == at limit)
        """
        health_index = float(max(0.0, min(1.0, health_index)))
        reasons: List[str] = []

        # --- Health contribution -------------------------------------------------
        health_risk = 1.0 - health_index
        if health_index < HEALTH_NO_GO:
            reasons.append(
                f"Health index {health_index:.2f} is below the dispatch floor {HEALTH_NO_GO:.2f}."
            )
        elif health_index < HEALTH_WATCH:
            reasons.append(
                f"Health index {health_index:.2f} is degraded but above the dispatch floor."
            )

        # --- RUL contribution ----------------------------------------------------
        required = mission.total_required_s
        # Decide on the PESSIMISTIC bound when we have one. Planning a sortie on the
        # optimistic edge of an uncertainty band is how uncertainty gets ignored.
        rul_for_decision = rul_lower_seconds if rul_lower_seconds is not None else rul_seconds

        # A weakly-supported RUL must not be able to ground a serviceable aircraft.
        # A poorly-fitted trend is an absence of evidence, not evidence of decay.
        if rul_for_decision is not None and rul_confidence < self.min_rul_confidence:
            reasons.append(
                f"Remaining-life trend is weakly supported (fit quality "
                f"{rul_confidence:.2f} < {self.min_rul_confidence:.2f}); "
                "not used for the dispatch decision."
            )
            rul_for_decision = None

        if rul_for_decision is None:
            mission_coverable = None
            rul_risk = 0.0
            reasons.append("No measurable degradation trend; remaining life is unbounded on current evidence.")
        else:
            needed = required * self.reserve_factor
            mission_coverable = rul_for_decision >= needed
            if mission_coverable:
                rul_risk = 0.0
                reasons.append(
                    f"Worst-case remaining life {rul_for_decision:.0f}s covers the "
                    f"{required:.0f}s mission with the {self.reserve_factor:.1f}x reserve."
                )
            else:
                # Scale risk by how far short we fall.
                shortfall = 1.0 - (rul_for_decision / needed if needed > 0 else 0.0)
                rul_risk = float(max(0.0, min(1.0, shortfall)))
                reasons.append(
                    f"Worst-case remaining life {rul_for_decision:.0f}s does NOT cover the "
                    f"{required:.0f}s mission plus {self.reserve_factor:.1f}x reserve "
                    f"({needed:.0f}s required)."
                )

        # --- Fault contribution --------------------------------------------------
        fault = (predicted_fault or "HEALTHY").upper()
        weight = FAULT_SEVERITY_WEIGHT.get(fault, 0.60)
        fault_risk = weight * float(max(0.0, min(1.0, fault_confidence)))
        if fault != "HEALTHY" and fault_confidence > 0.0:
            reasons.append(
                f"Diagnosed {fault} fault at {fault_confidence:.0%} confidence "
                f"(failure-mode weight {weight:.2f})."
            )

        # --- Anomaly contribution ------------------------------------------------
        anomaly_risk = float(max(0.0, min(1.0, anomaly_score_ratio / 2.0)))
        if anomaly_flagged:
            anomaly_risk = max(anomaly_risk, 0.5)
            reasons.append(
                f"Anomaly detector is flagging (score is {anomaly_score_ratio:.2f}x its threshold)."
            )

        contributions = {
            "health": self.w_health * health_risk,
            "rul": self.w_rul * rul_risk,
            "fault": self.w_fault * fault_risk,
            "anomaly": self.w_anomaly * anomaly_risk,
        }
        score = float(max(0.0, min(1.0, sum(contributions.values()))))
        band = self._band(score)

        # --- Recommendation ------------------------------------------------------
        # Hard gates override the score: a weighted average must never be able to
        # average away a health index that is below the dispatch floor.
        if health_index < HEALTH_NO_GO:
            rec = Recommendation.NO_GO
        elif mission_coverable is False:
            rec = Recommendation.ABORT_OR_SHORTEN
        elif band is RiskBand.SEVERE:
            rec = Recommendation.NO_GO
        elif band is RiskBand.HIGH:
            rec = Recommendation.ABORT_OR_SHORTEN
        elif band is RiskBand.MODERATE or anomaly_flagged or health_index < HEALTH_WATCH:
            rec = Recommendation.GO_WITH_MONITORING
        else:
            rec = Recommendation.GO
            reasons.append("All indicators nominal for the requested mission.")

        return MissionRiskAssessment(
            risk_score=score,
            risk_band=band,
            recommendation=rec,
            mission_coverable=mission_coverable,
            health_index=health_index,
            predicted_fault=fault,
            fault_confidence=float(fault_confidence),
            rul_seconds=rul_seconds,
            required_duration_s=required,
            reasons=reasons,
            contributions=contributions,
        )

    def max_safe_duration_s(
        self, rul_lower_seconds: Optional[float]
    ) -> Optional[float]:
        """
        Longest mission the current worst-case remaining life supports, including
        the reserve factor. None when no decay trend is measurable.
        """
        if rul_lower_seconds is None:
            return None
        return max(0.0, rul_lower_seconds / self.reserve_factor)
