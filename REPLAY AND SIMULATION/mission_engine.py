"""
============================================================================
mission_engine.py — Mission Clearance & Flight Reliability Engine
============================================================================
Evaluates dynamic flight clearance (GO / CAUTION_GO / NO-GO) based on live
engine prognostics, health indices, and remaining useful life (RUL).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional


class ClearanceStatus(str, Enum):
    GO = "GO"
    CAUTION_GO = "CAUTION_GO"
    NO_GO = "NO_GO"


@dataclass
class MissionProfile:
    name: str
    required_duration_s: float
    target_altitude_m: float
    safety_reserve_s: float = 120.0  # 2-minute mandatory loiter reserve

    def total_required_hours(self) -> float:
        return (self.required_duration_s + self.safety_reserve_s) / 3600.0


@dataclass
class MissionAssessment:
    profile_name: str
    status: ClearanceStatus
    risk_score: float              # [0.0, 1.0]
    predicted_rul_minutes: float
    required_mission_minutes: float
    health_index: float
    active_fault: str
    reasons: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "profile_name": self.profile_name,
            "status": self.status.value,
            "risk_score": round(self.risk_score, 3),
            "predicted_rul_minutes": round(self.predicted_rul_minutes, 1),
            "required_mission_minutes": round(self.required_mission_minutes, 1),
            "health_index": round(self.health_index, 3),
            "active_fault": self.active_fault,
            "reasons": self.reasons,
        }


class MissionSimulator:
    """
    Evaluates dynamic flight clearance (GO/NO-GO) based on live engine prognostics.
    Enforces defense-grade reliability standards for MALE UAV sorties.
    """

    STANDARD_PROFILES: Dict[str, MissionProfile] = {
        "ISR_SURVEILLANCE": MissionProfile(
            name="ISR_SURVEILLANCE",
            required_duration_s=360.0,   # 6.0 minutes
            target_altitude_m=3500.0,
            safety_reserve_s=90.0,      # 1.5 minutes
        ),
        "BORDER_PATROL": MissionProfile(
            name="BORDER_PATROL",
            required_duration_s=240.0,   # 4.0 minutes
            target_altitude_m=1500.0,
            safety_reserve_s=60.0,
        ),
        "HIGH_ALT_RECON": MissionProfile(
            name="HIGH_ALT_RECON",
            required_duration_s=480.0,   # 8.0 minutes
            target_altitude_m=6000.0,
            safety_reserve_s=120.0,
        ),
    }

    def evaluate_clearance(
        self,
        profile_name: str,
        predicted_rul_hours: float,
        composite_health_index: float,
        active_fault_name: str = "Normal",
        fault_confidence_pct: float = 90.0,
    ) -> MissionAssessment:
        profile = self.STANDARD_PROFILES.get(
            profile_name,
            MissionProfile(name=profile_name, required_duration_s=300.0, target_altitude_m=2000.0),
        )

        rul_min = predicted_rul_hours * 60.0
        req_min = (profile.required_duration_s + profile.safety_reserve_s) / 60.0
        reasons: List[str] = []
        risk_score = 0.0

        # 1. Critical Fault Mode Penalty
        critical_faults = ["Overheating", "Lubrication", "Misfire", "Combustion", "Bearing"]
        if active_fault_name in critical_faults and fault_confidence_pct > 60.0:
            reasons.append(f"ACTIVE CRITICAL FAULT: {active_fault_name} detected with {fault_confidence_pct:.1f}% confidence.")
            risk_score += 0.70

        # 2. RUL Safety Margin Check
        if rul_min < req_min:
            deficit = req_min - rul_min
            reasons.append(f"INSUFFICIENT RUL: Predicted {rul_min:.1f} min vs required {req_min:.1f} min (Deficit: {deficit:.1f} min).")
            risk_score += 0.50
        elif rul_min < req_min * 1.25:
            reasons.append("MARGINAL RUL: Buffer is less than 25% above required mission profile.")
            risk_score += 0.25

        # 3. Health Index Degradation
        if composite_health_index < 0.40:
            reasons.append(f"SEVERE ENGINE WEAR: Health Index {composite_health_index:.2f} is below 0.40 critical threshold.")
            risk_score += 0.40
        elif composite_health_index < 0.70:
            reasons.append(f"MODERATE WEAR: Subsystem Health Index at {composite_health_index:.2f}.")
            risk_score += 0.20

        # 4. Final Status Determination
        risk_score = min(1.0, risk_score)

        if risk_score >= 0.50 or composite_health_index < 0.50 or rul_min < req_min:
            status = ClearanceStatus.NO_GO
        elif risk_score >= 0.20 or composite_health_index < 0.80:
            status = ClearanceStatus.CAUTION_GO
        else:
            status = ClearanceStatus.GO
            reasons.append(f"SORTIE CLEARED: Robust RUL margin ({rul_min:.1f} min vs {req_min:.1f} min) and nominal health ({composite_health_index*100:.1f}%).")

        return MissionAssessment(
            profile_name=profile.name,
            status=status,
            risk_score=risk_score,
            predicted_rul_minutes=rul_min,
            required_mission_minutes=req_min,
            health_index=composite_health_index,
            active_fault=active_fault_name,
            reasons=reasons,
        )
