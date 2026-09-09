"""
AeroTwin-4 Maintenance Advisory.

Converts a diagnosis into the maintenance actions a line engineer can actually
carry out. A dashboard that reports "BEARING fault, 0.94 confidence" and stops
has handed the problem back to the reader; the useful output names the
inspection, the priority, and why that inspection follows from the evidence.

Actions are drawn from routine piston-engine maintenance practice for the four
modelled subsystems. They are advisory only: this is a reduced-order model, not
an approved maintenance manual, and nothing here replaces the operator's AMM.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class Priority(str, Enum):
    ROUTINE = "ROUTINE"
    SCHEDULED = "SCHEDULED"
    PRIORITY = "PRIORITY"
    IMMEDIATE = "IMMEDIATE"


PRIORITY_ORDER = {
    Priority.IMMEDIATE: 0,
    Priority.PRIORITY: 1,
    Priority.SCHEDULED: 2,
    Priority.ROUTINE: 3,
}


@dataclass
class AdvisoryItem:
    """One recommended maintenance action."""

    action: str
    subsystem: str
    priority: Priority
    rationale: str
    reference: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "subsystem": self.subsystem,
            "priority": self.priority.value,
            "rationale": self.rationale,
            "reference": self.reference,
        }


# Ordered most-diagnostic-first: the first entry is the check that best confirms
# or rules out the diagnosed family, so a time-limited crew starts there.
ACTIONS_BY_FAMILY: Dict[str, List[Dict[str, str]]] = {
    "CYLINDER": [
        {
            "action": "Differential compression test on the affected cylinder",
            "subsystem": "Cylinder assembly",
            "rationale": "Combustion torque contribution has fallen against the healthy twin, which points to a compression loss.",
            "reference": "Compression check",
        },
        {
            "action": "Borescope inspection of piston crown, bore and valve faces",
            "subsystem": "Cylinder assembly",
            "rationale": "Confirms whether the compression loss is valve seating, ring wear or a scored bore.",
            "reference": "Internal inspection",
        },
        {
            "action": "Inspect and gap or replace spark plugs on the affected cylinder",
            "subsystem": "Ignition",
            "rationale": "Fouled or worn plugs reproduce the same torque deficit without any mechanical wear.",
            "reference": "Ignition service",
        },
        {
            "action": "Check valve clearance and rocker gear",
            "subsystem": "Valvetrain",
            "rationale": "Clearance drift changes effective valve timing and shows up as a per-cylinder torque imbalance.",
            "reference": "Valve clearance",
        },
    ],
    "BEARING": [
        {
            "action": "Spectrographic oil analysis for bearing metals",
            "subsystem": "Lubrication",
            "rationale": "Rising friction torque with bearing metals in the oil confirms material loss rather than a viscosity effect.",
            "reference": "Oil analysis",
        },
        {
            "action": "Vibration spectrum survey at cruise and takeoff power",
            "subsystem": "Rotating assembly",
            "rationale": "Bearing defects present as discrete frequency content that the broadband RMS level alone does not separate.",
            "reference": "Vibration survey",
        },
        {
            "action": "Inspect oil filter and magnetic chip detector",
            "subsystem": "Lubrication",
            "rationale": "Captured debris gives direct physical evidence and grades the severity.",
            "reference": "Filter inspection",
        },
        {
            "action": "Measure main and rod bearing clearances at next access",
            "subsystem": "Rotating assembly",
            "rationale": "Quantifies remaining margin once the indirect indicators agree.",
            "reference": "Clearance check",
        },
    ],
    "COOLING": [
        {
            "action": "Inspect cooling baffles, seals and inter-cylinder ducting",
            "subsystem": "Cooling",
            "rationale": "Cylinder head temperature is running above the healthy twin at matched ambient, which is a heat rejection loss rather than a hot day.",
            "reference": "Baffle inspection",
        },
        {
            "action": "Clean cooling fins and check for obstruction or debris",
            "subsystem": "Cooling",
            "rationale": "Fouled fin surfaces reduce the effective heat transfer area directly.",
            "reference": "Fin cleaning",
        },
        {
            "action": "Verify cylinder head temperature probe calibration",
            "subsystem": "Instrumentation",
            "rationale": "Rules out an instrumentation bias before any cooling hardware is disturbed.",
            "reference": "Sensor calibration",
        },
        {
            "action": "Check cowl flap travel and actuation",
            "subsystem": "Cooling",
            "rationale": "Restricted cooling airflow reproduces the same temperature rise with serviceable cylinders.",
            "reference": "Cowl flap check",
        },
    ],
    "LUBRICATION": [
        {
            "action": "Verify oil pressure against the indicated reading with a calibrated gauge",
            "subsystem": "Lubrication",
            "rationale": "Separates a genuine pressure loss from a failing transducer before further work.",
            "reference": "Pressure verification",
        },
        {
            "action": "Inspect oil pump, pressure relief valve and pickup screen",
            "subsystem": "Lubrication",
            "rationale": "Pressure below the healthy twin at matched oil temperature points to pump delivery or relief valve leakage.",
            "reference": "Pump inspection",
        },
        {
            "action": "Replace oil and filter, retain sample for analysis",
            "subsystem": "Lubrication",
            "rationale": "Degraded oil loses viscosity and both lowers pressure and raises friction torque.",
            "reference": "Oil change",
        },
        {
            "action": "Check for external leaks and oil quantity",
            "subsystem": "Lubrication",
            "rationale": "The simplest cause of a pressure loss, and the fastest to eliminate.",
            "reference": "Leak check",
        },
    ],
}

# Actions offered when the engine is serviceable, so the panel is never empty.
HEALTHY_ACTIONS = [
    {
        "action": "No maintenance action required",
        "subsystem": "Engine",
        "rationale": "All monitored channels are tracking the healthy twin within calibrated limits.",
        "reference": "Continue scheduled maintenance",
    }
]


class MaintenanceAdvisor:
    """Builds a prioritised maintenance advisory from the live assessment."""

    def __init__(
        self,
        health_priority: float = 0.55,
        health_immediate: float = 0.35,
        min_confidence: float = 0.35,
    ):
        # Health below which work is raised to PRIORITY, and below which it is
        # IMMEDIATE. These mirror the dispatch bands in risk.py so the advisory
        # cannot contradict the GO/NO-GO shown beside it.
        self.health_priority = health_priority
        self.health_immediate = health_immediate
        # Below this confidence the diagnosis is too weak to justify pulling an
        # engine apart, so the advisory recommends confirming before acting.
        self.min_confidence = min_confidence

    def _priority_for(
        self, health_index: float, confidence: float, rul_seconds: Optional[float],
        mission_required_s: float,
    ) -> Priority:
        if health_index < self.health_immediate:
            return Priority.IMMEDIATE
        # Remaining life that does not cover the next sortie is a before-next-flight
        # item regardless of how good the health index looks in isolation.
        if rul_seconds is not None and rul_seconds < mission_required_s:
            return Priority.IMMEDIATE
        if health_index < self.health_priority:
            return Priority.PRIORITY
        if confidence >= 0.75:
            return Priority.SCHEDULED
        return Priority.ROUTINE

    def advise(
        self,
        predicted_fault: str = "HEALTHY",
        fault_confidence: float = 0.0,
        health_index: float = 1.0,
        rul_seconds: Optional[float] = None,
        mission_required_s: float = 600.0,
        anomaly_flagged: bool = False,
        max_items: int = 4,
    ) -> List[AdvisoryItem]:
        fault = (predicted_fault or "HEALTHY").upper()
        items: List[AdvisoryItem] = []

        if fault == "HEALTHY" or fault not in ACTIONS_BY_FAMILY:
            if anomaly_flagged:
                items.append(
                    AdvisoryItem(
                        action="Review recorded telemetry for the flagged window",
                        subsystem="Engine",
                        priority=Priority.ROUTINE,
                        rationale=(
                            "The detector flagged a deviation the classifier could not "
                            "attribute to a modelled subsystem. Confirm before dispatch."
                        ),
                        reference="Telemetry review",
                    )
                )
            for a in HEALTHY_ACTIONS:
                items.append(
                    AdvisoryItem(
                        action=a["action"],
                        subsystem=a["subsystem"],
                        priority=Priority.ROUTINE,
                        rationale=a["rationale"],
                        reference=a["reference"],
                    )
                )
            return items[:max_items]

        priority = self._priority_for(
            health_index, fault_confidence, rul_seconds, mission_required_s
        )

        if fault_confidence < self.min_confidence:
            items.append(
                AdvisoryItem(
                    action=f"Confirm the {fault.title()} indication before committing to component work",
                    subsystem="Engine",
                    priority=Priority.PRIORITY if priority is Priority.IMMEDIATE else Priority.SCHEDULED,
                    rationale=(
                        f"Classifier confidence is {fault_confidence:.0%}, below the "
                        f"{self.min_confidence:.0%} threshold for acting on the attribution alone."
                    ),
                    reference="Diagnosis confirmation",
                )
            )

        for idx, a in enumerate(ACTIONS_BY_FAMILY[fault]):
            # Only the leading, most diagnostic action inherits the full priority;
            # the follow-ups are one step lower so the list reads as a sequence.
            if idx == 0:
                p = priority
            elif priority is Priority.IMMEDIATE:
                p = Priority.PRIORITY
            elif priority is Priority.PRIORITY:
                p = Priority.SCHEDULED
            else:
                p = Priority.ROUTINE

            items.append(
                AdvisoryItem(
                    action=a["action"],
                    subsystem=a["subsystem"],
                    priority=p,
                    rationale=a["rationale"],
                    reference=a["reference"],
                )
            )

        items.sort(key=lambda i: PRIORITY_ORDER[i.priority])
        return items[:max_items]
