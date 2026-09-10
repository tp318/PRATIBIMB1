"""
AeroTwin-4 Efficiency Metrics, Alert Log and Mission Reporting.

Three operator-facing concerns that all read from the same live assessment stream:

  EfficiencyTracker  - how much power the engine is making for the fuel it burns,
                       measured against the healthy twin rather than against a
                       fixed book figure, so ambient and build variation do not
                       masquerade as efficiency loss.
  AlertLog           - discrete, timestamped events. A dashboard that only shows
                       instantaneous state loses the transition that mattered;
                       the log keeps what changed and when.
  MissionReport      - per-sortie summary for the debrief: how long was spent in
                       each risk band, what was detected, and how the engine
                       finished relative to how it started.
"""

import math
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Deque, Dict, List, Optional

# Below this shaft power the engine is effectively idling and specific fuel
# consumption is dominated by the fixed fuel flow, so the ratio is not meaningful.
MIN_POWER_KW_FOR_SFC = 0.5


def shaft_power_kw(torque_nm: Optional[float], rpm: Optional[float]) -> Optional[float]:
    """P = T * omega, with omega in rad/s. Returns kW."""
    if torque_nm is None or rpm is None:
        return None
    omega = float(rpm) * 2.0 * math.pi / 60.0
    return float(torque_nm) * omega / 1000.0


def bsfc(fuel_flow_kg_s: Optional[float], power_kw: Optional[float]) -> Optional[float]:
    """Brake specific fuel consumption in kg per kW-hour."""
    if fuel_flow_kg_s is None or power_kw is None or power_kw < MIN_POWER_KW_FOR_SFC:
        return None
    return float(fuel_flow_kg_s) * 3600.0 / power_kw


@dataclass
class EfficiencySample:
    """Efficiency of one window, observed against the healthy twin."""

    simulation_time: float
    power_kw: Optional[float]
    power_expected_kw: Optional[float]
    power_deficit_pct: Optional[float]
    bsfc: Optional[float]
    bsfc_expected: Optional[float]
    bsfc_penalty_pct: Optional[float]
    fuel_flow_lph: Optional[float]

    def to_dict(self) -> Dict[str, Any]:
        def r(v, n=4):
            return None if v is None else round(v, n)

        return {
            "simulation_time": round(self.simulation_time, 2),
            "power_kw": r(self.power_kw, 3),
            "power_expected_kw": r(self.power_expected_kw, 3),
            "power_deficit_pct": r(self.power_deficit_pct, 2),
            "bsfc": r(self.bsfc, 4),
            "bsfc_expected": r(self.bsfc_expected, 4),
            "bsfc_penalty_pct": r(self.bsfc_penalty_pct, 2),
            "fuel_flow_lph": r(self.fuel_flow_lph, 3),
        }


class EfficiencyTracker:
    """
    Tracks shaft power and specific fuel consumption against the twin.

    Both are expressed as a percentage deviation from the twin rather than as an
    absolute figure. An absolute BSFC number means nothing without knowing the
    ambient conditions and the build; the deviation from a condition-matched twin
    is directly interpretable as efficiency lost to degradation.
    """

    def __init__(self, maxlen: int = 600):
        self.samples: Deque[EfficiencySample] = deque(maxlen=maxlen)

    def update(self, observed: Dict[str, Any], expected: Dict[str, Any], sim_time: float) -> EfficiencySample:
        p_obs = shaft_power_kw(observed.get("mean_torque"), observed.get("rpm"))
        p_exp = shaft_power_kw(expected.get("mean_torque"), expected.get("rpm"))

        deficit = None
        if p_obs is not None and p_exp is not None and abs(p_exp) > 1e-6:
            deficit = (p_exp - p_obs) / p_exp * 100.0

        ff_obs = observed.get("fuel_flow")
        ff_exp = expected.get("fuel_flow")
        b_obs = bsfc(ff_obs, p_obs)
        b_exp = bsfc(ff_exp, p_exp)

        penalty = None
        if b_obs is not None and b_exp is not None and b_exp > 1e-9:
            penalty = (b_obs - b_exp) / b_exp * 100.0

        s = EfficiencySample(
            simulation_time=sim_time,
            power_kw=p_obs,
            power_expected_kw=p_exp,
            power_deficit_pct=deficit,
            bsfc=b_obs,
            bsfc_expected=b_exp,
            bsfc_penalty_pct=penalty,
            fuel_flow_lph=observed.get("fuel_flow_lph"),
        )
        self.samples.append(s)
        return s

    def series(self) -> List[Dict[str, Any]]:
        snapshot = list(self.samples)
        return [s.to_dict() for s in snapshot]

    def summary(self) -> Dict[str, Any]:
        snapshot = list(self.samples)
        def mean_of(attr):
            vals = [getattr(s, attr) for s in snapshot if getattr(s, attr) is not None]
            return sum(vals) / len(vals) if vals else None

        def peak_of(attr):
            vals = [getattr(s, attr) for s in snapshot if getattr(s, attr) is not None]
            return max(vals) if vals else None

        return {
            "n_samples": len(snapshot),
            "mean_power_kw": mean_of("power_kw"),
            "mean_power_deficit_pct": mean_of("power_deficit_pct"),
            "peak_power_deficit_pct": peak_of("power_deficit_pct"),
            "mean_bsfc": mean_of("bsfc"),
            "mean_bsfc_penalty_pct": mean_of("bsfc_penalty_pct"),
            "peak_bsfc_penalty_pct": peak_of("bsfc_penalty_pct"),
        }

    def reset(self):
        self.samples.clear()


class Severity(str, Enum):
    INFO = "INFO"
    CAUTION = "CAUTION"
    WARNING = "WARNING"


@dataclass
class Alert:
    """A discrete, timestamped operational event."""

    seq: int
    simulation_time: float
    wall_time: float
    severity: Severity
    category: str
    message: str
    detail: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "seq": self.seq,
            "simulation_time": round(self.simulation_time, 2),
            "wall_time": self.wall_time,
            "severity": self.severity.value,
            "category": self.category,
            "message": self.message,
            "detail": self.detail,
        }


class AlertLog:
    """
    Records state transitions rather than state.

    Raising an alert on every window while a condition persists buries the moment
    it started under hundreds of duplicates, so an alert fires only when the
    condition changes.
    """

    def __init__(self, maxlen: int = 300):
        self.alerts: Deque[Alert] = deque(maxlen=maxlen)
        self._seq = 0
        self._last_state: Dict[str, Any] = {}

    def _emit(self, sim_time: float, severity: Severity, category: str, message: str, detail: str = ""):
        self._seq += 1
        self.alerts.appendleft(
            Alert(
                seq=self._seq,
                simulation_time=sim_time,
                wall_time=time.time(),
                severity=severity,
                category=category,
                message=message,
                detail=detail,
            )
        )

    def evaluate(self, assessment: Dict[str, Any]) -> List[Alert]:
        """Compare this assessment against the last and log what changed."""
        sim_time = float(assessment.get("simulation_time", 0.0))
        raised: List[Alert] = []
        before = len(self.alerts)

        anomaly = assessment.get("anomaly") or {}
        diagnosis = assessment.get("diagnosis") or {}
        risk = assessment.get("mission_risk") or {}
        rul = assessment.get("rul") or {}
        health = (assessment.get("health") or {}).get("health_index")

        # --- anomaly flag transitions
        flagged = bool(anomaly.get("flagged"))
        if flagged != self._last_state.get("anomaly_flagged"):
            if flagged:
                self._emit(
                    sim_time, Severity.WARNING, "Anomaly",
                    "Anomaly detector asserted",
                    f"Score {anomaly.get('score', 0):.2f} against threshold {anomaly.get('threshold', 0):.2f}.",
                )
            elif self._last_state.get("anomaly_flagged") is not None:
                self._emit(
                    sim_time, Severity.INFO, "Anomaly",
                    "Anomaly cleared",
                    "Score returned below the calibrated threshold.",
                )
        self._last_state["anomaly_flagged"] = flagged

        # --- diagnosed fault family transitions
        fault = diagnosis.get("predicted_fault")
        if fault and fault != self._last_state.get("fault"):
            if fault == "HEALTHY":
                if self._last_state.get("fault"):
                    self._emit(
                        sim_time, Severity.INFO, "Diagnosis",
                        "Fault indication cleared",
                        "Classifier returned to a healthy attribution.",
                    )
            else:
                conf = float(diagnosis.get("confidence", 0.0))
                sev = Severity.WARNING if conf >= 0.6 else Severity.CAUTION
                self._emit(
                    sim_time, sev, "Diagnosis",
                    f"{fault.title()} fault indicated",
                    f"Confidence {conf:.0%}, runner-up {diagnosis.get('runner_up', 'n/a')}.",
                )
        if fault:
            self._last_state["fault"] = fault

        # --- dispatch recommendation transitions
        rec = risk.get("recommendation")
        if rec and rec != self._last_state.get("recommendation"):
            sev = {
                "GO": Severity.INFO,
                "GO_WITH_MONITORING": Severity.CAUTION,
                "ABORT_OR_SHORTEN": Severity.WARNING,
                "NO_GO": Severity.WARNING,
            }.get(rec, Severity.INFO)
            self._emit(
                sim_time, sev, "Dispatch",
                f"Recommendation changed to {rec.replace('_', ' ').title()}",
                f"Risk band {risk.get('risk_band', 'n/a')}, score {risk.get('risk_score', 0):.3f}.",
            )
            self._last_state["recommendation"] = rec

        # --- health band crossings, reported once per crossing
        if health is not None:
            band = "NORMAL" if health >= 0.70 else "DEGRADED" if health >= 0.35 else "CRITICAL"
            if band != self._last_state.get("health_band"):
                if self._last_state.get("health_band") is not None:
                    sev = {
                        "NORMAL": Severity.INFO,
                        "DEGRADED": Severity.CAUTION,
                        "CRITICAL": Severity.WARNING,
                    }[band]
                    self._emit(
                        sim_time, sev, "Health",
                        f"Health index entered {band.title()} band",
                        f"Health index {health:.3f}.",
                    )
                self._last_state["health_band"] = band

        # --- first time a finite remaining life is asserted
        decaying = bool(rul.get("is_decaying"))
        if decaying and not self._last_state.get("rul_decaying"):
            self._emit(
                sim_time, Severity.CAUTION, "Remaining life",
                "Degradation trend detected",
                f"Projected {rul.get('rul_seconds')} s remaining, trend fit {rul.get('confidence')}.",
            )
        self._last_state["rul_decaying"] = decaying

        for a in list(self.alerts)[: len(self.alerts) - before]:
            raised.append(a)
        return raised

    def sortie_started(self, sim_time: float = 0.0, note: str = ""):
        self._last_state.clear()
        self._emit(sim_time, Severity.INFO, "Sortie", "Sortie started", note)

    def series(self, limit: int = 60) -> List[Dict[str, Any]]:
        return [a.to_dict() for a in list(self.alerts)[:limit]]

    def counts(self) -> Dict[str, int]:
        out = {s.value: 0 for s in Severity}
        for a in self.alerts:
            out[a.severity.value] += 1
        return out

    def reset(self):
        self.alerts.clear()
        self._seq = 0
        self._last_state.clear()


class MissionReport:
    """
    Accumulates a per-sortie record for the debrief.

    Time spent in each dispatch state is tracked by integrating the interval
    between assessments rather than counting them, so a report stays correct if
    the scoring rate ever changes.
    """

    def __init__(self):
        self.reset()

    def reset(self, mission_name: str = "SORTIE", required_duration_s: float = 600.0):
        self.mission_name = mission_name
        self.required_duration_s = required_duration_s
        self.started_wall = time.time()
        self.start_health: Optional[float] = None
        self.end_health: Optional[float] = None
        self.min_health: Optional[float] = None
        self.peak_anomaly: Optional[float] = None
        self.anomaly_windows = 0
        self.total_windows = 0
        self.faults_seen: Dict[str, int] = {}
        self.time_in_recommendation: Dict[str, float] = {}
        self.worst_recommendation: Optional[str] = None
        self.last_sim_time: Optional[float] = None
        self.duration_s = 0.0

    _REC_RANK = {
        "GO": 0,
        "GO_WITH_MONITORING": 1,
        "ABORT_OR_SHORTEN": 2,
        "NO_GO": 3,
    }

    def update(self, assessment: Dict[str, Any]):
        sim_time = float(assessment.get("simulation_time", 0.0))
        anomaly = assessment.get("anomaly") or {}
        diagnosis = assessment.get("diagnosis") or {}
        risk = assessment.get("mission_risk") or {}
        health = (assessment.get("health") or {}).get("health_index")

        dt = 0.0 if self.last_sim_time is None else max(0.0, sim_time - self.last_sim_time)
        self.last_sim_time = sim_time
        self.duration_s = sim_time
        self.total_windows += 1

        if health is not None:
            if self.start_health is None:
                self.start_health = health
            self.end_health = health
            self.min_health = health if self.min_health is None else min(self.min_health, health)

        score = anomaly.get("score")
        if score is not None:
            self.peak_anomaly = score if self.peak_anomaly is None else max(self.peak_anomaly, score)
        if anomaly.get("flagged"):
            self.anomaly_windows += 1

        fault = diagnosis.get("predicted_fault")
        if fault:
            self.faults_seen[fault] = self.faults_seen.get(fault, 0) + 1

        rec = risk.get("recommendation")
        if rec:
            self.time_in_recommendation[rec] = self.time_in_recommendation.get(rec, 0.0) + dt
            if (
                self.worst_recommendation is None
                or self._REC_RANK.get(rec, 0) > self._REC_RANK.get(self.worst_recommendation, 0)
            ):
                self.worst_recommendation = rec

    def build(self, efficiency: Optional[Dict[str, Any]] = None,
              alert_counts: Optional[Dict[str, int]] = None) -> Dict[str, Any]:
        health_change = None
        if self.start_health is not None and self.end_health is not None:
            health_change = self.end_health - self.start_health

        dominant_fault = None
        # Healthy is excluded when picking the dominant finding: a sortie that was
        # healthy for most of its length but developed a fault should report the
        # fault, which is the part a maintainer needs.
        non_healthy = {k: v for k, v in self.faults_seen.items() if k != "HEALTHY"}
        if non_healthy:
            dominant_fault = max(non_healthy, key=non_healthy.get)

        return {
            "mission_name": self.mission_name,
            "required_duration_s": self.required_duration_s,
            "elapsed_s": round(self.duration_s, 2),
            "windows_assessed": self.total_windows,
            "start_health": None if self.start_health is None else round(self.start_health, 4),
            "end_health": None if self.end_health is None else round(self.end_health, 4),
            "min_health": None if self.min_health is None else round(self.min_health, 4),
            "health_change": None if health_change is None else round(health_change, 4),
            "peak_anomaly_score": None if self.peak_anomaly is None else round(self.peak_anomaly, 3),
            "anomaly_windows": self.anomaly_windows,
            "anomaly_rate": (
                round(self.anomaly_windows / self.total_windows, 4) if self.total_windows else None
            ),
            "dominant_finding": dominant_fault or "HEALTHY",
            "fault_window_counts": dict(self.faults_seen),
            "time_in_recommendation_s": {
                k: round(v, 2) for k, v in self.time_in_recommendation.items()
            },
            "worst_recommendation": self.worst_recommendation,
            "efficiency": efficiency or {},
            "alert_counts": alert_counts or {},
        }
