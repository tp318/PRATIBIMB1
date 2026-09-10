"""
============================================================================
mission_scenario.py — Mission Scenario Presets & Throttle Profile Driver
============================================================================
Provides named scenario configurations for standard MALE UAV flight types:
  • HIGH ALTITUDE      — 7500m, ISA standard, 70% cruise throttle, 480s
  • ENDURANCE          — 3000m, ISA standard, 65% cruise throttle, 1800s
  • HOT WEATHER        — 500m,  +30°C ISA delta, 80% cruise throttle, 600s
  • RAPID THROTTLE     — 1500m, step transitions every 30s, 300s
  • CUSTOM             — explicit user-configurable overrides
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional


class ThrottleProfile(str, Enum):
    CRUISE     = "CRUISE"       # Constant cruise throttle
    RAMP       = "RAMP"         # Linear 40% → 100% over duration
    STEP       = "STEP"         # Alternating 20% / 100% every 30 s
    SINUSOIDAL = "SINUSOIDAL"   # Smooth sine oscillation between 40% and 90%


class ScenarioType(str, Enum):
    HIGH_ALTITUDE  = "HIGH_ALTITUDE"
    ENDURANCE      = "ENDURANCE"
    HOT_WEATHER    = "HOT_WEATHER"
    RAPID_THROTTLE = "RAPID_THROTTLE"
    CUSTOM         = "CUSTOM"


@dataclass
class ScenarioConfig:
    """Complete description of a mission scenario."""
    scenario_type: ScenarioType
    altitude_m: float               # cruise altitude (m)
    isa_delta_c: float              # ISA temperature deviation (+30 = hot day)
    sea_level_temp_c: float         # = 15.0 + isa_delta_c
    duration_s: float               # total sortie duration (s)
    throttle_profile: ThrottleProfile
    cruise_throttle: float          # fraction [0.0, 1.0] for CRUISE/RAMP/SINE
    inject_fault: Optional[str]     # e.g. "LUBRICATION", None = healthy
    fault_severity: float           # [0.0, 1.0]
    fault_at_s: float               # seconds from T-0 to trigger fault
    description: str                # human-readable summary
    display_name: str               # short UI label

    @property
    def ambient_temp_at_altitude_c(self) -> float:
        """ISA temperature at cruise altitude (°C)."""
        lapse_rate = 0.0065          # K/m
        t_sl = self.sea_level_temp_c + 273.15
        temp_k = t_sl - lapse_rate * self.altitude_m
        return temp_k - 273.15

    def throttle_at(self, t: float) -> float:
        """Returns commanded throttle fraction at mission time `t` (seconds)."""
        if self.throttle_profile == ThrottleProfile.CRUISE:
            return self.cruise_throttle

        elif self.throttle_profile == ThrottleProfile.RAMP:
            return 0.40 + 0.60 * min(1.0, t / max(1.0, self.duration_s))

        elif self.throttle_profile == ThrottleProfile.STEP:
            cycle = int(t // 30) % 2
            return 1.00 if cycle == 1 else 0.20

        elif self.throttle_profile == ThrottleProfile.SINUSOIDAL:
            # 60s period sine wave between 40% and 90%
            omega = 2.0 * math.pi / 60.0
            return 0.65 + 0.25 * math.sin(omega * t)

        return self.cruise_throttle

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario_type": self.scenario_type.value,
            "display_name": self.display_name,
            "description": self.description,
            "altitude_m": self.altitude_m,
            "isa_delta_c": self.isa_delta_c,
            "sea_level_temp_c": self.sea_level_temp_c,
            "ambient_temp_c": round(self.ambient_temp_at_altitude_c, 1),
            "duration_s": self.duration_s,
            "throttle_profile": self.throttle_profile.value,
            "cruise_throttle": self.cruise_throttle,
            "inject_fault": self.inject_fault,
            "fault_severity": self.fault_severity,
            "fault_at_s": self.fault_at_s,
        }


PRESETS: Dict[ScenarioType, ScenarioConfig] = {
    ScenarioType.HIGH_ALTITUDE: ScenarioConfig(
        scenario_type=ScenarioType.HIGH_ALTITUDE,
        altitude_m=7500.0,
        isa_delta_c=0.0,
        sea_level_temp_c=15.0,
        duration_s=480.0,
        throttle_profile=ThrottleProfile.CRUISE,
        cruise_throttle=0.70,
        inject_fault=None,
        fault_severity=0.0,
        fault_at_s=0.0,
        display_name="High Altitude (7,500 m)",
        description="High altitude cruise near ceiling (7,500 m, -33.8°C ambient, low air density).",
    ),
    ScenarioType.ENDURANCE: ScenarioConfig(
        scenario_type=ScenarioType.ENDURANCE,
        altitude_m=3000.0,
        isa_delta_c=0.0,
        sea_level_temp_c=15.0,
        duration_s=1800.0,
        throttle_profile=ThrottleProfile.CRUISE,
        cruise_throttle=0.65,
        inject_fault=None,
        fault_severity=0.0,
        fault_at_s=0.0,
        display_name="Long Endurance Loiter",
        description="Extended station keeping at 3,000 m and optimal specific fuel consumption.",
    ),
    ScenarioType.HOT_WEATHER: ScenarioConfig(
        scenario_type=ScenarioType.HOT_WEATHER,
        altitude_m=500.0,
        isa_delta_c=30.0,
        sea_level_temp_c=45.0,
        duration_s=600.0,
        throttle_profile=ThrottleProfile.CRUISE,
        cruise_throttle=0.80,
        inject_fault=None,
        fault_severity=0.0,
        fault_at_s=0.0,
        display_name="Desert / Hot Weather",
        description="Low altitude high ambient heat (+30°C ISA delta = 41.8°C ambient), stressing cooling.",
    ),
    ScenarioType.RAPID_THROTTLE: ScenarioConfig(
        scenario_type=ScenarioType.RAPID_THROTTLE,
        altitude_m=1500.0,
        isa_delta_c=0.0,
        sea_level_temp_c=15.0,
        duration_s=300.0,
        throttle_profile=ThrottleProfile.STEP,
        cruise_throttle=0.60,
        inject_fault=None,
        fault_severity=0.0,
        fault_at_s=0.0,
        display_name="Rapid Throttle Cycles",
        description="Dynamic step throttle transitions (20% <-> 100% every 30s) testing mechanical stability.",
    ),
}


def build_scenario(
    scenario_type: str | ScenarioType = "HIGH_ALTITUDE",
    altitude_m: Optional[float] = None,
    isa_delta_c: Optional[float] = None,
    duration_s: Optional[float] = None,
    throttle_profile: Optional[str | ThrottleProfile] = None,
    cruise_throttle: Optional[float] = None,
    inject_fault: Optional[str] = None,
    fault_severity: float = 0.6,
    fault_at_s: float = 60.0,
) -> ScenarioConfig:
    """Builds a ScenarioConfig, using named preset as base with optional overrides."""
    if isinstance(scenario_type, str):
        try:
            st = ScenarioType[scenario_type.upper()]
        except KeyError:
            st = ScenarioType.CUSTOM
    else:
        st = scenario_type

    base = PRESETS.get(st, PRESETS[ScenarioType.HIGH_ALTITUDE])

    alt = float(altitude_m if altitude_m is not None else base.altitude_m)
    delta_c = float(isa_delta_c if isa_delta_c is not None else base.isa_delta_c)
    dur = float(duration_s if duration_s is not None else base.duration_s)

    tp = base.throttle_profile
    if throttle_profile is not None:
        if isinstance(throttle_profile, str):
            try:
                tp = ThrottleProfile[throttle_profile.upper()]
            except KeyError:
                tp = ThrottleProfile.CRUISE
        else:
            tp = throttle_profile

    thr = float(cruise_throttle if cruise_throttle is not None else base.cruise_throttle)

    return ScenarioConfig(
        scenario_type=st,
        altitude_m=max(0.0, min(7600.0, alt)),
        isa_delta_c=delta_c,
        sea_level_temp_c=15.0 + delta_c,
        duration_s=max(10.0, dur),
        throttle_profile=tp,
        cruise_throttle=max(0.0, min(1.0, thr)),
        inject_fault=inject_fault,
        fault_severity=max(0.0, min(1.0, fault_severity)),
        fault_at_s=max(0.0, fault_at_s),
        description=base.description if st != ScenarioType.CUSTOM else "Custom mission scenario.",
        display_name=base.display_name if st != ScenarioType.CUSTOM else "Custom Sortie",
    )


def list_presets() -> List[Dict[str, Any]]:
    """Returns list of preset dictionaries for API consumption."""
    return [cfg.to_dict() for cfg in PRESETS.values()]
