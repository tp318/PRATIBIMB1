"""
AeroTwin-4 Engine Operating Scenarios & Flight Profiles.

Defines engine operating modes (IDLE, TAXI, TAKEOFF, CLIMB, CRUISE, DESCENT)
and configurable flight mission profiles.
"""

from enum import Enum
from typing import List, Tuple, Dict, Any


class OperatingMode(str, Enum):
    IDLE = "IDLE"
    TAXI = "TAXI"
    TAKEOFF = "TAKEOFF"
    CLIMB = "CLIMB"
    CRUISE = "CRUISE"
    DESCENT = "DESCENT"


class FlightProfile:
    """
    Configurable flight mission scenario scheduling operating mode transitions,
    throttle commands, load modifiers, altitude, and airspeed trajectories.
    """

    def __init__(self, segments: List[Tuple[float, float, OperatingMode, float, float]] = None):
        """
        Parameters
        ----------
        segments : List[Tuple[start_time, end_time, mode, target_throttle, load_modifier]]
        """
        if segments is None:
            # Default mission profile sequence
            segments = [
                (0.0, 10.0, OperatingMode.IDLE, 0.15, 1.0),
                (10.0, 25.0, OperatingMode.TAKEOFF, 0.95, 1.0),
                (25.0, 50.0, OperatingMode.CLIMB, 0.85, 1.0),
                (50.0, 120.0, OperatingMode.CRUISE, 0.65, 1.0),
                (120.0, 150.0, OperatingMode.DESCENT, 0.35, 1.0),
            ]
        self.segments = sorted(segments, key=lambda s: s[0])

    def get_state_at(self, simulation_time: float) -> Dict[str, Any]:
        """
        Get profile operating state at specified simulation_time.

        Returns
        -------
        dict
            Dict with operating_mode, throttle, load_modifier, altitude, airspeed.
        """
        t = max(0.0, simulation_time)

        # Find active segment
        active_seg = self.segments[-1]
        for seg in self.segments:
            start_t, end_t, mode, thr, load_mult = seg
            if start_t <= t < end_t:
                active_seg = seg
                break

        start_t, end_t, mode, target_thr, load_mult = active_seg

        # Calculate smooth linear throttle transition within segment if transitioning
        if t < start_t + 2.0 and self.segments.index(active_seg) > 0:
            prev_seg = self.segments[self.segments.index(active_seg) - 1]
            prev_thr = prev_seg[3]
            alpha = (t - start_t) / 2.0
            current_thr = prev_thr + alpha * (target_thr - prev_thr)
        else:
            current_thr = target_thr

        # Mode specific altitude (m) and airspeed (m/s) estimates
        alt_map = {
            OperatingMode.IDLE: 0.0,
            OperatingMode.TAXI: 0.0,
            OperatingMode.TAKEOFF: 50.0,
            OperatingMode.CLIMB: 1500.0,
            OperatingMode.CRUISE: 3000.0,
            OperatingMode.DESCENT: 1000.0,
        }
        speed_map = {
            OperatingMode.IDLE: 0.0,
            OperatingMode.TAXI: 10.0,
            OperatingMode.TAKEOFF: 45.0,
            OperatingMode.CLIMB: 55.0,
            OperatingMode.CRUISE: 65.0,
            OperatingMode.DESCENT: 40.0,
        }

        return {
            "operating_mode": mode.value,
            "throttle": min(1.0, max(0.0, current_thr)),
            "load_modifier": load_mult,
            "altitude": alt_map.get(mode, 0.0),
            "airspeed": speed_map.get(mode, 0.0),
        }
