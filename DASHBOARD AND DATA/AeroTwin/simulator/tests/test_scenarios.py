"""
Unit tests for AeroTwin-4 Operating Modes & Flight Profiles.
"""

import os
import sys
import unittest

_test_dir = os.path.dirname(os.path.abspath(__file__))
_sim_dir = os.path.dirname(_test_dir)
_aerotwin_dir = os.path.dirname(_sim_dir)
_root_dir = os.path.dirname(_aerotwin_dir)

for _p in [_sim_dir, _aerotwin_dir, _root_dir]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from simulator.scenarios.profiles import FlightProfile, OperatingMode
except ImportError:
    from scenarios.profiles import FlightProfile, OperatingMode


class TestFlightProfile(unittest.TestCase):

    def test_default_profile_sequence(self):
        profile = FlightProfile()

        # At t=5.0s -> IDLE
        s1 = profile.get_state_at(5.0)
        self.assertEqual(s1["operating_mode"], OperatingMode.IDLE.value)

        # At t=20.0s -> TAKEOFF
        s2 = profile.get_state_at(20.0)
        self.assertEqual(s2["operating_mode"], OperatingMode.TAKEOFF.value)
        self.assertGreater(s2["throttle"], 0.80)

        # At t=40.0s -> CLIMB
        s3 = profile.get_state_at(40.0)
        self.assertEqual(s3["operating_mode"], OperatingMode.CLIMB.value)

        # At t=80.0s -> CRUISE
        s4 = profile.get_state_at(80.0)
        self.assertEqual(s4["operating_mode"], OperatingMode.CRUISE.value)

        # At t=130.0s -> DESCENT
        s5 = profile.get_state_at(130.0)
        self.assertEqual(s5["operating_mode"], OperatingMode.DESCENT.value)

    def test_custom_profile_segments(self):
        custom_segments = [
            (0.0, 10.0, OperatingMode.IDLE, 0.15, 1.0),
            (10.0, 30.0, OperatingMode.CRUISE, 0.60, 1.0),
        ]
        profile = FlightProfile(segments=custom_segments)
        s = profile.get_state_at(15.0)
        self.assertEqual(s["operating_mode"], OperatingMode.CRUISE.value)


if __name__ == "__main__":
    unittest.main()
