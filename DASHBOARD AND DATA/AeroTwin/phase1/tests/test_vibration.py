"""
Unit tests for AeroTwin-4 Mechanical Vibration Subsystem Model.
"""

import os
import sys
import unittest

_test_dir = os.path.dirname(os.path.abspath(__file__))
_phase1_dir = os.path.dirname(_test_dir)
_aerotwin_dir = os.path.dirname(_phase1_dir)
_root_dir = os.path.dirname(_aerotwin_dir)

for _p in [_phase1_dir, _aerotwin_dir, _root_dir]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from engine.vibration import VibrationModel
except ImportError:
    from AeroTwin.phase1.engine.vibration import VibrationModel

import numpy as np


class TestVibrationModel(unittest.TestCase):

    def setUp(self):
        self.model = VibrationModel(inertia=0.20, torque_gain=0.0035, rot_gain=0.15)

    def test_vibration_increases_with_torque_fluctuation(self):
        """
        Verify vibration amplitude increases with instantaneous torque fluctuation |T_instant - T_mean|.
        """
        st_low = self.model.update(instant_torque=100.0, mean_torque=100.0, rpm=2500.0)
        st_high = self.model.update(instant_torque=180.0, mean_torque=100.0, rpm=2500.0)

        self.assertGreater(
            st_high["vibration"],
            st_low["vibration"],
            "Higher torque fluctuation should yield higher mechanical vibration"
        )

    def test_vibration_increases_with_rpm(self):
        """
        Verify rotational vibration increases with RPM.
        """
        st_low = self.model.update(instant_torque=100.0, mean_torque=100.0, rpm=1000.0)
        st_high = self.model.update(instant_torque=100.0, mean_torque=100.0, rpm=3000.0)

        self.assertGreater(
            st_high["vibration"],
            st_low["vibration"],
            "Higher RPM should yield higher vibration amplitude"
        )

    def test_numerical_integrity(self):
        """
        Verify outputs are finite and positive.
        """
        for rpm in [900, 2500, 3500]:
            st = self.model.update(instant_torque=120.0, mean_torque=100.0, rpm=float(rpm))
            for k, v in st.items():
                self.assertTrue(np.isfinite(v), f"Non-finite vibration value in {k}")
                self.assertGreater(v, 0.0, f"Non-positive vibration value in {k}")


if __name__ == "__main__":
    unittest.main()
