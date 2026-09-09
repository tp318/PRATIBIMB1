"""
Unit tests for AeroTwin-4 Fuel Subsystem Model.
"""

import os
import sys
import unittest
import numpy as np

_test_dir = os.path.dirname(os.path.abspath(__file__))
_phase1_dir = os.path.dirname(_test_dir)
_aerotwin_dir = os.path.dirname(_phase1_dir)
_root_dir = os.path.dirname(_aerotwin_dir)

for _p in [_phase1_dir, _aerotwin_dir, _root_dir]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from engine.fuel import FuelModel
except ImportError:
    from AeroTwin.phase1.engine.fuel import FuelModel


class TestFuelModel(unittest.TestCase):

    def setUp(self):
        self.model = FuelModel(bsfc=0.28, fuel_density=0.72, nominal_pressure=320.0)

    def test_fuel_flow_increases_with_power(self):
        """
        Verify fuel flow rate increases with throttle, RPM, and mean torque.
        """
        st_low = self.model.update(throttle=0.3, rpm=1800.0, mean_torque=50.0)
        st_high = self.model.update(throttle=0.8, rpm=3000.0, mean_torque=140.0)

        self.assertGreater(
            st_high["fuel_flow"],
            st_low["fuel_flow"],
            "Fuel flow should increase with power output"
        )
        self.assertGreater(
            st_high["fuel_flow_lph"],
            st_low["fuel_flow_lph"],
            "Fuel flow (L/h) should increase with power output"
        )

    def test_fuel_pressure_stability(self):
        """
        Verify fuel pressure remains near nominal regulated pressure (320 kPa).
        """
        st = self.model.update(throttle=0.6, rpm=2600.0, mean_torque=108.0)
        self.assertAlmostEqual(st["fuel_pressure"], 320.0, delta=10.0)

    def test_numerical_integrity(self):
        """
        Verify outputs are finite and non-negative.
        """
        for thr in [0.0, 0.5, 1.0]:
            st = self.model.update(throttle=thr, rpm=2000.0, mean_torque=80.0)
            for k, v in st.items():
                self.assertTrue(np.isfinite(v), f"Non-finite fuel value in {k}")
                self.assertGreaterEqual(v, 0.0, f"Negative fuel value in {k}")


if __name__ == "__main__":
    unittest.main()
