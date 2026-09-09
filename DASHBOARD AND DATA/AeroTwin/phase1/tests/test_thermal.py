"""
Unit tests for AeroTwin-4 Thermal Subsystem Model.
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
    from engine.thermal import ThermalModel
except ImportError:
    from AeroTwin.phase1.engine.thermal import ThermalModel


class TestThermalModel(unittest.TestCase):

    def setUp(self):
        self.model = ThermalModel(ambient_temp=20.0, cht_init=25.0, oil_init=25.0)

    def test_thermal_heating_under_load(self):
        """
        Verify CHT, EGT, and Oil Temperature increase when running under throttle and load.
        """
        initial_cht = self.model.cht
        initial_oil = self.model.oil_temp

        # Simulate 100 seconds (10,000 steps at dt=0.01s)
        for _ in range(10000):
            st = self.model.update(throttle=0.7, rpm=2600.0, mean_torque=110.0, dt=0.01)

        self.assertGreater(st["cht"], initial_cht, "CHT should increase under load")
        self.assertGreater(st["oil_temperature"], initial_oil, "Oil temp should increase under load")
        self.assertGreater(st["egt"], 400.0, "EGT should be elevated under load")

    def test_cooling_towards_ambient(self):
        """
        Verify temperatures cool down toward ambient when engine is idle/stopped.
        """
        hot_model = ThermalModel(ambient_temp=20.0, cht_init=150.0, oil_init=95.0)

        # Simulate 500 seconds cooling at 0 throttle, 0 RPM
        for _ in range(50000):
            st = hot_model.update(throttle=0.0, rpm=0.0, mean_torque=0.0, dt=0.01)

        self.assertLess(st["cht"], 150.0, "CHT should cool down")
        self.assertLess(st["oil_temperature"], 95.0, "Oil temp should cool down")

    def test_numerical_integrity(self):
        """
        Verify outputs are finite and non-negative.
        """
        for _ in range(1000):
            st = self.model.update(throttle=0.5, rpm=2200.0, mean_torque=90.0, dt=0.001)
            for k, v in st.items():
                self.assertTrue(np.isfinite(v), f"Non-finite thermal value in {k}")
                self.assertGreaterEqual(v, 0.0, f"Negative thermal value in {k}")


if __name__ == "__main__":
    unittest.main()
