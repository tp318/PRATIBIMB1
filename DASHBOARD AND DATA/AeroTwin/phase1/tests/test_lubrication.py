"""
Unit tests for AeroTwin-4 Lubrication & Oil Subsystem Model.
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
    from engine.lubrication import LubricationModel
except ImportError:
    from AeroTwin.phase1.engine.lubrication import LubricationModel


class TestLubricationModel(unittest.TestCase):

    def setUp(self):
        self.model = LubricationModel(p_idle=280.0, p_max=410.0, nominal_temp=95.0)

    def test_oil_pressure_increases_with_rpm(self):
        """
        Verify oil pressure increases with engine speed (RPM).
        """
        st_idle = self.model.update(rpm=900.0, oil_temperature=95.0)
        st_cruise = self.model.update(rpm=2600.0, oil_temperature=95.0)

        self.assertGreater(
            st_cruise["oil_pressure"],
            st_idle["oil_pressure"],
            "Oil pressure should increase with RPM"
        )

    def test_oil_temp_viscosity_friction_coupling(self):
        """
        Verify elevated oil temperature reduces oil pressure and increases friction multiplier.
        """
        st_norm = self.model.update(rpm=2600.0, oil_temperature=95.0)
        st_hot = self.model.update(rpm=2600.0, oil_temperature=125.0)

        self.assertLess(
            st_hot["oil_pressure"],
            st_norm["oil_pressure"],
            "Hotter oil should lower viscosity and oil pressure"
        )
        self.assertGreater(
            st_hot["friction_multiplier"],
            st_norm["friction_multiplier"],
            "Hotter oil should increase friction multiplier"
        )

    def test_numerical_integrity(self):
        """
        Verify outputs are finite and non-negative.
        """
        for rpm in [0, 900, 2500, 3500]:
            st = self.model.update(rpm=float(rpm), oil_temperature=90.0)
            for k, v in st.items():
                self.assertTrue(np.isfinite(v), f"Non-finite lubrication value in {k}")
                self.assertGreaterEqual(v, 0.0, f"Negative lubrication value in {k}")


if __name__ == "__main__":
    unittest.main()
