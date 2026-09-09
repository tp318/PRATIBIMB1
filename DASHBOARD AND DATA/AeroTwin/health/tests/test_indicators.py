"""
Unit tests for ResidualIndicatorEngine and cylinder balance.
"""

import os
import sys
import unittest

_test_dir = os.path.dirname(os.path.abspath(__file__))
_health_dir = os.path.dirname(_test_dir)
_aerotwin_dir = os.path.dirname(_health_dir)
_root_dir = os.path.dirname(_aerotwin_dir)

for _p in [_health_dir, _aerotwin_dir, _root_dir]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from health.indicators import ResidualIndicatorEngine
from health.models import ResidualState


class TestIndicators(unittest.TestCase):

    def test_cylinder_balance_calculation(self):
        # Balanced cylinder torques
        balanced_torques = [150.0, 150.0, 150.0, 150.0]
        bal_balanced = ResidualIndicatorEngine.calculate_cylinder_balance(balanced_torques)
        self.assertAlmostEqual(bal_balanced, 0.0)

        # Imbalanced cylinder torques (Cyl 3 drop)
        imbalanced_torques = [150.0, 150.0, 75.0, 150.0]
        bal_imbalanced = ResidualIndicatorEngine.calculate_cylinder_balance(imbalanced_torques)
        self.assertGreater(bal_imbalanced, 0.10)

    def test_indicator_processing(self):
        engine = ResidualIndicatorEngine()
        raw = {"cht": 15.0, "oil_pressure": -30000.0, "vibration": 0.10, "friction_torque": 5.0}
        norm = {"cht": 3.0, "oil_pressure": -2.5, "vibration": 2.0, "friction_torque": 1.5}
        res_state = ResidualState(raw_signed=raw, absolute=raw, normalized=norm)

        indicators = engine.process_frame(res_state, observed_torques=[150.0, 150.0, 75.0, 150.0])
        self.assertEqual(indicators.thermal_deviation, 3.0)
        self.assertEqual(indicators.oil_deviation, 2.5)
        self.assertEqual(indicators.vibration_deviation, 2.0)
        self.assertGreater(indicators.cylinder_balance_deviation, 0.10)


if __name__ == "__main__":
    unittest.main()
