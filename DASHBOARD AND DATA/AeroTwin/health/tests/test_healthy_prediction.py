"""
Unit tests for HealthyStateModel (Mode A & Mode B).
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

from health.healthy_state import HealthyStateModel
from health.models import OperatingState, ExpectedState


class TestHealthyPrediction(unittest.TestCase):

    def test_mode_a_counterfactual_step(self):
        healthy_model = HealthyStateModel(dt=0.01, seed=42)
        op_state = OperatingState(
            timestamp=0.01,
            simulation_time=0.01,
            engine_id="AEROTWIN-001",
            operating_mode="CRUISE",
            throttle=0.60,
            rpm=2500.0,
        )

        expected = healthy_model.predict_counterfactual_step(op_state)
        self.assertIsInstance(expected, ExpectedState)
        self.assertGreater(expected.rpm, 0.0)
        self.assertGreater(expected.mean_torque, 0.0)
        self.assertGreater(expected.cht, 0.0)

    def test_mode_b_reference_point(self):
        healthy_model = HealthyStateModel(dt=0.01, seed=42)
        op_state = OperatingState(
            timestamp=1.0,
            simulation_time=1.0,
            engine_id="AEROTWIN-001",
            operating_mode="TAKEOFF",
            throttle=1.00,
            rpm=3200.0,
        )

    def test_mode_a_counterfactual_independence(self):
        # Mode A counterfactual twin must NOT be driven by observed/degraded RPM internally.
        healthy_model1 = HealthyStateModel(dt=0.01, seed=42)
        op_state_high_rpm = OperatingState(
            timestamp=0.01, simulation_time=0.01, engine_id="AEROTWIN-001",
            operating_mode="CRUISE", throttle=0.60, rpm=5000.0
        )
        exp1 = healthy_model1.predict_counterfactual_step(op_state_high_rpm)

        healthy_model2 = HealthyStateModel(dt=0.01, seed=42)
        op_state_low_rpm = OperatingState(
            timestamp=0.01, simulation_time=0.01, engine_id="AEROTWIN-001",
            operating_mode="CRUISE", throttle=0.60, rpm=100.0
        )
        exp2 = healthy_model2.predict_counterfactual_step(op_state_low_rpm)

        # Counterfactual expected RPM must be 100% identical regardless of passed observed RPM
        self.assertEqual(exp1.rpm, exp2.rpm)


if __name__ == "__main__":
    unittest.main()
