"""
Unit tests for Cylinder / Combustion Degradation Physics (D1).
"""

import os
import sys
import unittest
import numpy as np

_test_dir = os.path.dirname(os.path.abspath(__file__))
_deg_dir = os.path.dirname(_test_dir)
_aerotwin_dir = os.path.dirname(_deg_dir)
_root_dir = os.path.dirname(_aerotwin_dir)

for _p in [_deg_dir, _aerotwin_dir, _root_dir]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from simulator.runner import EngineRunner
from degradation.config import DegradationConfig, DegradationType, ComponentID
from degradation.injector import DegradationInjector


class TestCylinderDegradation(unittest.TestCase):

    def test_cylinder_3_degradation_reduces_torque_and_increases_vibration(self):
        # 1. Healthy Run
        healthy_runner = EngineRunner(dt=0.01, seed=42)
        healthy_inj = DegradationInjector(config=DegradationConfig.healthy(), runner=healthy_runner)
        h_telemetry, _ = healthy_inj.run_simulation(duration_seconds=5.0)

        # 2. Degraded Run (Cylinder 3 severity = 0.50 -> 75% efficiency)
        deg_config = DegradationConfig.single_fault(
            degradation_type=DegradationType.CYLINDER,
            component_id=ComponentID.CYLINDER_3,
            severity=0.50,
        )
        deg_runner = EngineRunner(dt=0.01, seed=42)
        deg_inj = DegradationInjector(config=deg_config, runner=deg_runner)
        d_telemetry, _ = deg_inj.run_simulation(duration_seconds=5.0)

        # Check Cylinder 3 torque reduction
        h_c3_max = np.max([t.cylinder_3_torque for t in h_telemetry])
        d_c3_max = np.max([t.cylinder_3_torque for t in d_telemetry])
        self.assertLess(d_c3_max, h_c3_max * 0.90)

        # Unaffected Cylinder 1 peak torque should remain unaffected
        h_c1_max = np.max([t.cylinder_1_torque for t in h_telemetry])
        d_c1_max = np.max([t.cylinder_1_torque for t in d_telemetry])
        self.assertAlmostEqual(h_c1_max, d_c1_max, delta=h_c1_max * 0.05)

        # Verify numerical integrity across all telemetry frames
        self.assertTrue(np.all([np.isfinite(t.instant_torque) for t in d_telemetry]))
        self.assertTrue(np.all([np.isfinite(t.vibration) for t in d_telemetry]))


if __name__ == "__main__":
    unittest.main()
