"""
Unit tests for Bearing / Mechanical Degradation Physics (D2).
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


class TestMechanicalDegradation(unittest.TestCase):

    def test_bearing_degradation_increases_friction_torque(self):
        # Healthy
        h_runner = EngineRunner(dt=0.01, seed=42)
        h_inj = DegradationInjector(config=DegradationConfig.healthy(), runner=h_runner)
        h_telemetry, _ = h_inj.run_simulation(duration_seconds=5.0)

        # Bearing Degraded (severity = 0.50 -> friction multiplier = 1.50)
        deg_config = DegradationConfig.single_fault(
            degradation_type=DegradationType.BEARING,
            component_id=ComponentID.BEARING,
            severity=0.50,
        )
        d_runner = EngineRunner(dt=0.01, seed=42)
        d_inj = DegradationInjector(config=deg_config, runner=d_runner)
        d_telemetry, _ = d_inj.run_simulation(duration_seconds=5.0)

        h_fric_mean = np.mean([t.friction_torque for t in h_telemetry[100:]])
        d_fric_mean = np.mean([t.friction_torque for t in d_telemetry[100:]])

        self.assertGreater(d_fric_mean, h_fric_mean * 1.20)
        self.assertTrue(np.all([np.isfinite(t.friction_torque) for t in d_telemetry]))


if __name__ == "__main__":
    unittest.main()
