"""
Unit tests for Lubrication / Oil Degradation Physics (D4).
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


class TestLubricationDegradation(unittest.TestCase):

    def test_lubrication_degradation_reduces_oil_pressure(self):
        # Healthy
        h_runner = EngineRunner(dt=0.01, seed=42)
        h_inj = DegradationInjector(config=DegradationConfig.healthy(), runner=h_runner)
        h_telemetry, _ = h_inj.run_simulation(duration_seconds=5.0)

        # Lubrication Degraded (severity = 0.50 -> 75% pump pressure capacity)
        deg_config = DegradationConfig.single_fault(
            degradation_type=DegradationType.LUBRICATION,
            component_id=ComponentID.LUBRICATION_SYSTEM,
            severity=0.50,
        )
        d_runner = EngineRunner(dt=0.01, seed=42)
        d_inj = DegradationInjector(config=deg_config, runner=d_runner)
        d_telemetry, _ = d_inj.run_simulation(duration_seconds=5.0)

        h_oil_p = np.mean([t.oil_pressure for t in h_telemetry[100:]])
        d_oil_p = np.mean([t.oil_pressure for t in d_telemetry[100:]])

        self.assertLess(d_oil_p, h_oil_p * 0.90)
        self.assertTrue(np.all([np.isfinite(t.oil_pressure) for t in d_telemetry]))


if __name__ == "__main__":
    unittest.main()
