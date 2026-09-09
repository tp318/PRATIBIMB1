"""
Unit tests for Cooling-System Degradation Physics (D3).
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


class TestThermalDegradation(unittest.TestCase):

    def test_cooling_degradation_increases_cht(self):
        # Healthy
        h_runner = EngineRunner(dt=0.01, seed=42)
        h_inj = DegradationInjector(config=DegradationConfig.healthy(), runner=h_runner)
        h_telemetry, _ = h_inj.run_simulation(duration_seconds=15.0)

        # Cooling Degraded (severity = 0.50 -> 75% cooling efficiency)
        deg_config = DegradationConfig.single_fault(
            degradation_type=DegradationType.COOLING,
            component_id=ComponentID.COOLING_SYSTEM,
            severity=0.50,
        )
        d_runner = EngineRunner(dt=0.01, seed=42)
        d_inj = DegradationInjector(config=deg_config, runner=d_runner)
        d_telemetry, _ = d_inj.run_simulation(duration_seconds=15.0)

        h_cht_final = h_telemetry[-1].cht
        d_cht_final = d_telemetry[-1].cht

        self.assertGreater(d_cht_final, h_cht_final + 3.0)
        self.assertTrue(np.all([np.isfinite(t.cht) for t in d_telemetry]))


if __name__ == "__main__":
    unittest.main()
