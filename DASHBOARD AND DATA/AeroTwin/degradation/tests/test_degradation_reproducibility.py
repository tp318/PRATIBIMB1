"""
Unit tests for deterministic seed reproducibility under degradation.
"""

import os
import sys
import unittest

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


class TestDegradationReproducibility(unittest.TestCase):

    def test_identical_seed_reproducibility_with_degradation(self):
        deg_config = DegradationConfig.single_fault(
            degradation_type=DegradationType.CYLINDER,
            component_id=ComponentID.CYLINDER_3,
            severity=0.50,
        )

        # Run 1
        runner1 = EngineRunner(dt=0.01, seed=42)
        inj1 = DegradationInjector(config=deg_config, runner=runner1, run_id="RUN_1")
        t1_list, g1_list = inj1.run_simulation(duration_seconds=3.0)

        # Run 2
        runner2 = EngineRunner(dt=0.01, seed=42)
        inj2 = DegradationInjector(config=deg_config, runner=runner2, run_id="RUN_2")
        t2_list, g2_list = inj2.run_simulation(duration_seconds=3.0)

        self.assertEqual(len(t1_list), len(t2_list))
        for t1, t2, g1, g2 in zip(t1_list, t2_list, g1_list, g2_list):
            self.assertEqual(t1.simulation_time, t2.simulation_time)
            self.assertEqual(t1.rpm, t2.rpm)
            self.assertEqual(t1.cylinder_3_torque, t2.cylinder_3_torque)
            self.assertEqual(t1.vibration, t2.vibration)
            self.assertEqual(g1.active_severity, g2.active_severity)


if __name__ == "__main__":
    unittest.main()
