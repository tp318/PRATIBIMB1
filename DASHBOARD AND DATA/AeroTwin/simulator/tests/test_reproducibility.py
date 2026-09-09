"""
Unit tests for deterministic reproducibility in AeroTwin-4 simulation runs.
"""

import os
import sys
import unittest

_test_dir = os.path.dirname(os.path.abspath(__file__))
_sim_dir = os.path.dirname(_test_dir)
_aerotwin_dir = os.path.dirname(_sim_dir)
_root_dir = os.path.dirname(_aerotwin_dir)

for _p in [_sim_dir, _aerotwin_dir, _root_dir]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from simulator.runner import EngineRunner
except ImportError:
    from runner import EngineRunner


class TestReproducibility(unittest.TestCase):

    def test_identical_seed_reproducibility(self):
        # Run 1
        runner1 = EngineRunner(seed=42, dt=0.01)
        history1 = runner1.run_for(5.0)

        # Run 2
        runner2 = EngineRunner(seed=42, dt=0.01)
        history2 = runner2.run_for(5.0)

        self.assertEqual(len(history1), len(history2))
        for t1, t2 in zip(history1, history2):
            self.assertEqual(t1.simulation_time, t2.simulation_time)
            self.assertEqual(t1.rpm, t2.rpm)
            self.assertEqual(t1.cht, t2.cht)
            self.assertEqual(t1.instant_torque, t2.instant_torque)


if __name__ == "__main__":
    unittest.main()
