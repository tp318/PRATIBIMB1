"""
Unit tests for AeroTwin-4 Simulation Clock.
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
    from simulator.clock import SimulationClock
except ImportError:
    from clock import SimulationClock


class TestSimulationClock(unittest.TestCase):

    def test_clock_progression(self):
        clock = SimulationClock(dt=0.01)
        self.assertEqual(clock.simulation_time, 0.0)
        self.assertEqual(clock.step_count, 0)

        t1 = clock.step()
        self.assertAlmostEqual(t1, 0.01)
        self.assertEqual(clock.step_count, 1)

        for _ in range(99):
            clock.step()

        self.assertAlmostEqual(clock.simulation_time, 1.0)
        self.assertEqual(clock.step_count, 100)

    def test_clock_reset(self):
        clock = SimulationClock(dt=0.05, initial_time=10.0)
        for _ in range(20):
            clock.step()
        self.assertAlmostEqual(clock.simulation_time, 11.0)

        clock.reset()
        self.assertEqual(clock.simulation_time, 10.0)
        self.assertEqual(clock.step_count, 0)

    def test_custom_step_size(self):
        clock = SimulationClock(dt=0.001)
        clock.step()
        self.assertAlmostEqual(clock.simulation_time, 0.001)


if __name__ == "__main__":
    unittest.main()
