"""
Unit tests for AeroTwin-4 Engine Runner lifecycle & telemetry generation.
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
    from simulator.runner import EngineRunner, RunnerState
    from simulator.telemetry.schema import EngineTelemetry
except ImportError:
    from runner import EngineRunner, RunnerState
    from telemetry.schema import EngineTelemetry


class TestEngineRunner(unittest.TestCase):

    def test_runner_initialization_and_states(self):
        runner = EngineRunner(dt=0.01)
        self.assertEqual(runner.state, RunnerState.STOPPED)

        runner.start()
        self.assertEqual(runner.state, RunnerState.RUNNING)

        runner.pause()
        self.assertEqual(runner.state, RunnerState.PAUSED)

        runner.resume()
        self.assertEqual(runner.state, RunnerState.RUNNING)

        runner.stop()
        self.assertEqual(runner.state, RunnerState.STOPPED)

    def test_runner_step_execution(self):
        runner = EngineRunner(dt=0.01)
        telemetry = runner.step()

        self.assertIsInstance(telemetry, EngineTelemetry)
        self.assertEqual(telemetry.simulation_time, 0.0)
        self.assertGreater(telemetry.rpm, 0.0)
        self.assertGreater(telemetry.oil_pressure, 0.0)

        t2 = runner.step()
        self.assertEqual(t2.simulation_time, 0.01)
        self.assertEqual(len(runner.history), 2)

    def test_runner_pause_resume_behavior(self):
        runner = EngineRunner(dt=0.01)
        t1 = runner.step()
        runner.pause()

        # Step while paused returns last telemetry without advancing history length
        t_paused = runner.step()
        self.assertEqual(t_paused.simulation_time, t1.simulation_time)
        self.assertEqual(len(runner.history), 1)

        runner.resume()
        t2 = runner.step()
        self.assertGreater(t2.simulation_time, t1.simulation_time)

    def test_runner_manual_overrides(self):
        runner = EngineRunner(dt=0.01)
        runner.set_throttle(0.85)
        runner.set_operating_mode("TAKEOFF")

        t = runner.step()
        self.assertEqual(t.throttle, 0.85)
        self.assertEqual(t.operating_mode, "TAKEOFF")

        runner.clear_overrides()
        t2 = runner.step()
        self.assertNotEqual(t2.throttle, 0.85)

    def test_per_cylinder_telemetry_fields(self):
        import numpy as np
        runner = EngineRunner(dt=0.01)
        telemetry = runner.step()

        self.assertTrue(hasattr(telemetry, "cylinder_1_torque"))
        self.assertTrue(hasattr(telemetry, "cylinder_2_torque"))
        self.assertTrue(hasattr(telemetry, "cylinder_3_torque"))
        self.assertTrue(hasattr(telemetry, "cylinder_4_torque"))
        self.assertTrue(np.isfinite(telemetry.cylinder_1_torque))
        self.assertTrue(np.isfinite(telemetry.cylinder_2_torque))

    def test_realtime_wall_clock_pacing(self):
        import time
        runner = EngineRunner(dt=0.05)
        t_start = time.perf_counter()
        
        # Run 0.2 seconds of simulation in 1.0x real-time mode
        history = runner.run_realtime(duration_seconds=0.2, playback_speed=1.0)
        t_elapsed = time.perf_counter() - t_start

        self.assertGreaterEqual(len(history), 4)
        # Wall clock elapsed time should be at least ~0.18s due to real-time pacing
        self.assertGreaterEqual(t_elapsed, 0.15)


if __name__ == "__main__":
    unittest.main()
