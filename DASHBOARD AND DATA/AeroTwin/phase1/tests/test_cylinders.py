"""
Unit tests for AeroTwin-4 four-cylinder combustion torque model.
"""

import os
import sys
import math
import unittest
import numpy as np

_test_dir = os.path.dirname(os.path.abspath(__file__))
_phase1_dir = os.path.dirname(_test_dir)
_aerotwin_dir = os.path.dirname(_phase1_dir)
_root_dir = os.path.dirname(_aerotwin_dir)

for _p in [_phase1_dir, _aerotwin_dir, _root_dir]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from engine.cylinders import FourCylinderModel
    from engine.dynamics import EngineDynamics
except ImportError:
    from AeroTwin.phase1.engine.cylinders import FourCylinderModel
    from AeroTwin.phase1.engine.dynamics import EngineDynamics


class TestFourCylinderModel(unittest.TestCase):

    def setUp(self):
        self.model = FourCylinderModel()

    def test_crank_angle_wrapping(self):
        """
        Verify crank angle wraps continuously within [0, 720) over long runs.
        """
        eng = EngineDynamics()
        for _ in range(100000):
            st = eng.update(throttle=0.6, dt=0.001)
            angle = st["crank_angle"]
            self.assertGreaterEqual(angle, 0.0)
            self.assertLess(angle, 720.0)
            self.assertTrue(np.isfinite(angle))

    def test_cylinder_phase_offsets(self):
        """
        Verify the 4 cylinders have explicit 180° phase offsets.
        """
        expected_offsets = {1: 0.0, 3: 180.0, 4: 360.0, 2: 540.0}
        self.assertEqual(self.model.phase_offsets, expected_offsets)

    def test_torque_pulsation_periodicity(self):
        """
        Verify instantaneous total engine torque exhibits 4 distinct periodic peak pulses
        across a single 720° combustion cycle.
        """
        angles = np.linspace(0, 720, 720, endpoint=False)
        torques = []
        for a in angles:
            _, total_t = self.model.calculate_torques(a, mean_torque=100.0)
            torques.append(total_t)

        torques = np.array(torques)
        
        # Identify local peak pulses (higher than neighbors)
        peaks = []
        for i in range(len(torques)):
            prev_val = torques[(i - 1) % len(torques)]
            next_val = torques[(i + 1) % len(torques)]
            if torques[i] > prev_val and torques[i] > next_val and torques[i] > 100.0:
                peaks.append(angles[i])

        self.assertEqual(
            len(peaks), 4,
            f"Expected exactly 4 combustion torque pulses per 720° cycle, found {len(peaks)} at angles {peaks}"
        )

    def test_cycle_average_torque_conservation(self):
        """
        Verify cycle-averaged total instantaneous torque over 720° matches target mean torque.
        """
        target_mean = 120.0
        steps = 720
        total_sum = 0.0
        for deg in range(steps):
            _, total_t = self.model.calculate_torques(float(deg), mean_torque=target_mean)
            total_sum += total_t

        computed_mean = total_sum / steps
        self.assertAlmostEqual(computed_mean, target_mean, delta=0.5)

    def test_non_negative_and_finite_torques(self):
        """
        Verify all individual cylinder and total instant torques are finite.
        """
        for deg in range(0, 720, 10):
            cyl_torques, total_t = self.model.calculate_torques(float(deg), mean_torque=150.0)
            self.assertTrue(np.isfinite(total_t))
            self.assertGreaterEqual(total_t, 0.0)
            for cyl_id, t_val in cyl_torques.items():
                self.assertTrue(np.isfinite(t_val), f"Cylinder {cyl_id} torque is not finite")


if __name__ == "__main__":
    unittest.main()
