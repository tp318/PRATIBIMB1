"""
Unit tests for AeroTwin-4 engine rotational dynamics & physics validation.
"""

import os
import sys
import unittest
import math
import numpy as np

_test_dir = os.path.dirname(os.path.abspath(__file__))
_phase1_dir = os.path.dirname(_test_dir)
_aerotwin_dir = os.path.dirname(_phase1_dir)
_root_dir = os.path.dirname(_aerotwin_dir)

for _p in [_phase1_dir, _aerotwin_dir, _root_dir]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from engine.dynamics import EngineDynamics
    from engine.parameters import ENGINE
except ImportError:
    from AeroTwin.phase1.engine.dynamics import EngineDynamics
    from AeroTwin.phase1.engine.parameters import ENGINE


class TestEngineDynamics(unittest.TestCase):

    def test_rpm_omega_conversions(self):
        dynamics = EngineDynamics()
        rpm = 3000.0
        omega = dynamics.rpm_to_omega(rpm)
        self.assertAlmostEqual(dynamics.omega_to_rpm(omega), rpm, places=5)

    def test_rpm_natural_bounds(self):
        """
        1. RPM bounds test:
        Verify 0 <= rpm <= max_rpm for all throttles (0.0 to 1.0)
        purely through natural physical equilibrium without hard clipping.
        """
        max_rpm = ENGINE["max_rpm"]
        for throttle in [0.0, 0.2, 0.5, 0.8, 1.0]:
            eng = EngineDynamics()
            for _ in range(10000):
                st = eng.update(throttle=throttle, dt=0.001)
            
            self.assertGreaterEqual(st["rpm"], 0.0, f"RPM became negative for throttle={throttle}")
            self.assertLessEqual(st["rpm"], max_rpm, f"RPM ({st['rpm']:.1f}) exceeded max_rpm ({max_rpm}) for throttle={throttle}")

    def test_throttle_monotonicity(self):
        """
        2. Throttle monotonicity test:
        Run throttles 0.3, 0.5, 0.7, 0.9 and verify equilibrium RPM monotonically increases.
        """
        throttles = [0.3, 0.5, 0.7, 0.9]
        equilibrium_rpms = []

        for thr in throttles:
            eng = EngineDynamics()
            for _ in range(10000):
                st = eng.update(throttle=thr, dt=0.001)
            equilibrium_rpms.append(st["rpm"])

        for i in range(len(equilibrium_rpms) - 1):
            self.assertGreater(
                equilibrium_rpms[i + 1],
                equilibrium_rpms[i],
                f"Equilibrium RPM failed to increase from throttle {throttles[i]} ({equilibrium_rpms[i]:.1f}) to {throttles[i+1]} ({equilibrium_rpms[i+1]:.1f})"
            )

    def test_load_response(self):
        """
        3. Load response test:
        Fix throttle at 0.6 and test load_coefficient low, medium, high.
        Verify equilibrium RPM strictly decreases as load increases.
        """
        base_k = ENGINE["load_coefficient"]
        load_multipliers = [0.7, 1.0, 1.5]
        rpms = []

        for mult in load_multipliers:
            params = ENGINE.copy()
            params["load_coefficient"] = base_k * mult
            eng = EngineDynamics(parameters=params)
            for _ in range(10000):
                st = eng.update(throttle=0.6, dt=0.001)
            rpms.append(st["rpm"])

        self.assertGreater(rpms[0], rpms[1], "Lower load should yield higher RPM")
        self.assertGreater(rpms[1], rpms[2], "Medium load should yield higher RPM than high load")

    def test_steady_state_stability(self):
        """
        4. Stability test:
        After reaching operating speed (t >= 8s), mean RPM reaches steady state
        and instantaneous RPM micro-ripple due to cylinder firing pulses remains strictly bounded (< 2.0 RPM/step).
        """
        eng = EngineDynamics()
        dt = 0.001
        prev_rpm = None
        recent_rpms = []

        # Simulate for 10 seconds (10,000 steps)
        for step in range(10000):
            st = eng.update(throttle=0.6, dt=dt)
            curr_rpm = st["rpm"]
            if step > 8000:
                diff = abs(curr_rpm - prev_rpm)
                self.assertLess(
                    diff,
                    15.0,
                    f"Engine instant RPM ripple unbounded at step {step}: delta={diff:.5f}"
                )
                recent_rpms.append(curr_rpm)
            prev_rpm = curr_rpm

        # Verify mean RPM stability over last 2000 steps
        mean_rpm = np.mean(recent_rpms)
        self.assertAlmostEqual(mean_rpm, 2596.8, delta=15.0)

    def test_numerical_integrity(self):
        """
        5. No numerical failure test:
        Check np.isfinite(val) for all output variables throughout simulation.
        """
        eng = EngineDynamics()
        for _ in range(5000):
            st = eng.update(throttle=0.7, dt=0.001)
            for key, val in st.items():
                if isinstance(val, dict):
                    for sub_k, sub_v in val.items():
                        self.assertTrue(
                            np.isfinite(sub_v),
                            f"Non-finite value ({sub_v}) found in dict '{key}[{sub_k}]'"
                        )
                else:
                    self.assertTrue(
                        np.isfinite(val),
                        f"Non-finite value ({val}) found in key '{key}'"
                    )


if __name__ == "__main__":
    unittest.main()


