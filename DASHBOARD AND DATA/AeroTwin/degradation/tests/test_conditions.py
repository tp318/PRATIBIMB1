"""
Unit tests for RunConditionSampler (build variation + environmental conditions).
"""

import os
import sys
import unittest

_test_dir = os.path.dirname(os.path.abspath(__file__))
_deg_dir = os.path.dirname(_test_dir)
_aerotwin_dir = os.path.dirname(_deg_dir)
_root_dir = os.path.dirname(_aerotwin_dir)
_phase1_dir = os.path.join(_aerotwin_dir, "phase1")

for _p in [_deg_dir, _aerotwin_dir, _phase1_dir, _root_dir]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from degradation.conditions import (
    BUILD_CLIP_SIGMA,
    BUILD_TOLERANCE,
    ISA_LAPSE_RATE_C_PER_M,
    RunConditionSampler,
)
from phase1.engine.parameters import ENGINE


class TestRunConditions(unittest.TestCase):

    def setUp(self):
        self.sampler = RunConditionSampler()

    def test_same_run_and_seed_is_reproducible(self):
        a = self.sampler.sample("HEALTHY_001", 43)
        b = self.sampler.sample("HEALTHY_001", 43)
        self.assertEqual(a.build_factors, b.build_factors)
        self.assertEqual(a.altitude_m, b.altitude_m)
        self.assertEqual(a.ambient_temperature_c, b.ambient_temperature_c)

    def test_different_runs_are_genuinely_different(self):
        """The bug this guards: all healthy runs used to be bit-identical."""
        a = self.sampler.sample("HEALTHY_001", 43)
        b = self.sampler.sample("HEALTHY_002", 44)
        self.assertNotEqual(a.build_factors, b.build_factors)
        self.assertNotAlmostEqual(a.altitude_m, b.altitude_m, places=3)

    def test_build_variation_stays_within_tolerance(self):
        """Build spread is manufacturing tolerance; it must never imitate a fault."""
        for i in range(1, 60):
            c = self.sampler.sample(f"RUN_{i:03d}", 100 + i)
            for name, factor in c.build_factors.items():
                bound = BUILD_CLIP_SIGMA * BUILD_TOLERANCE[name]
                self.assertGreaterEqual(factor, 1.0 - bound - 1e-9, f"{name} below tolerance")
                self.assertLessEqual(factor, 1.0 + bound + 1e-9, f"{name} above tolerance")

    def test_ambient_follows_isa_lapse_rate(self):
        c = self.sampler.sample("HEALTHY_001", 43)
        expected = c.sea_level_temp_c - ISA_LAPSE_RATE_C_PER_M * c.altitude_m
        self.assertAlmostEqual(c.ambient_temperature_c, expected, places=9)

    def test_build_engine_parameters_does_not_mutate_base(self):
        base_friction = ENGINE["friction_coefficient"]
        c = self.sampler.sample("HEALTHY_001", 43)
        params = self.sampler.build_engine_parameters(c)

        self.assertEqual(ENGINE["friction_coefficient"], base_friction)
        self.assertNotEqual(params["friction_coefficient"], base_friction)
        self.assertEqual(params["ambient_temperature"], c.ambient_temperature_c)

    def test_altitude_and_ambient_are_physically_sane(self):
        for i in range(1, 40):
            c = self.sampler.sample(f"RUN_{i:03d}", 200 + i)
            self.assertGreaterEqual(c.altitude_m, 0.0)
            self.assertLessEqual(c.altitude_m, 7600.0)
            # MALE UAV envelope: sea-level heat down to high-altitude cold soak.
            self.assertGreater(c.ambient_temperature_c, -60.0)
            self.assertLess(c.ambient_temperature_c, 60.0)


if __name__ == "__main__":
    unittest.main()
