"""
Unit tests for Ground-Truth Data Contract & Schema.
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
from degradation.ground_truth import RunGroundTruth, SampleGroundTruth


class TestGroundTruth(unittest.TestCase):

    def test_healthy_ground_truth_labeling(self):
        runner = EngineRunner(dt=0.01, seed=42)
        inj = DegradationInjector(config=DegradationConfig.healthy(), runner=runner, run_id="RUN_HEALTHY")
        _, gt_list = inj.run_simulation(duration_seconds=2.0)

        self.assertEqual(inj.run_ground_truth.degradation_type, "NONE")
        self.assertEqual(inj.run_ground_truth.max_severity, 0.0)

        for sample_gt in gt_list:
            self.assertIsInstance(sample_gt, SampleGroundTruth)
            self.assertEqual(sample_gt.degradation_type, "NONE")
            self.assertEqual(sample_gt.active_severity, 0.0)
            self.assertEqual(sample_gt.current_health, 1.0)
            self.assertFalse(sample_gt.is_degraded)

    def test_degraded_ground_truth_labeling(self):
        config = DegradationConfig.single_fault(
            degradation_type=DegradationType.CYLINDER,
            component_id=ComponentID.CYLINDER_3,
            severity=0.60,
        )
        runner = EngineRunner(dt=0.01, seed=42)
        inj = DegradationInjector(config=config, runner=runner, run_id="RUN_CYL3_060")
        _, gt_list = inj.run_simulation(duration_seconds=2.0)

        self.assertEqual(inj.run_ground_truth.degradation_type, "CYLINDER")
        self.assertEqual(inj.run_ground_truth.target_component, "CYLINDER_3")
        self.assertEqual(inj.run_ground_truth.max_severity, 0.60)

        for sample_gt in gt_list:
            self.assertEqual(sample_gt.degradation_type, "CYLINDER")
            self.assertEqual(sample_gt.target_component, "CYLINDER_3")
            self.assertEqual(sample_gt.active_severity, 0.60)
            self.assertEqual(sample_gt.current_health, 0.40)
            self.assertTrue(sample_gt.is_degraded)


if __name__ == "__main__":
    unittest.main()
