"""
Unit tests for Dataset Builder & Sliding Window Pipeline.
"""

import os
import sys
import tempfile
import unittest
import pandas as pd

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
from degradation.dataset import DatasetBuilder


class TestDatasetBuilder(unittest.TestCase):

    def test_window_generation_and_export(self):
        config = DegradationConfig.healthy()
        runner = EngineRunner(dt=0.01, seed=42)
        inj = DegradationInjector(config=config, runner=runner, run_id="TEST_RUN_001")
        telemetry_list, gt_list = inj.run_simulation(duration_seconds=7.0)

        with tempfile.TemporaryDirectory() as tmp_dir:
            builder = DatasetBuilder(output_dir=tmp_dir)
            raw_csv, win_csv = builder.export_run_dataset(
                telemetry_list, gt_list, inj.run_ground_truth, subfolder="healthy"
            )

            self.assertTrue(os.path.exists(raw_csv))
            self.assertTrue(os.path.exists(win_csv))

            df_win = pd.read_csv(win_csv)
            self.assertGreater(len(df_win), 0)
            self.assertIn("rpm_mean", df_win.columns)
            self.assertIn("cht_mean", df_win.columns)
            self.assertIn("degradation_type", df_win.columns)

            # Strict Anti-Leakage Rule Verification:
            # Physical degradation configuration parameters (combustion_efficiency, etc.)
            # must NEVER be included as ML telemetry input features
            self.assertNotIn("combustion_efficiency", df_win.columns)
            self.assertNotIn("bearing_friction_multiplier", df_win.columns)


if __name__ == "__main__":
    unittest.main()
