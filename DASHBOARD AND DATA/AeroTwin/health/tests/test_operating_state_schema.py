"""
Unit tests for OperatingStateExtractor & Ground Truth Isolation.
"""

import os
import sys
import unittest

_test_dir = os.path.dirname(os.path.abspath(__file__))
_health_dir = os.path.dirname(_test_dir)
_aerotwin_dir = os.path.dirname(_health_dir)
_root_dir = os.path.dirname(_aerotwin_dir)

for _p in [_health_dir, _aerotwin_dir, _root_dir]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from health.operating_state import OperatingStateExtractor
from health.models import OperatingState


class TestOperatingStateSchema(unittest.TestCase):

    def test_operating_state_extraction(self):
        sample_telemetry = {
            "timestamp": 12.5,
            "simulation_time": 12.5,
            "engine_id": "AEROTWIN-TEST-001",
            "operating_mode": "CRUISE",
            "throttle": 0.65,
            "rpm": 2750.0,
            # Ground truth fields (must NOT be passed to OperatingState)
            "gt_degradation_type": "CYLINDER",
            "gt_severity": 0.50,
        }

        op_state = OperatingStateExtractor.extract(sample_telemetry)
        self.assertIsInstance(op_state, OperatingState)
        self.assertEqual(op_state.timestamp, 12.5)
        self.assertEqual(op_state.operating_mode, "CRUISE")
        self.assertEqual(op_state.throttle, 0.65)
        self.assertEqual(op_state.rpm, 2750.0)

        # Ensure ground truth fields are NOT attributes of OperatingState
        self.assertFalse(hasattr(op_state, "gt_degradation_type"))
        self.assertFalse(hasattr(op_state, "gt_severity"))


if __name__ == "__main__":
    unittest.main()
