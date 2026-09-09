"""
Unit tests for AeroTwin-4 Degradation Configuration.
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

from degradation.config import DegradationConfig, ComponentDegradation, DegradationType, ComponentID, SeverityLevel


class TestDegradationConfig(unittest.TestCase):

    def test_healthy_config(self):
        config = DegradationConfig.healthy()
        self.assertEqual(len(config.degradation_list), 1)
        self.assertEqual(config.degradation_list[0].degradation_type, DegradationType.NONE)
        self.assertEqual(config.degradation_list[0].severity, 0.0)

    def test_single_fault_config(self):
        config = DegradationConfig.single_fault(
            degradation_type=DegradationType.CYLINDER,
            component_id=ComponentID.CYLINDER_3,
            severity=0.40,
        )
        self.assertEqual(len(config.degradation_list), 1)
        self.assertEqual(config.degradation_list[0].degradation_type, DegradationType.CYLINDER)
        self.assertEqual(config.degradation_list[0].component_id, ComponentID.CYLINDER_3)
        self.assertEqual(config.degradation_list[0].severity, 0.40)

    def test_invalid_severity_raises(self):
        with self.assertRaises(ValueError):
            ComponentDegradation(
                degradation_type=DegradationType.BEARING,
                component_id=ComponentID.BEARING,
                severity=1.5,
            )


if __name__ == "__main__":
    unittest.main()
