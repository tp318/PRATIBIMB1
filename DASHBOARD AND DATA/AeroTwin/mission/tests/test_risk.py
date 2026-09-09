"""
Unit tests for mission risk assessment and dispatch recommendation.
"""

import os
import sys
import unittest

_test_dir = os.path.dirname(os.path.abspath(__file__))
_mission_dir = os.path.dirname(_test_dir)
_aerotwin_dir = os.path.dirname(_mission_dir)
_root_dir = os.path.dirname(_aerotwin_dir)

for _p in [_mission_dir, _aerotwin_dir, _root_dir]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from mission.risk import (
    HEALTH_NO_GO,
    MissionProfile,
    MissionRiskAssessor,
    Recommendation,
    RiskBand,
)


class TestMissionRisk(unittest.TestCase):

    def setUp(self):
        self.assessor = MissionRiskAssessor()
        self.mission = MissionProfile(name="ISR", required_duration_s=100.0)

    def test_weights_must_sum_to_one(self):
        with self.assertRaises(ValueError):
            MissionRiskAssessor(w_health=0.5, w_rul=0.5, w_fault=0.5, w_anomaly=0.5)

    def test_healthy_engine_is_cleared(self):
        a = self.assessor.assess(self.mission, health_index=0.99)
        self.assertEqual(a.recommendation, Recommendation.GO)
        self.assertEqual(a.risk_band, RiskBand.LOW)
        self.assertTrue(any("nominal" in r.lower() for r in a.reasons))

    def test_health_below_floor_is_hard_no_go(self):
        """A weighted average must never be able to average away a hard limit."""
        a = self.assessor.assess(
            self.mission, health_index=HEALTH_NO_GO - 0.01, rul_seconds=1e9,
            rul_lower_seconds=1e9, predicted_fault="HEALTHY", fault_confidence=0.0,
        )
        self.assertEqual(a.recommendation, Recommendation.NO_GO)

    def test_insufficient_rul_triggers_abort_or_shorten(self):
        a = self.assessor.assess(
            self.mission, health_index=0.80, rul_seconds=60.0, rul_lower_seconds=40.0
        )
        self.assertFalse(a.mission_coverable)
        self.assertEqual(a.recommendation, Recommendation.ABORT_OR_SHORTEN)

    def test_sufficient_rul_covers_mission(self):
        a = self.assessor.assess(
            self.mission, health_index=0.95, rul_seconds=400.0, rul_lower_seconds=300.0
        )
        self.assertTrue(a.mission_coverable)

    def test_decision_uses_pessimistic_bound_not_point_estimate(self):
        """Planning on the optimistic edge of an interval ignores the uncertainty."""
        a = self.assessor.assess(
            self.mission, health_index=0.90, rul_seconds=10000.0, rul_lower_seconds=50.0
        )
        self.assertFalse(a.mission_coverable)

    def test_unknown_rul_is_not_treated_as_failure(self):
        a = self.assessor.assess(self.mission, health_index=0.97, rul_seconds=None)
        self.assertIsNone(a.mission_coverable)
        self.assertEqual(a.contributions["rul"], 0.0)

    def test_abrupt_fault_families_carry_more_risk(self):
        """Oil starvation ends a flight faster than a slowly fouling cylinder."""
        lub = self.assessor.assess(
            self.mission, health_index=0.75, predicted_fault="LUBRICATION", fault_confidence=0.9
        )
        cyl = self.assessor.assess(
            self.mission, health_index=0.75, predicted_fault="CYLINDER", fault_confidence=0.9
        )
        self.assertGreater(lub.risk_score, cyl.risk_score)

    def test_anomaly_flag_raises_risk_and_is_explained(self):
        clean = self.assessor.assess(self.mission, health_index=0.9)
        flagged = self.assessor.assess(
            self.mission, health_index=0.9, anomaly_flagged=True, anomaly_score_ratio=1.4
        )
        self.assertGreater(flagged.risk_score, clean.risk_score)
        self.assertTrue(any("anomaly" in r.lower() for r in flagged.reasons))

    def test_every_assessment_is_explained(self):
        a = self.assessor.assess(self.mission, health_index=0.5, predicted_fault="BEARING",
                                 fault_confidence=0.8)
        self.assertGreater(len(a.reasons), 0)
        self.assertEqual(set(a.contributions), {"health", "rul", "fault", "anomaly"})

    def test_risk_score_stays_in_unit_range(self):
        worst = self.assessor.assess(
            self.mission, health_index=0.0, rul_seconds=0.0, rul_lower_seconds=0.0,
            predicted_fault="LUBRICATION", fault_confidence=1.0,
            anomaly_flagged=True, anomaly_score_ratio=99.0,
        )
        self.assertGreaterEqual(worst.risk_score, 0.0)
        self.assertLessEqual(worst.risk_score, 1.0)
        self.assertEqual(worst.recommendation, Recommendation.NO_GO)

    def test_reserve_is_included_in_requirement(self):
        m = MissionProfile(required_duration_s=100.0, reserve_duration_s=50.0)
        self.assertEqual(m.total_required_s, 150.0)
        a = self.assessor.assess(m, health_index=0.9, rul_seconds=200.0, rul_lower_seconds=200.0)
        self.assertFalse(a.mission_coverable)  # needs 150 * 1.5 = 225s

    def test_max_safe_duration(self):
        self.assertAlmostEqual(self.assessor.max_safe_duration_s(300.0), 200.0)
        self.assertIsNone(self.assessor.max_safe_duration_s(None))

    def test_to_dict_is_serialisable(self):
        import json

        a = self.assessor.assess(self.mission, health_index=0.6, predicted_fault="COOLING",
                                 fault_confidence=0.7, rul_seconds=90.0, rul_lower_seconds=60.0)
        json.dumps(a.to_dict())


if __name__ == "__main__":
    unittest.main()
