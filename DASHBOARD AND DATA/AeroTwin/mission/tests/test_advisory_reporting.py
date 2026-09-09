"""
Unit tests for maintenance advisory, efficiency tracking, alert log and reporting.
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

from mission.advisory import ACTIONS_BY_FAMILY, MaintenanceAdvisor, Priority
from mission.reporting import (
    AlertLog,
    EfficiencyTracker,
    MissionReport,
    Severity,
    bsfc,
    shaft_power_kw,
)


class TestMaintenanceAdvisory(unittest.TestCase):

    def setUp(self):
        self.advisor = MaintenanceAdvisor()

    def test_healthy_engine_gets_a_no_action_advisory(self):
        """The panel must never be empty; an empty list reads as a failure."""
        items = self.advisor.advise(predicted_fault="HEALTHY", health_index=0.99)
        self.assertGreater(len(items), 0)
        self.assertTrue(all(i.priority is Priority.ROUTINE for i in items))

    def test_every_modelled_family_has_actions(self):
        for fam in ("CYLINDER", "BEARING", "COOLING", "LUBRICATION"):
            with self.subTest(family=fam):
                items = self.advisor.advise(predicted_fault=fam, fault_confidence=0.9, health_index=0.6)
                self.assertGreater(len(items), 0)
                self.assertTrue(all(i.action for i in items))
                self.assertTrue(all(i.rationale for i in items))

    def test_low_health_escalates_to_immediate(self):
        items = self.advisor.advise(predicted_fault="BEARING", fault_confidence=0.9, health_index=0.20)
        self.assertEqual(items[0].priority, Priority.IMMEDIATE)

    def test_rul_shorter_than_mission_escalates(self):
        """Remaining life below the next sortie is a before-flight item."""
        items = self.advisor.advise(
            predicted_fault="COOLING", fault_confidence=0.9, health_index=0.85,
            rul_seconds=100.0, mission_required_s=600.0,
        )
        self.assertEqual(items[0].priority, Priority.IMMEDIATE)

    def test_weak_attribution_recommends_confirmation_first(self):
        items = self.advisor.advise(predicted_fault="BEARING", fault_confidence=0.20, health_index=0.8)
        joined = " ".join(i.action for i in items).lower()
        self.assertIn("confirm", joined)

    def test_advisory_is_sorted_by_priority(self):
        from mission.advisory import PRIORITY_ORDER

        items = self.advisor.advise(predicted_fault="CYLINDER", fault_confidence=0.9, health_index=0.5)
        ranks = [PRIORITY_ORDER[i.priority] for i in items]
        self.assertEqual(ranks, sorted(ranks))

    def test_anomaly_without_attribution_still_advises_review(self):
        items = self.advisor.advise(predicted_fault="HEALTHY", health_index=0.95, anomaly_flagged=True)
        self.assertIn("review", " ".join(i.action for i in items).lower())

    def test_items_serialise(self):
        import json

        items = self.advisor.advise(predicted_fault="LUBRICATION", fault_confidence=0.8, health_index=0.5)
        json.dumps([i.to_dict() for i in items])


class TestEfficiency(unittest.TestCase):

    def test_shaft_power_formula(self):
        # 100 Nm at 3000 rpm -> 100 * 314.159 / 1000 = 31.416 kW
        self.assertAlmostEqual(shaft_power_kw(100.0, 3000.0), 31.4159, places=3)

    def test_power_is_none_without_inputs(self):
        self.assertIsNone(shaft_power_kw(None, 3000.0))
        self.assertIsNone(shaft_power_kw(100.0, None))

    def test_bsfc_suppressed_at_negligible_power(self):
        """Below idle power the ratio is dominated by fixed fuel flow, not efficiency."""
        self.assertIsNone(bsfc(0.001, 0.01))

    def test_bsfc_formula(self):
        # 0.001 kg/s at 10 kW -> 0.001*3600/10 = 0.36 kg/kWh
        self.assertAlmostEqual(bsfc(0.001, 10.0), 0.36, places=6)

    def test_identical_observed_and_expected_gives_zero_deviation(self):
        tr = EfficiencyTracker()
        obs = {"mean_torque": 80.0, "rpm": 2500.0, "fuel_flow": 0.0012, "fuel_flow_lph": 6.0}
        s = tr.update(obs, dict(obs), 1.0)
        self.assertAlmostEqual(s.power_deficit_pct, 0.0, places=9)
        self.assertAlmostEqual(s.bsfc_penalty_pct, 0.0, places=9)

    def test_power_loss_reports_positive_deficit(self):
        tr = EfficiencyTracker()
        obs = {"mean_torque": 72.0, "rpm": 2500.0, "fuel_flow": 0.0012}
        exp = {"mean_torque": 80.0, "rpm": 2500.0, "fuel_flow": 0.0012}
        s = tr.update(obs, exp, 1.0)
        self.assertAlmostEqual(s.power_deficit_pct, 10.0, places=6)
        # Same fuel for less work must show as a fuel penalty.
        self.assertGreater(s.bsfc_penalty_pct, 0.0)

    def test_summary_reports_mean_and_peak(self):
        tr = EfficiencyTracker()
        for tq in (80.0, 76.0, 72.0):
            tr.update(
                {"mean_torque": tq, "rpm": 2500.0, "fuel_flow": 0.0012},
                {"mean_torque": 80.0, "rpm": 2500.0, "fuel_flow": 0.0012},
                1.0,
            )
        s = tr.summary()
        self.assertEqual(s["n_samples"], 3)
        self.assertAlmostEqual(s["peak_power_deficit_pct"], 10.0, places=6)

    def test_reset_clears_samples(self):
        tr = EfficiencyTracker()
        tr.update({"mean_torque": 80.0, "rpm": 2500.0}, {"mean_torque": 80.0, "rpm": 2500.0}, 1.0)
        tr.reset()
        self.assertEqual(tr.summary()["n_samples"], 0)


def _assessment(sim_time=1.0, flagged=False, fault="HEALTHY", conf=0.9,
                rec="GO", band="LOW", health=0.99, decaying=False):
    return {
        "simulation_time": sim_time,
        "anomaly": {"score": 12.0 if flagged else 1.0, "threshold": 5.0, "flagged": flagged},
        "diagnosis": {"predicted_fault": fault, "confidence": conf, "runner_up": "BEARING", "margin": 0.5},
        "health": {"health_index": health},
        "rul": {"is_decaying": decaying, "rul_seconds": 120.0 if decaying else None, "confidence": 0.8},
        "mission_risk": {"recommendation": rec, "risk_band": band, "risk_score": 0.1},
    }


class TestAlertLog(unittest.TestCase):

    def test_alert_fires_on_transition_not_on_every_window(self):
        """Re-logging a persistent condition buries the moment it started."""
        log = AlertLog()
        log.evaluate(_assessment(1.0, flagged=True, fault="BEARING", rec="GO_WITH_MONITORING"))
        first = len(log.alerts)
        for i in range(5):
            log.evaluate(_assessment(2.0 + i, flagged=True, fault="BEARING", rec="GO_WITH_MONITORING"))
        self.assertEqual(len(log.alerts), first)

    def test_anomaly_assertion_and_clear_are_both_logged(self):
        log = AlertLog()
        log.evaluate(_assessment(1.0, flagged=False))
        log.evaluate(_assessment(2.0, flagged=True))
        log.evaluate(_assessment(3.0, flagged=False))
        msgs = [a.message for a in log.alerts]
        self.assertTrue(any("asserted" in m.lower() for m in msgs))
        self.assertTrue(any("cleared" in m.lower() for m in msgs))

    def test_fault_indication_is_logged_with_severity(self):
        log = AlertLog()
        log.evaluate(_assessment(1.0))
        log.evaluate(_assessment(2.0, fault="COOLING", conf=0.9))
        top = log.alerts[0]
        self.assertEqual(top.severity, Severity.WARNING)
        self.assertIn("Cooling", top.message)

    def test_low_confidence_fault_logs_as_caution(self):
        log = AlertLog()
        log.evaluate(_assessment(1.0))
        log.evaluate(_assessment(2.0, fault="COOLING", conf=0.4))
        self.assertEqual(log.alerts[0].severity, Severity.CAUTION)

    def test_health_band_crossing_is_logged_once(self):
        log = AlertLog()
        log.evaluate(_assessment(1.0, health=0.95))
        log.evaluate(_assessment(2.0, health=0.50))
        log.evaluate(_assessment(3.0, health=0.48))
        crossings = [a for a in log.alerts if a.category == "Health"]
        self.assertEqual(len(crossings), 1)

    def test_newest_alert_is_first(self):
        log = AlertLog()
        log.evaluate(_assessment(1.0, flagged=True))
        log.evaluate(_assessment(2.0, flagged=False))
        self.assertGreater(log.alerts[0].seq, log.alerts[1].seq)

    def test_counts_by_severity(self):
        log = AlertLog()
        log.evaluate(_assessment(1.0, flagged=True, fault="BEARING"))
        c = log.counts()
        self.assertEqual(set(c), {"INFO", "CAUTION", "WARNING"})
        self.assertGreaterEqual(sum(c.values()), 1)

    def test_reset_clears_the_log_and_state(self):
        log = AlertLog()
        log.evaluate(_assessment(1.0, flagged=True))
        log.reset()
        self.assertEqual(len(log.alerts), 0)


class TestMissionReport(unittest.TestCase):

    def test_tracks_health_start_end_and_minimum(self):
        r = MissionReport()
        r.reset("ISR", 600.0)
        for i, hv in enumerate([0.99, 0.60, 0.75]):
            r.update(_assessment(float(i + 1), health=hv))
        out = r.build()
        self.assertAlmostEqual(out["start_health"], 0.99)
        self.assertAlmostEqual(out["end_health"], 0.75)
        self.assertAlmostEqual(out["min_health"], 0.60)
        self.assertLess(out["health_change"], 0.0)

    def test_dominant_finding_ignores_healthy_windows(self):
        """A sortie that developed a fault must report the fault, not the majority."""
        r = MissionReport()
        r.reset()
        for i in range(8):
            r.update(_assessment(float(i + 1), fault="HEALTHY"))
        for i in range(3):
            r.update(_assessment(float(i + 9), fault="COOLING"))
        self.assertEqual(r.build()["dominant_finding"], "COOLING")

    def test_time_in_disposition_is_integrated_not_counted(self):
        r = MissionReport()
        r.reset()
        r.update(_assessment(0.0, rec="GO"))
        r.update(_assessment(10.0, rec="GO"))
        r.update(_assessment(20.0, rec="ABORT_OR_SHORTEN"))
        times = r.build()["time_in_recommendation_s"]
        self.assertAlmostEqual(times["GO"], 10.0)
        self.assertAlmostEqual(times["ABORT_OR_SHORTEN"], 10.0)

    def test_worst_recommendation_is_retained(self):
        r = MissionReport()
        r.reset()
        r.update(_assessment(1.0, rec="GO"))
        r.update(_assessment(2.0, rec="NO_GO"))
        r.update(_assessment(3.0, rec="GO"))
        self.assertEqual(r.build()["worst_recommendation"], "NO_GO")

    def test_anomaly_rate(self):
        r = MissionReport()
        r.reset()
        for i in range(4):
            r.update(_assessment(float(i + 1), flagged=(i < 1)))
        self.assertAlmostEqual(r.build()["anomaly_rate"], 0.25)

    def test_report_serialises(self):
        import json

        r = MissionReport()
        r.reset()
        r.update(_assessment(1.0))
        json.dumps(r.build(efficiency={"mean_power_kw": 1.0}, alert_counts={"INFO": 1}))


if __name__ == "__main__":
    unittest.main()
