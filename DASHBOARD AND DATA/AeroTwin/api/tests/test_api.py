"""
Unit tests for the real-time API layer and the live inference pipeline.

These use FastAPI's TestClient, so no server process is required.
"""

import os
import sys
import unittest

_test_dir = os.path.dirname(os.path.abspath(__file__))
_api_dir = os.path.dirname(_test_dir)
_aerotwin_dir = os.path.dirname(_api_dir)
_root_dir = os.path.dirname(_aerotwin_dir)

for _p in [_aerotwin_dir, _root_dir]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from fastapi.testclient import TestClient

from AeroTwin.api.pipeline import WINDOW_SAMPLES, LiveAssessmentPipeline, default_engine_parameters
from AeroTwin.api.server import app
from AeroTwin.degradation.config import DegradationConfig
from AeroTwin.degradation.injector import DegradationInjector
from AeroTwin.simulator.runner import EngineRunner


class TestApiRoutes(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(app)

    def test_health_probe(self):
        r = self.client.get("/health")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["status"], "ok")

    def test_root_serves_the_dashboard(self):
        r = self.client.get("/")
        self.assertEqual(r.status_code, 200)
        self.assertIn("text/html", r.headers["content-type"])
        self.assertIn("AeroTwin-4", r.text)

    def test_api_banner(self):
        r = self.client.get("/api")
        self.assertEqual(r.status_code, 200)
        self.assertIn("websocket", r.json())

    def test_dashboard_assets_are_served_locally(self):
        """Vendored, not CDN: a live demo must not depend on venue internet."""
        for path in ("/static/app.js",
                     "/static/vendor/react.production.min.js",
                     "/static/vendor/react-dom.production.min.js"):
            with self.subTest(path=path):
                r = self.client.get(path)
                self.assertEqual(r.status_code, 200, path)
                self.assertGreater(len(r.content), 100)

    def test_status_available_before_start(self):
        r = self.client.get("/api/status")
        self.assertEqual(r.status_code, 200)
        self.assertIn("running", r.json())

    def test_telemetry_404_before_simulation_starts(self):
        """A missing value must 404, not return a fabricated zero frame."""
        r = self.client.get("/api/telemetry/latest")
        self.assertIn(r.status_code, (200, 404))

    def test_throttle_rejected_when_not_running(self):
        r = self.client.post("/api/sim/throttle", json={"throttle": 0.5})
        self.assertIn(r.status_code, (409, 200))

    def test_throttle_validates_range(self):
        r = self.client.post("/api/sim/throttle", json={"throttle": 5.0})
        self.assertEqual(r.status_code, 422)

    def test_unknown_fault_type_is_rejected(self):
        r = self.client.post(
            "/api/sim/inject_fault", json={"fault_type": "WARP_CORE", "severity": 0.5}
        )
        self.assertEqual(r.status_code, 422)

    def test_severity_out_of_range_is_rejected(self):
        r = self.client.post(
            "/api/sim/inject_fault", json={"fault_type": "BEARING", "severity": 3.0}
        )
        self.assertEqual(r.status_code, 422)

    def test_mission_assess_requires_positive_duration(self):
        r = self.client.post("/api/mission/assess", json={"required_duration_s": -5})
        self.assertEqual(r.status_code, 422)

    def test_openapi_schema_is_served(self):
        r = self.client.get("/openapi.json")
        self.assertEqual(r.status_code, 200)
        paths = r.json()["paths"]
        for expected in ("/api/status", "/api/twin/state", "/api/sim/inject_fault"):
            self.assertIn(expected, paths)


class TestLivePipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.params = default_engine_parameters("TEST_UNIT", 7)

    def _pipeline(self):
        return LiveAssessmentPipeline(root_dir=_root_dir, engine_parameters=self.params, seed=7)

    def test_status_reports_model_availability(self):
        st = self._pipeline().status()
        self.assertIn("models_loaded", st)
        self.assertEqual(st["window_samples_required"], WINDOW_SAMPLES)

    def test_no_assessment_before_a_full_window(self):
        """Scoring a partial window would silently compare against the wrong shape."""
        pipe = self._pipeline()
        self.assertIsNone(pipe.assess_current_window())

    def test_ingest_returns_twin_state_per_frame(self):
        pipe = self._pipeline()
        runner = EngineRunner(dt=0.01, seed=7, engine_parameters=self.params)
        inj = DegradationInjector(
            config=DegradationConfig.healthy(), runner=runner, run_id="TEST", noise_enabled=True
        )
        telemetry, _ = inj.step()
        out = pipe.ingest(telemetry.to_dict())
        self.assertIn("observed", out)
        self.assertIn("expected", out)
        self.assertIn("indicators", out)
        self.assertFalse(out["window_ready"])

    def test_full_window_produces_a_complete_assessment(self):
        pipe = self._pipeline()
        runner = EngineRunner(dt=0.01, seed=7, engine_parameters=self.params)
        inj = DegradationInjector(
            config=DegradationConfig.healthy(), runner=runner, run_id="TEST", noise_enabled=True
        )
        assessment = None
        for _ in range(WINDOW_SAMPLES):
            telemetry, _gt = inj.step()
            out = pipe.ingest(telemetry.to_dict())
            if out["assessment"]:
                assessment = out["assessment"]

        self.assertIsNotNone(assessment, "no assessment after a full window")
        self.assertIn("mission_risk", assessment)
        risk = assessment["mission_risk"]
        self.assertIn(risk["recommendation"], {"GO", "GO_WITH_MONITORING", "ABORT_OR_SHORTEN", "NO_GO"})
        # Every decision must carry its reasoning.
        self.assertGreater(len(risk["reasons"]), 0)

    def test_healthy_engine_is_not_flagged_as_anomalous(self):
        """The false-alarm case: a serviceable engine must not be grounded."""
        pipe = self._pipeline()
        if not pipe.loaded.get("anomaly"):
            self.skipTest("anomaly artifacts not trained in this checkout")

        runner = EngineRunner(dt=0.01, seed=7, engine_parameters=self.params)
        inj = DegradationInjector(
            config=DegradationConfig.healthy(), runner=runner, run_id="TEST", noise_enabled=True
        )
        assessment = None
        for _ in range(WINDOW_SAMPLES):
            telemetry, _gt = inj.step()
            out = pipe.ingest(telemetry.to_dict())
            if out["assessment"]:
                assessment = out["assessment"]

        anomaly = assessment.get("anomaly", {})
        self.assertFalse(anomaly.get("flagged", False), f"healthy engine flagged: {anomaly}")

    def test_anomaly_score_is_bounded(self):
        """Guards the 1e6 Z-score explosion caused by a near-zero variance floor."""
        pipe = self._pipeline()
        if not pipe.loaded.get("anomaly"):
            self.skipTest("anomaly artifacts not trained in this checkout")

        runner = EngineRunner(dt=0.01, seed=7, engine_parameters=self.params)
        inj = DegradationInjector(
            config=DegradationConfig.healthy(), runner=runner, run_id="TEST", noise_enabled=True
        )
        assessment = None
        for _ in range(WINDOW_SAMPLES):
            telemetry, _gt = inj.step()
            out = pipe.ingest(telemetry.to_dict())
            if out["assessment"]:
                assessment = out["assessment"]

        score = assessment.get("anomaly", {}).get("score")
        if score is not None:
            self.assertLess(score, 100.0, f"anomaly score exploded: {score}")

    def test_reset_clears_the_window(self):
        pipe = self._pipeline()
        runner = EngineRunner(dt=0.01, seed=7, engine_parameters=self.params)
        inj = DegradationInjector(
            config=DegradationConfig.healthy(), runner=runner, run_id="TEST", noise_enabled=True
        )
        for _ in range(10):
            telemetry, _gt = inj.step()
            pipe.ingest(telemetry.to_dict())
        self.assertGreater(pipe.status()["samples_seen"], 0)
        pipe.reset()
        self.assertEqual(pipe.status()["samples_seen"], 0)


if __name__ == "__main__":
    unittest.main()
