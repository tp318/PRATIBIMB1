"""
test_packaged_suite.py — Standalone Test Suite for REPLAY AND SIMULATION
========================================================================
Validates all sub-modules in complete isolation:
  - EnvironmentalSimulator (ISA atmospheric physics)
  - MissionSimulator (GO/NO-GO clearance rules)
  - Mission Scenario Presets & Throttle Curves
  - FlightReplayer (streaming & synthetic generation)
  - SqliteMissionDatabase (complete time-series read/write/stream lifecycle)
  - Database Factory (auto fallback logic)
"""

import sys
import tempfile
from pathlib import Path
import pytest

# Add package root to sys.path
_PKG_DIR = Path(__file__).resolve().parent.parent
if str(_PKG_DIR) not in sys.path:
    sys.path.insert(0, str(_PKG_DIR))

from env_simulator import EnvironmentalSimulator
from mission_engine import MissionSimulator, ClearanceStatus
from mission_scenario import (
    build_scenario,
    list_presets,
    ScenarioType,
    ThrottleProfile,
)
from flight_replay import FlightReplayer, FlightSample
from database import (
    SqliteMissionDatabase,
    get_mission_database,
)


def test_environmental_simulator():
    """Verify ISA temperature lapse, pressure, and density ratios."""
    env = EnvironmentalSimulator(sea_level_temp_c=15.0)

    # Sea level
    sl = env.get_atmosphere(0.0)
    assert abs(sl.ambient_temp_c - 15.0) < 0.1
    assert abs(sl.pressure_hpa - 1013.25) < 1.0
    assert abs(sl.density_ratio - 1.0) < 0.01

    # 5,000m cruise
    fl160 = env.get_atmosphere(5000.0)
    # T = 15 - 0.0065 * 5000 = -17.5°C
    assert abs(fl160.ambient_temp_c - (-17.5)) < 0.5
    assert 530.0 < fl160.pressure_hpa < 550.0
    assert 0.55 < fl160.density_ratio < 0.65


def test_mission_clearance_scenarios():
    """Verify defense-grade GO, CAUTION_GO, and NO-GO clearance decisions."""
    msim = MissionSimulator()

    # 1. Healthy sortie -> GO
    healthy_go = msim.evaluate_clearance(
        profile_name="BORDER_PATROL",
        predicted_rul_hours=0.20,
        composite_health_index=0.98,
        active_fault_name="Normal",
    )
    assert healthy_go.status == ClearanceStatus.GO
    assert healthy_go.risk_score < 0.20

    # 2. Critical fault -> NO_GO
    fault_nogo = msim.evaluate_clearance(
        profile_name="ISR_SURVEILLANCE",
        predicted_rul_hours=0.15,
        composite_health_index=0.75,
        active_fault_name="Overheating",
        fault_confidence_pct=95.0,
    )
    assert fault_nogo.status == ClearanceStatus.NO_GO
    assert fault_nogo.risk_score >= 0.50

    # 3. Insufficient RUL -> NO_GO
    short_rul = msim.evaluate_clearance(
        profile_name="HIGH_ALT_RECON",
        predicted_rul_hours=0.02,  # 1.2 min vs 10 min required
        composite_health_index=0.90,
    )
    assert short_rul.status == ClearanceStatus.NO_GO
    assert any("INSUFFICIENT RUL" in r for r in short_rul.reasons)


def test_mission_scenario_presets():
    """Verify standard mission presets and throttle profiles."""
    presets = list_presets()
    assert len(presets) >= 4

    # High Altitude
    cfg_ha = build_scenario("HIGH_ALTITUDE")
    assert cfg_ha.altitude_m == 7500.0
    assert cfg_ha.throttle_at(10.0) == 0.70

    # Rapid Throttle Step
    cfg_step = build_scenario("RAPID_THROTTLE")
    assert cfg_step.throttle_profile == ThrottleProfile.STEP
    assert cfg_step.throttle_at(10.0) == 0.20
    assert cfg_step.throttle_at(35.0) == 1.00


def test_flight_replayer_streaming():
    """Verify streaming samples from flight replayer (with fallback)."""
    replayer = FlightReplayer()
    samples = []
    for s in replayer.stream_engine(engine_id=1, max_steps=15):
        samples.append(s)

    assert len(samples) == 15
    assert isinstance(samples[0], FlightSample)
    assert samples[0].engine_id == 1
    assert samples[0].step == 0
    assert samples[14].step == 14
    assert "rpm_residual" in samples[0].residuals
    assert "cht_residual" in samples[0].residuals
    assert "oil_pressure_residual" in samples[0].residuals


def test_sqlite_mission_database_lifecycle():
    """Verify full CRUD, batch insert, and stream replay in SQLite database."""
    with tempfile.NamedTemporaryFile(suffix=".sqlite3", delete=False) as tf:
        temp_db_path = tf.name

    try:
        db = SqliteMissionDatabase(db_path=temp_db_path)
        mission_id = "TEST_SORTIE_99"

        # 1. Create mission
        db.create_mission(
            mission_id=mission_id,
            mission_name="Unit Test Sortie",
            flight_profile="ENDURANCE",
            injected_fault="LUBRICATION",
            fault_severity=0.4,
            engine_id=3,
        )

        # 2. Insert batch telemetry
        frames = [
            {
                "mission_id": mission_id,
                "step": i,
                "timestamp_s": i * 0.1,
                "rpm": 2500.0 + i,
                "cht": 90.0,
                "egt": 710.0,
                "oil_pressure_psi": 50.0 - (i * 0.1),
                "oil_temperature": 80.0,
                "fuel_flow_lph": 25.0,
                "vibration": 1.0,
                "m1_anomaly_score": 0.05,
                "m2_predicted_fault": "Normal",
                "m3_rul_minutes": 30.0,
            }
            for i in range(25)
        ]
        db.insert_telemetry_batch(frames)

        # 3. Finalize mission
        db.finalize_mission(
            mission_id=mission_id,
            duration_s=2.5,
            disposition="CAUTION_GO",
            start_health=1.0,
            end_health=0.85,
            min_health=0.85,
            dominant_fault="LUBRICATION",
            total_frames=25,
            summary_dict={"notes": "All checks passed"},
        )

        # 4. Read mission record
        m = db.get_mission(mission_id)
        assert m is not None
        assert m["mission_id"] == mission_id
        assert m["disposition"] == "CAUTION_GO"
        assert m["end_health"] == 0.85

        # 5. Stream telemetry
        replayed = list(db.stream_mission_telemetry(mission_id, chunk_size=10))
        assert len(replayed) == 25
        assert replayed[0]["step"] == 0
        assert replayed[24]["step"] == 24

        # 6. Delete mission
        deleted = db.delete_mission(mission_id)
        assert deleted is True
        assert db.get_mission(mission_id) is None

    finally:
        try:
            Path(temp_db_path).unlink(missing_ok=True)
        except Exception:
            pass


def test_database_factory_fallback():
    """Verify that get_mission_database gracefully creates SQLite when prefer_sqlite=True."""
    db = get_mission_database(prefer_sqlite=True, force_new=True)
    assert isinstance(db, SqliteMissionDatabase)
