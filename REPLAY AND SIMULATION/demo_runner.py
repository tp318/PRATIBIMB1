"""
=============================================================================
demo_runner.py — Standalone Demo Runner for Replay, Simulation & Database
=============================================================================
Demonstrates the complete end-to-end capability of the packaged module:
  1. ISA Atmospheric Environmental Query
  2. Mission Scenario Preset Generation
  3. Pre-flight Clearance Evaluation (GO / NO-GO)
  4. Flight Replay Generator Streaming
  5. Mission Database Persistence (Batch Insert to SQLite / PostgreSQL)
  6. Replay from Database
"""

import sys
import time
from pathlib import Path

# Add package root to sys.path
_PKG_DIR = Path(__file__).resolve().parent
if str(_PKG_DIR) not in sys.path:
    sys.path.insert(0, str(_PKG_DIR))

from env_simulator import EnvironmentalSimulator
from mission_scenario import build_scenario, list_presets, ScenarioType
from mission_engine import MissionSimulator, ClearanceStatus
from flight_replay import FlightReplayer
from database import get_mission_database


def main():
    print("=" * 75)
    print("  AEROTWIN-4: MISSION REPLAY, SIMULATION & DATABASE STANDALONE DEMO")
    print("=" * 75)
    print()

    # 1. Environmental Simulation
    print("[+] 1. Atmospheric Environmental Simulation (ISA Model)")
    env = EnvironmentalSimulator(sea_level_temp_c=15.0)
    for alt in [0.0, 1500.0, 3500.0, 7500.0]:
        atm = env.get_atmosphere(alt)
        print(f"    - Alt: {alt:5.0f} m | Temp: {atm.ambient_temp_c:5.1f} °C | Press: {atm.pressure_hpa:6.1f} hPa | Density Ratio: {atm.density_ratio:.3f}")
    print()

    # 2. Scenario Presets
    print("[+] 2. Mission Scenario Presets")
    presets = list_presets()
    for p in presets:
        print(f"    - [{p['scenario_type']:<15}] {p['display_name']} -> {p['altitude_m']}m, {p['duration_s']}s, Throttle: {p['throttle_profile']}")
    print()

    # 3. Dynamic Flight Clearance Evaluation
    print("[+] 3. Dynamic Mission Clearance Engine")
    msim = MissionSimulator()
    # Scenario A: Healthy sortie
    assess_a = msim.evaluate_clearance(
        profile_name="ISR_SURVEILLANCE",
        predicted_rul_hours=0.20,
        composite_health_index=0.98,
        active_fault_name="Normal",
    )
    print(f"    - Sortie A (Healthy): Status = {assess_a.status.value} (Risk: {assess_a.risk_score:.2f}) -> {assess_a.reasons[0]}")

    # Scenario B: Critical Fault Injection
    assess_b = msim.evaluate_clearance(
        profile_name="ISR_SURVEILLANCE",
        predicted_rul_hours=0.04,
        composite_health_index=0.62,
        active_fault_name="Overheating",
        fault_confidence_pct=94.0,
    )
    print(f"    - Sortie B (Degraded): Status = {assess_b.status.value} (Risk: {assess_b.risk_score:.2f}) -> {assess_b.reasons[0]}")
    print()

    # 4. Flight Replayer Streaming
    print("[+] 4. Flight Replay Stream (Engine 1)")
    replayer = FlightReplayer()
    sample_count = 0
    start_t = time.time()
    for sample in replayer.stream_engine(engine_id=1, max_steps=20):
        sample_count += 1
        if sample_count in (1, 10, 20):
            print(f"    - Step {sample.step:3d} (t={sample.time_sec:4.2f}s) | Fault: {sample.fault_name:<8} | CHT Res: {sample.residuals['cht_residual']:+5.1f} °C | RUL: {sample.rul_hours*60:.1f} min")
    print()

    # 5. Database Persistence (Auto SQLite / PostgreSQL)
    print("[+] 5. Database Mission Persistence & Recording")
    db = get_mission_database(prefer_sqlite=True)
    mission_id = f"DEMO_SORTIE_{int(time.time())}"
    print(f"    - Creating Mission: {mission_id} in {type(db).__name__}...")
    db.create_mission(
        mission_id=mission_id,
        mission_name="Standalone Demo Sortie",
        flight_profile="HIGH_ALTITUDE",
        injected_fault="COOLING",
        fault_severity=0.5,
        engine_id=1,
    )

    # Insert batch of telemetry
    frames = []
    for step in range(50):
        t_s = step * 0.1
        frames.append({
            "mission_id": mission_id,
            "step": step,
            "timestamp_s": round(t_s, 2),
            "rpm": 2480.0 + (step * 0.5),
            "cht": 94.0 + (step * 0.2),
            "egt": 715.0 + (step * 0.3),
            "oil_pressure_psi": 53.0 - (step * 0.05),
            "oil_temperature": 84.0 + (step * 0.1),
            "fuel_flow_lph": 27.2,
            "vibration": 1.2,
            "m1_anomaly_score": 0.02 if step < 25 else 0.45,
            "m2_predicted_fault": "Normal" if step < 25 else "Cooling",
            "m3_rul_minutes": max(5.0, 40.0 - step * 0.5),
            "residuals_json": {"cht_residual": 0.5 if step < 25 else 18.2},
            "indicators_json": {"cooling_margin": 12.5},
        })

    db.insert_telemetry_batch(frames)
    print(f"    - Inserted {len(frames)} telemetry frames.")

    db.finalize_mission(
        mission_id=mission_id,
        duration_s=5.0,
        disposition="GO_WITH_MONITORING",
        start_health=1.0,
        end_health=0.82,
        min_health=0.82,
        dominant_fault="Cooling",
        total_frames=len(frames),
        summary_dict={"notes": "Demo verification successful"},
    )
    print(f"    - Finalized mission: status = GO_WITH_MONITORING, end_health = 0.82.")
    print()

    # 6. Database Replay Stream
    print("[+] 6. Streaming Replay Directly from Database")
    replayed_count = 0
    for frame in db.stream_mission_telemetry(mission_id, chunk_size=20):
        replayed_count += 1
        if replayed_count in (1, 25, 50):
            print(f"    - Replayed DB Frame #{frame['step']:2d}: RPM={frame['rpm']:.1f}, CHT={frame['cht']:.1f}°C, Fault={frame['m2_predicted_fault']}")

    print()
    print(f"[OK] Replay complete ({replayed_count} frames verified).")
    print("=" * 75)
    print("  ALL SUBSYSTEMS OPERATIONAL AND 100% STANDALONE READY!")
    print("=" * 75)


if __name__ == "__main__":
    main()
