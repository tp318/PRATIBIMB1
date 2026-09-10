"""
=============================================================================
init_db.py — CLI Database Initializer for PostgreSQL & SQLite
=============================================================================
Usage:
    python init_db.py --type sqlite
    python init_db.py --type postgres --password YOUR_PASSWORD
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Add REPLAY AND SIMULATION to sys.path
_PKG_DIR = Path(__file__).resolve().parent.parent
if str(_PKG_DIR) not in sys.path:
    sys.path.insert(0, str(_PKG_DIR))

from database.postgres_db import HAS_PSYCOPG2, PostgresMissionDatabase, load_env_file
from database.sqlite_db import SqliteMissionDatabase


def main():
    parser = argparse.ArgumentParser(description="Initialize AeroTwin Mission Database")
    parser.add_argument(
        "--type", "-t", choices=["postgres", "sqlite", "auto"], default="auto",
        help="Database engine to initialize (default: auto)"
    )
    parser.add_argument("--password", "-p", help="PostgreSQL password", default=None)
    parser.add_argument("--host", default="localhost", help="PostgreSQL host (default: localhost)")
    parser.add_argument("--port", default="5432", help="PostgreSQL port (default: 5432)")
    parser.add_argument("--user", "-u", default="postgres", help="PostgreSQL user (default: postgres)")
    parser.add_argument("--db", "-d", default="aerotwin", help="Database name (default: aerotwin)")
    parser.add_argument("--sqlite-path", help="Path to SQLite db file", default=None)
    args = parser.parse_args()

    load_env_file()

    chosen_type = args.type
    if chosen_type == "auto":
        chosen_type = "postgres" if (HAS_PSYCOPG2 and ("DATABASE_URL" in os.environ or args.password)) else "sqlite"

    print(f"[*] Initializing {chosen_type.upper()} database...")

    if chosen_type == "postgres":
        if not HAS_PSYCOPG2:
            print("[!] psycopg2 is not installed. Run: pip install psycopg2-binary")
            sys.exit(1)

        password = args.password or os.environ.get("POSTGRES_PASSWORD", "")
        conn_url = os.environ.get(
            "DATABASE_URL",
            f"postgresql://{args.user}:{password}@{args.host}:{args.port}/{args.db}"
        )
        try:
            db = PostgresMissionDatabase(connection_url=conn_url)
            db.connect()
            print("[OK] PostgreSQL Schema initialized successfully!")
            _run_smoke_test(db)
        except Exception as e:
            print(f"[!] PostgreSQL initialization error: {e}")
            sys.exit(1)

    elif chosen_type == "sqlite":
        try:
            db = SqliteMissionDatabase(db_path=args.sqlite_path)
            print(f"[OK] SQLite database initialized at: {db.db_path}")
            _run_smoke_test(db)
        except Exception as e:
            print(f"[!] SQLite initialization error: {e}")
            sys.exit(1)


def _run_smoke_test(db):
    """Smoke test read/write on newly initialized database."""
    test_id = "SMOKE_TEST_001"
    db.create_mission(
        mission_id=test_id,
        mission_name="Database Verification Sortie",
        flight_profile="BENCHMARK",
        injected_fault="NONE",
    )
    sample_frame = [{
        "mission_id": test_id,
        "step": 0,
        "timestamp_s": 0.0,
        "rpm": 2450.0,
        "cht": 92.5,
        "egt": 710.0,
        "oil_pressure_psi": 52.0,
        "oil_temperature": 82.0,
        "fuel_flow_lph": 26.5,
        "vibration": 1.1,
        "m1_anomaly_score": 0.015,
        "m2_predicted_fault": "Normal",
        "m3_rul_minutes": 45.0,
    }]
    db.insert_telemetry_batch(sample_frame)
    missions = db.list_missions(limit=5)
    print(f"[OK] Smoke test passed: Retrieved {len(missions)} mission(s) successfully.")


if __name__ == "__main__":
    main()
