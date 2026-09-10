"""
=============================================================================
postgres_db.py — PostgreSQL Mission Persistence & Replay Engine
=============================================================================
Provides high-throughput time-series recording and sub-millisecond replay
for the AeroTwin-4 / PRATIBIMB Digital Twin & Mission Simulation System.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

try:
    import psycopg2
    from psycopg2 import pool
    from psycopg2.extras import RealDictCursor, execute_values
    HAS_PSYCOPG2 = True
except ImportError:
    HAS_PSYCOPG2 = False
    pool = None
    RealDictCursor = None
    execute_values = None

logger = logging.getLogger("AeroTwin.PostgresDB")


def load_env_file(env_path: Optional[Path] = None) -> None:
    """Lightweight .env parser avoiding extra third-party dependencies."""
    if env_path is None:
        # Search parent directories for .env
        cur = Path(__file__).resolve().parent
        for _ in range(4):
            candidate = cur / ".env"
            if candidate.exists():
                env_path = candidate
                break
            cur = cur.parent

    if env_path and env_path.exists():
        try:
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, val = line.split("=", 1)
                    key = key.strip()
                    val = val.strip().strip("'\"")
                    if key not in os.environ:
                        os.environ[key] = val
        except Exception as e:
            logger.warning(f"Could not read .env at {env_path}: {e}")


class PostgresMissionDatabase:
    """
    PostgreSQL mission database manager.
    Handles master sortie manifests, 50 Hz telemetry time-series, and flight events.
    """

    def __init__(self, connection_url: Optional[str] = None) -> None:
        if not HAS_PSYCOPG2:
            raise ImportError(
                "psycopg2 is required for PostgresMissionDatabase. "
                "Install it via: pip install psycopg2-binary"
            )
        load_env_file()

        self.connection_url = connection_url or os.environ.get(
            "DATABASE_URL",
            f"postgresql://{os.environ.get('POSTGRES_USER', 'postgres')}:{os.environ.get('POSTGRES_PASSWORD', '')}@{os.environ.get('POSTGRES_HOST', 'localhost')}:{os.environ.get('POSTGRES_PORT', '5432')}/{os.environ.get('POSTGRES_DB', 'aerotwin')}"
        )
        self._pool: Optional[pool.SimpleConnectionPool] = None

    def connect(self) -> None:
        """Establishes connection pool and verifies connection."""
        if self._pool is not None:
            return

        try:
            self._pool = pool.SimpleConnectionPool(
                minconn=1,
                maxconn=10,
                dsn=self.connection_url,
            )
            logger.info("Successfully connected to PostgreSQL database.")
            self.init_schema()
        except psycopg2.OperationalError as e:
            msg = str(e)
            if "password authentication failed" in msg:
                raise ConnectionError(
                    "PostgreSQL password authentication failed. "
                    "Please set your password in the .env file: "
                    "DATABASE_URL=postgresql://postgres:YOUR_PASSWORD@localhost:5432/aerotwin"
                ) from e
            elif 'database "aerotwin" does not exist' in msg:
                raise ConnectionError(
                    "Database 'aerotwin' not found. Please create it in pgAdmin or via createdb."
                ) from e
            raise ConnectionError(f"Could not connect to PostgreSQL: {e}") from e

    def get_connection(self):
        """Borrow a connection from the pool."""
        if self._pool is None:
            self.connect()
        return self._pool.getconn()

    def release_connection(self, conn):
        """Return connection to pool."""
        if self._pool is not None and conn is not None:
            self._pool.putconn(conn)

    def init_schema(self) -> None:
        """Creates the tables and indexes required for mission persistence if not exists."""
        sql_schema = """
        -- 1. Master Sorties / Missions Table
        CREATE TABLE IF NOT EXISTS missions (
            mission_id VARCHAR(64) PRIMARY KEY,
            mission_name VARCHAR(128) NOT NULL,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            duration_s DOUBLE PRECISION DEFAULT 0.0,
            engine_id INTEGER DEFAULT 1,
            flight_profile VARCHAR(64) DEFAULT 'BENCHMARK',
            injected_fault VARCHAR(64) DEFAULT 'NONE',
            fault_severity DOUBLE PRECISION DEFAULT 0.0,
            disposition VARCHAR(32) DEFAULT 'PENDING',
            start_health DOUBLE PRECISION DEFAULT 1.0,
            end_health DOUBLE PRECISION DEFAULT 1.0,
            min_health DOUBLE PRECISION DEFAULT 1.0,
            dominant_fault VARCHAR(64) DEFAULT 'Normal',
            total_frames INTEGER DEFAULT 0,
            summary_json JSONB DEFAULT '{}'::jsonb
        );

        -- 2. Telemetry Frames Table (Time-Series)
        CREATE TABLE IF NOT EXISTS mission_telemetry (
            id BIGSERIAL PRIMARY KEY,
            mission_id VARCHAR(64) REFERENCES missions(mission_id) ON DELETE CASCADE,
            step INTEGER NOT NULL,
            timestamp_s DOUBLE PRECISION NOT NULL,
            rpm DOUBLE PRECISION,
            cht DOUBLE PRECISION,
            egt DOUBLE PRECISION,
            oil_pressure_psi DOUBLE PRECISION,
            oil_temperature DOUBLE PRECISION,
            fuel_flow_lph DOUBLE PRECISION,
            vibration DOUBLE PRECISION,
            m1_anomaly_score DOUBLE PRECISION,
            m2_predicted_fault VARCHAR(64),
            m3_rul_minutes DOUBLE PRECISION,
            residuals_json JSONB,
            indicators_json JSONB
        );

        -- Indexes for ultra-fast replay and querying
        CREATE INDEX IF NOT EXISTS idx_telemetry_mission_step
            ON mission_telemetry (mission_id, step ASC);

        CREATE INDEX IF NOT EXISTS idx_missions_created_at
            ON missions (created_at DESC);
        """
        conn = self.get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(sql_schema)
            conn.commit()
            logger.info("Database schema initialized and verified.")
        finally:
            self.release_connection(conn)

    def create_mission(
        self,
        mission_id: str,
        mission_name: str,
        flight_profile: str = "BENCHMARK",
        injected_fault: str = "NONE",
        fault_severity: float = 0.0,
        engine_id: int = 1,
    ) -> None:
        """Inserts a new sortie record into the database."""
        sql = """
        INSERT INTO missions (
            mission_id, mission_name, flight_profile, injected_fault,
            fault_severity, engine_id, disposition
        ) VALUES (%s, %s, %s, %s, %s, %s, 'RUNNING')
        ON CONFLICT (mission_id) DO UPDATE SET
            mission_name = EXCLUDED.mission_name,
            flight_profile = EXCLUDED.flight_profile,
            injected_fault = EXCLUDED.injected_fault,
            fault_severity = EXCLUDED.fault_severity,
            disposition = 'RUNNING';
        """
        conn = self.get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    sql,
                    (
                        mission_id,
                        mission_name,
                        flight_profile,
                        injected_fault,
                        fault_severity,
                        engine_id,
                    ),
                )
            conn.commit()
        finally:
            self.release_connection(conn)

    def insert_telemetry_batch(self, frames: List[Dict[str, Any]]) -> None:
        """High-speed batch insertion using psycopg2 execute_values."""
        if not frames:
            return

        sql = """
        INSERT INTO mission_telemetry (
            mission_id, step, timestamp_s, rpm, cht, egt,
            oil_pressure_psi, oil_temperature, fuel_flow_lph, vibration,
            m1_anomaly_score, m2_predicted_fault, m3_rul_minutes,
            residuals_json, indicators_json
        ) VALUES %s;
        """
        records = []
        for f in frames:
            records.append((
                f["mission_id"],
                f["step"],
                f["timestamp_s"],
                f.get("rpm"),
                f.get("cht"),
                f.get("egt"),
                f.get("oil_pressure_psi"),
                f.get("oil_temperature"),
                f.get("fuel_flow_lph"),
                f.get("vibration"),
                f.get("m1_anomaly_score"),
                f.get("m2_predicted_fault"),
                f.get("m3_rul_minutes"),
                json.dumps(f.get("residuals_json", {})),
                json.dumps(f.get("indicators_json", {})),
            ))

        conn = self.get_connection()
        try:
            with conn.cursor() as cur:
                execute_values(cur, sql, records, page_size=1000)
            conn.commit()
        finally:
            self.release_connection(conn)

    def finalize_mission(
        self,
        mission_id: str,
        duration_s: float,
        disposition: str,
        start_health: float,
        end_health: float,
        min_health: float,
        dominant_fault: str,
        total_frames: int,
        summary_dict: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Updates the mission record upon sortie completion."""
        sql = """
        UPDATE missions
        SET duration_s = %s,
            disposition = %s,
            start_health = %s,
            end_health = %s,
            min_health = %s,
            dominant_fault = %s,
            total_frames = %s,
            summary_json = %s
        WHERE mission_id = %s;
        """
        conn = self.get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    sql,
                    (
                        duration_s,
                        disposition,
                        start_health,
                        end_health,
                        min_health,
                        dominant_fault,
                        total_frames,
                        json.dumps(summary_dict or {}),
                        mission_id,
                    ),
                )
            conn.commit()
        finally:
            self.release_connection(conn)

    def list_missions(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Returns catalogue of recent recorded missions with actual telemetry row counts."""
        sql = """
        SELECT m.mission_id, m.mission_name, m.created_at, m.duration_s, m.engine_id,
               m.flight_profile, m.injected_fault, m.fault_severity, m.disposition,
               m.start_health, m.end_health, m.min_health, m.dominant_fault,
               m.total_frames,
               COUNT(t.id) AS actual_frames,
               m.summary_json
        FROM missions m
        LEFT JOIN mission_telemetry t ON t.mission_id = m.mission_id
        GROUP BY m.mission_id, m.mission_name, m.created_at, m.duration_s, m.engine_id,
                 m.flight_profile, m.injected_fault, m.fault_severity, m.disposition,
                 m.start_health, m.end_health, m.min_health, m.dominant_fault,
                 m.total_frames, m.summary_json
        ORDER BY m.created_at DESC
        LIMIT %s;
        """
        conn = self.get_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(sql, (limit,))
                rows = cur.fetchall()
            return [dict(r) for r in rows]
        finally:
            self.release_connection(conn)

    def get_mission(self, mission_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves single mission metadata and summary report."""
        sql = "SELECT * FROM missions WHERE mission_id = %s;"
        conn = self.get_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(sql, (mission_id,))
                row = cur.fetchone()
            return dict(row) if row else None
        finally:
            self.release_connection(conn)

    def stream_mission_telemetry(
        self, mission_id: str, chunk_size: int = 100
    ) -> Generator[Dict[str, Any], None, None]:
        """Streams telemetry frames sequentially ordered by step for live replay."""
        conn = self.get_connection()
        try:
            with conn.cursor(name=f"stream_{mission_id}", cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    "SELECT * FROM mission_telemetry WHERE mission_id = %s ORDER BY step ASC;",
                    (mission_id,),
                )
                while True:
                    rows = cur.fetchmany(chunk_size)
                    if not rows:
                        break
                    for r in rows:
                        yield dict(r)
        finally:
            self.release_connection(conn)

    def delete_mission(self, mission_id: str) -> bool:
        """Deletes a mission and its cascaded telemetry frames."""
        sql = "DELETE FROM missions WHERE mission_id = %s;"
        conn = self.get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(sql, (mission_id,))
                affected = cur.rowcount
            conn.commit()
            return affected > 0
        finally:
            self.release_connection(conn)
