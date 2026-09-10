"""
=============================================================================
sqlite_db.py — Zero-Config SQLite Mission Persistence & Replay Engine
=============================================================================
Provides a drop-in, zero-dependency alternative to PostgreSQL for local
testing, standalone deployment, or lightweight environments.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

logger = logging.getLogger("AeroTwin.SqliteDB")

DEFAULT_DB_PATH = Path(__file__).resolve().parent / "aerotwin_missions.sqlite3"


class SqliteMissionDatabase:
    """
    SQLite mission database manager implementing the exact same interface
    as PostgresMissionDatabase for maximum cross-compatibility.
    """

    def __init__(self, db_path: Optional[str | Path] = None) -> None:
        self.db_path = str(db_path or DEFAULT_DB_PATH)
        self.init_schema()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute("PRAGMA journal_mode = WAL;")
        return conn

    def connect(self) -> None:
        """Verifies DB connection and schema."""
        self.init_schema()

    def init_schema(self) -> None:
        """Initializes tables and indexes."""
        sql = """
        CREATE TABLE IF NOT EXISTS missions (
            mission_id TEXT PRIMARY KEY,
            mission_name TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            duration_s REAL DEFAULT 0.0,
            engine_id INTEGER DEFAULT 1,
            flight_profile TEXT DEFAULT 'BENCHMARK',
            injected_fault TEXT DEFAULT 'NONE',
            fault_severity REAL DEFAULT 0.0,
            disposition TEXT DEFAULT 'PENDING',
            start_health REAL DEFAULT 1.0,
            end_health REAL DEFAULT 1.0,
            min_health REAL DEFAULT 1.0,
            dominant_fault TEXT DEFAULT 'Normal',
            total_frames INTEGER DEFAULT 0,
            summary_json TEXT DEFAULT '{}'
        );

        CREATE TABLE IF NOT EXISTS mission_telemetry (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            mission_id TEXT NOT NULL REFERENCES missions(mission_id) ON DELETE CASCADE,
            step INTEGER NOT NULL,
            timestamp_s REAL NOT NULL,
            rpm REAL,
            cht REAL,
            egt REAL,
            oil_pressure_psi REAL,
            oil_temperature REAL,
            fuel_flow_lph REAL,
            vibration REAL,
            m1_anomaly_score REAL,
            m2_predicted_fault TEXT,
            m3_rul_minutes REAL,
            residuals_json TEXT,
            indicators_json TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_telemetry_mission_step
            ON mission_telemetry (mission_id, step ASC);

        CREATE INDEX IF NOT EXISTS idx_missions_created_at
            ON missions (created_at DESC);
        """
        with self._get_conn() as conn:
            conn.executescript(sql)
            conn.commit()

    def create_mission(
        self,
        mission_id: str,
        mission_name: str,
        flight_profile: str = "BENCHMARK",
        injected_fault: str = "NONE",
        fault_severity: float = 0.0,
        engine_id: int = 1,
    ) -> None:
        sql = """
        INSERT INTO missions (
            mission_id, mission_name, flight_profile, injected_fault,
            fault_severity, engine_id, disposition
        ) VALUES (?, ?, ?, ?, ?, ?, 'RUNNING')
        ON CONFLICT(mission_id) DO UPDATE SET
            mission_name = excluded.mission_name,
            flight_profile = excluded.flight_profile,
            injected_fault = excluded.injected_fault,
            fault_severity = excluded.fault_severity,
            disposition = 'RUNNING';
        """
        with self._get_conn() as conn:
            conn.execute(sql, (
                mission_id, mission_name, flight_profile,
                injected_fault, fault_severity, engine_id
            ))
            conn.commit()

    def insert_telemetry_batch(self, frames: List[Dict[str, Any]]) -> None:
        if not frames:
            return

        sql = """
        INSERT INTO mission_telemetry (
            mission_id, step, timestamp_s, rpm, cht, egt,
            oil_pressure_psi, oil_temperature, fuel_flow_lph, vibration,
            m1_anomaly_score, m2_predicted_fault, m3_rul_minutes,
            residuals_json, indicators_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
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

        with self._get_conn() as conn:
            conn.executemany(sql, records)
            conn.commit()

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
        sql = """
        UPDATE missions
        SET duration_s = ?,
            disposition = ?,
            start_health = ?,
            end_health = ?,
            min_health = ?,
            dominant_fault = ?,
            total_frames = ?,
            summary_json = ?
        WHERE mission_id = ?;
        """
        with self._get_conn() as conn:
            conn.execute(sql, (
                duration_s, disposition, start_health, end_health,
                min_health, dominant_fault, total_frames,
                json.dumps(summary_dict or {}), mission_id
            ))
            conn.commit()

    def list_missions(self, limit: int = 50) -> List[Dict[str, Any]]:
        sql = """
        SELECT m.mission_id, m.mission_name, m.created_at, m.duration_s, m.engine_id,
               m.flight_profile, m.injected_fault, m.fault_severity, m.disposition,
               m.start_health, m.end_health, m.min_health, m.dominant_fault,
               m.total_frames,
               COUNT(t.id) AS actual_frames,
               m.summary_json
        FROM missions m
        LEFT JOIN mission_telemetry t ON t.mission_id = m.mission_id
        GROUP BY m.mission_id
        ORDER BY m.created_at DESC
        LIMIT ?;
        """
        with self._get_conn() as conn:
            cur = conn.execute(sql, (limit,))
            rows = cur.fetchall()
            result = []
            for r in rows:
                item = dict(r)
                if isinstance(item.get("summary_json"), str):
                    try:
                        item["summary_json"] = json.loads(item["summary_json"])
                    except Exception:
                        pass
                result.append(item)
            return result

    def get_mission(self, mission_id: str) -> Optional[Dict[str, Any]]:
        sql = "SELECT * FROM missions WHERE mission_id = ?;"
        with self._get_conn() as conn:
            cur = conn.execute(sql, (mission_id,))
            row = cur.fetchone()
            if not row:
                return None
            item = dict(row)
            if isinstance(item.get("summary_json"), str):
                try:
                    item["summary_json"] = json.loads(item["summary_json"])
                except Exception:
                    pass
            return item

    def stream_mission_telemetry(
        self, mission_id: str, chunk_size: int = 100
    ) -> Generator[Dict[str, Any], None, None]:
        conn = self._get_conn()
        try:
            cur = conn.execute(
                "SELECT * FROM mission_telemetry WHERE mission_id = ? ORDER BY step ASC;",
                (mission_id,),
            )
            while True:
                rows = cur.fetchmany(chunk_size)
                if not rows:
                    break
                for r in rows:
                    item = dict(r)
                    if isinstance(item.get("residuals_json"), str):
                        try:
                            item["residuals_json"] = json.loads(item["residuals_json"])
                        except Exception:
                            pass
                    if isinstance(item.get("indicators_json"), str):
                        try:
                            item["indicators_json"] = json.loads(item["indicators_json"])
                        except Exception:
                            pass
                    yield item
        finally:
            conn.close()

    def delete_mission(self, mission_id: str) -> bool:
        sql = "DELETE FROM missions WHERE mission_id = ?;"
        with self._get_conn() as conn:
            cur = conn.execute(sql, (mission_id,))
            conn.commit()
            return cur.rowcount > 0
