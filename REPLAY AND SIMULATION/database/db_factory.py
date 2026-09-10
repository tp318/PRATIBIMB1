"""
=============================================================================
db_factory.py — Resilient Database Factory with Auto-Fallback
=============================================================================
Instantiates either PostgreSQL (production) or SQLite (zero-config local)
depending on environment configuration and driver availability.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional, Union

from .postgres_db import HAS_PSYCOPG2, PostgresMissionDatabase, load_env_file
from .sqlite_db import SqliteMissionDatabase

logger = logging.getLogger("AeroTwin.DBFactory")

_active_database_instance: Optional[Union[PostgresMissionDatabase, SqliteMissionDatabase]] = None


def get_mission_database(
    connection_url: Optional[str] = None,
    prefer_sqlite: bool = False,
    sqlite_path: Optional[str | Path] = None,
    force_new: bool = False,
) -> Union[PostgresMissionDatabase, SqliteMissionDatabase]:
    """
    Returns an initialized database manager.
    
    1. If `prefer_sqlite` is True, directly returns SqliteMissionDatabase.
    2. Otherwise, attempts PostgreSQL if DATABASE_URL or Postgres credentials exist.
    3. Gracefully falls back to SqliteMissionDatabase if PostgreSQL cannot be reached.
    """
    global _active_database_instance
    if _active_database_instance is not None and not force_new:
        return _active_database_instance

    if prefer_sqlite:
        logger.info("Using SQLite mission database (prefer_sqlite=True).")
        _active_database_instance = SqliteMissionDatabase(db_path=sqlite_path)
        return _active_database_instance

    load_env_file()
    has_postgres_env = (
        "DATABASE_URL" in os.environ
        or "POSTGRES_PASSWORD" in os.environ
        or connection_url is not None
    )

    if HAS_PSYCOPG2 and has_postgres_env:
        try:
            db = PostgresMissionDatabase(connection_url=connection_url)
            db.connect()
            logger.info("Connected to PostgreSQL mission database.")
            _active_database_instance = db
            return db
        except Exception as exc:
            logger.warning(
                f"PostgreSQL connection failed ({exc}). Falling back to local SQLite database."
            )

    logger.info("Initializing fallback local SQLite database.")
    _active_database_instance = SqliteMissionDatabase(db_path=sqlite_path)
    return _active_database_instance


def get_database() -> Union[PostgresMissionDatabase, SqliteMissionDatabase]:
    """Convenience alias matching DATABASE.postgres_db.get_database()."""
    return get_mission_database()
