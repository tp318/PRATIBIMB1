"""
bus.py
======
DT CORE — Real-Time Telemetry Bus

Implements an asyncio-based telemetry streaming bus that works like CAN bus:

  - PRODUCER: Engine physics (MISSION_SIMULATOR) publishes frames at 10 Hz.
  - BUS: asyncio.Queue per subscriber — any component can tap the stream.
  - CONSUMERS:
      1. DT Engine — reads sensor frame, produces predictions + residuals
      2. WebSocket broadcaster — streams to frontend dashboard
      3. Log buffer — accumulates rows in memory (NOT pre-written to disk)

On sortie end (or on-demand download):
  - Log buffer is flushed to a timestamped CSV file.
  - Buffer is then cleared.

Architecture (mimics CAN bus node model):

    [Producer]
        │  publish(frame)
        ▼
    [TelemetryBus]
        ├── subscriber_queue_1  → [DT Engine consumer]
        ├── subscriber_queue_2  → [WebSocket broadcaster]
        └── subscriber_queue_3  → [Log buffer]

Usage:
    bus = TelemetryBus()
    q   = bus.subscribe("dt_engine")
    ...
    bus.publish(frame_dict)
    frame = await q.get()
"""

from __future__ import annotations

import asyncio
import csv
import io
import logging
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# Maximum in-memory log rows before auto-flush to prevent OOM
MAX_LOG_ROWS = 100_000

# Directory to save CSVs (relative to DT CORE); can be overridden
DEFAULT_LOG_DIR = Path(__file__).parent.parent / "DT CORE" / "logs"


class TelemetryBus:
    """
    Central telemetry bus.  Thread-safe via asyncio.

    All operations must be called from within the same asyncio event loop.
    """

    def __init__(self, max_queue_depth: int = 200):
        self._queues: Dict[str, asyncio.Queue] = {}
        self._max_q  = max_queue_depth
        self._log: List[Dict[str, Any]] = []
        self._sortie_id: Optional[str] = None
        self._started_at: Optional[float] = None

    # ---------------------------------------------------------------- lifecycle

    def start_sortie(self, sortie_id: Optional[str] = None) -> str:
        """Clear log buffer and mark sortie start."""
        self._log.clear()
        self._started_at = time.time()
        self._sortie_id  = sortie_id or f"SORTIE_{int(self._started_at)}"
        logger.info("Telemetry bus: sortie %s started.", self._sortie_id)
        return self._sortie_id

    def end_sortie(self) -> Optional[Path]:
        """
        End the current sortie and flush log to CSV.
        Returns the path of the written file, or None if log was empty.
        """
        if not self._log:
            logger.info("Telemetry bus: sortie ended with empty log.")
            return None
        path = self._flush_csv()
        self._log.clear()
        logger.info("Telemetry bus: sortie %s ended, log saved to %s.",
                    self._sortie_id, path)
        return path

    # ---------------------------------------------------------------- subscribe

    def subscribe(self, name: str) -> asyncio.Queue:
        """Register a named consumer and get its queue."""
        if name not in self._queues:
            self._queues[name] = asyncio.Queue(maxsize=self._max_q)
        return self._queues[name]

    def unsubscribe(self, name: str) -> None:
        self._queues.pop(name, None)

    # ---------------------------------------------------------------- publish

    def publish(self, frame: Dict[str, Any]) -> None:
        """
        Publish one telemetry frame to all subscriber queues and the log.

        Non-blocking: if a subscriber queue is full, the oldest item is
        dropped (oldest-first, like a CAN bus overrun).
        """
        for name, q in self._queues.items():
            if q.full():
                try:
                    q.get_nowait()   # drop oldest
                except asyncio.QueueEmpty:
                    pass
            try:
                q.put_nowait(frame)
            except asyncio.QueueFull:
                pass  # shouldn't happen after the get_nowait above

        # Append to in-memory log
        self._log.append(frame)
        if len(self._log) > MAX_LOG_ROWS:
            # Flush mid-sortie if log is huge; keep a rolling tail in memory
            self._flush_csv(append=True)
            self._log.clear()

    # ---------------------------------------------------------------- log access

    @property
    def log_rows(self) -> int:
        return len(self._log)

    def get_csv_bytes(self) -> bytes:
        """
        Return the in-memory log as UTF-8 CSV bytes — for live download
        without ending the sortie.
        """
        return self._log_to_csv_bytes(self._log)

    # ---------------------------------------------------------------- internals

    @staticmethod
    def _log_to_csv_bytes(rows: List[Dict[str, Any]]) -> bytes:
        if not rows:
            return b""
        buf = io.StringIO()
        fieldnames = list(rows[0].keys())
        writer = csv.DictWriter(buf, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        return buf.getvalue().encode("utf-8")

    def _flush_csv(self, append: bool = False) -> Path:
        log_dir = DEFAULT_LOG_DIR
        log_dir.mkdir(parents=True, exist_ok=True)
        fname = f"{self._sortie_id}.csv"
        path  = log_dir / fname

        rows = self._log
        if not rows:
            return path

        mode = "a" if (append and path.exists()) else "w"
        fieldnames = list(rows[0].keys())

        with open(path, mode, newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            if mode == "w":
                writer.writeheader()
            writer.writerows(rows)

        return path
