"""
backend.py
==========
Real-time FastAPI/WebSocket streaming server for the MALE UAV Piston Engine
Mission Simulator — now with integrated DT CORE and real-time telemetry bus.

Architecture
------------
  MISSION_SIMULATOR (engine_physics.py)
      ↓  10 Hz  (real engine readings)
  [TELEMETRY BUS]   ← asyncio.Queue per subscriber, like CAN bus
      ↓                  ↓                   ↓
  DT CORE          WebSocket            Log Buffer
  (Y_pred,          broadcaster         (in-memory)
   residuals)        (frontend)              ↓
                                      CSV on demand
                                      GET /download_log

Data flow per tick:
  1. step(engine_state, controls, rng) → actual sensor readings
  2. DT Core → predicted readings (same throttle/altitude inputs, same timestamp)
  3. compute_residuals(actual, predicted) → r_i, z_i, severity per channel
  4. Broadcast full frame (actual + predicted + residuals) to all WS clients
  5. Append to in-memory log buffer (NOT written to disk until download)
  6. GET /download_log → return CSV bytes; log buffer is NOT cleared (sortie continues)
  7. POST /stop_sortie OR WebSocket reset → flush log to disk, clear buffer

Supported WebSocket commands (JSON):
  start          – begin simulation, accept initial controls
  update_controls – live throttle / altitude update
  inject_fault   – set fault_id + fault_severity
  clear_fault    – reset fault to 0 (Healthy)
  pause          – freeze ticker (state preserved)
  resume         – unpause
  reset          – full state + RNG reset, flushes log to disk

Outgoing telemetry frame (10 Hz, JSON):
  {
    "timestamp":         <wall-clock epoch seconds>,
    "mission_time":      <sim seconds since start>,
    "throttle":          <0.0 – 1.0>,
    "altitude_ft":       <0 – 25000>,
    "ambient_c":         <-30 – 60>,

    -- Real engine (MVEM) readings --
    "real_RPM":          <RPM>,
    "real_EGT":          <°C>,
    "real_CHT":          <°C>,
    "real_OilPress":     <bar>,
    "real_OilTemp":      <°C>,
    "real_FuelFlow":     <L/hr>,
    "real_Vibration":    <g RMS>,
    "real_AltVoltage":   <V>,

    -- DT CORE predictions --
    "dt_RPM":            <RPM>,
    "dt_EGT":            <°C>,
    "dt_CHT":            <°C>,
    "dt_OilPress":       <bar>,
    "dt_OilTemp":        <°C>,
    "dt_FuelFlow":       <L/hr>,

    -- Residuals --
    "r_RPM":    <RPM>,   "z_RPM":    <σ>,  "sev_RPM":    <NORMAL|CAUTION|WARNING>,
    "r_EGT":    <°C>,    "z_EGT":    <σ>,  "sev_EGT":    ...,
    "r_CHT":    <°C>,    "z_CHT":    <σ>,  "sev_CHT":    ...,
    "r_OilPress":<bar>,  "z_OilPress":<σ>, "sev_OilPress":...,
    "r_OilTemp": <°C>,   "z_OilTemp": <σ>, "sev_OilTemp": ...,
    "r_FuelFlow":<L/hr>, "z_FuelFlow":<σ>, "sev_FuelFlow":...,
    "max_z":             <float>,
    "flagged":           <bool>,

    "fault_label":       <string>,
    "fault_active":      <bool>,
    "log_rows":          <int>        -- rows buffered so far
  }

Run with:
    python -m uvicorn backend:app --port 8000 --reload
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional, Set

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

# ── DT CORE import ───────────────────────────────────────────────────────────
_DT_CORE_DIR = Path(__file__).parent.parent / "DT CORE"
sys.path.insert(0, str(_DT_CORE_DIR))

from engine    import DigitalTwinEngine
from bus       import TelemetryBus
from residuals import compute_residuals

# ── MISSION_SIMULATOR physics ─────────────────────────────────────────────────
from engine_physics import (
    DT,
    FAULT_NAMES,
    Controls,
    EngineState,
    create_initial_state,
    step,
)

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="UAV Piston Engine — Mission Simulator + DT Core",
    description=(
        "Real-time 10 Hz MVEM simulator with integrated Digital Twin Core. "
        "Streams real engine vs DT predictions + residuals on a CAN-bus-style "
        "telemetry bus. CSV log available on demand via GET /download_log."
    ),
    version="3.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Global state — all accessed from the same asyncio event loop
# ---------------------------------------------------------------------------

engine_state: EngineState       = create_initial_state()
controls: Controls               = Controls()
rng: np.random.Generator         = np.random.default_rng(seed=42)

simulation_running: bool = False
simulation_paused:  bool = False

# DT CORE — Digital Twin physics engine
dt_engine = DigitalTwinEngine()

# Telemetry bus
bus = TelemetryBus(max_queue_depth=300)

# WebSocket clients (for direct broadcast without subscribing to bus queue)
_clients: Set[WebSocket] = set()

# Auto-mission coroutine handle
_auto_mission_task: Optional[asyncio.Task] = None

# ---------------------------------------------------------------------------
# Auto-mission waypoints
# ---------------------------------------------------------------------------

_WP_TIMES = [0, 120, 300, 600, 1500, 1800, 1860, 1920, 2400, 2580, 2640, 2680, 2700]
_WP_THR   = [0.12, 0.15, 0.90, 0.75, 0.65, 0.80, 0.50, 0.75, 0.60, 0.35, 0.25, 0.15, 0.12]
_WP_ALT   = [0, 200, 1000, 10000, 15000, 15000, 14000, 13000, 10000, 5000, 3000, 500, 0]
_WP_AMB   = [15.0, 14.8, 14.0, 9.5, 6.5, 6.5, 7.5, 8.0, 9.5, 12.0, 13.5, 14.8, 15.0]


async def _auto_mission_coroutine() -> None:
    global controls
    mission_start = engine_state.t
    while True:
        elapsed = engine_state.t - mission_start
        if elapsed >= _WP_TIMES[-1]:
            break
        controls.throttle    = round(float(np.interp(elapsed, _WP_TIMES, _WP_THR)), 4)
        controls.altitude_ft = round(float(np.interp(elapsed, _WP_TIMES, _WP_ALT)), 1)
        controls.ambient_c   = round(float(np.interp(elapsed, _WP_TIMES, _WP_AMB)), 2)
        await asyncio.sleep(DT)

# ---------------------------------------------------------------------------
# Telemetry frame builder — REAL ENGINE + DT CORE + RESIDUALS
# ---------------------------------------------------------------------------

_RESIDUAL_CHANNELS = ["RPM", "EGT", "CHT", "OilPress", "OilTemp", "FuelFlow"]


def _build_and_publish_frame() -> dict:
    """
    One 0.1s tick:
      1. Step MVEM physics → actual sensor readings
      2. Step DT Core with same inputs → predicted readings
      3. Compute residuals: r_i = actual - predicted, z_i = r_i / σ_i
      4. Publish full frame to the telemetry bus
      5. Return frame dict for direct broadcast to WebSocket clients
    """
    # -- Step 1: MVEM (real engine) --
    expected_mvem, actual = step(engine_state, controls, rng)
    fault_id     = controls.fault_id
    fault_label  = FAULT_NAMES.get(fault_id, "Unknown")

    # -- Step 2: DT Core (physics-based prediction, no noise, no fault) --
    measured_for_dt = {
        "RPM":      actual["RPM"],
        "EGT":      actual["EGT"],
        "CHT":      actual["CHT"],
        "OilPress": actual["OilPress"],
        "OilTemp":  actual["OilTemp"],
        "FuelFlow":  actual["FuelFlow"],
    }
    dt_result = dt_engine.step(
        throttle    = controls.throttle,
        altitude_ft = controls.altitude_ft,
        ambient_c   = controls.ambient_c,
        measured    = measured_for_dt,
    )

    # -- Step 3: Residuals --
    residuals = dt_result.residuals   # computed inside dt_engine.step()

    # -- Step 4: Build flat frame dict --
    frame: dict = {
        "timestamp":    round(time.time(),          3),
        "mission_time": round(engine_state.t,       2),
        "throttle":     round(controls.throttle,    4),
        "altitude_ft":  round(controls.altitude_ft, 1),
        "ambient_c":    round(controls.ambient_c,   2),
        "fault_label":  fault_label,
        "fault_active": fault_id != 0,
    }

    # Real engine readings
    for ch in ["RPM", "EGT", "CHT", "OilPress", "OilTemp", "FuelFlow"]:
        frame[f"real_{ch}"] = round(float(actual.get(ch, 0)), 4)
    frame["real_Vibration"]   = round(float(actual.get("Vibration",  0)), 5)
    frame["real_AltVoltage"]  = round(float(actual.get("AltVoltage", 0)), 3)

    # DT Core predictions
    for ch, v in dt_result.predicted.items():
        frame[f"dt_{ch}"] = round(float(v), 4)

    # Residuals (r, z, severity per channel)
    if residuals:
        for ch in _RESIDUAL_CHANNELS:
            frame[f"r_{ch}"]   = round(residuals.raw.get(ch,        0.0), 4)
            frame[f"z_{ch}"]   = round(residuals.normalized.get(ch, 0.0), 4)
            frame[f"sev_{ch}"] = residuals.severity.get(ch, "NORMAL")
        frame["max_z"]   = round(residuals.max_z, 3)
        frame["flagged"] = residuals.flagged

    frame["log_rows"] = bus.log_rows

    # Publish to bus (log buffer + subscriber queues)
    bus.publish(frame)

    return frame


# ---------------------------------------------------------------------------
# Simulation lifecycle
# ---------------------------------------------------------------------------

def _reset() -> None:
    global engine_state, controls, rng, simulation_running, simulation_paused
    global _auto_mission_task

    if _auto_mission_task and not _auto_mission_task.done():
        _auto_mission_task.cancel()
        _auto_mission_task = None

    bus.end_sortie()   # flush log to disk

    engine_state       = create_initial_state()
    controls           = Controls()
    rng                = np.random.default_rng(seed=42)
    dt_engine.reset()
    simulation_running = False
    simulation_paused  = False


def _apply_command(msg: dict) -> None:
    global simulation_running, simulation_paused, _auto_mission_task

    command = msg.get("command", "")

    if "throttle" in msg:
        controls.throttle    = float(np.clip(msg["throttle"],    0.0, 1.0))
    if "altitude_ft" in msg:
        controls.altitude_ft = float(np.clip(msg["altitude_ft"], 0.0, 25000.0))
    if "ambient_c" in msg:
        controls.ambient_c   = float(np.clip(msg["ambient_c"],  -30.0, 60.0))
    if "fault_id" in msg:
        controls.fault_id = int(np.clip(int(msg["fault_id"]), 0, 8))
    if "fault_severity" in msg:
        controls.severity = float(np.clip(msg["fault_severity"], 0.0, 1.0))
    if "severity" in msg:
        controls.severity = float(np.clip(msg["severity"], 0.0, 1.0))

    if command == "start":
        if not simulation_running:
            bus.start_sortie()   # begin in-memory log
        simulation_running    = True
        simulation_paused     = False
        engine_state.running  = True

        if msg.get("mission_type") == "auto":
            if _auto_mission_task is None or _auto_mission_task.done():
                _auto_mission_task = asyncio.create_task(_auto_mission_coroutine())

        if "fault_id" in msg:
            controls.fault_id = int(np.clip(int(msg["fault_id"]), 0, 8))
        if "fault_severity" in msg:
            controls.severity = float(np.clip(msg["fault_severity"], 0.0, 1.0))

    elif command == "inject_fault":
        controls.fault_id = int(np.clip(int(msg.get("fault_id", 0)), 0, 8))
        controls.severity = float(np.clip(msg.get("fault_severity", msg.get("severity", 0.5)), 0.0, 1.0))

    elif command == "clear_fault":
        controls.fault_id = 0
        controls.severity = 0.0

    elif command == "pause":
        simulation_paused = True

    elif command == "resume":
        simulation_paused = False

    elif command == "reset":
        _reset()


# ---------------------------------------------------------------------------
# Broadcast helper
# ---------------------------------------------------------------------------

async def _broadcast(payload: dict) -> None:
    if not _clients:
        return
    text = json.dumps(payload)
    dead: Set[WebSocket] = set()
    for ws in list(_clients):
        try:
            await ws.send_text(text)
        except Exception:
            dead.add(ws)
    _clients.difference_update(dead)


# ---------------------------------------------------------------------------
# Main simulation loop (10 Hz)
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def _start_simulation_loop() -> None:
    asyncio.create_task(_simulation_loop())


async def _simulation_loop() -> None:
    while True:
        loop_start = time.perf_counter()

        if simulation_running and not simulation_paused:
            frame = _build_and_publish_frame()
            await _broadcast(frame)

        elapsed = time.perf_counter() - loop_start
        await asyncio.sleep(max(0.0, DT - elapsed))


# ---------------------------------------------------------------------------
# WebSocket endpoint
# ---------------------------------------------------------------------------

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    _clients.add(websocket)

    await websocket.send_text(json.dumps({
        "type":    "connected",
        "running": simulation_running,
        "paused":  simulation_paused,
        "faults":  FAULT_NAMES,
        "message": "Connected — DT CORE integrated. Real engine + Digital Twin readings stream in real-time.",
    }))

    async def _receiver() -> None:
        while True:
            try:
                raw = await websocket.receive_text()
                msg = json.loads(raw)
                _apply_command(msg)
            except (json.JSONDecodeError, TypeError, ValueError, KeyError):
                continue

    receiver_task = asyncio.create_task(_receiver())
    try:
        await asyncio.wait({receiver_task}, return_when=asyncio.FIRST_COMPLETED)
    except WebSocketDisconnect:
        pass
    finally:
        receiver_task.cancel()
        _clients.discard(websocket)
        try:
            await websocket.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# HTTP endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health_check() -> dict:
    return {"status": "ok", "running": simulation_running, "paused": simulation_paused}


@app.get("/status")
async def simulation_status() -> dict:
    return {
        "running":       simulation_running,
        "paused":        simulation_paused,
        "mission_time":  round(engine_state.t, 2),
        "throttle":      controls.throttle,
        "altitude_ft":   controls.altitude_ft,
        "ambient_c":     controls.ambient_c,
        "fault_id":      controls.fault_id,
        "fault_label":   FAULT_NAMES.get(controls.fault_id, "Unknown"),
        "fault_severity": controls.severity,
        "fault_active":  controls.fault_id != 0,
        "log_rows":      bus.log_rows,
        "sortie_id":     bus._sortie_id,
    }


@app.get("/faults")
async def fault_catalog() -> dict:
    return {"faults": [{"id": fid, "name": name} for fid, name in FAULT_NAMES.items()]}


@app.get("/download_log")
async def download_log() -> Response:
    """
    Download the in-memory telemetry log as a CSV file.

    The sortie continues — the log buffer is NOT cleared.
    Call this at any point to get a snapshot, or after stopping the sortie
    for the complete record.

    The CSV contains per-row:
      timestamp, mission_time, throttle, altitude_ft, ambient_c,
      real_RPM, real_EGT, real_CHT, real_OilPress, real_OilTemp, real_FuelFlow,
      real_Vibration, real_AltVoltage,
      dt_RPM,   dt_EGT,   dt_CHT,   dt_OilPress,   dt_OilTemp,   dt_FuelFlow,
      r_RPM, z_RPM, sev_RPM, ...(per channel residuals)...,
      max_z, flagged, fault_label, fault_active, log_rows
    """
    csv_bytes = bus.get_csv_bytes()
    sortie_id = bus._sortie_id or "sortie"
    filename  = f"{sortie_id}.csv"
    return Response(
        content     = csv_bytes,
        media_type  = "text/csv",
        headers     = {"Content-Disposition": f"attachment; filename={filename}"},
    )


@app.post("/stop_sortie")
async def stop_sortie() -> dict:
    """Stop the simulation and flush the full log to disk."""
    global simulation_running, simulation_paused
    simulation_running = False
    simulation_paused  = False
    path = bus.end_sortie()
    return {"stopped": True, "log_saved_to": str(path) if path else None}


@app.post("/control")
async def http_control(body: dict) -> dict:
    _apply_command(body)
    return {"status": "ok", "command": body.get("command")}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend:app", host="0.0.0.0", port=8000, reload=True)
