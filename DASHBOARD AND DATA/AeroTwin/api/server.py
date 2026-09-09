"""
AeroTwin-4 Real-Time Digital Twin API.

FastAPI + WebSocket service exposing the live engine twin: telemetry in, health /
anomaly / diagnosis / RUL / mission-risk out.

Run:
    .venv/Scripts/python.exe -m uvicorn AeroTwin.api.server:app --reload --port 8000

The operator dashboard is served from the same process at ``/``, so a demo needs
one command and no separate frontend build or dev server.

Endpoints:
    GET  /                      operator dashboard (React)
    GET  /api                   service banner
    GET  /health                liveness probe
    GET  /api/status            simulation + model-loading state
    GET  /api/telemetry/latest  most recent telemetry frame
    GET  /api/twin/state        latest full twin assessment
    POST /api/sim/start         start / restart the simulation
    POST /api/sim/stop          stop the simulation
    POST /api/sim/throttle      manual throttle override
    POST /api/sim/inject_fault  inject a degradation for demonstration
    POST /api/mission/assess    assess an arbitrary mission against live state
    GET  /api/alerts            timestamped fault and state-change alert log
    GET  /api/efficiency        power and specific-fuel-consumption trend
    GET  /api/maintenance       prioritised maintenance advisory
    GET  /api/mission/report    per-sortie health report for the debrief
    WS   /ws/telemetry          live telemetry + assessment stream
"""

import asyncio
import os
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from AeroTwin.api.pipeline import LiveAssessmentPipeline, default_engine_parameters
from AeroTwin.degradation.config import (
    ComponentID,
    DegradationConfig,
    DegradationType,
    TrajectoryType,
)
from AeroTwin.degradation.injector import DegradationInjector
from AeroTwin.mission.risk import MissionProfile, MissionRiskAssessor
from AeroTwin.simulator.runner import EngineRunner

_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Telemetry is generated at 100 Hz; broadcasting every frame would flood clients
# for no benefit, so the stream is decimated to this rate.
STREAM_HZ = 10.0
SIM_DT = 0.01


class SimulationService:
    """Owns the engine simulation loop and fans results out to websocket clients."""

    def __init__(self):
        self.pipeline: Optional[LiveAssessmentPipeline] = None
        self.injector: Optional[DegradationInjector] = None
        self.runner: Optional[EngineRunner] = None

        self.latest_telemetry: Optional[Dict[str, Any]] = None
        self.latest_assessment: Optional[Dict[str, Any]] = None
        self.running = False
        self.started_at: Optional[float] = None

        self._task: Optional[asyncio.Task] = None
        self._clients: List[WebSocket] = []
        self._lock = asyncio.Lock()

    # ---------------------------------------------------------------- lifecycle

    def build(
        self,
        seed: int = 42,
        mission_duration_s: float = 600.0,
        fault: Optional[DegradationConfig] = None,
    ):
        params = default_engine_parameters("LIVE_001", seed)
        self.runner = EngineRunner(dt=SIM_DT, seed=seed, engine_parameters=params)
        self.injector = DegradationInjector(
            config=fault or DegradationConfig.healthy(),
            runner=self.runner,
            run_id="LIVE_001",
            noise_enabled=True,
        )
        self.pipeline = LiveAssessmentPipeline(
            root_dir=_ROOT_DIR,
            engine_parameters=params,
            seed=seed,
            dt=SIM_DT,
            mission=MissionProfile(name="ISR_SORTIE", required_duration_s=mission_duration_s),
        )
        self.latest_telemetry = None
        self.latest_assessment = None

    async def start(self, **kwargs):
        await self.stop()
        self.build(**kwargs)
        self.running = True
        self.started_at = time.time()
        self._task = asyncio.create_task(self._loop())

    async def stop(self):
        self.running = False
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
            self._task = None

    # ------------------------------------------------------------------- loop

    async def _loop(self):
        """
        Advance the simulation and broadcast. Frames are produced in small batches
        and awaited between them so the event loop stays responsive - a tight
        synchronous loop here would starve the websocket handlers.
        """
        frames_per_broadcast = max(1, int((1.0 / STREAM_HZ) / SIM_DT))
        try:
            while self.running:
                payload = None
                for _ in range(frames_per_broadcast):
                    telemetry, _gt = self.injector.step()
                    tel_dict = telemetry.to_dict()
                    self.latest_telemetry = tel_dict
                    result = self.pipeline.ingest(tel_dict)
                    if result.get("assessment"):
                        self.latest_assessment = result["assessment"]
                    payload = result

                await self._broadcast(
                    {
                        "type": "telemetry",
                        "telemetry": self.latest_telemetry,
                        "twin": {
                            "simulation_time": payload["simulation_time"],
                            "expected": payload["expected"],
                            "indicators": payload["indicators"],
                        },
                        "efficiency": payload.get("efficiency"),
                        "assessment": self.latest_assessment,
                    }
                )
                await asyncio.sleep(1.0 / STREAM_HZ)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # keep the service alive; report to clients
            self.running = False
            await self._broadcast({"type": "error", "message": str(exc)})

    # -------------------------------------------------------------- websockets

    async def register(self, ws: WebSocket):
        async with self._lock:
            self._clients.append(ws)

    async def unregister(self, ws: WebSocket):
        async with self._lock:
            if ws in self._clients:
                self._clients.remove(ws)

    async def _broadcast(self, message: Dict[str, Any]):
        async with self._lock:
            targets = list(self._clients)
        dead = []
        for ws in targets:
            try:
                await ws.send_json(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            await self.unregister(ws)


service = SimulationService()


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    await service.stop()


app = FastAPI(
    title="AeroTwin-4 Digital Twin API",
    description="Real-time health monitoring, fault diagnosis, RUL and mission risk "
                "for a representative 4-cylinder aero piston engine.",
    version="1.0.0",
    lifespan=lifespan,
)

# The dashboard is expected to be served from a different origin in development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------- schemas

class StartRequest(BaseModel):
    seed: int = 42
    mission_duration_s: float = Field(600.0, gt=0)


class ThrottleRequest(BaseModel):
    throttle: float = Field(..., ge=0.0, le=1.0)


class InjectFaultRequest(BaseModel):
    fault_type: str = Field(..., description="CYLINDER | BEARING | COOLING | LUBRICATION")
    severity: float = Field(..., ge=0.0, le=1.0)
    component: Optional[str] = Field(None, description="e.g. CYLINDER_3; defaults per family")
    trajectory: str = Field("CONSTANT", description="CONSTANT | LINEAR | STEP | EXPONENTIAL")
    ramp_duration_s: float = Field(60.0, gt=0)
    seed: int = 42
    mission_duration_s: float = Field(600.0, gt=0)


class MissionAssessRequest(BaseModel):
    required_duration_s: float = Field(..., gt=0)
    reserve_duration_s: float = Field(0.0, ge=0)


# -------------------------------------------------------------------- routes

_STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")


@app.get("/")
def dashboard():
    """Serve the operator dashboard so the demo is a single command."""
    return FileResponse(os.path.join(_STATIC_DIR, "index.html"))


@app.get("/api")
def root():
    return {
        "service": "AeroTwin-4 Digital Twin API",
        "version": "1.0.0",
        "dashboard": "/",
        "docs": "/docs",
        "websocket": "/ws/telemetry",
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/api/status")
def status():
    body: Dict[str, Any] = {
        "running": service.running,
        "started_at": service.started_at,
        "stream_hz": STREAM_HZ,
        "sim_dt": SIM_DT,
    }
    body["pipeline"] = service.pipeline.status() if service.pipeline else None
    return body


@app.get("/api/telemetry/latest")
def latest_telemetry():
    if service.latest_telemetry is None:
        raise HTTPException(status_code=404, detail="No telemetry yet; start the simulation.")
    return service.latest_telemetry


@app.get("/api/twin/state")
def twin_state():
    if service.latest_assessment is None:
        raise HTTPException(
            status_code=404,
            detail="No assessment yet; a full 5 s window must accumulate first.",
        )
    return service.latest_assessment


@app.post("/api/sim/start")
async def sim_start(req: StartRequest):
    await service.start(seed=req.seed, mission_duration_s=req.mission_duration_s)
    return {"running": True, "seed": req.seed, "mission_duration_s": req.mission_duration_s}


@app.post("/api/sim/stop")
async def sim_stop():
    await service.stop()
    return {"running": False}


@app.post("/api/sim/throttle")
def sim_throttle(req: ThrottleRequest):
    if service.runner is None:
        raise HTTPException(status_code=409, detail="Simulation is not running.")
    service.runner.set_throttle(req.throttle)
    return {"throttle": req.throttle}


@app.post("/api/sim/inject_fault")
async def inject_fault(req: InjectFaultRequest):
    """Restart the sortie with a degradation injected - the demo path for the PS."""
    try:
        deg_type = DegradationType[req.fault_type.upper()]
    except KeyError:
        raise HTTPException(status_code=422, detail=f"Unknown fault_type {req.fault_type!r}")

    defaults = {
        DegradationType.CYLINDER: ComponentID.CYLINDER_3,
        DegradationType.BEARING: ComponentID.BEARING,
        DegradationType.COOLING: ComponentID.COOLING_SYSTEM,
        DegradationType.LUBRICATION: ComponentID.LUBRICATION_SYSTEM,
    }
    if req.component:
        try:
            comp = ComponentID[req.component.upper()]
        except KeyError:
            raise HTTPException(status_code=422, detail=f"Unknown component {req.component!r}")
    else:
        comp = defaults.get(deg_type)
        if comp is None:
            raise HTTPException(status_code=422, detail=f"No default component for {deg_type}")

    try:
        traj = TrajectoryType[req.trajectory.upper()]
    except KeyError:
        raise HTTPException(status_code=422, detail=f"Unknown trajectory {req.trajectory!r}")

    cfg = DegradationConfig.single_fault(
        degradation_type=deg_type,
        component_id=comp,
        severity=req.severity,
        trajectory_type=traj,
        ramp_duration=req.ramp_duration_s,
    )
    await service.stop()
    service.build(seed=req.seed, mission_duration_s=req.mission_duration_s, fault=cfg)
    service.running = True
    service.started_at = time.time()
    service._task = asyncio.create_task(service._loop())

    return {
        "running": True,
        "fault_type": deg_type.value,
        "component": comp.value,
        "severity": req.severity,
        "trajectory": traj.value,
    }


@app.post("/api/mission/assess")
def mission_assess(req: MissionAssessRequest):
    """Re-assess the live engine state against a different mission requirement."""
    if service.latest_assessment is None:
        raise HTTPException(status_code=404, detail="No assessment yet; start the simulation.")

    latest = service.latest_assessment
    rul = latest.get("rul") or {}
    diag = latest.get("diagnosis") or {}
    anom = latest.get("anomaly") or {}
    health = (latest.get("health") or {}).get("health_index", 1.0)

    assessment = MissionRiskAssessor().assess(
        mission=MissionProfile(
            name="AD_HOC",
            required_duration_s=req.required_duration_s,
            reserve_duration_s=req.reserve_duration_s,
        ),
        health_index=float(health),
        rul_seconds=rul.get("rul_seconds"),
        rul_lower_seconds=rul.get("rul_lower_seconds"),
        predicted_fault=diag.get("predicted_fault", "HEALTHY"),
        fault_confidence=float(diag.get("confidence", 0.0)),
        anomaly_flagged=bool(anom.get("flagged", False)),
        anomaly_score_ratio=float(anom.get("score_ratio", 0.0)),
    )
    return assessment.to_dict()


@app.get("/api/alerts")
def alerts(limit: int = 60):
    """Timestamped log of state changes: anomalies, diagnoses, dispatch changes."""
    if service.pipeline is None:
        return {"alerts": [], "counts": {}}
    return {
        "alerts": service.pipeline.alerts.series(limit=limit),
        "counts": service.pipeline.alerts.counts(),
    }


@app.get("/api/efficiency")
def efficiency(limit: int = 240):
    """Shaft power and BSFC against the healthy twin, plus a running summary."""
    if service.pipeline is None:
        return {"series": [], "summary": {}}
    return {
        "series": service.pipeline.efficiency_series(limit=limit),
        "summary": service.pipeline.efficiency.summary(),
    }


@app.get("/api/maintenance")
def maintenance():
    """Prioritised maintenance advisory derived from the latest assessment."""
    if service.latest_assessment is None:
        raise HTTPException(status_code=404, detail="No assessment yet; start the simulation.")
    return {"items": service.latest_assessment.get("maintenance", [])}


@app.get("/api/mission/report")
def mission_report():
    """Per-sortie health report for the debrief."""
    if service.pipeline is None:
        raise HTTPException(status_code=404, detail="No sortie in progress.")
    return service.pipeline.mission_report()


@app.websocket("/ws/telemetry")
async def ws_telemetry(ws: WebSocket):
    await ws.accept()
    await service.register(ws)
    try:
        await ws.send_json({"type": "connected", "status": status()})
        while True:
            # Keep the socket open; the broadcast task does the sending.
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        await service.unregister(ws)
