# PRATIBIMB — Component Integration Guide

## System Architecture

```
PRATIBIMB/
├── DASHBOARD AND DATA/          ← The complete Digital Twin backend + frontend
│   ├── AeroTwin/                ← Python package (physics, ML, API)
│   │   ├── api/server.py        ← FastAPI server + dashboard
│   │   ├── simulator/           ← 4-cylinder crankshaft engine model
│   │   ├── health/              ← Digital Twin state engine
│   │   ├── ml/                  ← Anomaly detection, fault diagnosis, RUL
│   │   ├── mission/             ← Mission risk, advisory, reporting
│   │   └── degradation/         ← Fault injection
│   ├── models/                  ← Trained ML model artifacts
│   ├── data/generated/          ← Synthetic training datasets
│   └── run_dashboard.py         ← ONE-COMMAND LAUNCHER ← START HERE
│
├── MISSION_SIMULATOR/           ← Standalone real-time streaming server
│   ├── engine_physics.py        ← MVEM first-order-lag model (8 fault modes)
│   ├── backend.py               ← FastAPI + WebSocket server (port 8000)
│   └── data/                    ← CSV mission logs
│
├── FAULT DETECTION/             ← CNN-LSTM fault classifier (PyTorch)
├── ANOMALY DETECTION/           ← Additional anomaly detectors
├── HEALTH MONITORING/           ← Health index models
└── RUL ESTIMATION/              ← Remaining Useful Life models
```

---

## How to Run the Dashboard

```powershell
cd "e:\PRATIBIMB\DASHBOARD AND DATA"
python run_dashboard.py
```

Open **http://localhost:8001** in your browser.

**What you'll see:**
1. Click **"Start simulation"** → engine begins running
2. Watch live telemetry (RPM, CHT, EGT, oil pressure, vibration) vs. Digital Twin expected values
3. After ~5 seconds, health index and anomaly score appear
4. Click **"Inject fault"** → pick COOLING / BEARING / LUBRICATION / CYLINDER + severity
5. Watch the anomaly detector trigger, fault diagnosis kick in, and RUL decrease

---

## How to Run the Mission Simulator (standalone)

```powershell
cd "e:\PRATIBIMB\MISSION_SIMULATOR"
python -m uvicorn backend:app --port 8000 --reload
```

**WebSocket API** at `ws://localhost:8000/ws`:
```json
{"command": "start", "throttle": 0.65, "altitude_ft": 3000}
{"command": "inject_fault", "fault_id": 2, "fault_severity": 0.7}
{"command": "pause"}
{"command": "resume"}
{"command": "reset"}
```

**REST endpoints:**
- `http://localhost:8000/health`
- `http://localhost:8000/status`
- `http://localhost:8000/faults`
- `http://localhost:8000/docs`

---

## Component Mapping

| Component | Purpose | Connected to |
|---|---|---|
| `AeroTwin/api/server.py` | Full assessment backend | `AeroTwin/api/static/` dashboard |
| `MISSION_SIMULATOR/backend.py` | Real-time MVEM simulator | Future React dashboard (Phase 11) |
| `FAULT DETECTION/` | Trained CNN-LSTM classifier | Batch inference |
| `RUL ESTIMATION/` | RUL regression models | AeroTwin pipeline |

---

## Two Different Physics Engines

| | AeroTwin Simulator | MISSION_SIMULATOR |
|---|---|---|
| Model | 4-cylinder crankshaft (ODE) | MVEM first-order lag |
| Faults | CYLINDER, BEARING, COOLING, LUBRICATION | 8 modes incl. misfire, sensor drift, injector |
| Altitude | No | Yes (rho_ratio) |
| Purpose | Dashboard demo + ML training | Real-time pilot control simulation |
| Port | 8001 | 8000 |
