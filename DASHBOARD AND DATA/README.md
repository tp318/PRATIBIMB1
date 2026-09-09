# AeroTwin-4 — AI-Enabled Real-Time Digital Twin for Aero Piston Engines

**SIH 2026 · Problem Statement 26054 · DRDO / Department of Defence Production (IDEX)**
*AI-Enabled Real-Time Digital Twin System for Health Monitoring, Fault Prediction and
Mission Reliability Enhancement of Aero Piston Engines used in MALE UAVs.*

AeroTwin-4 is a physics-informed Digital Twin of a representative 4-cylinder
4-stroke aero piston engine. It runs a counterfactual healthy twin alongside the
observed engine, turns the difference into physically meaningful residuals, and uses
those residuals to answer four questions in real time:

> **Is something wrong?** → **What is wrong?** → **How long do I have?** → **Can I fly this mission?**

---

## Coverage against the problem statement

| PS requirement | Status | Where |
| :--- | :--- | :--- |
| Real-time Digital Twin system | Implemented | `AeroTwin/api/` — FastAPI + WebSocket, 10 Hz stream |
| Health monitoring | Implemented | `AeroTwin/health/` — counterfactual twin, residuals, indicators |
| Anomaly detection (beyond thresholds) | Implemented | `AeroTwin/ml/anomaly/` — 3 unsupervised detectors |
| Fault prediction / diagnosis | Implemented | `AeroTwin/ml/diagnosis/` — 5-class attribution + explainability |
| Remaining Useful Life (RUL) | Implemented | `AeroTwin/ml/rul/` — health regression + trend projection |
| Degradation trend prediction | Implemented | `scripts/generate_rul_dataset.py` — LINEAR / EXPONENTIAL trajectories |
| Mission reliability enhancement | Implemented | `AeroTwin/mission/` — risk score, GO/NO-GO, reasoning |
| Varying environmental & operating conditions | Implemented | `AeroTwin/degradation/conditions.py` — 0–7,600 m, ISA lapse rate |
| Dashboard: real-time health status | Implemented | Status strip plus observed against twin overlay |
| Dashboard: fault alerts | Implemented | Timestamped event log, logged on transition |
| Dashboard: efficiency trends | Implemented | Shaft power and specific fuel consumption against twin |
| Dashboard: maintenance advisory | Implemented | Prioritised actions with the evidence behind each |
| Dashboard: mission health reports | Implemented | Per sortie debrief record |
| 3D engine visualisation | Not built | out of current scope |

The PS notes that conventional UAV engine monitoring is *"primarily threshold-based
and reactive… indicating failures only after abnormality has already occurred."*
The residual-based approach here is the direct answer: a raw threshold on CHT cannot
distinguish a hot day from a failing cooling system, whereas a residual against a
condition-matched twin can. The ablation in
[docs/FIXES_AND_ARCHITECTURE.md](docs/FIXES_AND_ARCHITECTURE.md) quantifies exactly
that — ROC-AUC **0.7031 → 1.0000** moving from raw telemetry to twin residuals.

---

## Test suite

**176 / 176 passing.**

```bash
.venv/Scripts/python.exe -m pytest AeroTwin/ -q
```

| Module | Tests |
| :--- | ---: |
| Phase 1 — engine physics | 23 |
| Phase 2 — simulator runtime | 13 |
| Phase 3 — degradation & conditions | 17 |
| Phase 4 — Digital Twin residuals | 12 |
| Phase 5 — anomaly detection | 13 |
| Fault diagnosis | 17 |
| RUL estimation | 18 |
| Mission risk, advisory, reporting | 44 |
| Real-time API + dashboard | 19 |

---

## Results

### Anomaly detection ablation — held-out test set

The test set contains **168 healthy and 840 degraded windows**. Both classes are
present, so precision, FPR and ROC-AUC are all well-defined.

| Features | Model | Precision | Recall | F1 | FPR | ROC-AUC |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| Raw telemetry | Statistical | 0.8765 | 0.0845 | 0.1542 | 0.0595 | 0.7031 |
| Raw telemetry | Isolation Forest | 0.8571 | 0.0929 | 0.1676 | 0.0774 | 0.6451 |
| Raw telemetry | Autoencoder | 0.9966 | 0.6952 | 0.8191 | 0.0119 | 0.9225 |
| **Twin residuals** | **Statistical** | **1.0000** | **0.8667** | **0.9286** | **0.0000** | **1.0000** |
| Twin residuals | Isolation Forest | 0.9898 | 0.8048 | 0.8877 | 0.0417 | 0.9635 |
| **Twin residuals** | **Autoencoder** | **1.0000** | **0.8690** | **0.9299** | **0.0000** | **1.0000** |
| Hybrid | Statistical | 1.0000 | 0.8667 | 0.9286 | 0.0000 | 1.0000 |
| Hybrid | Isolation Forest | 0.9929 | 0.6702 | 0.8003 | 0.0238 | 0.9158 |
| Hybrid | Autoencoder | 1.0000 | 0.8643 | 0.9272 | 0.0000 | 1.0000 |

### Fault diagnosis — held out on unseen severity (SEV080)

Accuracy **0.9665** · balanced accuracy **0.9464** · macro-F1 **0.9490**

| Class | Precision | Recall | F1 |
| :--- | ---: | ---: | ---: |
| HEALTHY | 1.0000 | 1.0000 | 1.0000 |
| COOLING | 1.0000 | 1.0000 | 1.0000 |
| CYLINDER | 0.9492 | 1.0000 | 0.9739 |
| BEARING | 0.8615 | 1.0000 | 0.9256 |
| LUBRICATION | 1.0000 | 0.7321 | 0.8454 |

Trained on SEV020/SEV040, validated on SEV060, tested on **SEV080 which the model
never saw** — the question is whether the fault *signature* generalises across
severity, not whether the model can interpolate within a severity it memorised.

**Known limitation:** LUBRICATION recall 0.73; 9 windows are called BEARING. This is
physically coherent — lubrication degradation raises friction torque, the same
primary signature as bearing wear — and it is reported rather than tuned away.

### RUL — held out on an unseen engine unit

Health estimation MAE **0.0225** · RUL MAE **52.12 s** over 1,160 mid-flight
predictions · healthy-engine false decay calls **4/171 (2.3%)**.

RUL is in **simulation seconds, not engine flight hours**. The uncertainty band is a
**trend band with 68.5% measured coverage — it is not a calibrated 95% interval**,
and it is labelled as such everywhere it appears.

---

## Live system

```bash
.venv/Scripts/python.exe -m uvicorn AeroTwin.api.server:app --port 8000
```

That single command serves **both** the API and the operator dashboard:

| URL | What |
| :--- | :--- |
| `http://127.0.0.1:8000/` | **Operator dashboard** |
| `http://127.0.0.1:8000/docs` | Interactive API reference |

### Dashboard

A persistent status strip carries health index, anomaly state, fault attribution,
remaining life and mission disposition. Below it, five tabs:

| Tab | Contents |
| :--- | :--- |
| Monitoring | Observed against twin expected for RPM, CHT, EGT, oil pressure and vibration; dispatch assessment with contribution breakdown; per cylinder torque |
| Efficiency | Shaft power and specific fuel consumption against the twin, power deficit and fuel penalty trends, sortie summary |
| Fault alerts | Timestamped log of anomalies, attributions, dispatch changes and health band crossings |
| Maintenance advisory | Prioritised actions with the subsystem, the evidence behind each, and the confirmation step when attribution is weak |
| Mission report | Health at start, end and minimum; findings; efficiency over the sortie; time in each disposition |

The centrepiece is the **observed against twin expected overlay**. The solid line is
the real engine, the dashed line is what a healthy twin of that unit in those
conditions would be doing. The gap between them is the detection principle made
visible: with a cooling fault injected, CHT reads 133.2 °C against a twin expecting
91.8 °C, while a fixed threshold has no way to separate that from a hot day.

Efficiency is reported as deviation from the twin rather than against a book figure.
Measured peak deviation at severity 0.80, which is a physically coherent ordering:

| Injected fault | Power deficit | Fuel penalty |
| :--- | ---: | ---: |
| Cylinder | 8.63% | 7.69% |
| Bearing | 6.98% | 6.19% |
| Lubrication | 1.96% | 1.96% |
| Cooling | 0.69% | 0.69% |

Cooling degradation costs almost no shaft power, which is correct rather than a
defect: it is a heat rejection failure, not a combustion one. It is also the reason
efficiency monitoring alone is insufficient and the thermal residual channel is
needed beside it.

Alerts are logged on **transition**, not per window. Re-logging a persistent
condition every second buries the moment it started under hundreds of duplicates.

React is loaded from **locally vendored UMD builds**, not a CDN, and there is no
build step, because a demo must not depend on venue internet or on a frontend dev
server behaving.

| Endpoint | Purpose |
| :--- | :--- |
| `GET /api/status` | simulation state + which model stages loaded |
| `GET /api/telemetry/latest` | most recent telemetry frame |
| `GET /api/twin/state` | full assessment: anomaly, diagnosis, health, RUL, risk |
| `POST /api/sim/start` | start / restart the sortie |
| `POST /api/sim/throttle` | manual throttle override |
| `POST /api/sim/inject_fault` | inject a degradation (the demo path) |
| `POST /api/mission/assess` | re-assess live state against another mission |
| `GET /api/alerts` | timestamped fault and state change log |
| `GET /api/efficiency` | power and fuel consumption trend against the twin |
| `GET /api/maintenance` | prioritised maintenance advisory |
| `GET /api/mission/report` | per sortie health report |
| `WS /ws/telemetry` | live 10 Hz telemetry + assessment stream |

Verified end-to-end behaviour:

| Injected fault | Diagnosed | Conf. | Anomaly | Health | Decision |
| :--- | :--- | ---: | :--- | ---: | :--- |
| *(none)* | HEALTHY | 0.98 | — | 0.987 | **GO** |
| BEARING | BEARING | 0.96 | flagged | 0.725 | GO_WITH_MONITORING |
| COOLING | COOLING | 0.97 | — | 0.806 | GO_WITH_MONITORING |
| CYLINDER | CYLINDER | 0.98 | flagged | 0.740 | GO_WITH_MONITORING |
| LUBRICATION | *BEARING* | 0.46 | flagged | 0.489 | GO_WITH_MONITORING |

---

## Architecture

```
telemetry (100 Hz)
      |
      v
DigitalTwinStateEngine  ---- condition-matched counterfactual healthy twin
      |                      (same build tolerance + ambient as the observed unit)
      v
  residuals + indicators
      |
      +--> FeatureExtractor (5 s window, 1 s stride)
              |
              +--> Anomaly detection      "is something wrong?"
              +--> Fault diagnosis        "what is wrong, and how sure?"
              +--> Health estimation --> RUL projection    "how long left?"
                                              |
                                              v
                                    MissionRiskAssessor
                                              |
                                              v
                              risk band + GO / NO-GO + reasons
```

```text
AeroTwin/
├── phase1/        Engine physics: crank dynamics, thermal, lubrication, fuel, vibration
├── simulator/     Real-time runtime, flight profiles, 25-field telemetry contract
├── degradation/   Fault injection, trajectories, ground truth, run conditions
├── health/        Digital Twin state engine, residual generation, indicators
├── ml/
│   ├── anomaly/   Statistical · Isolation Forest · PyTorch Autoencoder
│   ├── diagnosis/ 5-class fault attribution + feature importance
│   └── rul/       Health regression + trend projection with uncertainty
├── mission/       Mission reliability, risk scoring, dispatch recommendation
│   ├── advisory.py    Maintenance actions with the evidence behind each
│   └── reporting.py   Efficiency tracking, alert log, per sortie report
└── api/           FastAPI + WebSocket service
    └── static/    Operator dashboard (React, vendored, no build step)
```

---

## Quick start

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
.venv/Scripts/python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu

.venv/Scripts/python.exe -m pytest AeroTwin/ -q          # 176 tests

# Regenerate everything from scratch
.venv/Scripts/python.exe scripts/generate_phase3_dataset.py --full
.venv/Scripts/python.exe scripts/generate_phase4_dataset.py --full
.venv/Scripts/python.exe scripts/generate_rul_dataset.py
.venv/Scripts/python.exe scripts/build_phase5_features.py
.venv/Scripts/python.exe scripts/train_phase5_anomaly_models.py
.venv/Scripts/python.exe scripts/train_fault_diagnosis.py
.venv/Scripts/python.exe scripts/train_rul_model.py
.venv/Scripts/python.exe scripts/evaluate_phase5_models.py
```

`scikit-learn` is **pinned to 1.6.1**. IsolationForest's tree construction changed
between 1.6 and 1.9 and moves the reported ablation F1 from 0.8276 to 0.0822 — the
documented numbers cannot be reproduced without the pin.

---

## Engineering disclaimer

The engine model is a **reduced-order phenomenological** representation of a generic
4-cylinder aero piston engine, not a model of any specific UAV powerplant. Passing
tests confirm mathematical and physical self-consistency within the model's own
equations; they are **not** empirical validation against real or proprietary engine
test-cell data. Severity-to-parameter mappings are engineering assumptions, and the
mission-risk thresholds are demonstration defaults rather than certified
airworthiness limits.

A full record of defects found and corrected — including several previously reported
results that were artifacts of the evaluation setup rather than model performance —
is in **[docs/FIXES_AND_ARCHITECTURE.md](docs/FIXES_AND_ARCHITECTURE.md)**.
