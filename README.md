# PRATIBIMB (AeroTwin-4) — AI-Enabled Physics-Informed Digital Twin for Aero Piston Engines

<div align="center">

**Smart India Hackathon 2026 · Problem Statement 26054**  
**Ministry of Defence · DRDO / Department of Defence Production (IDEX)**  
*AI-Enabled Real-Time Digital Twin System for Health Monitoring, Fault Prediction, and Mission Reliability Enhancement of Aero Piston Engines used in MALE UAVs.*

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg)](https://fastapi.tiangolo.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-EE4C2C.svg)](https://pytorch.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-green.svg)](https://xgboost.readthedocs.io/)
[![TreeSHAP](https://img.shields.io/badge/XAI-TreeSHAP-orange.svg)](https://github.com/slundberg/shap)
[![Airworthiness](https://img.shields.io/badge/Compliance-DFSA--26054-red.svg)](docs/JURY_PITCH_AND_VIVA.md)

[🚀 Quick Start](#-quick-start) • [📐 Architecture](#-system-architecture) • [📊 Results & Benchmarks](#-benchmarks--validation) • [📑 PS Compliance](#-problem-statement-compliance-matrix) • [🎥 Demo Video](#-demonstration-video) • [🎙️ Defense Jury Pitch & Viva](docs/JURY_PITCH_AND_VIVA.md)

</div>

---

## 🎯 Executive Overview

Conventional unmanned aerial vehicle (UAV) engine monitoring relies on **static redline thresholding** (e.g., *CHT > 105 °C* or *Vibration > 1.2 g*). In tactical flight envelopes, static thresholds produce critical blind spots:
- They **cannot distinguish** between an ambient heatwave ($+40^\circ\text{C}$ desert loiter) and a failing cylinder cooling jacket.
- They are **purely reactive**, sounding alarms only after mechanical damage has occurred.

**PRATIBIMB** solves this by running a **condition-matched counterfactual healthy twin** alongside the live engine. Both engines experience identical throttle commands, airspeed, and atmospheric lapse rates (ISA standard: $-6.5^\circ\text{C} / 1,000\,\text{m}$ up to 7,600 m ceiling). The difference between the observed engine and the healthy twin yields **9 physical residual channels**:

$$\mathbf{r}_i(t) = \mathbf{y}_i^{\text{observed}}(t) - \mathbf{y}_i^{\text{twin}}\left(\text{Throttle}(t), \text{Altitude}(t), \text{ISA\_Lapse}(t)\right)$$

By monitoring **residuals instead of raw telemetry**, environmental weather shifts produce zero drift, turning mechanical degradation into clean, high-contrast fault signatures.

---

## 📐 System Architecture

```
                                  UAV PROPULSION SYSTEM
                               (Rotax 914 Turbocharged Engine)
                                             │  50-100 Hz Raw Telemetry
                                             ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ 1. TELEMETRY & SIGNAL CONDITIONING LAYER (< 1.0 ms budget)                             │
│    • Signal Conditioning & Anti-Glitch Filter (stuck sensor detection)                 │
│    • Hardware Safety Watchdog (Redline / Amber limit screening)                        │
│    • Lightweight Downlink Prioritisation Node                                          │
└────────────────────────────────────────────────────────────────────────────────────────┘
                                             │
                                             ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ 2. DIGITAL TWIN CORE (MVEM Physics Engine)                                             │
│    • Coupled 4-Cylinder Crank Dynamics (J dω/dt = T_comb - T_load - T_fric) @ 100 Hz    │
│    • International Standard Atmosphere (ISA) Model (0 - 7,600 m altitude scaling)      │
│    • Condition-Matched Counterfactual Healthy Twin Generator                           │
│    • 9 Physical Residual Channels (RPM, CHT, EGT, Oil P, Oil T, Fuel, Vib, Volt, Inj) │
└────────────────────────────────────────────────────────────────────────────────────────┘
                                             │
                                             ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ 3. AI / ML DIAGNOSTICS & PROGNOSTICS LAYER                                             │
│    ├── Model 1 (Unsupervised Anomaly): LSTM Autoencoder (Reconstruction MSE vs 3σ)     │
│    ├── Model 2 (9-Class Fault Attribution): 1D-CNN + BiLSTM + Attention (98.82% Acc)   │
│    ├── Model 3 (Prognostics / RUL): PINN-LSTM Regressor (0.0 to 50.0 Flight Hours)     │
│    └── Explainable AI (XAI): TreeSHAP & DeepSHAP Real-Time Feature Attribution Drivers │
└────────────────────────────────────────────────────────────────────────────────────────┘
                                             │
                                             ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ 4. MISSION RELIABILITY & TACTICAL DECISION SUPPORT                                     │
│    • Pre-Flight Airworthiness Clearance Engine (GO / CAUTION_GO / NO_GO)               │
│    • Actionable Maintenance Advisor (Prioritised actions, subsystem, AMM references)   │
│    • Form DFSA-26054 Official Sortie Debrief Generator (A4 Native PDF Export)          │
│    • 50 Hz Blackbox Scrubber & Military Operational Scenario Presets                   │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## ⚡ Problem Statement Compliance Matrix

PRATIBIMB fulfills **100% of the core requirements** specified in **SIH 2026 Problem Statement 26054 (DRDO / IDEX)**:

| PS Requirement | Engineering Implementation in PRATIBIMB | Status |
| :--- | :--- | :---: |
| **Real-Time Digital Twin System** | 100 Hz Mean Value Engine Model (MVEM) + 10 Hz WebSocket streaming | ✅ **Complete** |
| **Health Monitoring & Residuals** | 9-channel physical residual generator ($r_i = y_{\text{obs}} - y_{\text{twin}}$) | ✅ **Complete** |
| **Anomaly Detection Beyond Thresholds** | Unsupervised LSTM Autoencoder detecting novel deviations beyond static bounds | ✅ **Complete** |
| **Misfire Conditions** | Cycle-to-cycle torque deficit detection & EGT drop analysis | ✅ **Complete** |
| **Injector Abnormalities** | Fuel-flow / injection timing residual isolation | ✅ **Complete** |
| **Cooling Degradation** | Thermal dissipation ODE tracking with CHT heat-balance drift | ✅ **Complete** |
| **Lubrication Issues** | Oil gallery pressure drop & viscosity friction tracking | ✅ **Complete** |
| **Sensor Drift / Failure** | Decoupled single-channel drift isolation (distinguished from mechanical wear) | ✅ **Complete** |
| **Combustion Instability** | Cyclic combustion variability & torque pulse oscillation analysis | ✅ **Complete** |
| **Overheating Trends** | Coupled CHT/EGT thermal runaway predictor with slope estimation | ✅ **Complete** |
| **Abnormal Vibration Patterns** | Torque unbalance & RMS vibration burst kurtosis detection | ✅ **Complete** |
| **Remaining Useful Life (RUL)** | Physics-Informed Neural Network (PINN-LSTM) bounded prognostic regressor | ✅ **Complete** |
| **Varying Environmental Envelopes** | Full ISA atmospheric lapse rate from 0 m to 7,600 m ceiling (ambient $-33.8^\circ\text{C}$) | ✅ **Complete** |
| **Mission Reliability Enhancement** | Dynamic GO / CAUTION / NO-GO clearance considering 20-min safety reserve | ✅ **Complete** |
| **Operator HMI & Maintenance Reports** | Ground Control Station UI, TreeSHAP bar graphs, and DFSA-26054 PDF debrief | ✅ **Complete** |

---

## 📊 Benchmarks & Validation

### 1. Anomaly Detection Ablation (Held-Out Test Set)
Moving from raw telemetry to condition-matched twin residuals completely eliminates false alarms while maximizing detection recall:

| Features Used | Model | Precision | Recall | F1 Score | FPR | ROC-AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Raw Telemetry | Statistical Redlines | 0.8765 | 0.0845 | 0.1542 | 0.0595 | 0.7031 |
| Raw Telemetry | Isolation Forest | 0.8571 | 0.0929 | 0.1676 | 0.0774 | 0.6451 |
| Raw Telemetry | PyTorch Autoencoder | 0.9966 | 0.6952 | 0.8191 | 0.0119 | 0.9225 |
| **Twin Residuals** | **PyTorch Autoencoder** | **1.0000** | **0.8690** | **0.9299** | **0.0000** | **1.0000** |
| **Twin Residuals** | **XGBoost Classifier** | **0.9942** | **0.9882** | **0.9912** | **0.0012** | **0.9998** |

### 2. Multi-Class Fault Attribution (Unseen Test Engines)
Trained on 42 engines, validated on 9 engines, and tested on **9 completely held-out engines**:
- **Overall Test Accuracy:** **98.82%**
- **Balanced Macro-F1:** **0.9854**
- **Inference Latency:** **$< 3.8\,\text{ms}$** per 30-second rolling window.

---

## 🚀 Quick Start

### Prerequisites
- Python 3.9, 3.10, or 3.11
- Modern web browser (Chrome, Edge, Safari, Firefox)

### Installation & Launch

```bash
# 1. Clone the repository
git clone https://github.com/tp318/PRATIBIMB1.git
cd PRATIBIMB1

# 2. Create and activate a virtual environment
python -m venv .venv
# On macOS / Linux:
source .venv/bin/activate
# On Windows:
# .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the Turnkey Dashboard Server
cd "DASHBOARD AND DATA"
python run_dashboard.py
```

### Access Points
- **Tactical Operator Dashboard:** [http://localhost:8001/](http://localhost:8001/)
- **Interactive OpenAPI / Swagger Documentation:** [http://localhost:8001/docs](http://localhost:8001/docs)
- **High-Rate Telemetry WebSocket:** `ws://localhost:8001/ws/telemetry`

---

## 🎮 How to Demo the Prototype

1. **Launch the Engine:**  
   Click **"Start simulation"** on the dashboard. The Rotax 914 engine starts up; observe 10 Hz telemetry streaming on the dials and twin overlays.
2. **Explore the Digital Twin Overlay:**  
   The solid blue line indicates observed real-time telemetry; the dashed gray line indicates the condition-matched healthy twin.
3. **Test Altitude & Weather Physics:**  
   Adjust the altitude slider up to **7,500 m**. Notice how CHT drops according to the ISA atmospheric model without triggering a false alarm (residuals remain near zero).
4. **Inject In-Flight Faults:**  
   Click **"Inject fault"** and select **COOLING SYSTEM FAILURE** (Severity 0.75). Observe:
   - CHT begins rising and departs from the healthy twin.
   - Anomaly detector flags an alert.
   - 9-Class classifier shifts from `NORMAL` $\rightarrow$ `OVERHEATING`.
   - **TreeSHAP Explainability** visualizes `cht` (+45.8%) and `oil_temperature_slope` (+24.9%) as the primary positive risk drivers.
5. **Export Official Sortie Debrief (PDF):**  
   Navigate to the **"Mission report"** tab and click **"🖨️ Generate Flight Debrief (PDF)"**. An official Directorate of Flight Safety & Airworthiness (DFSA) military sortie record generates with vector SVG health curves and maintenance sign-off blocks. Click **"Print / Save as PDF"** to export.
6. **Pre-Flight Clearance & 50 Hz Blackbox Scrubber:**  
   Navigate to the **"Replay & Simulation"** tab. Inspect the **GO / CAUTION / NO-GO Airworthiness Clearance**, run military preset scenarios (High Altitude, Desert Loiter), or scrub through 50 Hz fleet blackbox trajectories.

---

## 🎥 Demonstration Video

A comprehensive high-definition recording showcasing the working prototype, physics-informed digital twin, fault injection, explainable AI, and official PDF debrief generation is embedded below:

<div align="center">

![PRATIBIMB Prototype Live Walkthrough](docs/media/demo_prototype_walkthrough.webp)

*Interactive prototype walkthrough demonstrating real-time 10 Hz telemetry, counterfactual twin residuals, cooling fault injection, and Form DFSA-26054 military debrief generation.*

</div>

- **Walkthrough Media File in Repository:** [`docs/media/demo_prototype_walkthrough.webp`](docs/media/demo_prototype_walkthrough.webp)
- **Comprehensive Defense Jury Pitch & Viva Guide:** [`docs/JURY_PITCH_AND_VIVA.md`](docs/JURY_PITCH_AND_VIVA.md)

---

## 📂 Repository & Documentation Guide

| Subsystem / Document | Description | Documentation Link |
| :--- | :--- | :--- |
| **Tactical GCS & Dashboard** | Turnkey FastAPI server, WebSockets & React interface | [`DASHBOARD AND DATA/`](DASHBOARD%20AND%20DATA/README.md) |
| **GCS Tactical Frontend Client** | Standalone pilot telemetry UI & audio alert engine | [`frontend/`](frontend/) |
| **Physics Digital Twin Core** | 100 Hz Mean Value Engine Model (MVEM) ODE simulator | [`DT CORE/`](DT%20CORE/README.md) |
| **Anomaly Detection Engine** | Unsupervised LSTM autoencoders for zero-drift alerting | [`ANOMALY DETECTION/`](ANOMALY%20DETECTION/README.md) |
| **Fault Diagnostics & XAI** | 9-class CNN-BiLSTM & XGBoost with TreeSHAP explainability | [`FAULT DETECTION/`](FAULT%20DETECTION/README.md) |
| **Prognostics & RUL** | Physics-Informed LSTM RUL degradation predictor (0-50h) | [`RUL ESTIMATION/`](RUL%20ESTIMATION/README.md) |
| **Mission Replay & Scrubber** | 50 Hz fleet blackbox scrubber & SQLite mission DB | [`REPLAY AND SIMULATION/`](REPLAY%20AND%20SIMULATION/README.md) |
| **Health Monitoring Layer** | Health score calculation & sensor validation | [`HEALTH MONITORING/`](HEALTH%20MONITORING/README.md) |
| **Mission Simulator** | Standalone real-time MVEM streaming simulator | [`MISSION_SIMULATOR/`](MISSION_SIMULATOR/README.md) |
| **Integration Architecture** | Multi-service orchestration & data pipeline guide | [`README_INTEGRATION.md`](README_INTEGRATION.md) |
| **Jury Pitch & Viva Guide** | 2-minute defense pitch, live demo script & 10 hostile Q&As | [`docs/JURY_PITCH_AND_VIVA.md`](docs/JURY_PITCH_AND_VIVA.md) |
| **Engineering Specifications**| Multi-phase architecture, physics formulas & benchmark specs | [`DASHBOARD AND DATA/docs/`](DASHBOARD%20AND%20DATA/docs/) |

---

## 🛡️ Airworthiness & Engineering Disclaimer

PRATIBIMB is a **physics-informed reduced-order phenomenological digital twin** calibrated against published specifications of the **Rotax 914** aircraft piston engine. Operational dispatch thresholds, maintenance intervals, and degradation trajectories demonstrate mathematical and algorithmic self-consistency and represent demonstration baselines rather than certified civilian/military airworthiness limits.

---

<div align="center">
<b>Project PRATIBIMB · Smart India Hackathon 2026 · Defence Research & Development Organisation (DRDO)</b>
</div>
