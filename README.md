# PRATIBIMB (AeroTwin-4)
### AI-Enabled Physics-Informed Digital Twin for Aero Piston Engines

**Smart India Hackathon 2026 | Problem Statement 26054**  
**Ministry of Defence | Defence Research & Development Organisation (DRDO) / IDEX**  
*Health Monitoring, Fault Prediction, and Mission Reliability Enhancement of Aero Piston Engines used in MALE UAVs.*

<div align="center">

[![Live Web Application](https://img.shields.io/badge/Live%20Demo-pratibimb--1.vercel.app-blue?style=for-the-badge&logo=vercel&logoColor=white)](https://pratibimb-1.vercel.app/)
[![Cloud Backend](https://img.shields.io/badge/Cloud%20API-Render%20Live-46E3B7?style=for-the-badge&logo=render&logoColor=white)](https://pratibimb1-3.onrender.com/docs)

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-EE4C2C.svg?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-green.svg?style=flat)](https://xgboost.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/XAI-TreeSHAP-orange.svg?style=flat)](https://github.com/slundberg/shap)
[![Standard](https://img.shields.io/badge/Compliance-DFSA--26054-red.svg?style=flat)]()

</div>

---

## Live Deployment & Access Points

The complete PRATIBIMB digital twin system is deployed and accessible online:

- **Live Tactical Ground Station (Web):** [https://pratibimb-1.vercel.app/](https://pratibimb-1.vercel.app/)
- **Live Cloud Backend API (Render):** [https://pratibimb1-3.onrender.com](https://pratibimb1-3.onrender.com)
- **Interactive OpenAPI / Swagger Documentation:** [https://pratibimb1-3.onrender.com/docs](https://pratibimb1-3.onrender.com/docs)
- **High-Rate Cloud Telemetry Stream:** `wss://pratibimb1-3.onrender.com/ws/telemetry`

*(For local execution instructions, see [Local Development & Setup](#local-development--setup)).*

---

## Executive Overview

Conventional unmanned aerial vehicle (UAV) engine monitoring relies on static redline thresholds (e.g., Cylinder Head Temperature > 105 °C or vibration > 1.2 g). In operational flight regimes, static thresholds exhibit critical blind spots:
1. **Environmental Blind Spots:** Changes in ambient conditions between low-altitude desert loiter (+40 °C) and ceiling altitude (7,600 m, -34 °C under ISA lapse rates) induce large temperature shifts. Static redlines cannot distinguish between atmospheric shifts and genuine mechanical cooling jacket failure.
2. **Reactive Failure Warning:** Static alarms trigger only after component degradation has breached safety limits, leaving zero margin for preemptive mission aborts.

**PRATIBIMB** resolves this by operating a **condition-matched counterfactual healthy digital twin** in real time alongside observed engine telemetry. Both the live engine and the digital twin share identical throttle commands, flight velocity, altitude, and ISA atmospheric conditions. Subtracting the twin prediction from observed telemetry yields **9 physical residual channels**:

$$r_i(t) = y_i^{\text{observed}}(t) - y_i^{\text{twin}}\left(\text{Throttle}(t), \text{Altitude}(t), \text{ISA}(t)\right)$$

By evaluating **residuals instead of raw sensor values**, environmental weather shifts produce zero baseline drift, turning incipient mechanical faults into sharp, high-contrast diagnostic signatures.

---

## System Architecture

```
                                  UAV PROPULSION SYSTEM
                             (Rotax 914 Turbocharged Engine)
                                           |  50-100 Hz Raw Telemetry
                                           v
+----------------------------------------------------------------------------------------+
| 1. TELEMETRY & SIGNAL CONDITIONING LAYER (< 1.0 ms latency)                            |
|    - Anti-glitch and stuck-sensor filtering                                            |
|    - Hardware safety boundary screening (amber and redline limits)                     |
|    - Channel isolation and telemetry downsampling                                      |
+----------------------------------------------------------------------------------------+
                                           |
                                           v
+----------------------------------------------------------------------------------------+
| 2. DIGITAL TWIN CORE (Mean Value Engine Model)                                         |
|    - 4-Cylinder coupled crank dynamics: J * domega/dt = T_comb - T_load - T_fric       |
|    - International Standard Atmosphere (ISA) model (0 to 7,600 m ceiling)              |
|    - Condition-matched counterfactual healthy reference generator                      |
|    - 9 Physical residual channels (RPM, CHT, EGT, Oil P, Oil T, Fuel, Vib, Volt, Inj)  |
+----------------------------------------------------------------------------------------+
                                           |
                                           v
+----------------------------------------------------------------------------------------+
| 3. DIAGNOSTICS & PROGNOSTICS LAYER                                                     |
|    - Unsupervised Anomaly Detection: PyTorch LSTM Autoencoder (reconstruction error)   |
|    - 9-Class Fault Classification: CNN-BiLSTM & XGBoost Classifier (98.82% accuracy)   |
|    - Remaining Useful Life (RUL): Physics-Informed Neural Network (0.0 to 50.0 hours)   |
|    - Explainable AI: Real-time TreeSHAP feature attribution bar plots                  |
+----------------------------------------------------------------------------------------+
                                           |
                                           v
+----------------------------------------------------------------------------------------+
| 4. MISSION RELIABILITY & TACTICAL DECISION SUPPORT                                     |
|    - Pre-flight dispatch clearance engine (GO / CAUTION / NO-GO status)                |
|    - Actionable maintenance advisor linked to standard aircraft maintenance manuals    |
|    - Form DFSA-26054 official sortie debrief report generator (native A4 PDF export)   |
|    - 50 Hz flight blackbox scrubber and operational military preset scenarios          |
+----------------------------------------------------------------------------------------+
```

---

## Physics & Mathematical Foundation

The physics core models the Rotax 914 four-stroke, four-cylinder turbocharged piston engine using coupled ordinary differential equations (ODEs):

1. **Rotational Dynamics:**
   The crankshaft angular velocity $\omega$ is modeled by:
   $$J \frac{d\omega}{dt} = T_{\text{comb}}(\text{throttle}, P_{\text{manifold}}, \lambda) - T_{\text{prop}}(\omega, V_{\text{air}}, \rho) - T_{\text{fric}}(\omega, \mu_{\text{oil}})$$
   where $J$ is the rotational inertia of the assembly.

2. **Thermodynamics & Heat Balance:**
   Cylinder Head Temperature (CHT) and Exhaust Gas Temperature (EGT) are derived from fuel combustion energy and convection heat dissipation:
   $$C_{\text{thermal}} \frac{d\text{CHT}}{dt} = \dot{Q}_{\text{combustion}}(\dot{m}_f, \text{AFR}) - h_{\text{cooling}}(V_{\text{air}}, \rho_{\text{ambient}}) \cdot (\text{CHT} - T_{\text{ambient}})$$

3. **Atmospheric Modeling (ISA):**
   Ambient conditions adjust with geometric altitude $h$ according to the International Standard Atmosphere model up to the troposphere boundary:
   - $T(h) = T_0 - L \cdot h$, with $T_0 = 288.15\,\text{K}$ and lapse rate $L = 0.0065\,\text{K/m}$.
   - $P(h) = P_0 \left(1 - \frac{L \cdot h}{T_0}\right)^{\frac{g_0}{R \cdot L}}$.

4. **Residual Extraction:**
   Nine continuous residual channels decouple environmental variations from physical faults:
   - $\Delta \text{RPM}$ (Speed discrepancy)
   - $\Delta \text{CHT}$ (Thermal dissipation anomaly)
   - $\Delta \text{EGT}$ (Combustion energy deviation)
   - $\Delta P_{\text{oil}}$ (Lubrication gallery pressure drop)
   - $\Delta T_{\text{oil}}$ (Oil friction thermal rise)
   - $\Delta \dot{m}_{\text{fuel}}$ (Fuel flow rate anomaly)
   - $\Delta \text{Vib}$ (Vibrational RMS & kurtosis shift)
   - $\Delta V_{\text{bus}}$ (Alternator / electrical bus load)
   - $\Delta \text{PW}_{\text{inj}}$ (Injector pulse width compensation)

---

## Problem Statement Compliance Matrix

| Problem Statement Requirement | Technical Implementation | Status |
| :--- | :--- | :---: |
| Real-Time Digital Twin System | 100 Hz Mean Value Engine Model with 10 Hz WebSocket telemetry streaming | Complete |
| Health Monitoring & Residuals | 9-channel physical residual generator isolating mechanical wear from ambient conditions | Complete |
| Anomaly Detection Beyond Thresholds | PyTorch LSTM Autoencoder identifying deviations before static redline violations | Complete |
| Misfire Conditions | Cycle-to-cycle torque deficit detection and EGT drop tracking | Complete |
| Injector Abnormalities | Fuel flow rate and injection pulse width residual analysis | Complete |
| Cooling Degradation | Heat balance ODE tracking with cylinder head thermal dissipation monitoring | Complete |
| Lubrication Issues | Oil gallery pressure drop and viscosity-induced friction tracking | Complete |
| Sensor Drift / Calibration Loss | Single-channel residual drift isolation distinguished from multi-sensor mechanical faults | Complete |
| Combustion Instability | Torque pulse variance and cyclic combustion variability detection | Complete |
| Overheating Trends | Coupled CHT and EGT thermal runaway prediction with rate-of-rise tracking | Complete |
| Abnormal Vibration Patterns | Torque unbalance and RMS vibration burst kurtosis detection | Complete |
| Remaining Useful Life (RUL) | Physics-informed prognostic regressor calibrated for 0 to 50 flight hours | Complete |
| Variable Environmental Envelopes | International Standard Atmosphere lapse model from sea level to 7,600 m ceiling | Complete |
| Mission Reliability Enhancement | Dynamic GO / CAUTION / NO-GO dispatch clearance engine with safety margin assessment | Complete |
| Operator Interface & Reporting | Ground control station web interface and Form DFSA-26054 sortie debrief PDF export | Complete |

---

## Benchmarks & Validation Results

### 1. Anomaly Detection Ablation (Held-Out Test Engines)
Evaluating residuals instead of raw telemetry removes ambient environmental drift, reducing false alarms to zero while maximizing detection recall:

| Input Feature Set | Evaluation Model | Precision | Recall | F1 Score | False Positive Rate | ROC-AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Raw Telemetry | Static Redlines | 0.8765 | 0.0845 | 0.1542 | 0.0595 | 0.7031 |
| Raw Telemetry | Isolation Forest | 0.8571 | 0.0929 | 0.1676 | 0.0774 | 0.6451 |
| Raw Telemetry | PyTorch Autoencoder | 0.9966 | 0.6952 | 0.8191 | 0.0119 | 0.9225 |
| **Twin Residuals** | **PyTorch Autoencoder** | **1.0000** | **0.8690** | **0.9299** | **0.0000** | **1.0000** |
| **Twin Residuals** | **XGBoost Classifier** | **0.9942** | **0.9882** | **0.9912** | **0.0012** | **0.9998** |

### 2. Multi-Class Fault Classification (9 Completely Held-Out Engines)
- **Overall Test Accuracy:** 98.82%
- **Macro-Averaged F1:** 0.9854
- **Inference Latency:** < 3.8 ms per 30-second rolling window

---

## Prototype Walkthrough Guide

Follow these steps on either the [Live Vercel Deployment](https://pratibimb-1.vercel.app/) or a local instance:

1. **Initialize Engine Simulation:**  
   Click **"Start simulation"** on the dashboard. The Rotax 914 model initializes, streaming telemetry at 10 Hz across primary flight instruments and digital twin comparison plots.
2. **Observe Digital Twin Residual Overlays:**  
   The solid blue line indicates observed telemetry, while the dashed gray line indicates the condition-matched healthy twin baseline.
3. **Verify Environmental Invariance (Altitude Scaling):**  
   Move the altitude slider to **7,500 m**. Notice how ambient temperature and pressure adjust according to the ISA model. Cylinder temperatures drop as expected, yet residuals remain centered around zero, demonstrating that weather changes do not produce false alarms.
4. **Inject In-Flight Mechanical Fault:**  
   Click **"Inject fault"** and select **"COOLING SYSTEM FAILURE"** (Severity 0.75).
   - Cylinder Head Temperature departs from the healthy twin baseline.
   - The anomaly detector flags the deviation.
   - The 9-class classifier shifts from `NORMAL` to `OVERHEATING`.
   - TreeSHAP feature importance displays `cht` (+45.8%) and `oil_temperature_slope` (+24.9%) as primary risk drivers.
5. **Generate Official Sortie Debrief (Form DFSA-26054):**  
   Navigate to the **"Mission report"** tab and click **"Generate Flight Debrief (PDF)"**. A standardized military maintenance debrief document generates with vector SVG health degradation curves, dispatch airworthiness status, and 3-party maintenance sign-off blocks. Use the browser print button to save as a PDF.
6. **Airworthiness Clearance & 50 Hz Blackbox Replay:**  
   Navigate to the **"Replay & Simulation"** tab to inspect the GO / CAUTION / NO-GO dispatch clearance calculations, test military flight scenarios (Desert Loiter, High Altitude), or scrub through 50 Hz recorded blackbox data.

---

## Demonstration Walkthrough Recording

A demonstration recording showcasing the live prototype, digital twin tracking, fault injection, explainable AI, and official debrief generation is embedded below:

<div align="center">

![PRATIBIMB Prototype Walkthrough](docs/media/demo_prototype_walkthrough.webp)

*Interactive prototype demonstration: 10 Hz telemetry streaming, counterfactual twin residuals, cooling fault injection, and Form DFSA-26054 debrief export.*

</div>

- **Walkthrough Media File:** [`docs/media/demo_prototype_walkthrough.webp`](docs/media/demo_prototype_walkthrough.webp)

---

## Repository & Module Directory

| Directory / File | Description | Link |
| :--- | :--- | :--- |
| **frontend/** | Standalone ground control station web client deployed on Vercel | [`frontend/`](frontend/) |
| **DASHBOARD AND DATA/** | FastAPI backend, WebSockets, and integrated operator interface | [`DASHBOARD AND DATA/`](DASHBOARD%20AND%20DATA/README.md) |
| **DT CORE/** | 100 Hz Mean Value Engine Model solving coupled rotational and thermal ODEs | [`DT CORE/`](DT%20CORE/README.md) |
| **ANOMALY DETECTION/** | PyTorch LSTM Autoencoders for unsupervised novelty detection | [`ANOMALY DETECTION/`](ANOMALY%20DETECTION/README.md) |
| **FAULT DETECTION/** | 9-Class fault classifier with TreeSHAP feature attribution | [`FAULT DETECTION/`](FAULT%20DETECTION/README.md) |
| **RUL ESTIMATION/** | Physics-informed prognostic models predicting remaining safe flight hours | [`RUL ESTIMATION/`](RUL%20ESTIMATION/README.md) |
| **REPLAY AND SIMULATION/** | 50 Hz flight blackbox scrubber and SQLite mission database | [`REPLAY AND SIMULATION/`](REPLAY%20AND%20SIMULATION/README.md) |
| **HEALTH MONITORING/** | Health score calculation modules and sensor cross-validation | [`HEALTH MONITORING/`](HEALTH%20MONITORING/README.md) |
| **MISSION_SIMULATOR/** | Standalone real-time MVEM streaming simulator | [`MISSION_SIMULATOR/`](MISSION_SIMULATOR/README.md) |
| **README_INTEGRATION.md** | Multi-service orchestration guide and data pipeline documentation | [`README_INTEGRATION.md`](README_INTEGRATION.md) |
| **DASHBOARD AND DATA/docs/** | Detailed mathematical specifications and validation reports | [`DASHBOARD AND DATA/docs/`](DASHBOARD%20AND%20DATA/docs/) |

---

## Local Development & Setup

### Prerequisites
- Python 3.9, 3.10, or 3.11
- Modern web browser (Chrome, Edge, Firefox, Safari)

### Setup & Launch

```bash
# 1. Clone the repository
git clone https://github.com/tp318/PRATIBIMB1.git
cd PRATIBIMB1

# 2. Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate    # On Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start the local server (Port 8001)
cd "DASHBOARD AND DATA"
python run_dashboard.py
```

### Local URLs
- Local Operator Dashboard: `http://localhost:8001/`
- Local API Documentation: `http://localhost:8001/docs`
- Local Telemetry WebSocket: `ws://localhost:8001/ws/telemetry`

---

## Airworthiness & Engineering Disclaimer

PRATIBIMB is a physics-informed reduced-order digital twin calibrated against published technical data for the Rotax 914 aircraft piston engine. Dispatch thresholds, maintenance recommendations, and degradation models are implemented to demonstrate algorithmic integrity and mathematical self-consistency for research and evaluation purposes.

---

<div align="center">
<b>Project PRATIBIMB | Smart India Hackathon 2026 | Defence Research & Development Organisation (DRDO)</b>
</div>
