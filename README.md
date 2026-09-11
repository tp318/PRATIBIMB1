# PRATIBIMB: AI-Enabled Physics-Informed Digital Twin for Aero Piston Engines

Smart India Hackathon 2026 | Problem Statement 26054  
Ministry of Defence | DRDO / Department of Defence Production (IDEX)  
*Health Monitoring, Fault Prediction, and Mission Reliability Enhancement for Aero Piston Engines in MALE UAVs.*

[System Overview](#system-overview) | [System Architecture](#system-architecture) | [Physics and Mathematical Foundation](#physics-and-mathematical-foundation) | [Machine Learning and Diagnostics](#machine-learning-and-diagnostics) | [Compliance Matrix](#compliance-matrix) | [Installation and Quick Start](#installation-and-quick-start) | [Prototype Demonstration](#prototype-demonstration) | [Demonstration Video](#demonstration-video) | [Repository Structure](#repository-structure)

---

## System Overview

Medium Altitude Long Endurance (MALE) Unmanned Aerial Vehicles rely heavily on turbocharged four-cylinder piston engines (such as the Rotax 914) to maintain sustained patrol endurance. Traditional health monitoring relies on static redline thresholds (for example, Cylinder Head Temperature exceeding 105 degrees Celsius or vibration amplitude exceeding 1.2 g). 

In operational flight, static thresholds introduce two major drawbacks:
1. **Environmental ambiguity:** Ambient temperature changes between low-altitude loiter (+40 deg C) and ceiling altitude (7,600 m, -34 deg C under ISA standard lapse rates) shift baseline temperatures significantly, causing false alarms or masking genuine cooling failures.
2. **Reactive warning:** Static thresholds trigger only after thermal or mechanical limits have been violated, leaving minimal time for tactical recovery.

PRATIBIMB addresses this by running a **condition-matched counterfactual healthy digital twin** in real time alongside live telemetry. Both the aircraft engine and the computational twin receive identical inputs: throttle command, airspeed, altitude, and ISA atmospheric conditions. The residual difference between the observed engine parameters and the healthy twin predictions isolates mechanical degradation from environmental variations:

```
Residual r_i(t) = y_observed_i(t) - y_twin_i(Throttle(t), Altitude(t), Ambient(t))
```

Evaluating residuals rather than raw telemetry eliminates altitude and temperature drift, allowing machine learning models to detect subtle degradation long before critical thresholds are reached.

---

## System Architecture

```
                                  UAV PROPULSION SYSTEM
                             (Rotax 914 Turbocharged Engine)
                                           |  50-100 Hz Raw Telemetry
                                           v
+----------------------------------------------------------------------------------------+
| 1. TELEMETRY & SIGNAL CONDITIONING LAYER                                                |
|    - Signal conditioning and anti-glitch filtering (stuck sensor detection)            |
|    - Safety boundary screening (amber and redline checks)                              |
|    - Data validation and channel isolation                                            |
+----------------------------------------------------------------------------------------+
                                           |
                                           v
+----------------------------------------------------------------------------------------+
| 2. DIGITAL TWIN CORE (Mean Value Engine Model)                                         |
|    - Coupled 4-cylinder rotational dynamics: J * domega/dt = T_comb - T_load - T_fric   |
|    - International Standard Atmosphere (ISA) model (0 to 7,600 m ceiling)              |
|    - Condition-matched counterfactual healthy reference generator                      |
|    - 9 Physical residual channels (RPM, CHT, EGT, Oil P, Oil T, Fuel, Vib, Volt, Inj)  |
+----------------------------------------------------------------------------------------+
                                           |
                                           v
+----------------------------------------------------------------------------------------+
| 3. DIAGNOSTICS & PROGNOSTICS LAYER                                                     |
|    - Unsupervised Anomaly Detection: PyTorch LSTM Autoencoder (reconstruction error)   |
|    - 9-Class Fault Attribution: CNN-BiLSTM & XGBoost Classifier                        |
|    - Remaining Useful Life (RUL): Physics-Informed Neural Network (PINN-LSTM)          |
|    - Explainable AI: TreeSHAP real-time feature attribution                            |
+----------------------------------------------------------------------------------------+
                                           |
                                           v
+----------------------------------------------------------------------------------------+
| 4. MISSION RELIABILITY & TACTICAL DECISION SUPPORT                                     |
|    - Pre-flight dispatch clearance engine (GO / CAUTION / NO-GO status)                |
|    - Actionable maintenance recommendations linked to subsystem procedures             |
|    - Form DFSA-26054 official sortie debrief report generator (native PDF)             |
|    - 50 Hz blackbox flight replay and operational scenario scrubber                    |
+----------------------------------------------------------------------------------------+
```

---

## Physics and Mathematical Foundation

The physics core models the Rotax 914 four-stroke, four-cylinder turbocharged piston engine using coupled ordinary differential equations (ODEs):

1. **Rotational Dynamics:**
   The crankshaft angular velocity `omega` is governed by:
   `J * (domega / dt) = T_comb(throttle, manifold_p, lambda) - T_prop(omega, airspeed, rho) - T_friction(omega, oil_viscosity)`
   where `J` is the combined rotational inertia of the crank, flywheel, and propeller.

2. **Thermodynamic & Heat Balance:**
   Cylinder Head Temperature (CHT) and Exhaust Gas Temperature (EGT) are derived from fuel energy input and heat dissipation:
   `C_thermal * (dCHT / dt) = Q_combustion(fuel_flow, AF_ratio) - h_cooling(airspeed, rho_ambient) * (CHT - T_ambient)`
   
3. **Atmospheric Modeling (ISA):**
   Ambient temperature and pressure vary with geometric altitude `h` according to the International Standard Atmosphere model up to the troposphere boundary (11,000 m):
   - `T(h) = T_0 - L * h`, where `T_0 = 288.15 K` and lapse rate `L = 0.0065 K/m`.
   - `P(h) = P_0 * (1 - L * h / T_0)^(g_0 / (R * L))`.

4. **Residual Generation:**
   Nine continuous residual channels are tracked:
   - RPM residual (`Delta RPM`)
   - Cylinder Head Temperature residual (`Delta CHT`)
   - Exhaust Gas Temperature residual (`Delta EGT`)
   - Oil Pressure residual (`Delta Oil_P`)
   - Oil Temperature residual (`Delta Oil_T`)
   - Fuel Flow residual (`Delta Fuel_Rate`)
   - Vibration RMS residual (`Delta Vib`)
   - Bus Voltage residual (`Delta Volt`)
   - Fuel Injector Pulse Width residual (`Delta Inj_PW`)

---

## Machine Learning and Diagnostics

The diagnostic pipeline combines unsupervised anomaly detection, supervised fault classification, and prognostic estimation:

1. **Unsupervised Anomaly Detection:**
   A PyTorch-based LSTM Autoencoder models normal flight dynamics over rolling 30-second temporal windows. When engine degradation causes the reconstruction error to exceed an empirically calibrated 3-sigma threshold, an anomaly alert is raised.

2. **9-Class Fault Classification:**
   The classification engine categorizes operational status into 9 distinct classes:
   - Normal Operation
   - Cooling System Degradation
   - Lubrication System Failure
   - Spark / Ignition Misfire
   - Fuel Injector Abnormality
   - Sensor Drift / Failure
   - Turbocharger / Manifold Leak
   - Structural / Bearing Vibration
   - Severe Overheating Trend

3. **Explainable AI (TreeSHAP):**
   Predictions are accompanied by real-time SHAP feature importance values, identifying the specific physical sensors driving the model's decision (for example, isolating whether an alert is driven by cooling jacket failure or oil pressure loss).

4. **Remaining Useful Life (RUL) Prognostics:**
   A Physics-Informed LSTM regressor predicts the remaining safe flight hours (0.0 to 50.0 hours) before critical maintenance intervention is mandatory.

---

## Compliance Matrix

| Problem Statement Requirement | System Implementation | Verification Status |
| :--- | :--- | :--- |
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

## Benchmarks and Validation

### 1. Anomaly Detection Performance (Held-Out Test Engines)

Evaluating residuals instead of raw telemetry removes ambient environmental drift, reducing false alarms while maximizing detection rate:

| Input Feature Set | Evaluation Model | Precision | Recall | F1 Score | False Positive Rate | ROC-AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Raw Telemetry | Static Redlines | 0.8765 | 0.0845 | 0.1542 | 0.0595 | 0.7031 |
| Raw Telemetry | Isolation Forest | 0.8571 | 0.0929 | 0.1676 | 0.0774 | 0.6451 |
| Raw Telemetry | PyTorch Autoencoder | 0.9966 | 0.6952 | 0.8191 | 0.0119 | 0.9225 |
| **Twin Residuals** | **PyTorch Autoencoder** | **1.0000** | **0.8690** | **0.9299** | **0.0000** | **1.0000** |
| **Twin Residuals** | **XGBoost Classifier** | **0.9942** | **0.9882** | **0.9912** | **0.0012** | **0.9998** |

### 2. Multi-Class Fault Classification (Unseen Test Engines)
Trained across 42 engines, validated on 9 engines, and evaluated on 9 completely held-out engines:
- **Test Accuracy:** 98.82%
- **Macro-Averaged F1:** 0.9854
- **Inference Latency:** < 3.8 ms per 30-second rolling window

---

## Installation and Quick Start

### Prerequisites
- Python 3.9, 3.10, or 3.11
- Modern web browser (Chrome, Edge, Firefox, Safari)

### Setup Instructions

```bash
# 1. Clone the repository
git clone https://github.com/tp318/PRATIBIMB1.git
cd PRATIBIMB1

# 2. Set up virtual environment
python -m venv .venv
# On Linux / macOS:
source .venv/bin/activate
# On Windows:
# .venv\Scripts\activate

# 3. Install required dependencies
pip install -r requirements.txt

# 4. Start the dashboard server
cd "DASHBOARD AND DATA"
python run_dashboard.py
```

### Access URLs
- **Operator Dashboard:** [http://localhost:8001/](http://localhost:8001/)
- **Interactive API Documentation:** [http://localhost:8001/docs](http://localhost:8001/docs)
- **Real-Time Telemetry WebSocket:** `ws://localhost:8001/ws/telemetry`

---

## Prototype Demonstration

Follow these steps to evaluate the system:

1. **Start Engine Simulation:**  
   Click "Start simulation" on the dashboard. The Rotax 914 engine model will initialize, streaming live telemetry at 10 Hz across the primary gauges and digital twin comparison plots.
2. **Examine Twin Residual Overlays:**  
   The solid blue line indicates observed telemetry, while the dashed gray line indicates the condition-matched healthy twin. Notice how closely the values track during normal operation.
3. **Evaluate Altitude Scaling:**  
   Move the altitude slider to 7,500 m. The ambient temperature and pressure will adjust according to the ISA model. Cylinder temperatures decrease as expected, but residuals remain centered around zero, confirming that environmental changes do not trigger false alarms.
4. **Inject Mechanical Fault:**  
   Click "Inject fault" and choose "COOLING SYSTEM FAILURE" with severity 0.75.
   - Cylinder Head Temperature will begin diverging from the healthy twin baseline.
   - The anomaly detector will flag the deviation.
   - The fault classifier will transition from "NORMAL" to "OVERHEATING".
   - TreeSHAP feature importance will update, displaying `cht` and `oil_temperature_slope` as the primary risk contributors.
5. **Generate Flight Debrief:**  
   Navigate to the "Mission report" tab and click "Generate Flight Debrief (PDF)". The system generates a standardized Form DFSA-26054 military maintenance record containing the health degradation curve, TreeSHAP diagnostic attribution, dispatch recommendation, and engineering sign-off fields. Use the browser print function to save the document as a PDF.
6. **Airworthiness Clearance and Flight Replay:**  
   Navigate to the "Replay & Simulation" tab to review the GO / CAUTION / NO-GO dispatch clearance calculations, run preset operational profiles (such as Desert Loiter or High-Altitude Patrol), or replay recorded 50 Hz blackbox data.

---

## Demonstration Video

A demonstration recording showcasing the live prototype, digital twin tracking, fault injection, explainability, and report generation is available in the repository:

<div align="center">

![PRATIBIMB Prototype Walkthrough](docs/media/demo_prototype_walkthrough.webp)

*Interactive prototype demonstration: 10 Hz telemetry streaming, counterfactual twin residuals, cooling fault injection, and Form DFSA-26054 debrief export.*

</div>

- **Walkthrough Media File in Repository:** [`docs/media/demo_prototype_walkthrough.webp`](docs/media/demo_prototype_walkthrough.webp)

---

## Repository Structure

| Directory / File | Description | Link |
| :--- | :--- | :--- |
| **DASHBOARD AND DATA/** | FastApi backend server, WebSocket pipeline, and embedded operator interface | [`DASHBOARD AND DATA/`](DASHBOARD%20AND%20DATA/README.md) |
| **frontend/** | Standalone ground control station web client and audio alert engine | [`frontend/`](frontend/) |
| **DT CORE/** | 100 Hz Mean Value Engine Model solving coupled rotational and thermal ODEs | [`DT CORE/`](DT%20CORE/README.md) |
| **ANOMALY DETECTION/** | PyTorch LSTM Autoencoders for unsupervised novelty detection | [`ANOMALY DETECTION/`](ANOMALY%20DETECTION/README.md) |
| **FAULT DETECTION/** | 9-Class fault classifier with TreeSHAP feature attribution | [`FAULT DETECTION/`](FAULT%20DETECTION/README.md) |
| **RUL ESTIMATION/** | Physics-informed prognostic models predicting remaining safe flight hours | [`RUL ESTIMATION/`](RUL%20ESTIMATION/README.md) |
| **REPLAY AND SIMULATION/** | 50 Hz flight blackbox scrubber and SQLite mission database | [`REPLAY AND SIMULATION/`](REPLAY%20AND%20SIMULATION/README.md) |
| **HEALTH MONITORING/** | Health score calculation modules and sensor cross-validation | [`HEALTH MONITORING/`](HEALTH%20MONITORING/README.md) |
| **MISSION_SIMULATOR/** | Standalone real-time MVEM streaming simulator | [`MISSION_SIMULATOR/`](MISSION_SIMULATOR/README.md) |
| **README_INTEGRATION.md** | Architecture guide for multi-service execution and pipelines | [`README_INTEGRATION.md`](README_INTEGRATION.md) |
| **DASHBOARD AND DATA/docs/** | Detailed mathematical specifications and validation reports | [`DASHBOARD AND DATA/docs/`](DASHBOARD%20AND%20DATA/docs/) |

---

## Airworthiness & Engineering Note

PRATIBIMB is a physics-informed reduced-order digital twin calibrated against published technical data for the Rotax 914 aircraft piston engine. Dispatch thresholds, maintenance recommendations, and degradation models are implemented to demonstrate algorithmic integrity and mathematical self-consistency for research and demonstration purposes.

---

Smart India Hackathon 2026 | Defence Research & Development Organisation (DRDO)
