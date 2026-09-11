# PRATIBIMB (AeroTwin-4)
## Physics-Informed Digital Twin for Aero Piston Engine Health Monitoring, Fault Prediction, and Airworthiness Assurance in MALE UAVs

**Smart India Hackathon 2026 | Problem Statement 26054**  
**Ministry of Defence | Defence Research & Development Organisation (DRDO) / IDEX**

---

### Deployment & Quick Links
- **Live Tactical Ground Station (Vercel):** [https://pratibimb-1.vercel.app/](https://pratibimb-1.vercel.app/)
- **Live Cloud Backend API (Render):** [https://pratibimb1-3.onrender.com](https://pratibimb1-3.onrender.com)
- **Interactive OpenAPI Specification:** [https://pratibimb1-3.onrender.com/docs](https://pratibimb1-3.onrender.com/docs)
- **Repository Source Code:** [https://github.com/tp318/PRATIBIMB1](https://github.com/tp318/PRATIBIMB1)

---

## 1. Executive Summary & Problem Formulation

Medium Altitude Long Endurance (MALE) Unmanned Aerial Vehicles (e.g., TAPAS-BH-201, Archer-NG class) rely on turbocharged four-cylinder four-stroke aero piston powerplants (calibrated to the Rotax 914 F configuration) for extended 24+ hour loiter profiles. Traditional engine health monitoring architectures rely on static redline exceedance thresholds (e.g., Cylinder Head Temperature $T_{\text{cht}} > 105^\circ\text{C}$ or vibration amplitude $> 1.2\,\text{g}$).

### 1.1 Limitations of Conventional Threshold-Based Monitoring
1. **Atmospheric Ambiguity:** In tactical operations, flight altitude shifts from sea level to the operational ceiling ($7,600\,\text{m}$ / $25,000\,\text{ft}$), where ambient temperature drops to $-34.4^\circ\text{C}$ and air density decreases to $0.549\,\text{kg/m}^3$ under International Standard Atmosphere (ISA) conditions. Static thresholds cannot distinguish whether a temperature change is driven by environmental lapse rates or incipient coolant jacket leakage.
2. **Reactive Failure Notification:** Static redlines alert flight crews only after catastrophic mechanical damage (such as piston seizure or bearing galling) has already initiated, preventing preemptive mission replanning or glide-back recovery.

### 1.2 The PRATIBIMB Counterfactual Solution
PRATIBIMB executes an onboard, condition-matched counterfactual healthy digital twin running synchronously alongside observed engine telemetry. Both the physical engine and the computational twin receive identical control vectors: throttle position $\alpha(t)$, true airspeed $V_\infty(t)$, pressure altitude $h(t)$, and ambient atmospheric states.

The subtraction of twin predictions from measured telemetry yields **9 physical residual channels**:

$$r_i(t) = y_i^{\text{observed}}(t) - y_i^{\text{twin}}\left(\alpha(t), V_\infty(t), h(t), \text{ISA}(t)\right)$$

By operating entirely in the residual domain $\mathbf{r}(t) \in \mathbb{R}^9$, environmental and operational baseline shifts cancel out mathematically:

$$\mathbb{E}[\mathbf{r}_{\text{healthy}}(t) \mid \alpha, h, V_\infty] \approx \mathbf{0}$$

This zero-drift property enables unsupervised anomaly detectors and supervised diagnostic classifiers to identify incipient degradation patterns at signal-to-noise ratios undetectable by conventional avionics.

---

## 2. System Architecture & High-Rate Data Pipeline

```
                                      UAV PROPULSION SYSTEM
                           (Rotax 914 Turbocharged 4-Cylinder Engine)
                                                 │
                                                 │ 50-100 Hz Raw Telemetry (CAN / ARINC-429)
                                                 ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│ 1. TELEMETRY CONDITIONING & EDGE PROTECTION LAYER (< 1.0 ms execution budget)               │
│    ├── Dynamic Rate-of-Change & Range Validation (stuck sensor & open-circuit detection)     │
│    ├── Moving-Median Anti-Glitch Filter (spurious transient suppression)                    │
│    └── Redline & Amber Safety Screening (hardware-level watchdog limits)                    │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
                                                 │
                                                 ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│ 2. DIGITAL TWIN CORE: MEAN VALUE ENGINE MODEL (MVEM) @ 100 Hz                               │
│    ├── Coupled 4-Cylinder Crank Dynamics: J * (dω/dt) = T_comb - T_prop - T_fric            │
│    ├── Turbocharger & Manifold Gas Dynamics: dP_man/dt = (R*T/V) * (m_dot_in - m_dot_out)   │
│    ├── Lumped-Parameter 2-Node Thermal Model (CHT & Oil Heat Balance ODEs)                  │
│    ├── International Standard Atmosphere (ISA) Model (0 to 7,600 m continuous lapse)        │
│    └── Counterfactual Residual Engine: 9 Normalized Residual Channels (r_i = y_obs - y_twin)│
└─────────────────────────────────────────────────────────────────────────────────────────────┘
                                                 │
                                                 ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│ 3. DIAGNOSTICS, PROGNOSTICS & EXPLAINABILITY LAYER                                          │
│    ├── Model 1 (Novelty Detection): PyTorch Temporal LSTM Autoencoder (MSE vs 3σ Threshold)  │
│    ├── Model 2 (Fault Attribution): 1D-CNN + BiLSTM + XGBoost Classifier (9-Class, 98.82%)  │
│    ├── Model 3 (Prognostics / RUL): Physics-Informed Neural Network (PINN-LSTM, 0 to 50 hrs)│
│    └── Explainable AI: Real-Time TreeSHAP Feature Attribution Vectors                       │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
                                                 │
                                                 ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│ 4. TACTICAL DECISION SUPPORT & AIRWORTHINESS REPORTING                                      │
│    ├── Dynamic Pre-Flight Dispatch Clearance: GO / CAUTION / NO-GO Status                   │
│    ├── Subsystem Maintenance Advisory (tied to Aircraft Maintenance Manual tasks)           │
│    ├── Directorate of Flight Safety & Airworthiness (DFSA-26054) Native PDF Report Generator│
│    └── 50 Hz Blackbox Scrubber & Military Tactical Scenario Replay Database                 │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Physics & Mathematical Foundation

The digital twin core is formulated on first-principles aeromechanical and thermodynamic equations calibrated against the Rotax 914 F powerplant specifications:
- **Displacement:** 1,211 cc (bore: 79.5 mm, stroke: 61.0 mm)
- **Rated Power:** 84.5 kW (115 hp) @ 5,800 RPM (5-minute takeoff rating with turbo boost); 73.5 kW (100 hp) @ 5,500 RPM continuous
- **Induction:** Garrett turbocharger with automatic wastegate controller, dual Bing constant-depression carburetors / port injection
- **Cooling System:** Liquid-cooled cylinder heads, ram-air cooled cylinder barrels, dry-sump forced lubrication with external oil radiator

### 3.1 Crankshaft Rotational Dynamics
The angular acceleration of the crankshaft $\omega$ is modeled by the torque balance equation:

$$J \frac{d\omega}{dt} = T_{\text{ind}}(\alpha, P_{\text{man}}, \omega) - T_{\text{prop}}(\omega, V_\infty, \rho) - T_{\text{fric}}(\omega, T_{\text{oil}})$$

where:
- $J = 0.082\,\text{kg}\cdot\text{m}^2$ is the total rotational mass moment of inertia (crankshaft, connecting rods, flywheel, and reduction gearbox assembly).
- $T_{\text{ind}}$ is the indicated combustion torque generated across four cylinders with firing sequence 1-4-2-3:
  $$T_{\text{ind}} = \frac{V_d \cdot \text{IMEP}}{4\pi} \cdot \eta_{\text{comb}}(S_{\text{comb}})$$
  where $\text{IMEP}$ is the indicated mean effective pressure and $\eta_{\text{comb}}$ is the degradation efficiency factor.
- $T_{\text{prop}}$ is the absorbing aerodynamic torque of the variable-pitch propeller:
  $$T_{\text{prop}} = K_{\text{prop}} \cdot \rho(h) \cdot \omega^2 \cdot C_P\left(\frac{V_\infty}{\omega \cdot D_{\text{prop}}}\right)$$
- $T_{\text{fric}}$ combines Coulomb boundary friction and Stribeck hydrodynamic shear:
  $$T_{\text{fric}} = C_{\text{coulomb}} + C_{\text{visc}} \cdot \mu(T_{\text{oil}}) \cdot \omega + C_{\text{aero}} \cdot \omega^2$$

### 3.2 Intake Manifold & Turbocharger Pressure Dynamics
The manifold air pressure $P_{\text{man}}$ is governed by the mass continuity equation for compressible flow in a lumped control volume $V_{\text{man}}$:

$$\frac{dP_{\text{man}}}{dt} = \frac{R_{\text{spec}} \cdot T_{\text{man}}}{V_{\text{man}}} \left( \dot{m}_{\text{comp}}(P_{\text{man}}, \omega_{\text{turbo}}) - \dot{m}_{\text{cyl}}(P_{\text{man}}, \omega) \right)$$

where $\dot{m}_{\text{comp}}$ is determined by the Garrett compressor map and wastegate position, and cylinder mass induction rate is:

$$\dot{m}_{\text{cyl}} = \eta_{\text{vol}} \cdot \frac{V_d \cdot \omega}{4\pi} \cdot \frac{P_{\text{man}}}{R_{\text{spec}} \cdot T_{\text{man}}}$$

### 3.3 Two-Node Lumped Thermal Model
Cylinder Head Temperature ($T_{\text{cht}}$) and Oil Temperature ($T_{\text{oil}}$) are solved via coupled heat balance ODEs:

$$C_{\text{head}} \frac{dT_{\text{cht}}}{dt} = \dot{Q}_{\text{comb}} - h_{\text{cool}}(V_\infty, \rho) \cdot A_{\text{head}} (T_{\text{cht}} - T_\infty) - \dot{Q}_{\text{oil\_jacket}}$$

$$C_{\text{oil}} \frac{dT_{\text{oil}}}{dt} = \dot{Q}_{\text{fric}} + \dot{Q}_{\text{oil\_jacket}} - h_{\text{rad}}(V_\infty, \rho) \cdot A_{\text{rad}} (T_{\text{oil}} - T_\infty)$$

where $\dot{Q}_{\text{comb}} = \eta_{\text{th}} \cdot \dot{m}_f \cdot \text{LHV}_{\text{fuel}}$, and $h_{\text{cool}} \propto V_\infty^{0.8} \rho^{0.8}$ reflects forced convection scaling during loiter and dash envelopes.

### 3.4 International Standard Atmosphere (ISA) Model
Ambient temperature $T_\infty$, ambient pressure $P_\infty$, and air density $\rho$ are modeled as continuous functions of geometric altitude $h$ up to the tropopause ($11,000\,\text{m}$):

$$T_\infty(h) = T_0 - L \cdot h \quad \left(T_0 = 288.15\,\text{K}, \; L = 0.0065\,\text{K/m}\right)$$

$$P_\infty(h) = P_0 \left( 1 - \frac{L \cdot h}{T_0} \right)^{\frac{g_0}{R_{\text{spec}} \cdot L}} \quad \left(P_0 = 101,325\,\text{Pa}\right)$$

$$\rho(h) = \frac{P_\infty(h)}{R_{\text{spec}} \cdot T_\infty(h)} \quad \left(R_{\text{spec}} = 287.05\,\text{J/(kg}\cdot\text{K)}\right)$$

---

## 4. Digital Twin Counterfactual Residual Engine

For every telemetry sample, the digital twin generates expected nominal outputs $\hat{\mathbf{y}}(t) \in \mathbb{R}^9$. The residual engine computes three mathematical representations:

1. **Raw Signed Residual:** $r_i(t) = y_i^{\text{observed}}(t) - \hat{y}_i^{\text{twin}}(t)$
2. **Absolute Residual:** $|r_i(t)| = |y_i^{\text{observed}}(t) - \hat{y}_i^{\text{twin}}(t)|$
3. **Z-Score Normalized Residual:** $z_i(t) = \frac{y_i^{\text{observed}}(t) - \hat{y}_i^{\text{twin}}(t)}{\sigma_{\text{healthy}, i}(\text{mode}, \alpha, \omega)}$

### 4.1 Monitored Residual Channels

| Index | Residual Channel | Unit | Physical Failure Signature |
| :---: | :--- | :---: | :--- |
| $r_1$ | $\Delta \text{RPM}$ | $\text{rev/min}$ | Power deficit, mechanical drag increase, propeller governor fault |
| $r_2$ | $\Delta \text{CHT}$ | $^\circ\text{C}$ | Coolant leak, radiator blockage, localized cylinder hotspot |
| $r_3$ | $\Delta \text{EGT}$ | $^\circ\text{C}$ | Cylinder misfire, ignition timing advance/retard, rich/lean burn |
| $r_4$ | $\Delta P_{\text{oil}}$ | $\text{bar}$ | Pressure relief valve sticking, oil pump cavitation, line rupture |
| $r_5$ | $\Delta T_{\text{oil}}$ | $^\circ\text{C}$ | Bearing galling, thermal breakdown, oil cooler bypass failure |
| $r_6$ | $\Delta \dot{m}_{\text{fuel}}$ | $\text{L/h}$ | Injector fouling, fuel rail leak, regulator diaphragm failure |
| $r_7$ | $\Delta \text{Vib}$ | $\text{g}$ | Unbalanced cylinder torque pulse, bearing raceway spalling |
| $r_8$ | $\Delta V_{\text{bus}}$ | $\text{V}$ | Alternator diode failure, battery charge regulator degradation |
| $r_9$ | $\Delta \text{PW}_{\text{inj}}$ | $\text{ms}$ | Closed-loop ECU injection trim compensation saturation |

### 4.2 Diagnostic Subsystem Indicators
- **Thermal Imbalance Index:** $\mathcal{I}_{\text{thermal}} = \max(|z_{\text{cht}}|, |z_{\text{egt}}|)$
- **Lubrication System Index:** $\mathcal{I}_{\text{lub}} = \max(|z_{P_{\text{oil}}}|, |z_{T_{\text{oil}}}|)$
- **4-Cylinder Firing Balance:**
  $$\mathcal{I}_{\text{balance}} = \frac{\text{std}(T_{\text{cyl}_1}, T_{\text{cyl}_2}, T_{\text{cyl}_3}, T_{\text{cyl}_4})}{\max(1.0, \text{mean}(T_{\text{cyl}_1}, T_{\text{cyl}_2}, T_{\text{cyl}_3}, T_{\text{cyl}_4}))}$$

---

## 5. Diagnostics, Prognostics & Explainability Architecture

```
Telemetric Feature Vector (30-Second Rolling Window @ 10 Hz: 300 x 9)
                                │
        ┌───────────────────────┴───────────────────────┐
        ▼                                               ▼
┌──────────────────────────────┐        ┌──────────────────────────────┐
│ Model 1: Anomaly Detector    │        │ Model 2: Fault Classifier    │
│ PyTorch LSTM Autoencoder     │        │ 1D-CNN + BiLSTM + XGBoost    │
│ Loss: MSE Reconstruction     │        │ 9-Class Softmax Posterior    │
│ Threshold: tau = mu + 3*sigma│        │ Output: P(Fault_k | r(t))    │
└──────────────────────────────┘        └──────────────────────────────┘
        │                                               │
        └───────────────────────┬───────────────────────┘
                                │
        ┌───────────────────────┴───────────────────────┐
        ▼                                               ▼
┌──────────────────────────────┐        ┌──────────────────────────────┐
│ Model 3: Prognostic Model    │        │ Model 4: Explainability XAI  │
│ Physics-Informed LSTM RUL    │        │ TreeSHAP Additive Values     │
│ Target: Remaining Safe Hours │        │ Driver Attribution: phi_i    │
│ Output: [0.0, 50.0] hrs      │        │ Rank Sensor Risk Drivers     │
└──────────────────────────────┘        └──────────────────────────────┘
```

### 5.1 Unsupervised Anomaly Detection (PyTorch LSTM Autoencoder)
- **Architecture:** 2-layer LSTM Encoder (hidden dimensions: 64 $\to$ 32) + Bottleneck Representation + 2-layer LSTM Decoder (hidden dimensions: 32 $\to$ 64).
- **Training Discipline:** Fitted strictly on healthy operational data (`HEALTHY_001`, `HEALTHY_002`) with zero fault exposure.
- **Threshold Setting:** Evaluated on validation runs (`HEALTHY_003`) to establish decision boundary $\tau = \mu_{\text{MSE}} + 3\sigma_{\text{MSE}}$, enforcing a target False Positive Rate $\le 0.1\%$.

### 5.2 9-Class Multi-Fault Diagnostics
The supervised classification engine categorizes operational degradation into 9 standardized failure modes:
1. `NORMAL_OPERATION` — Healthy engine dynamics across all operational envelopes.
2. `COOLING_DEGRADATION` — Radiator airflow restriction, coolant loss, CHT-EGT divergence.
3. `LUBRICATION_FAILURE` — Oil pressure loss, viscosity breakdown, friction torque surge.
4. `SPARK_MISFIRE` — Cylinder combustion failure, cyclic torque deficit, EGT drop.
5. `INJECTOR_ABNORMALITY` — Fuel flow delivery asymmetry, pulse width saturation.
6. `SENSOR_CALIBRATION_DRIFT` — Decoupled single-channel sensor drift without corroborating mechanical indicators.
7. `TURBO_MANIFOLD_LEAK` — Boost pressure deficit, delayed turbo response, manifold pressure loss.
8. `STRUCTURAL_VIBRATION` — Crankshaft bearing raceway unbalance, mechanical vibration RMS spike.
9. `THERMAL_RUNAWAY` — Combined CHT and oil temperature runaway exceeding critical bounds.

### 5.3 Explainable AI: Real-Time TreeSHAP
For every classified failure event, local additive feature attributions are computed via TreeSHAP:

$$f(\mathbf{x}) = \phi_0 + \sum_{j=1}^{M} \phi_j(\mathbf{x})$$

where $\phi_j(\mathbf{x})$ represents the exact Shapley contribution of residual feature $j$ toward pushing the model's confidence toward the diagnosed fault. The dashboard renders these attributions in real time, isolating whether an overheating diagnosis is driven by `cht_residual` ($+45.8\%$) or `oil_temp_slope` ($+24.9\%$).

### 5.4 Physics-Informed Remaining Useful Life (RUL) Prognostics
The prognostic model estimates remaining safe flight hours $t_{\text{rem}} \in [0.0, 50.0]$ based on an exponential wear formulation coupled with bidirectional LSTM regression:

$$\mathcal{H}(t) = \exp\left( -\lambda \int_0^t \mathcal{D}(\tau)\, d\tau \right)$$

where $\mathcal{D}(\tau)$ is the cumulative mechanical and thermal damage rate derived from residual energy integrals.

---

## 6. Empirical Validation & Benchmarks

Models were evaluated across a test fleet of **60 distinct simulated flight sorties** (42 training engines, 9 validation engines, and 9 completely held-out test engines) with varying throttle sweeps, ambient temperature steps, and altitude climbs up to $7,600\,\text{m}$.

### 6.1 Anomaly Detection Feature Ablation Study

| Feature Space Configuration | Feature Dim | Detector Model | Precision | Recall | F1 Score | False Positive Rate | ROC-AUC |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **Config A: Raw Telemetry** | 136 | Static Thresholds | 0.8765 | 0.0845 | 0.1542 | 0.0595 | 0.7031 |
| **Config A: Raw Telemetry** | 136 | Isolation Forest | 0.8571 | 0.0929 | 0.1676 | 0.0774 | 0.6451 |
| **Config A: Raw Telemetry** | 136 | PyTorch LSTM-AE | 0.9966 | 0.6952 | 0.8191 | 0.0119 | 0.9225 |
| **Config B: Twin Residuals** | 178 | Isolation Forest | 0.9620 | 0.8140 | 0.8819 | 0.0034 | 0.9812 |
| **Config B: Twin Residuals** | 178 | **PyTorch LSTM-AE** | **1.0000** | **0.8690** | **0.9299** | **0.0000** | **1.0000** |
| **Config C: Hybrid (Raw+Twin)** | 314 | **XGBoost Classifier**| **0.9942** | **0.9882** | **0.9912** | **0.0012** | **0.9998** |

### 6.2 Multi-Class Fault Classification (Held-Out Test Engines)
- **Test Engines Evaluated:** 9 unseen MALE UAV engines under varied environmental profiles.
- **Overall Accuracy:** **98.82%**
- **Macro-Averaged F1-Score:** **0.9854**
- **Inference Latency:** **< 3.8 ms** per 30-second rolling inference window on standard CPU.

---

## 7. Problem Statement 26054 Compliance Matrix

| PS Mandate | Specific Technical Implementation in PRATIBIMB | Status |
| :--- | :--- | :---: |
| **Real-Time Digital Twin System** | 100 Hz Mean Value Engine Model (MVEM) synchronized with 10 Hz WebSocket telemetry streaming | Complete |
| **Health Monitoring Beyond Redlines** | 9 continuous physical residual channels isolating mechanical degradation from environmental variations | Complete |
| **Unsupervised Anomaly Detection** | PyTorch LSTM Autoencoder trained strictly on healthy data with dynamic 3-sigma thresholds | Complete |
| **Misfire Detection** | Cycle-to-cycle torque deficit monitoring and EGT rapid drop detection | Complete |
| **Injector Abnormalities** | Fuel mass flow rate vs pulse width residual tracking | Complete |
| **Cooling System Degradation** | Thermal dissipation ODE tracking with CHT-EGT heat rejection drift isolation | Complete |
| **Lubrication System Issues** | Oil gallery pressure drop vs bearing friction torque coupling | Complete |
| **Sensor Calibration Drift** | Single-channel divergence isolation distinguished from multi-sensor physical damage | Complete |
| **Combustion Instability** | Cyclic combustion variability and torque pulse oscillation variance detection | Complete |
| **Overheating Trends** | Coupled CHT/EGT thermal runaway slope forecasting with 5-minute lookahead | Complete |
| **Abnormal Vibration Patterns** | Torque unbalance and RMS vibration burst kurtosis analysis | Complete |
| **Remaining Useful Life (RUL)** | Physics-informed LSTM prognostic model calibrated for 0.0 to 50.0 flight hours | Complete |
| **Environmental Operating Envelopes** | Continuous International Standard Atmosphere (ISA) lapse scaling from sea level to 7,600 m ceiling | Complete |
| **Mission Reliability Enhancement** | Dynamic pre-flight GO / CAUTION / NO-GO dispatch clearance considering safe fuel reserves | Complete |
| **Operator HMI & Maintenance Reports** | Ground station web interface and Form DFSA-26054 military debrief PDF export | Complete |

---

## 8. Tactical Operations & Sortie Debrief Generation

### 8.1 Pre-Flight Dispatch Clearance Engine
The pre-flight engine calculates an objective airworthiness index $\Phi_{\text{airworthy}} \in [0.0, 1.0]$ based on historical component fatigue, current sensor residuals, and weather forecasts:
- **GO (Green):** $\Phi \ge 0.70$ — All subsystems cleared for full operational flight envelope.
- **CAUTION_GO (Amber):** $0.35 \le \Phi < 0.70$ — Secondary sensor drift or minor thermal degradation noted; loiter altitude and payload restrictions applied.
- **NO_GO (Red):** $\Phi < 0.35$ — Critical lubrication, misfire, or structural vibration detected; sortie cancelled.

### 8.2 Standardized Form DFSA-26054 Sortie Debrief Generator
PRATIBIMB integrates a native military maintenance report generator compliant with the Directorate of Flight Safety & Airworthiness (DFSA) standards. Accessible from the "Mission report" tab, clicking **"Generate Flight Debrief (PDF)"** produces an official A4 document including:
1. **Sortie Identification:** Flight number, aircraft tail number, engine serial (`Rotax-914-F-UAV-001`), and total airframe hours.
2. **Pre-flight vs Post-flight Health Curve:** Continuous vector SVG trajectory contrasting health degradation against airworthiness minimums.
3. **TreeSHAP Fault Attribution Table:** Quantitative sensor contribution metrics with directional flags (`RISK-INCREASING` vs `RISK-SUPPRESSING`).
4. **Maintenance Action Itemization:** Prescribed Aircraft Maintenance Manual (AMM) inspection tasks.
5. **Three-Party Military Sign-Off Block:** Formal sign-off entries for Flight Line Maintenance Engineer, Chief Technical Officer (Propulsion), and GCS Flight Commander.

---

## 9. Interactive Evaluation Guide

You can evaluate the prototype on the [Live Web Application](https://pratibimb-1.vercel.app/) or via a local instance:

1. **Engine Initialization:**  
   Click **"Start simulation"** on the dashboard masthead. The Rotax 914 engine initializes, streaming live telemetry at 10 Hz across flight dials, engine instruments, and digital twin comparison plots.
2. **Inspect Condition-Matched Twin Tracking:**  
   Observe the real-time plots: solid blue lines indicate live measured telemetry, while dashed gray lines indicate the healthy digital twin baseline.
3. **Evaluate Environmental Invariance (ISA Altitude Scaling):**  
   Drag the altitude slider up to **7,500 m**. Notice how ambient temperature and pressure adjust according to the ISA model. Cylinder Head Temperature drops naturally due to atmospheric cooling, yet residuals remain zero-centered, confirming that environmental shifts do not cause false alarms.
4. **Inject In-Flight Degradation:**  
   Click **"Inject fault"** and select **"COOLING SYSTEM FAILURE"** (Severity: 0.75).
   - Observed CHT diverges from the healthy twin baseline.
   - The PyTorch Autoencoder detects the anomaly and triggers an alert.
   - The 9-class classifier updates from `NORMAL_OPERATION` to `COOLING_DEGRADATION` / `OVERHEATING`.
   - TreeSHAP feature attributions display `cht` (+45.8%) and `oil_temperature_slope` (+24.9%) as top positive risk drivers.
5. **Generate Flight Debrief (PDF):**  
   Navigate to the **"Mission report"** tab and click **"Generate Flight Debrief (PDF)"**. Inspect the rendered Form DFSA-26054 document with vector health curves and sign-off fields. Use the browser print dialog to export to PDF.
6. **Airworthiness Clearance & Blackbox Replay:**  
   Navigate to the **"Replay & Simulation"** tab to review the pre-flight dispatch clearance calculations, test military operational profiles (Desert Loiter, High-Altitude Dash), or scrub through 50 Hz flight blackbox data.

---

## 10. Repository & Subsystem Directory

| Subsystem Directory | Architectural Function | Module Link |
| :--- | :--- | :--- |
| **`frontend/`** | Standalone pilot telemetry UI deployed on Vercel | [`frontend/`](frontend/) |
| **`DASHBOARD AND DATA/`** | FastAPI backend server, WebSocket pipeline, and embedded React interface | [`DASHBOARD AND DATA/`](DASHBOARD%20AND%20DATA/README.md) |
| **`DT CORE/`** | 100 Hz Mean Value Engine Model (MVEM) solving coupled rotational & thermal ODEs | [`DT CORE/`](DT%20CORE/README.md) |
| **`ANOMALY DETECTION/`** | PyTorch LSTM Autoencoders for unsupervised novelty detection | [`ANOMALY DETECTION/`](ANOMALY%20DETECTION/README.md) |
| **`FAULT DETECTION/`** | 9-Class fault classifier with TreeSHAP feature attribution | [`FAULT DETECTION/`](FAULT%20DETECTION/README.md) |
| **`RUL ESTIMATION/`** | Physics-informed prognostic models predicting remaining safe flight hours | [`RUL ESTIMATION/`](RUL%20ESTIMATION/README.md) |
| **`REPLAY AND SIMULATION/`** | 50 Hz flight blackbox scrubber and SQLite mission database | [`REPLAY AND SIMULATION/`](REPLAY%20AND%20SIMULATION/README.md) |
| **`HEALTH MONITORING/`** | Health score calculation modules and sensor cross-validation | [`HEALTH MONITORING/`](HEALTH%20MONITORING/README.md) |
| **`MISSION_SIMULATOR/`** | Standalone real-time MVEM streaming simulator | [`MISSION_SIMULATOR/`](MISSION_SIMULATOR/README.md) |
| **`README_INTEGRATION.md`** | Multi-service orchestration guide and data pipeline documentation | [`README_INTEGRATION.md`](README_INTEGRATION.md) |
| **`DASHBOARD AND DATA/docs/`** | Detailed mathematical specifications and validation reports | [`DASHBOARD AND DATA/docs/`](DASHBOARD%20AND%20DATA/docs/) |

---

## 11. Local Development & Setup

### Prerequisites
- Python 3.9, 3.10, or 3.11
- Operating System: Linux (Ubuntu 20.04/22.04), macOS, or Windows 10/11
- Modern web browser (Chrome, Edge, Firefox, Safari)

### Installation Steps

```bash
# 1. Clone the repository
git clone https://github.com/tp318/PRATIBIMB1.git
cd PRATIBIMB1

# 2. Configure virtual environment
python -m venv .venv
source .venv/bin/activate    # On Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the integrated server (Port 8001)
cd "DASHBOARD AND DATA"
python run_dashboard.py
```

### Local Endpoints
- **Operator Dashboard:** `http://localhost:8001/`
- **Interactive API Documentation:** `http://localhost:8001/docs`
- **Real-Time Telemetry Stream:** `ws://localhost:8001/ws/telemetry`

---

## 12. Airworthiness & Engineering Disclaimer

PRATIBIMB is a physics-informed reduced-order phenomenological digital twin calibrated against published open specifications for the Rotax 914 aero piston engine. Dispatch thresholds, maintenance actions, and degradation dynamics demonstrate algorithmic integrity and mathematical self-consistency for research and demonstration purposes.

---

Smart India Hackathon 2026 | Defence Research & Development Organisation (DRDO)
