# PRATIBIMB: Digital Twin Framework for UAV Aero Piston Engines

> **Indigenous Physics-Informed Digital Twin for Real-Time Health Monitoring, Early Anomaly Detection, and Predictive Diagnostics in Medium-Altitude Long-Endurance (MALE) UAV Propulsion Systems.**

---

## 🚀 Live Prototype & Deployment

Experience the interactive digital twin dashboard running live telemetry streams, fault injection, real-time residual tracking, and health estimation:

- **Live Web Prototype**: [PRATIBIMB Cloud Dashboard](https://pratibimb-1.vercel.app/) *(Alternative: [Vercel Mirror](https://pratibimb1.vercel.app/))*
- **Live API & WebSocket Backend**: [PRATIBIMB Backend on Render](https://pratibimb1-3.onrender.com)
- **Local One-Command Launch**:
  ```bash
  cd "DASHBOARD AND DATA"
  python run_dashboard.py
  ```
  *Access locally at: `http://localhost:8001` (Dashboard) and `http://localhost:8001/docs` (Interactive API)*

---

## 📁 Codebase Architecture

The repository is modularly organized across the digital twin telemetry and analytics lifecycle:

```
PRATIBIMB/
├── DASHBOARD AND DATA/          # Core Digital Twin engine & operator console
│   ├── AeroTwin/                # Python package (physics, ML inference, API)
│   │   ├── simulator/           # 4-cylinder MVEM crankshaft & thermal dynamics
│   │   ├── health/              # Digital Twin state tracking & health index
│   │   ├── ml/                  # Covariance gating, CNN-LSTM, XGBoost, RUL
│   │   └── api/server.py        # FastAPI + WebSocket streaming engine
│   └── run_dashboard.py         # Primary launcher for local testing
├── MISSION_SIMULATOR/           # High-rate hardware telemetry stream emulator
│   ├── engine_physics.py        # Thermodynamic MVEM model (8 degradation modes)
│   └── backend.py               # Asynchronous WebSocket telemetry service
├── DT CORE/                     # Physics engines & Kalman state estimators
├── FAULT DETECTION/             # Diagnostic neural networks & XGBoost classifiers
├── ANOMALY DETECTION/           # Statistical & autoencoder residual gates
├── HEALTH MONITORING/           # Multi-sensor health index synthesis
├── RUL ESTIMATION/              # Remaining useful life Weibull hazard estimators
└── experiments/                 # Automated test harness & validation suites
```

---

## ⚙️ Technical Approach

PRATIBIMB replaces fragile static thresholding with a **three-tier physics-informed architecture**:

1. **Deterministic Physics Reference**: A real-time Mean-Value Engine Model (MVEM) executes alongside the engine, computing expected thermodynamic values (manifold pressure, CHT, EGT, oil pressure, torque) as a function of RPM, throttle, altitude ($0–7{,}500\text{ m}$), and ambient conditions.
2. **Adaptive State Estimation (AUKF)**: An Adaptive Unscented Kalman Filter continuously tracks unit-to-unit manufacturing variations, preventing false alarms caused by natural component wear.
3. **Multi-Stage Diagnostic Pipeline**:
   - *Anomaly Gate*: Sliding-window Mahalanobis residual tracking with temporal persistence ($K=5$) filters transient sensor noise.
   - *Sensor Plausibility Gate*: Cross-checks thermodynamic relationships (e.g., CHT increase must correlate with oil temperature or EGT) to isolate sensor drift from actual engine failure.
   - *Classification & RUL*: Diagnoses root cause across 8 fault classes and projects remaining flight hours before operational redlines are crossed.

---

## 📊 Performance Numbers & Verified Benchmarks

All metrics are deterministically validated across 480 Monte Carlo sorties spanning the complete envelope ($0–7{,}500\text{ m}$ altitude, ISA $\pm 15\,^\circ\text{C}$):

| Metric | Result | Benchmark Context |
|---|---|---|
| **Early Warning Lead Time** | **4.4 min (262 s)** median | 0 misses across 480 runs; alerts at just 2.1–4.5% degradation |
| **Slow Degradation Lead Time**| **Up to 17.4 min** | Captures 58–60% of fault runway on 30-minute ramps |
| **Operational False Alert Rate**| **0.00 / 100 flight hours** | Validated over 110.0 flight hours across full flight envelope |
| **Unit-to-Unit Tolerance** | **$\le 1.2\%$ alerts at $\pm 20\%$ mismatch** | Static twins trigger 100% false alarms at only $\pm 5\%$ mismatch |
| **Sensor Drift Discrimination**| **100% Drift Recall, 0% Engine False Alarm** | Eliminates false engine-abort commands from faulty probes |
| **Diagnostic Accuracy** | **80.0% Recall (Thermal), 83.3% Precision (Bearing)** | Validated against realistic noise and engine variations |
| **Out-of-Distribution Generalization** | **0.0 FA/100h at 7,500 m** | ML polynomial baselines blow up to 1,200–2,800 FA/100h |

---

## ⚠️ Future Challenges & Strategies

1. **Extreme Operating Transients**: Rapid tactical maneuvers, combat throttling, and steep dives induce thermodynamic lags not captured by steady-state maps.
   - *Strategy*: Introduce dynamic enthalpy-lag states into the MVEM and train transient-aware residual estimators.
2. **Onboard Edge SWaP Constraints**: Deploying deep learning and numerical filters on low-power avionics ($\le 5\text{ W}$ budget) without thermal throttling.
   - *Strategy*: Quantize neural networks to INT8 via TensorRT/ONNX Runtime and implement fixed-point sparse Kalman formulations.
3. **Scarcity of Real Flight Failure Data**: Aircraft engines rarely run to catastrophic destruction in flight, creating an imbalanced training set.
   - *Strategy*: Anchor failure representations in physics first-principles, using synthetic fault injection to generate high-fidelity boundary cases.

---

## 📈 Roadmap: How We Will Improve Our Numbers

- **Real Engine Dynamometer Calibration**: Calibrate MVEM parameters using test-rig dynamometer telemetry from indigenous aero engines (e.g., DRDO/ADE testbeds) to push detection thresholds down from $2.0\%$ to $\le 0.8\%$ severity.
- **Physics-Informed Neural Networks (PINNs)**: Replace empirical heat transfer lookups with lightweight PINNs to model non-linear cylinder thermal gradients under severe ambient variations.
- **Sub-Millisecond Edge Optimization**: Refactor state estimation and diagnostic inference in C++/Rust with SIMD acceleration to achieve $\le 0.5\text{ ms}$ step latency on ARM Cortex-A53 avionics.
- **Fleet-Wide Federated Learning**: Aggregate edge residual models across multiple UAV sorties to continuously refine baseline wear curves without compromising mission data security.
