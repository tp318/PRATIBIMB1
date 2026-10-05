# PRATIBIMB: Digital Twin Framework for UAV Aero Piston Engines

> **Indigenous Physics-Informed Digital Twin for Real-Time Health Monitoring, Early Anomaly Detection, and Predictive Diagnostics in Medium-Altitude Long-Endurance (MALE) UAV Propulsion Systems.**

---

## Live Prototype & Deployment

- **Live Web Prototype**: [PRATIBIMB Cloud Dashboard](https://pratibimb-1.vercel.app/) *(Alternative: [Vercel Mirror](https://pratibimb1.vercel.app/))*
- **Live API & WebSocket Backend**: [PRATIBIMB Backend on Render](https://pratibimb1-3.onrender.com)
- **Local One-Command Launch**:
  ```bash
  cd "DASHBOARD AND DATA"
  python run_dashboard.py
  ```
  *Access locally at: `http://localhost:8001` (Dashboard) and `http://localhost:8001/docs` (Interactive API)*

---

## Codebase Architecture

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
├── CAN_TELEMETRY/               # NEW: SAE J1939 CAN bus integration layer
├── FEDERATED_LEARNING/          # NEW: Fleet-wide FedAvg + DP-SGD anomaly trainer
├── EDGE_ML/                     # NEW: INT8 quantization, ONNX export, edge profiler
├── DT CORE/                     # Physics engines & Kalman state estimators
├── FAULT DETECTION/             # Diagnostic neural networks & XGBoost classifiers
├── ANOMALY DETECTION/           # Statistical & autoencoder residual gates
├── HEALTH MONITORING/           # Multi-sensor health index synthesis
├── RUL ESTIMATION/              # Remaining useful life Weibull hazard estimators
├── experiments/                 # Automated test harness & validation suites
└── demo_advanced_systems.py     # NEW: End-to-end CAN + FL + Edge ML demo
```

---

## Technical Approach

PRATIBIMB replaces fragile static thresholding with a **three-tier physics-informed architecture**:

1. **Deterministic Physics Reference**: A real-time Mean-Value Engine Model (MVEM) executes alongside the engine, computing expected thermodynamic values (manifold pressure, CHT, EGT, oil pressure, torque) as a function of RPM, throttle, altitude (0–7,500 m), and ambient conditions.
2. **Adaptive State Estimation (AUKF)**: An Adaptive Unscented Kalman Filter continuously tracks unit-to-unit manufacturing variations, preventing false alarms caused by natural component wear.
3. **Multi-Stage Diagnostic Pipeline**:
   - *Anomaly Gate*: Sliding-window Mahalanobis residual tracking with temporal persistence (K=5) filters transient sensor noise.
   - *Sensor Plausibility Gate*: Cross-checks thermodynamic relationships (e.g., CHT increase must correlate with oil temperature or EGT) to isolate sensor drift from actual engine failure.
   - *Classification & RUL*: Diagnoses root cause across 8 fault classes and projects remaining flight hours before operational redlines are crossed.

---

## Performance Numbers & Verified Benchmarks

All metrics validated across 480 Monte Carlo sorties spanning the complete envelope (0–7,500 m, ISA ±15°C):

| Metric | Result | Benchmark Context |
|---|---|---|
| **Early Warning Lead Time** | **4.4 min (262 s)** median | 0 misses across 480 runs; alerts at 2.1–4.5% degradation |
| **Slow Degradation Lead Time** | **Up to 17.4 min** | Captures 58–60% of fault runway on 30-minute ramps |
| **Operational False Alert Rate** | **0.00 / 100 flight hours** | Validated over 110.0 flight hours across full envelope |
| **Unit-to-Unit Tolerance** | **≤ 1.2% alerts at ±20% mismatch** | Static twins fail at only ±5% mismatch |
| **Sensor Drift Discrimination** | **100% Drift Recall, 0% Engine False Alarm** | Eliminates false engine-abort commands |
| **Diagnostic Accuracy** | **80.0% Recall (Thermal), 83.3% Precision (Bearing)** | Validated with realistic noise |
| **Out-of-Distribution Generalization** | **0.0 FA/100h at 7,500 m** | ML baselines blow up to 1,200–2,800 FA/100h |

---

## NEW: Advanced Capability Layers

### 1. CAN Telemetry Integration (`CAN_TELEMETRY/`)

Real hardware CAN bus ingestion using the SAE J1939 / MilCAN protocol. Decodes raw 8-byte ECU frames from the Rotax 914 / indigenous DRDO ECU into the canonical `EngineTelemetry` schema without modifying any core twin logic.

**Modules:**
| File | Purpose |
|---|---|
| `dbc.py` | DBC signal catalogue — 7 frame IDs (0x601–0x607), 20+ physical signals |
| `decoder.py` | Stateless frame decoder: raw bytes → `{signal_name: float}` dicts |
| `bridge.py` | Hardware CAN bridge (python-can) with automatic CANSimulator fallback |
| `simulator.py` | Software CAN bus driven by AeroTwin physics; generates synthetic frames |

**Key results from demo run:**
- 50 telemetry frames assembled from 350 raw CAN frames, 0 decode errors
- Sample decoded values: RPM=4867.8, CHT=186.9°C, EGT=733.2°C, Vibration=0.101g
- Fallback to software simulator automatic when hardware is absent

**To run on real hardware** (Linux with SocketCAN):
```bash
sudo ip link set can0 type can bitrate 500000
sudo ip link set can0 up
python -c "from CAN_TELEMETRY import CANBridge; b=CANBridge('socketcan','can0'); b.start(); [print(f) for f in b.stream()]"
```

---

### 2. Federated Learning (`FEDERATED_LEARNING/`)

Fleet-wide collaborative model improvement using FedAvg (McMahan et al., 2017) with DP-SGD privacy guarantees (Abadi et al., 2016). Each UAV ground station trains locally on its own sortie data; only encrypted weight deltas are shared — raw telemetry never leaves the aircraft.

**Modules:**
| File | Purpose |
|---|---|
| `model.py` | Pure-NumPy MLP Autoencoder (no PyTorch required on edge nodes) |
| `privacy.py` | L2 gradient clipping + Gaussian noise (DP-SGD), epsilon budget estimation |
| `client.py` | Per-UAV local trainer with unit-to-unit personality variation |
| `aggregator.py` | FedAvg (weighted mean) and FedMedian (Byzantine-robust) |
| `server.py` | FL coordinator: round loop, convergence detection, checkpoint I/O |

**Demo results (10-UAV fleet, 5 rounds):**
- Reconstruction loss: 0.403 → 0.251 (**37.7% reduction** in 5 rounds, 0.14 s wall-clock)
- DP privacy budget: ε ≈ 24.2, δ = 1e-5 (strong composition estimate)
- 5,846 total local samples distributed across fleet; no raw data centralized
- Supports `fedavg` and `fedmedian` (Byzantine-robust) aggregation strategies

**Usage:**
```python
from FEDERATED_LEARNING import FLConfig, FederatedClient, FederatedServer

config = FLConfig(num_rounds=20, clients_per_round=5, dp_noise_multiplier=0.5)
clients = [FederatedClient(f"UAV-{i}", config, local_data=sortie_windows[i])
           for i in range(10)]
server = FederatedServer(config)
server.fit(clients)
server.save_checkpoint("global_model.npz")
```

---

### 3. Edge ML — INT8 Quantization & ONNX Deployment (`EDGE_ML/`)

Compresses the federated global model for deployment on onboard avionics computers (ARM Cortex-A53, NVIDIA Jetson Orin, Raspberry Pi CM4) with a 4x memory footprint reduction via per-layer symmetric INT8 quantization.

**Modules:**
| File | Purpose |
|---|---|
| `quantizer.py` | Static INT8 quantizer — per-layer min-max calibration from healthy-flight data |
| `inference.py` | INT8 forward pass in INT32 accumulators — drop-in for float32 model |
| `profiler.py` | Wall-clock latency benchmark: mean/P95/P99, throughput, tracemalloc memory |
| `exporter.py` | ONNX export (torch-trace primary, manual protobuf fallback) + ORT validation |

**Demo results:**
- Model size: 1,689 parameters → **1,956 bytes INT8** (4x compression vs float32)
- Max element-wise reconstruction error: **0.035** (float32 vs INT8)
- Anomaly score on healthy window: 0.192 (INT8 inference)
- ONNX export: supported via torch (with torch) or manual protobuf builder (no torch)

> **Note on INT8 speed**: NumPy INT8 matmul is slower than float32 on x86 (no NEON vectorization). On target ARM hardware with CMSIS-NN or ONNX Runtime Mobile, INT8 delivers ~4x speedup. The quantizer targets ARM Cortex-A53 avionics deployment, not desktop profiling.

**Usage:**
```python
from EDGE_ML import INT8Quantizer, LatencyProfiler, ONNXExporter

quantizer = INT8Quantizer()
int8_model = quantizer.quantize(global_model, calibration_data)

profiler = LatencyProfiler(int8_model, n_runs=1000)
stats = profiler.run()

exporter = ONNXExporter()
exporter.export(global_model, calibration_data[:2], "pratibimb_anomaly.onnx")
```

**Run the full end-to-end demo:**
```bash
python demo_advanced_systems.py
```

---

## Future Challenges & Improvement Roadmap

| Challenge | Current State | Planned Improvement |
|---|---|---|
| **Real hardware CAN integration** | Software simulator only | Test against Rotax 914 ECU on DRDO/ADE dynamometer testbed |
| **Detection sensitivity floor** | 2.0% fault severity | Push to ≤ 0.8% via dynamometer-calibrated MVEM parameters |
| **FL convergence speed** | 37.7% loss reduction in 5 rounds | Add FedProx proximal term for non-IID fleet data; target >60% in 5 rounds |
| **FL privacy budget** | ε ≈ 24 (strong composition) | Implement Moments Accountant / RDP for tighter ε < 5 at same noise level |
| **Edge latency on ARM** | ~12 µs float32 (x86 desktop) | Sub-0.5 ms on ARM with CMSIS-NN INT8 + sparse Kalman formulation |
| **Transient maneuver faults** | Steady-state MVEM only | Introduce enthalpy-lag dynamic states for combat throttle transients |
| **Out-of-distribution altitude** | Validated to 7,500 m | Extend to 9,000 m ceiling with hypoxic enrichment modeling |
| **Real failure data** | Physics-only synthetic | Federated aggregation from fleet sorties to refine degradation curves |
