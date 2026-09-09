# Anomaly Detection Subsystem — MALE UAV Digital Twin (PRATIBIMB)

---

## 1. Overview & Operational Role

The **Anomaly Detection** module acts as the continuous **first-stage watchdog** in the MALE UAV aero piston engine Digital Twin. 

Before diagnosing specific component failures or predicting Remaining Useful Life (RUL), the system must continuously assess whether the engine is operating within its healthy baseline envelope.

```
[Live Engine CAN Telemetry]
            ↓
[AUKF / MVEM State Estimator]
            ↓  residual_vector(t) = actual(t) − healthy_predicted(t)
┌────────────────────────────────────────────────────────┐
│ ANOMALY DETECTION (Unsupervised LSTM Autoencoder)     │
│  • Encoder compresses residual window to latent vector │
│  • Decoder reconstructs expected healthy residuals     │
│  • High Reconstruction Error (MSE) → Anomaly Alert     │
└───────────────────────────┬────────────────────────────┘
                            │
              Anomaly Ratio > 1.0 (Alert)
                            ↓
               [Triggers FAULT DETECTION & RUL]
```

---

## 2. Technical Architecture

* **Model Type:** Sequence-to-Sequence LSTM Autoencoder
* **Input Window:** Shape `(Batch, 9, 64)` — 9 physical AUKF sensor residual channels over 64 timesteps (1.28 s @ 50 Hz).
* **Encoder:** 2-layer LSTM ($9 \rightarrow 64 \rightarrow 32$ latent bottleneck vector).
* **Decoder:** RepeatVector across 64 steps + 2-layer LSTM ($32 \rightarrow 64 \rightarrow 64$) + Linear projection back to 9 channels.
* **Threshold Calibration:** 
  $$\text{Threshold} = \mu_{\text{val}} + 3.0 \cdot \sigma_{\text{val}}$$
  Calibrated dynamically on healthy validation residuals.
* **Outputs:**
  1. `anomaly_score`: Continuous normalized health index $\in [0.0, 1.0]$.
  2. `anomaly_ratio`: Error relative to calibrated threshold ($>1.0 \rightarrow$ Anomaly).
  3. `sensor_contributions_pct`: Reconstruction loss breakdown showing which sensor deviated most.

---

## 3. Directory Layout

```
ANOMALY DETECTION/
├── config.py           ← Central hyperparameters, sensor channels, and thresholds
├── preprocessing.py    ← ResidualScaler, sliding window generator, synthetic data
├── architecture.py     ← LSTMAutoencoder PyTorch model definition
├── training.py         ← Training loop, early stopping, threshold calibration
├── inference.py        ← AnomalyInferenceEngine, StreamingAnomalyMonitor, CLI demo
├── requirements.txt    ← Python package dependencies
└── README.md           ← This documentation
```

---

## 4. Quick Start

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Train Model & Calibrate Threshold
```bash
cd "ANOMALY DETECTION"
python training.py
```
This will train the LSTM Autoencoder on healthy flight sequences and generate:
- `best_anomaly_detector.pt`: Optimal model weights.
- `anomaly_scaler.pkl`: Fitted Z-score parameters.
- `anomaly_threshold.json`: Calibrated $3\sigma$ and $99\text{th}$-percentile decision boundaries.

### Step 3: Run Inference & Streaming Monitor Demo
```bash
python inference.py
```
Outputs an audit report showing healthy vs. anomalous window diagnostics and per-sensor divergence percentages.