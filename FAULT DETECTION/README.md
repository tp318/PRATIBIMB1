# Part C — Fault Detection & Predictive Analytics
## MALE UAV Aero Piston Engine Digital Twin

---

## Architecture Overview

```
[Physical Engine Sensors]
         ↓  CAN bus / FADEC
[AUKF / MVEM State Estimator]
         ↓  residual_vector(t) = sensor(t) - MVEM_predicted(t)
[preprocessing.py: ResidualScaler + Sliding Windows]
         ↓  (N, 64, 9) windows
[architecture.py: 1D-CNN + LSTM Model]
    ┌─────────────────────────────────────┐
    │ CNN Block-1 (64 filters, k=5)       │ ← cross-sensor spatial correlations
    │ CNN Block-2 (128 filters, k=5)      │ ← richer fault signatures
    │ MaxPool1D(2)                        │ ← halves sequence length
    │ LSTM (128 units, 2 layers)          │ ← temporal fault evolution
    │ FC (64) → FC (9) logits            │ ← 9-class softmax
    └─────────────────────────────────────┘
         ↓  class probabilities
[inference.py: ExplainableFaultInference + DeepSHAP]
         ↓  FaultReport (fault class + sensor attributions)
[Dashboard / GCS Operator]
```

---

## Fault Classes (9)

| ID | Name                     | Urgency  | Primary Sensors                       |
|----|--------------------------|----------|---------------------------------------|
| 0  | Normal operation         | NONE     | —                                     |
| 1  | Misfire conditions       | WARNING  | rpm, egt, vibration                   |
| 2  | Injector abnormalities   | WARNING  | fuel_flow, inj_timing                 |
| 3  | Cooling degradation      | CAUTION  | cht, egt                              |
| 4  | Lubrication issues       | WARNING  | oil_pressure, oil_temp                |
| 5  | Sensor drift / failure   | CAUTION  | rpm, oil_pressure, batt_voltage       |
| 6  | Combustion instability   | CRITICAL | egt, vibration                        |
| 7  | Overheating trends       | CRITICAL | cht, egt, oil_temp                    |
| 8  | Abnormal vibration       | WARNING  | vibration                             |

---

## File Structure

```
FAULT DETECTION/
├── config.py           ← All constants: sensors, classes, hyperparameters
├── preprocessing.py    ← ResidualScaler + sliding windows + DataLoader factory
├── architecture.py     ← CNNLSTMFaultDetector + save/load utilities
├── training.py         ← Training loop + EarlyStopping + run_training_pipeline()
├── inference.py        ← ExplainableFaultInference + DeepSHAP + StreamingFaultMonitor
├── requirements.txt    ← Python dependencies
└── README.md           ← This file
```

---

## Quick Start

### 1 — Install dependencies
```bash
pip install torch numpy shap scikit-learn matplotlib
```

### 2 — Train the model (synthetic data)
```bash
cd "FAULT DETECTION"
python training.py
# Saves: best_fault_detector.pt  residual_scaler.pkl
```

### 3 — Run explainable inference
```bash
python inference.py           # single-window audit report
python inference.py stream    # streaming monitor demo
```

### 4 — Use in your own pipeline
```python
from preprocessing import ResidualScaler, create_sliding_windows
from architecture import CNNLSTMFaultDetector, load_model
from inference import ExplainableFaultInference

# Load model + scaler (saved during training)
model  = load_model("best_fault_detector.pt")
scaler = ResidualScaler.load("residual_scaler.pkl")

# Your AUKF produces: raw_residuals shape (T, 9)
norm_residuals = scaler.transform(raw_residuals)
X, y = create_sliding_windows(norm_residuals, labels)
background = X[y == 0][:100]   # healthy windows for SHAP baseline

engine = ExplainableFaultInference(model, background_data=background)
report = engine.predict_and_explain(X[42])   # prints audit report
print(report.to_json())                      # JSON for dashboard
```

---

## DeepSHAP Explainability

The inference engine integrates `shap.DeepExplainer` to produce per-sensor attributions for every prediction.

### What it outputs:
```
  Sensor Residual Attributions (DeepSHAP):
  (% = contribution to this prediction; ▲=fault-driving, ▼=counter-evidence)
    egt_residual             ████████░░░░░░░░░░░░  38.4%  [▲ fault-driving  ] [°C]
    rpm_residual             █████░░░░░░░░░░░░░░░  24.1%  [▲ fault-driving  ] [RPM]
    vibration_residual       ████░░░░░░░░░░░░░░░░  18.2%  [▲ fault-driving  ] [g]
    ...
```

### Technical note on aggregation:
SHAP values have shape `(1, F, W)` — one value per sensor per timestep.
We aggregate as `sensor_importance[f] = mean_t(|SHAP[f, t]|)` and normalise
to percentages. Taking absolute value before averaging prevents positive and
negative contributions from different timesteps cancelling each other.

---

## Hyperparameters (config.py)

| Parameter          | Default | Description                           |
|--------------------|---------|---------------------------------------|
| WINDOW_SIZE        | 64      | Timesteps per window (1.28 s @ 50 Hz) |
| STRIDE             | 16      | Window overlap step (75% overlap)     |
| cnn_filters_1      | 64      | First CNN block feature maps          |
| cnn_filters_2      | 128     | Second CNN block feature maps         |
| cnn_kernel_size    | 5       | Temporal receptive field              |
| lstm_hidden        | 128     | LSTM hidden state dimension           |
| lstm_layers        | 2       | Stacked LSTM depth                    |
| BATCH_SIZE         | 64      | Mini-batch size                       |
| NUM_EPOCHS         | 50      | Max training epochs                   |
| LEARNING_RATE      | 1e-3    | AdamW initial LR                      |
| SHAP_BACKGROUND_SIZE | 100  | SHAP baseline reference windows       |

---

## Integration with Real AUKF Data

Replace the synthetic data generator with your actual pipeline:

```python
# Real AUKF output feed:
# residuals[t, :] = sensor_readings[t, :] - mvem_prediction[t, :]
# labels[t]       = ground_truth class (from maintenance logs / expert annotation)

real_residuals = np.load("flight_residuals.npy")   # (T, 9)
real_labels    = np.load("flight_labels.npy")       # (T,)

from training import run_training_pipeline
model, history = run_training_pipeline(
    residuals=real_residuals,
    labels=real_labels,
    verbose_f1=True,
)
```
