# AeroTwin-4 — Corrections Log & System Architecture

This document records what was wrong with the pipeline, why it mattered, and what
the system does now. It exists because several of the previously reported results
were artifacts of the evaluation setup rather than model performance, and anyone
presenting this work needs to know which numbers changed and why.

---

## Part 1 — Defects found and fixed

### 1.1 Every healthy run was bit-identical (root cause of most other problems)

`HEALTHY_001`, `HEALTHY_002` and `HEALTHY_003` were **byte-for-byte the same
telemetry**. The run seed was passed to `EngineRunner`, which seeded Python's global
`random` — but `random` was only consumed by the sensor-noise model, and noise was
disabled (`noise_enabled=False`). The physics itself is deterministic, so the seed
changed nothing.

Consequences that all trace back to this single defect:

| Symptom | Mechanism |
| :--- | :--- |
| Healthy feature variance ≈ 1e-18 | One run repeated has no variance |
| `FeatureScaler` values exploding to 1e19 | Dividing by ~zero std |
| Autoencoder loss reaching 1e33 | Consequence of the above |
| The `min_std = 1e-2` "Phase 5.1 fix" | Treated the symptom, not the cause |
| Validation threshold calibrated on training data | `HEALTHY_003` *was* `HEALTHY_001` |

**Fix.** Added `AeroTwin/degradation/conditions.py`. Every run now draws:

- **Build variation** — unit-to-unit manufacturing tolerance on friction, thermal
  mass, cooling coefficient, pump clearance, BSFC and vibration gains, clipped at
  ±2σ so a build stays a *healthy* engine and never imitates a fault.
- **Environmental conditions** — altitude 0–7,600 m and sea-level temperature
  5–40 °C, combined through the ISA lapse rate (6.5 °C/km) into ambient temperature.
- **A private seeded RNG for sensor noise**, instead of the global `random` state.

Measured effect on cross-run spread of healthy run means (previously exactly 0.0):

```
rpm               std =  1.44      range [2663.69, 2668.26]
cht               std = 14.14      range [  51.13,   96.13]
egt               std = 14.60      range [ 638.95,  685.97]
oil_pressure_psi  std =  1.13      range [  60.53,   64.57]
vibration         std =  0.021     range [   0.43,    0.50]
```

The `min_std` floor was **kept**, but it is now a legitimate guard for the 8
genuinely-constant features rather than a cover for a broken dataset. Median
feature std is now 3.28.

---

### 1.2 The held-out test set contained no healthy runs

The test partition was **840 windows, all degraded**. With no negative class:

- **Precision could only ever be 1.0000** — with zero true negatives, no false
  positive is possible.
- **FPR could only ever be 0.0000** — it is `FP/(FP+TN)` with `TN = 0`.
- **ROC-AUC was `nan`** in all nine ablation cells. `sklearn` emitted
  `UndefinedMetricWarning: Only one class is present in y_true` nine times per run,
  and the published results table simply had no ROC-AUC column.

The reported "Precision 1.0000 / F1 1.0000 / FPR 0.0000" was therefore a property
of the split, not of the models.

**Fix.** Nine healthy runs are now generated and distributed across all three
partitions, and `RunSplitter` **raises** if the test partition ends up single-class.
Two regression tests cover it.

```
train  224 windows   healthy 224   degraded   0     (unsupervised, healthy-only)
val    392 windows   healthy 112   degraded 280     (threshold calibration)
test  1008 windows   healthy 168   degraded 840     (both classes -> real metrics)
```

---

### 1.3 The Digital Twin was not a twin of the engine being observed

`HealthyStateModel` built its counterfactual healthy engine with **library default
parameters** — ambient 20 °C, nominal build. Once runs had real ambient conditions,
the observed engine might be flying at −11 °C while its own "twin" sat at 20 °C.

The residual therefore contained *(degradation) + (ambient mismatch) + (build
mismatch)*, and the ambient term swamped the thermal channels.

This was invisible before, because every run used identical default conditions.
It surfaced immediately as a **COOLING recall of 0.000** — every cooling fault was
classified as healthy, and 42 healthy windows were called cooling faults.

**Fix.** `DigitalTwinStateEngine` and `HealthyStateModel` accept
`engine_parameters`, and the Phase 4 pipeline reconstructs each run's build and
ambient conditions for its twin. Ambient temperature is directly measured on a real
UAV (OAT probe); build tolerance is what a twin is calibrated to from acceptance
tests, so both are legitimately available.

| Diagnosis metric | Default-parameter twin | Condition-matched twin |
| :--- | ---: | ---: |
| COOLING recall | 0.000 | **1.000** |
| Overall accuracy | 0.779 | **0.967** |
| Balanced accuracy | 0.749 | **0.946** |
| HEALTHY false positives | 42 | **0** |

---

### 1.4 Statistical detector: a 1e-6 variance floor produced 1e6 Z-scores

`StatisticalAnomalyDetector` used `std = np.std(X) + 1e-6`. For a constant feature
that makes the divisor ~1e-6, so any deviation becomes a Z-score of ~1e6, and one
dead channel dominates the RMS across all 178 features. Live scoring returned
**9,561,534** against a threshold of 5.6.

**Fix.** A `min_std = 1e-2` floor (consistent with `FeatureScaler`) plus a per-feature
Z clip at ±20, both persisted in the model artifact. Offline ROC-AUC was unaffected
(still 1.0000 on residual features); live scores are now bounded to ~1.1 healthy and
4.8–15.5 degraded.

---

### 1.5 Remaining Useful Life was not learnable from the dataset

Every Phase 3 run used `CONSTANT` severity — measured std of `gt_active_severity`
across a run was `1.1e-16`. Degradation never progressed, so there was no trend to
extrapolate and RUL could not be estimated at all.

**Fix.** `scripts/generate_rul_dataset.py` generates 24 progressive-degradation runs
with `LINEAR` and `EXPONENTIAL` trajectories over 180 s sorties, plus healthy
references. Health now genuinely decays (1.000 → 0.055 over a run).

---

### 1.6 Smaller defects

| Defect | Location | Fix |
| :--- | :--- | :--- |
| Hardcoded personal IDE path `~/.gemini/antigravity-ide/brain/c4239f03-…`; ran the whole simulation twice and wrote outside the repo | `plot_phase4_residuals.py:170` | Removed |
| Plots also written to the **parent of the repo** (the user's Desktop) | `plot_phase5_results.py` | Writes only to `docs/plots` |
| `ax.boxplot(labels=)` removed in matplotlib 3.11 — script crashed before the last plot | `plot_phase5_results.py:165` | Explicit tick labels |
| `to_parquet` caught bare `Exception` and silently returned a `.csv` path; caller printed "Exported Parquet" for a file that never existed | `telemetry/exporter.py` | Catches `ImportError`, warns, and the caller reports what it actually wrote |
| `requirements.txt` omitted `scikit-learn`, `torch`, `joblib` — Phase 5 could not run from a clean clone | `requirements.txt` | Complete, with `scikit-learn==1.6.1` pinned |
| Isolation Forest results not reproducible (F1 0.8276 under sklearn 1.6.1 → 0.0822 under 1.9.0) | dependency drift | sklearn pinned |
| Healthy-validation rows selected by hardcoded `run_id == "HEALTHY_003"` | `train_phase5_anomaly_models.py` | Label-driven, with an empty guard |
| `reame.md` byte-identical duplicate of `README.md` | repo root | Removed |
| Bare imports (`from simulator.runner import …`) required manual `sys.path` setup by every caller | package layout | `AeroTwin/__init__.py` does the bootstrap |
| A healthy engine was given a 28 s RUL from 6 noisy points and recommended `ABORT` | `rul/projector.py` | `min_points` 5→15, plus a significance gate requiring the observed health drop to exceed 3σ of estimator noise |
| Weak RUL trends drove dispatch decisions | `mission/risk.py` | An RUL with fit quality below R²=0.60 is treated as absence of evidence |

---

## Part 2 — Results after the corrections

### 2.1 Anomaly detection ablation (held-out test set, both classes present)

| Features | Model | Precision | Recall | F1 | FPR | ROC-AUC |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| RAW | Statistical | 0.8765 | 0.0845 | 0.1542 | 0.0595 | 0.7031 |
| RAW | Isolation Forest | 0.8571 | 0.0929 | 0.1676 | 0.0774 | 0.6451 |
| RAW | Autoencoder | 0.9966 | 0.6952 | 0.8191 | 0.0119 | 0.9225 |
| **RESIDUAL** | **Statistical** | **1.0000** | **0.8667** | **0.9286** | **0.0000** | **1.0000** |
| RESIDUAL | Isolation Forest | 0.9898 | 0.8048 | 0.8877 | 0.0417 | 0.9635 |
| **RESIDUAL** | **Autoencoder** | **1.0000** | **0.8690** | **0.9299** | **0.0000** | **1.0000** |
| HYBRID | Statistical | 1.0000 | 0.8667 | 0.9286 | 0.0000 | 1.0000 |
| HYBRID | Isolation Forest | 0.9929 | 0.6702 | 0.8003 | 0.0238 | 0.9158 |
| HYBRID | Autoencoder | 1.0000 | 0.8643 | 0.9272 | 0.0000 | 1.0000 |

These numbers are now meaningful because the test set contains 168 healthy windows.
The physics-informed claim survives the correction and is in fact **better
supported** than before: for the statistical detector, Digital-Twin residuals lift
ROC-AUC from 0.7031 to 1.0000 and F1 from 0.1542 to 0.9286 over raw telemetry.

### 2.2 Fault diagnosis (held out on UNSEEN severity SEV080)

Accuracy **0.9665** · balanced accuracy **0.9464** · macro-F1 **0.9490** · n = 448

| Class | Precision | Recall | F1 | Support |
| :--- | ---: | ---: | ---: | ---: |
| BEARING | 0.8615 | 1.0000 | 0.9256 | 56 |
| COOLING | 1.0000 | 1.0000 | 1.0000 | 56 |
| CYLINDER | 0.9492 | 1.0000 | 0.9739 | 112 |
| HEALTHY | 1.0000 | 1.0000 | 1.0000 | 168 |
| LUBRICATION | 1.0000 | 0.7321 | 0.8454 | 56 |

**Known limitation.** LUBRICATION recall is 0.73 — 9 windows are called BEARING and
6 CYLINDER. This is physically coherent rather than a bug: lubrication degradation
raises friction torque, which is the same primary signature as bearing wear. It is
reported rather than tuned away, and it reproduces in the live API.

### 2.3 RUL (held out on an UNSEEN engine unit, U02)

Health estimation MAE **0.0225**, RMSE 0.0409 on the unseen unit.

RUL over 1,160 mid-flight predictions: **MAE 52.12 s**, RMSE 113.00 s.

| Trajectory | Per-run RUL MAE |
| :--- | ---: |
| LINEAR runs | 11.9 – 23.2 s |
| EXPONENTIAL runs | 65.8 – 146.1 s |

Healthy-engine false decay calls: **4 / 171 predictions (2.3%)**.

**Interval honesty.** The reported band is **not** a calibrated 95% confidence
interval. Measured coverage is **68.5%**. It is built from three real error sources
(slope standard error, linear-vs-quadratic model disagreement, and an
extrapolation-horizon penalty) but is deliberately *not* tuned against the held-out
set, because tuning it there would make the coverage figure meaningless. It is
labelled a **trend uncertainty band** everywhere it appears.

The linear/exponential gap is expected and is stated rather than hidden: a trailing
linear extrapolation systematically over-predicts life for accelerating decay. Model
selection between the linear and quadratic fit uses **adjusted R²**, which charges
for the extra parameter — a textbook criterion with no hand-tuned threshold.

### 2.4 Units caveat

RUL is expressed in **simulation seconds**, not engine flight hours. Converting
between them needs a validated wear-rate mapping against real engine test-cell data,
which this project does not have.

---

## Part 3 — Architecture

```
telemetry (100 Hz)
      |
      v
DigitalTwinStateEngine  ---- condition-matched counterfactual twin
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

### Module map

| Path | Role |
| :--- | :--- |
| `AeroTwin/phase1/` | Engine physics (crank dynamics, thermal, lubrication, fuel, vibration) |
| `AeroTwin/simulator/` | Real-time runtime, flight profiles, telemetry contract |
| `AeroTwin/degradation/` | Fault injection, trajectories, ground truth, **run conditions** |
| `AeroTwin/health/` | Digital Twin state engine, residual generation, indicators |
| `AeroTwin/ml/anomaly/` | Unsupervised detection (statistical, isolation forest, autoencoder) |
| `AeroTwin/ml/diagnosis/` | **New** — multi-class fault attribution with explainability |
| `AeroTwin/ml/rul/` | **New** — health regression + trend projection with uncertainty |
| `AeroTwin/mission/` | **New** — mission reliability, risk scoring, dispatch recommendation |
| `AeroTwin/api/` | **New** — FastAPI + WebSocket real-time service |
| `AeroTwin/api/static/` | **New** — operator dashboard (React UMD, vendored, no build step) |

### Mission risk model

The risk score is a transparent weighted sum, not a learned model, because an
operator has to be able to see *why* the system said no:

```
risk = 0.35 * (1 - health)
     + 0.30 * rul_shortfall
     + 0.25 * (fault_confidence * failure_mode_weight)
     + 0.10 * anomaly_severity
```

Failure-mode weights reflect how abruptly a family ends a flight — lubrication 0.95
and bearing 0.90 outrank cylinder 0.55, because oil starvation or a seizing bearing
ends a sortie far faster than a slowly fouling cylinder.

Two hard gates override the weighted score, so an average can never dilute a limit:

- health below **0.35** → `NO_GO` regardless of everything else
- worst-case RUL not covering `mission × 1.5` reserve → `ABORT_OR_SHORTEN`

Dispatch decisions use the **pessimistic** bound of the RUL band, never the point
estimate — planning on the optimistic edge of an interval is how uncertainty gets
quietly discarded.

---

## Part 4 — Running it

```bash
# 1. Environment
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
.venv/Scripts/python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu

# 2. Full test suite (144 tests)
.venv/Scripts/python.exe -m pytest AeroTwin/ -q

# 3. Data generation
.venv/Scripts/python.exe scripts/generate_phase3_dataset.py --full
.venv/Scripts/python.exe scripts/generate_phase4_dataset.py --full
.venv/Scripts/python.exe scripts/generate_rul_dataset.py

# 4. Train everything
.venv/Scripts/python.exe scripts/build_phase5_features.py
.venv/Scripts/python.exe scripts/train_phase5_anomaly_models.py
.venv/Scripts/python.exe scripts/train_fault_diagnosis.py
.venv/Scripts/python.exe scripts/train_rul_model.py

# 5. Evaluate + plot
.venv/Scripts/python.exe scripts/evaluate_phase5_models.py
.venv/Scripts/python.exe scripts/plot_phase4_residuals.py
.venv/Scripts/python.exe scripts/plot_phase5_results.py

# 6. Real-time API + operator dashboard (one command serves both)
.venv/Scripts/python.exe -m uvicorn AeroTwin.api.server:app --port 8000
#    dashboard at http://127.0.0.1:8000/
#    API docs  at http://127.0.0.1:8000/docs
```

### Live demo sequence

```bash
curl -X POST localhost:8000/api/sim/start \
     -H "Content-Type: application/json" \
     -d '{"seed":42,"mission_duration_s":600}'

# wait ~6 s for the first 5 s window, then:
curl localhost:8000/api/twin/state

curl -X POST localhost:8000/api/sim/inject_fault \
     -H "Content-Type: application/json" \
     -d '{"fault_type":"BEARING","severity":0.8}'

curl localhost:8000/api/twin/state
```

Observed live behaviour:

| Injected | Diagnosed | Confidence | Anomaly | Health | Decision |
| :--- | :--- | ---: | :--- | ---: | :--- |
| *(none)* | HEALTHY | 0.98 | not flagged | 0.987 | **GO** |
| BEARING | BEARING | 0.96 | flagged | 0.725 | GO_WITH_MONITORING |
| COOLING | COOLING | 0.97 | not flagged | 0.806 | GO_WITH_MONITORING |
| CYLINDER | CYLINDER | 0.98 | flagged | 0.740 | GO_WITH_MONITORING |
| LUBRICATION | *BEARING* | 0.46 | flagged | 0.489 | GO_WITH_MONITORING |

The LUBRICATION row is the known confusion from §2.2, visible end-to-end. Note the
low confidence (0.46) and the fact that the runner-up is surfaced — the API reports
the margin between the top two classes precisely so a weak call looks weak.

---

## Part 5 — Scope and limitations

- The engine model is a **reduced-order phenomenological** model of a representative
  4-cylinder aero piston engine. It is not a model of any specific UAV powerplant.
- Passing tests confirm **mathematical and physical self-consistency** within the
  model's own equations. They are not empirical validation against real engine
  test-cell data.
- Severity-to-parameter mappings are engineering assumptions.
- Mission risk thresholds are demonstration defaults, not certified airworthiness
  limits.
- RUL is in simulation seconds and its uncertainty band is uncalibrated (68.5%
  measured coverage).
- Build variation is assumed exactly known to the twin. A real deployment would
  estimate it from healthy baseline data, which would add error this model does not
  currently carry.
