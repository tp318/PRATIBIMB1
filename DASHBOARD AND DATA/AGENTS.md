# AeroTwin-4 — AI Coding Agent Instructions

## 0. Mission

You are the primary engineering agent for **AeroTwin-4**, a prototype Digital Twin for a representative 4-cylinder aero piston engine intended for the Smart India Hackathon (SIH) project.

Your job is to build a technically credible, testable, modular prototype — not a collection of disconnected demo scripts.

The final system should evolve toward:

```text
Representative engine model
        ↓
Physics-informed telemetry
        ↓
Fault/degradation simulation
        ↓
Digital Twin state estimation
        ↓
Anomaly detection
        ↓
Fault diagnosis
        ↓
Remaining Useful Life (RUL)
        ↓
Maintenance + mission risk
        ↓
Interactive 3D engine
        ↓
React dashboard
```

Do not jump randomly between phases. Implement the current phase completely, test it, document it, and only then move forward.

---

# 1. Locked Architecture

## Core stack

### Simulation / engineering
- Python 3.11+ if available
- NumPy
- SciPy
- Pandas
- Matplotlib
- Optional: MATLAB for independent validation, not as the primary runtime

### Backend
- FastAPI
- Pydantic
- WebSocket for live telemetry later

### AI / ML
- scikit-learn
- XGBoost when needed
- SHAP when explainability is implemented
- PyTorch only when sequential/deep-learning models are justified

### Storage
- Start with CSV/Parquet for development
- SQLite for the first integrated prototype if useful
- PostgreSQL/TimescaleDB only when the data pipeline actually needs it

### Frontend
- React
- TypeScript
- Tailwind CSS
- Three.js / React Three Fiber
- Recharts or another lightweight charting library

### 3D
- Blender
- Export GLB/GLTF
- The web application is the final visualization layer

Do not introduce Docker, Kafka, Redis, Kubernetes, microservices, or other infrastructure unless there is a demonstrated need.

---

# 2. Engineering Integrity Rules

These rules are mandatory.

## Never fabricate realism

This is a representative simulation, NOT an exact replica of a classified or proprietary UAV engine.

Do not claim:
- exact military/UAV engine specifications
- exact engine performance
- real failure probabilities
- validated RUL numbers
- real operational limits

unless a source is explicitly available and documented.

Clearly label model assumptions.

Use terminology such as:

> "representative aero piston-engine model"

> "physics-informed reduced-order simulation"

> "prototype assumptions"

Do not hard-code fake sensor values merely to make the dashboard look realistic.

Every telemetry signal should have a causal relationship to the simulated engine state.

---

# 3. Physics Philosophy

The simulator should be **physics-informed**, not a random-number generator.

The core rotational dynamics are:

J * dω/dt = T_engine - T_load - T_friction

and:

RPM = ω * 60 / (2π)

where:

- J = rotational inertia
- ω = angular velocity
- T_engine = engine torque
- T_load = propeller/load torque
- T_friction = mechanical friction

The model should progressively add:

1. rotational dynamics
2. crank angle
3. 4-cylinder torque generation
4. combustion-cycle effects
5. propeller/load behavior
6. vibration
7. thermal dynamics
8. lubrication
9. fuel system
10. component degradation

Do not implement all ten at once.

---

# 4. Current Development Status

## Current status: PHASES 1-9 IMPLEMENTED

Physics, simulator, degradation, Digital Twin residuals, anomaly detection, fault
diagnosis, RUL and mission risk are all built and tested (144 tests). The real-time
FastAPI + WebSocket layer is live at `AeroTwin/api/`.

NOTE ON PHASE NUMBERING: the roadmap below lists "Phase 5 - Real-Time Telemetry" and
"Phase 6 - AI Anomaly Detection", but the code was built with anomaly detection AS
Phase 5 and the API layer deferred. The API now exists, so both are covered; the
module names (`ml/anomaly`, `api/`) are the reliable reference, not the phase numbers.

Not yet built: Phase 10 (3D engine) and Phase 11 (React dashboard).

See `docs/FIXES_AND_ARCHITECTURE.md` for the defect log and current measured results.

### Original Phase 1 objective (complete)

Build and validate the mathematical foundation of the representative 4-cylinder engine.

The first milestone was:

```text
Throttle
   ↓
Engine Torque
   ↓
Net Torque ← Load Torque
   ↓
Angular Acceleration
   ↓
Angular Velocity
   ↓
RPM
```

The initial model must be stable and testable. (Done: 23/23 Phase 1 tests pass.)

---

# 5. Phase Roadmap

Do not skip phases.

## Phase 1 — Engine Mathematical Model
Deliver:
- parameterized engine model
- crankshaft dynamics
- 4-cylinder torque model
- crank angle
- load model
- friction model
- basic thermal model
- oil model
- fuel model
- vibration model
- validation plots/tests

## Phase 2 — Engine Simulator
Deliver:
- simulation loop
- configurable operating scenarios
- idle/takeoff/climb/cruise/descent
- telemetry generation
- deterministic simulation seeds
- CSV/Parquet export
- simulation configuration

## Phase 3 — Fault & Degradation Model
Deliver:
- bearing degradation
- lubrication degradation
- overheating/cooling degradation
- fuel-system degradation
- severity levels
- gradual degradation
- fault event logging
- causal propagation into telemetry

## Phase 4 — Digital Twin State Engine
Deliver:
- current state
- expected state
- residual generation
- component health
- degradation state
- state estimation
- confidence

## Phase 5 — Real-Time Telemetry
Deliver:
- FastAPI
- WebSocket
- telemetry stream
- historical storage
- simulation clock
- live Digital Twin state

## Phase 6 — AI Anomaly Detection
Deliver:
- feature engineering
- baseline normal behavior
- anomaly detector
- anomaly score
- threshold strategy
- validation metrics

## Phase 7 — Fault Diagnosis
Deliver:
- multi-class fault model
- fault probabilities
- confusion matrix
- precision/recall/F1
- explainability

## Phase 8 — RUL Prediction
Deliver:
- degradation trajectory
- RUL model
- uncertainty interval
- evaluation against held-out degradation runs
- avoid unsupported precision

## Phase 9 — Maintenance + Mission Risk
Deliver:
- maintenance recommendation
- mission-duration comparison
- RUL vs mission requirement
- risk score
- decision explanation

## Phase 10 — 3D Engine
Deliver:
- representative 4-cylinder engine
- piston/crank animation
- component state mapping
- health visualization
- fault highlighting
- RPM-synchronized animation

## Phase 11 — React Dashboard
Deliver:
- 3D viewport
- telemetry charts
- engine health
- fault diagnosis
- RUL
- mission risk
- maintenance recommendation
- event timeline

## Phase 12 — Integration + Validation + SIH Demo
Deliver:
- end-to-end system
- reproducible demo scenarios
- validation report
- performance testing
- documentation
- architecture diagram
- demo script

---

# 6. Project Structure

Use this structure unless there is a strong technical reason not to:

```text
AeroTwin/
│
├── README.md
├── AGENTS.md
├── requirements.txt
├── .gitignore
│
├── phase1/
│   ├── engine/
│   │   ├── __init__.py
│   │   ├── parameters.py
│   │   ├── dynamics.py
│   │   ├── cylinders.py
│   │   ├── thermal.py
│   │   ├── lubrication.py
│   │   ├── fuel.py
│   │   └── vibration.py
│   │
│   ├── tests/
│   │   ├── test_dynamics.py
│   │   ├── test_cylinders.py
│   │   └── test_physics.py
│   │
│   ├── plots/
│   └── main.py
│
├── simulator/
│   ├── scenarios/
│   ├── telemetry/
│   └── simulator.py
│
├── digital_twin/
│   ├── state.py
│   ├── residuals.py
│   ├── health.py
│   └── estimator.py
│
├── faults/
│   ├── bearing.py
│   ├── lubrication.py
│   ├── thermal.py
│   └── fuel.py
│
├── ml/
│   ├── features.py
│   ├── anomaly.py
│   ├── diagnosis.py
│   └── rul.py
│
├── backend/
│   └── app/
│
├── frontend/
│
├── models/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── generated/
│
├── validation/
│
└── docs/
```

If the repository currently has a different structure, preserve working code and refactor incrementally rather than destroying it.

---

# 7. Parameter Management

All model parameters must be centralized.

Never scatter values such as:

```python
0.2
180
3500
0.00005
```

through the code.

Use a typed configuration object or parameter dictionary.

Prefer:

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class EngineParameters:
    cylinders: int
    inertia: float
    max_torque: float
    max_rpm: float
    idle_rpm: float
    load_coefficient: float
    friction_coefficient: float
    initial_rpm: float
    dt: float
```

Keep units explicit in comments/docstrings.

---

# 8. Unit Policy

Use SI units internally wherever practical.

Examples:

- angular velocity: rad/s
- torque: N·m
- temperature: °C for user-facing telemetry, K where thermodynamic equations require absolute temperature
- pressure: Pa internally, convert to PSI/bar only at presentation boundaries
- mass flow: kg/s internally
- time: seconds
- vibration: define the exact unit consistently

Do not mix units silently.

Create conversion utilities when needed.

---

# 9. 4-Cylinder Model

A four-stroke engine completes one combustion cycle every:

```text
720° crank rotation
```

Represent cylinder phase explicitly.

Do not merely add four identical torque values.

Each cylinder should have:
- crank-angle offset
- combustion/power-stroke window
- torque contribution
- optional combustion efficiency

A first simplified model may use smooth torque pulses.

Later it should become richer.

Keep the cylinder model isolated so the firing configuration can be changed without rewriting the entire simulator.

---

# 10. Numerical Stability

The simulator must not explode numerically.

Always check:
- negative RPM
- NaN
- infinity
- unstable integration
- unrealistic temperature
- unreasonable pressure
- runaway torque

Use:
- bounded inputs
- sensible initial conditions
- appropriate time steps
- explicit integration initially
- SciPy integration later if justified

Every numerical change must have a test.

---

# 11. Operating Modes

The model will eventually support:

```text
IDLE
TAKEOFF
CLIMB
CRUISE
DESCENT
```

Do not use only throttle to define operating mode forever.

Eventually operating state should consider:
- throttle
- RPM
- load
- altitude
- airspeed
- mission segment

For Phase 1, simple rules are acceptable.

---

# 12. Telemetry Contract

The canonical telemetry schema should eventually include:

```text
timestamp
engine_id
operating_mode

rpm
crank_angle
engine_torque
load_torque
engine_load
throttle

cht
egt
oil_temperature
oil_pressure

fuel_flow
fuel_pressure

vibration

bearing_health
lubrication_health
thermal_health
fuel_system_health

overall_health
```

Do not generate a signal unless there is a defined model for it.

Every signal should have:
- unit
- source equation/model
- expected operating range
- failure relationship
- validation test

---

# 13. Fault Modeling

Faults must propagate through the system.

Example:

```text
Bearing degradation
      ↓
mechanical resistance / imbalance
      ↓
torque fluctuation
      ↓
vibration increase
      ↓
temperature increase
      ↓
health degradation
```

Lubrication:

```text
Oil system degradation
      ↓
oil pressure reduction
      ↓
friction increase
      ↓
temperature increase
      ↓
vibration increase
      ↓
mechanical health reduction
```

Cooling:

```text
Cooling effectiveness ↓
      ↓
CHT ↑
      ↓
oil temperature ↑
      ↓
thermal stress ↑
```

Fuel:

```text
Fuel delivery degradation
      ↓
combustion efficiency ↓
      ↓
torque fluctuation
      ↓
RPM instability / EGT deviation
```

Do not implement:

```python
if fault:
    vibration = 2.0
```

unless the value is derived from a defined fault model.

---

# 14. Digital Twin Rules

The Digital Twin is NOT just a copy of telemetry.

It must maintain:

```text
actual state
expected state
residual
health state
degradation state
confidence
```

Example:

```text
Expected EGT = 690 °C
Actual EGT   = 724 °C
Residual     = +34 °C
```

Residuals should eventually be conditioned on operating state.

A 34 °C deviation at takeoff may have a different meaning than the same deviation during cruise.

---

# 15. ML Rules

Do not start with deep learning just because it looks impressive.

Start with interpretable baselines:

### Anomaly
- Isolation Forest
- statistical thresholds
- rolling z-score where appropriate

### Fault diagnosis
- Random Forest / XGBoost
- evaluate class imbalance
- confusion matrix
- precision
- recall
- F1

### RUL
- begin with degradation regression / survival-style baseline
- compare against stronger sequential models only when data justifies it

Never report fabricated accuracy.

Always split data properly:
- training
- validation
- test

Avoid leakage from adjacent time windows.

---

# 16. Data Generation

Because real UAV engine failure data is unlikely to be available, synthetic data is acceptable for the prototype.

But synthetic data must be generated from the engine model.

Generate:
- healthy runs
- multiple operating profiles
- multiple fault types
- multiple severity levels
- gradual degradation
- noise
- sensor bias
- missing samples
- realistic operating variation

Keep the random seed configurable.

Example:

```text
seed = 42
```

should reproduce the same dataset.

---

# 17. Validation Requirements

Every major model needs automated tests.

At minimum:

### Physics sanity
- RPM remains non-negative
- higher throttle generally increases equilibrium RPM
- higher load generally decreases equilibrium RPM
- increasing load increases heat generation
- oil pressure follows RPM/system health relationship
- vibration increases with mechanical degradation

### Numerical
- no NaN
- no infinity
- bounded states
- stable integration

### Fault
- healthy engine produces low fault indicators
- degradation produces the expected sensor trends
- fault severity produces monotonic/meaningful degradation where intended

Do not rely only on visual inspection.

Use `pytest`.

---

# 18. Logging

Use Python's `logging` module.

Do not scatter:

```python
print(...)
```

through production modules.

Use:

```python
logger.info(...)
logger.warning(...)
logger.error(...)
```

CLI/demo output may use print statements if appropriate.

---

# 19. Code Quality

Follow:
- PEP 8
- type hints
- docstrings
- small functions
- clear names
- no hidden global state
- no unnecessary classes
- no giant files
- no duplicated equations

Prefer simple, maintainable engineering code over clever abstractions.

---

# 20. Dependency Policy

Before adding a dependency:
1. Check whether the standard library already solves it.
2. Check whether NumPy/SciPy/Pandas already solve it.
3. Check whether the dependency is actively maintained.
4. Explain why it is needed.
5. Keep the dependency optional if it is only for visualization or experimentation.

Do not add large frameworks without a concrete requirement.

---

# 21. Git / Change Management

Make small logical commits.

Good examples:

```text
feat: add crankshaft rotational dynamics
feat: add four-cylinder torque model
feat: add thermal state model
test: validate engine equilibrium behavior
feat: add bearing degradation model
```

Do not mix:
- physics changes
- frontend redesign
- dependency changes
- unrelated cleanup

in one commit.

Never delete working reference code just to make the new implementation easier.

---

# 22. Agent Working Procedure

For EVERY task:

### Step 1 — Inspect
Read the existing relevant files before changing anything.

### Step 2 — Plan
State:
- what is changing
- why
- affected files
- expected behavior
- tests

### Step 3 — Implement
Write complete working code.

### Step 4 — Test
Run:
- unit tests
- relevant simulation
- lint/type checks if available

### Step 5 — Inspect outputs
Check plots/data numerically, not just whether the script exits.

### Step 6 — Document
Update README/docs if behavior or architecture changed.

### Step 7 — Report
Summarize:
- files changed
- functionality added
- tests run
- results
- remaining limitations
- next recommended step

Never say "done" if tests have not been run.

---

# 23. Current Task — Implement Phase 1 Correctly

If the repository already contains the basic implementation, DO NOT blindly replace it.

First inspect:
- `phase1/engine/parameters.py`
- `phase1/engine/dynamics.py`
- `phase1/main.py`
- existing tests

Then improve the implementation to satisfy these requirements.

## Milestone 1

Implement:

```text
Engine parameters
        ↓
EngineDynamics
        ↓
engine torque
        ↓
load torque
        ↓
friction
        ↓
net torque
        ↓
angular acceleration
        ↓
angular velocity
        ↓
RPM
```

Use:

```text
J dω/dt = T_engine - T_load - T_friction
```

with numerical integration.

## Milestone 2

Implement crank angle:

```text
dθ/dt = ω
```

and normalize it to:

```text
0°–720°
```

## Milestone 3

Implement four cylinder torque contributions.

Keep cylinder logic separate from crankshaft dynamics.

## Milestone 4

Add basic thermal state.

## Milestone 5

Add lubrication, fuel and vibration models.

## Milestone 6

Create validation tests and plots.

---

# 24. Phase 1 Acceptance Criteria

Phase 1 is complete ONLY when all of these are true:

- [ ] Engine parameters are centralized
- [ ] Crankshaft dynamics work
- [ ] RPM is calculated from angular velocity
- [ ] Crank angle is simulated
- [ ] Four cylinder torque contributions exist
- [ ] Torque is not simply a constant random signal
- [ ] Load torque exists
- [ ] Friction exists
- [ ] Thermal state exists
- [ ] Oil pressure exists
- [ ] Fuel flow exists
- [ ] Vibration exists
- [ ] Healthy simulation is stable
- [ ] Throttle test passes
- [ ] Load test passes
- [ ] No NaN/infinite values
- [ ] Unit tests pass
- [ ] Plots demonstrate expected behavior
- [ ] Assumptions are documented

Do NOT move to Phase 2 if these fail.

---

# 25. What "good" output looks like

The simulator should eventually produce structured telemetry such as:

```json
{
  "timestamp": 4.250,
  "engine_id": "AEROTWIN-4-001",
  "operating_mode": "CRUISE",

  "rpm": 3100.0,
  "crank_angle": 428.2,
  "engine_torque": 142.0,
  "load_torque": 137.0,
  "engine_load": 0.68,
  "throttle": 0.62,

  "cht": 168.0,
  "egt": 710.0,
  "oil_temperature": 96.0,
  "oil_pressure": 54.0,

  "fuel_flow": 0.0051,
  "fuel_pressure": 320000.0,

  "vibration": 0.18
}
```

The numbers above are examples of the schema only. Do NOT hard-code them.

---

# 26. Important Product Principle

The final dashboard should answer five questions immediately:

1. **Is the engine healthy?**
2. **What is wrong?**
3. **Why does the system think it is wrong?**
4. **How long can it continue operating?**
5. **Can it safely complete the planned mission?**

The 3D model should support these questions, not merely decorate the dashboard.

---

# 27. Future 3D Requirements

When Phase 10 begins, the 3D engine should expose identifiable components:

```text
Engine Block
Cylinder 1
Cylinder 2
Cylinder 3
Cylinder 4
Piston 1
Piston 2
Piston 3
Piston 4
Connecting Rods
Crankshaft
Bearing regions
Oil system
Cooling regions
```

The Digital Twin should be able to map health to components:

```text
component health
      ↓
3D state
```

Example:

```text
bearing_health = 0.42
```

should cause a corresponding warning/highlight in the 3D view.

---

# 28. Do Not Overbuild

The priority order is:

```text
Correctness
    >
Testability
    >
Causal behavior
    >
Integration
    >
Visualization
    >
Polish
```

A beautiful dashboard with fake physics is worse than a simple dashboard backed by a defensible model.

---

# 29. If You Are Unsure

Do not invent architecture.

Do not:
- silently change the project scope
- add unsupported aerospace claims
- replace the engine model with random formulas
- skip tests
- jump to another framework
- rewrite the entire repository

Instead:
1. inspect the existing implementation,
2. identify the smallest change that satisfies the requirement,
3. implement it,
4. test it,
5. document the limitation.

---

# 30. Immediate Instruction

Start by inspecting the current repository.

Then:

1. identify which Phase 1 files already exist;
2. compare them with this specification;
3. preserve correct existing work;
4. implement missing functionality;
5. add/repair tests;
6. run the Phase 1 simulation;
7. generate validation plots;
8. report exactly what passed and what remains.

**Do not start Phase 2 until Phase 1 acceptance criteria are satisfied.**
