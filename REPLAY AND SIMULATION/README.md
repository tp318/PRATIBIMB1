# REPLAY AND SIMULATION — Portable Sortie Replay, Mission Simulation & Persistence Package

A fully self-contained, modular Python package for **UAV Mission Scenario Simulation, Atmospheric Modeling (ISA), Dynamic Flight Clearance (GO/NO-GO), High-Frequency Telemetry Replay, and Time-Series Database Persistence**.

Designed to be **copied directly into any other repository** with zero friction.

---

## Package Directory Structure

```text
REPLAY AND SIMULATION/
├── __init__.py               # Unified package exports for all modules
├── requirements.txt          # Minimal lightweight dependencies
├── README.md                 # Complete integration guide & documentation
├── demo_runner.py            # Standalone end-to-end runnable verification script
│
├── flight_replay.py          # 50 Hz flight trajectory replayer (CSV streaming + synthetic fallback)
├── env_simulator.py          # ISA atmospheric environment model (0 to 7,600 m altitude)
├── mission_engine.py         # Dynamic flight clearance (GO / CAUTION_GO / NO-GO) & risk scoring
├── mission_scenario.py       # Standard scenario presets (High Altitude, Endurance, Hot Weather, Rapid Throttle)
│
├── database/                 # Time-series persistence & replay database layer
│   ├── __init__.py           # Database package exports
│   ├── postgres_db.py        # High-throughput PostgreSQL time-series persistence (production)
│   ├── sqlite_db.py          # Zero-config standalone SQLite persistence (local fallback)
│   ├── db_factory.py         # Auto-selection factory (PostgreSQL with graceful SQLite fallback)
│   └── init_db.py            # CLI database schema initializer
│
└── tests/                    # Standalone pytest suite
    ├── __init__.py
    └── test_packaged_suite.py # Unit and integration tests
```

---

## Key Features

1. **Zero-Configuration Standalone Operation**:
   - Includes **SQLite** persistence out-of-the-box (no external database installation required).
   - If PostgreSQL is available (`DATABASE_URL`), it automatically upgrades to high-throughput pooled PostgreSQL operations.
2. **Synthetic Replay Fallback**:
   - `FlightReplayer` streams real flight CSV dataset trajectories if present, or automatically generates physically consistent flight degradation trajectories if the CSV was not transferred.
3. **Defense-Grade Flight Clearance**:
   - Evaluates mission viability (`GO`, `CAUTION_GO`, `NO_GO`) using composite health indices, critical subsystem fault detections, and RUL reserve margins.
4. **Physics-Consistent Atmospheric Simulator**:
   - International Standard Atmosphere (ISA) calculating ambient temperature, barometric pressure, air density ratio, and Dryden turbulence proxies up to 7,600 m.

---

## How to Integrate into Another Repository

### Step 1: Copy the Folder
Copy the entire `REPLAY AND SIMULATION` folder into your destination repository:

```bash
cp -r "REPLAY AND SIMULATION" /path/to/your/new_repo/
```

### Step 2: Install Dependencies
```bash
pip install -r "REPLAY AND SIMULATION/requirements.txt"
```
*(Only `numpy` and `pandas` are strictly required. `psycopg2-binary` is optional if you want PostgreSQL support).*

### Step 3: Run the Verification Demo
```bash
python "REPLAY AND SIMULATION/demo_runner.py"
```

---

## Code Examples

### 1. Atmospheric Environmental Simulation
```python
from env_simulator import EnvironmentalSimulator

env = EnvironmentalSimulator(sea_level_temp_c=15.0)

# Calculate atmospheric properties at 5,000 meters cruise
state = env.get_atmosphere(altitude_m=5000.0)
print(f"Ambient Temp: {state.ambient_temp_c:.1f} °C")  # -17.5 °C
print(f"Pressure:     {state.pressure_hpa:.1f} hPa")   # ~540 hPa
print(f"Density Ratio: {state.density_ratio:.3f}")     # ~0.60
```

---

### 2. Mission Scenario Presets & Throttle Driver
```python
from mission_scenario import build_scenario, ScenarioType, list_presets

# List all available presets
for preset in list_presets():
    print(preset["scenario_type"], preset["display_name"])

# Build a High-Altitude cruise scenario
scenario = build_scenario("HIGH_ALTITUDE")
print(f"Altitude: {scenario.altitude_m} m, Duration: {scenario.duration_s} s")

# Query commanded throttle at elapsed time t = 45.0 seconds
cmd_throttle = scenario.throttle_at(t=45.0)
```

---

### 3. Flight Clearance Assessment (GO / NO-GO)
```python
from mission_engine import MissionSimulator, ClearanceStatus

sim = MissionSimulator()

# Evaluate clearance for an ISR surveillance flight
assessment = sim.evaluate_clearance(
    profile_name="ISR_SURVEILLANCE",
    predicted_rul_hours=0.20,       # 12 minutes RUL remaining
    composite_health_index=0.96,    # 96% health
    active_fault_name="Normal",
)

print("Disposition:", assessment.status.value)     # GO
print("Risk Score:", assessment.risk_score)         # < 0.20
print("Reasons:", assessment.reasons)
```

---

### 4. High-Rate Flight Telemetry Replay
```python
from flight_replay import FlightReplayer

replayer = FlightReplayer()

# Stream 50 Hz flight samples for Engine 1
for sample in replayer.stream_engine(engine_id=1, max_steps=100):
    print(f"Step {sample.step:3d} (t={sample.time_sec:5.2f}s) | Fault: {sample.fault_name} | CHT Residual: {sample.residuals['cht_residual']}")
```

---

### 5. Time-Series Mission Database Persistence & Replay
```python
from database import get_mission_database

# Connects to PostgreSQL if DATABASE_URL exists, or automatically uses local SQLite
db = get_mission_database()

# 1. Create a sortie record
mission_id = "SORTIE_ALPHA_001"
db.create_mission(
    mission_id=mission_id,
    mission_name="Border Recon Sortie",
    flight_profile="HIGH_ALTITUDE",
    injected_fault="NONE",
)

# 2. Insert telemetry in batches (high throughput)
frames = [
    {
        "mission_id": mission_id,
        "step": 0,
        "timestamp_s": 0.0,
        "rpm": 2500.0,
        "cht": 95.2,
        "egt": 720.0,
        "oil_pressure_psi": 54.0,
        "oil_temperature": 84.0,
        "fuel_flow_lph": 28.0,
        "vibration": 1.1,
        "m1_anomaly_score": 0.02,
        "m2_predicted_fault": "Normal",
        "m3_rul_minutes": 42.0,
    }
]
db.insert_telemetry_batch(frames)

# 3. Finalize mission summary
db.finalize_mission(
    mission_id=mission_id,
    duration_s=120.0,
    disposition="GO",
    start_health=1.0,
    end_health=0.98,
    min_health=0.97,
    dominant_fault="Normal",
    total_frames=1,
)

# 4. Stream historical telemetry directly back from the database
for frame in db.stream_mission_telemetry(mission_id):
    print(f"Replaying frame {frame['step']}: RPM={frame['rpm']}")
```

---

## Database Configuration (Optional PostgreSQL)

To point to a PostgreSQL database instead of the default local SQLite:

Create a `.env` file or export `DATABASE_URL`:
```bash
DATABASE_URL=postgresql://postgres:your_password@localhost:5432/aerotwin
```

Initialize tables:
```bash
python "REPLAY AND SIMULATION/database/init_db.py" --type postgres
# Or for SQLite:
python "REPLAY AND SIMULATION/database/init_db.py" --type sqlite
```

---

## Running Unit Tests

```bash
pytest "REPLAY AND SIMULATION/tests/" -v
```