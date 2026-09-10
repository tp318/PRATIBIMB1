# DT CORE — Digital Twin Core

Implements the full physics chain from **Absolutely.docx**:

```
Environment [altitude, Ta, Pa]
     ↓
air_path.py → Manifold Pressure, Air Mass Flow, Volumetric Efficiency
     ↓
fuel_model.py → Air/Fuel Ratio, Fuel Flow
     ↓
combustion.py → Torque (N·m), Heat to CHT, Heat to EGT
     ↓
crankshaft.py → RPM (J·dω/dt = T_engine − T_load − T_friction)
     ↓
thermal.py → EGT (°C), CHT (°C)
     ↓
oil.py → Oil Pressure (bar), Oil Temperature (°C)
     ↓
Y_predicted = {RPM, EGT, CHT, OilPress, OilTemp, FuelFlow}
     ↓
residuals.py → r_i = Y_measured − Y_predicted
               z_i = r_i / σ_i   (normalized)
               severity: NORMAL | CAUTION | WARNING
     ↓
AI/ML → Anomaly → Fault Diagnosis → RUL
```

## Key Insight (from Absolutely.docx)

> "Your MVEM is not the Digital Twin by itself. The MVEM is the physics engine **inside** the Digital Twin."

The MISSION_SIMULATOR engine_physics.py is the **real engine simulator** (with faults and noise).
The DT CORE engine.py is the **healthy reference model** that predicts what the engine *should* be doing.
The difference between them is the **residual** — the fault signal.

## Usage

```python
import sys
sys.path.insert(0, 'path/to/DT CORE')
from engine import DigitalTwinEngine

dt = DigitalTwinEngine()

# Each 0.1s tick
result = dt.step(
    throttle    = 0.65,
    altitude_ft = 3000,
    ambient_c   = 15.0,
    measured    = {
        "RPM": 4300, "EGT": 695, "CHT": 140,
        "OilPress": 4.1, "OilTemp": 85, "FuelFlow": 21.0
    }
)

print(result.predicted)   # {'RPM': 4520, 'EGT': 680, 'CHT': 138, ...}
print(result.residuals.raw)       # {'EGT': +15, 'CHT': +2, ...}
print(result.residuals.normalized) # {'EGT': 1.0, 'CHT': 0.5, ...}  (z-scores)
```

## Real-Time Bus

```python
from bus import TelemetryBus

bus = TelemetryBus()
bus.start_sortie("FLIGHT_001")

# Publish frames
bus.publish(result.to_dict())

# Download CSV (any time, non-destructive)
csv_bytes = bus.get_csv_bytes()

# End sortie (flushes to logs/ directory)
bus.end_sortie()
```

## Module Reference

| Module | Inputs | Outputs |
|--------|--------|---------|
| `environment.py` | altitude_ft, ambient_c | Pa, Ta, ρ, ρ_ratio |
| `air_path.py` | throttle, RPM, atm | P_man, η_v, ṁ_air |
| `fuel_model.py` | throttle, ṁ_air | ṁ_fuel, FuelFlow, AFR, λ |
| `combustion.py` | RPM, fuel, air | T_engine, Q_CHT, Q_EGT |
| `crankshaft.py` | T_engine, state | RPM (ODE step) |
| `thermal.py` | Q_CHT, Q_EGT, ṁ_air | CHT, EGT (lag) |
| `oil.py` | RPM, CHT | OilPress, OilTemp (lag) |
| `residuals.py` | measured, predicted | r_i, z_i, severity |
| `engine.py` | throttle, alt, measured | DTStepResult (all above) |
| `bus.py` | frames | async queue, in-memory log, CSV |