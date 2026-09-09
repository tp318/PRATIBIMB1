# AeroTwin-4 Phase 3 Specification: Degradation Physics & Ground-Truth System

## 1. Overview
Phase 3 implements a physics-informed degradation injection framework and dataset generation pipeline for AeroTwin-4.

It modifies low-level physical engine subsystem parameters directly (cylinder combustion pulse, bearing mechanical friction, cooling airflow dissipation, oil pump pressure), allowing sensor responses to emerge naturally while maintaining **strict ground-truth isolation**.

---

## 2. Architecture & Extension Points

```text
                  DEGRADATION CONFIG
                          │
                          ▼
            DEGRADATION INJECTOR S(t)
                          │
                          ▼
             PHYSICAL PARAMETER MAPPER
                          │
       ┌──────────────────┼──────────────────┐
       ↓                  ↓                  ↓
Cylinder Combustion  Bearing Friction  Cooling / Oil Subsystem
  (efficiencies)      (multiplier)         (efficiencies)
       │                  │                  │
       └──────────────────┼──────────────────┘
                          │
                          ▼
                   ENGINE DYNAMICS
                          │
                ┌─────────┴─────────┐
                ↓                   ↓
         Engine Telemetry    Ground Truth Label
```

---

## 3. Degradation Modes & Physical Mechanisms

| ID | Degradation Mode | Target Component | Parameter | Healthy | Degraded Range |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **D1** | Cylinder Combustion | `CYLINDER_1`..`CYLINDER_4` | `combustion_efficiency` | `1.0` | $[0.50, 1.00]$ |
| **D2** | Mechanical / Bearing | `BEARING` | `bearing_friction_multiplier` | `1.0` | $[1.00, 2.00]$ |
| **D3** | Cooling System | `COOLING_SYSTEM` | `cooling_efficiency` | `1.0` | $[0.50, 1.00]$ |
| **D4** | Lubrication / Oil | `LUBRICATION_SYSTEM` | `lubrication_efficiency` | `1.0` | $[0.50, 1.00]$ |

---

## 4. Ground-Truth Data Contracts

### Run Ground Truth (`RunGroundTruth`)
Tracks scenario-level metadata: `run_id`, `degradation_type`, `target_component`, `max_severity`, `trajectory_type`, `seed`, `operating_profile`.

### Sample Ground Truth (`SampleGroundTruth`)
Tracks time-step metadata: `timestamp`, `simulation_time`, `degradation_type`, `target_component`, `active_severity`, `current_health`, `is_degraded`.

Ground truth is stored separately and is **never** included in sensor feature inputs.
