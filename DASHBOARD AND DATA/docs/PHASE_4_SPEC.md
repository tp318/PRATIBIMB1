# AeroTwin-4 Phase 4 Specification: Digital Twin State Engine & Residual Generation

## 1. Overview
Phase 4 implements a **Healthy Digital Twin State Engine** that estimates expected engine physical outputs under current operating conditions ($t$, throttle, operating mode, ambient temperature), compares observed telemetry with healthy expected telemetry, and calculates **signed, normalized residuals** and **residual-derived health indicators**.

---

## 2. Dual-Mode Architecture

```text
               PHASE 3 TELEMETRY / INJECTOR
                     │
                     ▼
           Operating State Extractor
                     │
                     ├── Metadata (timestamp, mode, etc.) -> Preserved context (NO residuals)
                     ├── Operating Inputs (throttle, ambient) -> Input to Healthy Twin
                     │
                     ▼
         ┌───────────────────────────┐
         │ Healthy Twin State Engine │
         └─────────────┬─────────────┘
                       │
         ┌─────────────┴─────────────┐
         ▼                           ▼
  Mode A: Synchronized       Mode B: Pointwise
 Counterfactual Twin       Reference Predictor
 (Healthy physics runner,   (Pointwise lookup from
   degradation OFF)         operating conditions)
         │                           │
         └─────────────┬─────────────┘
                       ▼
                 EXPECTED STATE
                       │
                       ▼
                 RESIDUAL ENGINE
                       │
         ┌─────────────┼─────────────┐
         ▼             ▼             ▼
    Raw Signed     Absolute      Normalized
     Residual      Residual       Residual
         │             │             │
         └─────────────┼─────────────┘
                       ▼
             RESIDUAL INDICATORS
             (thermal, oil, vib,
              cylinder_balance)
```

---

## 3. Ground-Truth Isolation & Field Separation

- **Metadata Fields** (`timestamp`, `simulation_time`, `engine_id`, `operating_mode`): Preserved context. **Zero residuals calculated**.
- **Operating Inputs** (`throttle`, `ambient_temperature`): Inputs to the Healthy Twin.
- **Physical Output Channels**: `rpm`, `mean_torque`, `instant_torque`, `load_torque`, `friction_torque`, `net_torque`, `cht`, `egt`, `oil_temperature`, `oil_pressure`, `fuel_flow`, `fuel_pressure`, `vibration`, `cylinder_1_torque`..`cylinder_4_torque`.
- **Evaluation Metadata**: Ground-truth labels (`gt_degradation_type`, `gt_severity`, etc.) are strictly isolated for validation and are **never** passed to the Healthy Twin.
