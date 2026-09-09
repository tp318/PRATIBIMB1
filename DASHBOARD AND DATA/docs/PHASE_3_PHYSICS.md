# AeroTwin-4 Phase 3 Physics & Severity Mapping Specification

## 1. Important Engineering Principle & Disclaimer

> [!WARNING]
> **Mathematical/Model Validation vs Experimental Validation**:
> All physical degradation mechanisms and severity-to-parameter mappings ($S \in [0.0, 1.0]$) implemented in Phase 3 are **phenomenological reduced-order prototype simulation assumptions**.
>
> Passing unit tests and causality checks confirms **mathematical and physical self-consistency** within the reduced-order model equations. It does **NOT** constitute empirical validation against classified or proprietary UAV engine test cell data.

---

## 2. Severity-to-Parameter Mappings

The prototype mapping equations from severity $S \in [0.0, 1.0]$ to physical subsystem parameters are defined as follows:

### D1 — Cylinder Combustion Degradation
- **Target Parameter**: `combustion_efficiency` $E_{comb}$
- **Equation**: $E_{comb}(S) = 1.0 - 0.50 \cdot S$
- **Physical Effect**: Reduces gas expansion torque pulse in the power stroke of the degraded cylinder, inducing crankshaft torque pulsation imbalance and mechanical vibration.

### D2 — Mechanical Bearing Degradation
- **Target Parameter**: `bearing_friction_multiplier` $M_{bearing}$
- **Equation**: $M_{bearing}(S) = 1.0 + 1.00 \cdot S$
- **Physical Effect**: Increases mechanical dry/boundary friction on the crankshaft, reducing net torque and shifting equilibrium speed downward.

### D3 — Cooling-System Degradation
- **Target Parameter**: `cooling_efficiency` $E_{cool}$
- **Equation**: $E_{cool}(S) = 1.0 - 0.50 \cdot S$
- **Physical Effect**: Reduces cylinder head heat rejection rate $Q_{cool}$, leading to elevated Cylinder Head Temperature (CHT) and Oil Temperature.

### D4 — Lubrication-System Degradation
- **Target Parameter**: `lubrication_efficiency` $E_{lub}$
- **Equation**: $E_{lub}(S) = 1.0 - 0.50 \cdot S$
- **Physical Effect**: Reduces oil pump pressure generation capacity, altering fluid film viscosity friction coupling.
