# AeroTwin-4 Phase 4 Validation Specification

## 1. Important Engineering & Scientific Disclaimer

> [!WARNING]
> **Mathematical Residual vs Engine Fault Diagnosis**:
> A residual measures mathematical deviation from modeled healthy expectations.
> In Phase 4, a non-zero residual does **NOT** equal a final diagnosis or health score ($0–100\%$).
> Final fault classification, explainability (SHAP), and RUL belong to later phases.

---

## 2. Validation Test Matrix

1. **Healthy-vs-Healthy Equivalence**: Residuals $\approx 0$ under healthy simulation runs.
2. **Cylinder 3 Fault Isolation**: Cylinder 3 torque residual exhibits large negative deviation while Cylinder 1, 2, 4 residuals remain near zero.
3. **Bearing Fault Residual**: Positive mechanical friction torque residual.
4. **Cooling Fault Residual**: Positive CHT and EGT thermal residual.
5. **Lubrication Fault Residual**: Negative oil pressure residual.
6. **Zero Temporal Leakage**: Non-leakage run partitioning.
