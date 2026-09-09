# AeroTwin-4 Phase 4 Residuals & Indicator Specification

## 1. Residual Calculations

For every physical output channel $x$:

$$\text{Raw Signed Residual}(x) = x_{\text{observed}} - x_{\text{expected}}$$
$$\text{Absolute Residual}(x) = |x_{\text{observed}} - x_{\text{expected}}|$$
$$\text{Normalized Residual}(x) = \frac{x_{\text{observed}} - x_{\text{expected}}}{\sigma_{\text{healthy}}(\text{mode, throttle, rpm})}$$

---

## 2. Residual Indicators (`ResidualIndicatorEngine`)

- **Thermal Deviation**: $\max(|z_{\text{cht}}|, |z_{\text{egt}}|)$
- **Oil System Deviation**: $\max(|z_{\text{oil\_pressure}}|, |z_{\text{oil\_temp}}|)$
- **Vibration Deviation**: $|z_{\text{vibration}}|$
- **Torque Deviation**: $\max(|z_{\text{friction\_torque}}|, |z_{\text{mean\_torque}}|)$
- **4-Cylinder Balance Indicator**:
  $$\text{Cylinder Balance} = \frac{\text{std}(T_{\text{cyl1}}, T_{\text{cyl2}}, T_{\text{cyl3}}, T_{\text{cyl4}})}{\max\left(1.0, \text{mean}(T_{\text{cyl1}}, T_{\text{cyl2}}, T_{\text{cyl3}}, T_{\text{cyl4}})\right)}$$
