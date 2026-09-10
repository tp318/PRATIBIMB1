"""
residuals.py
============
DT CORE — Residual Generation and Normalization

Computes the residual vector:
    r_i = y_measured_i − y_predicted_i

And the normalized (z-score) residual:
    z_i = r_i / σ_i

where σ_i is the expected standard deviation of the residual for channel i
under healthy conditions (calibrated from historical data or engineering
judgment).

From Absolutely.docx (Section 23):
    "Suppose the normal EGT prediction error is σ_EGT = 15°C.
     Then 20°C may not be particularly significant.
     So normalize your residual: z_i = r_i / σ_i"

The z-score is the primary input to the anomaly detection layer:
    |z_i| < 2  → normal (green)
    2 ≤ |z_i| < 3  → caution (amber)
    |z_i| ≥ 3  → warning (red)

Channels modelled:
    RPM         rev/min
    EGT         °C
    CHT         °C
    OilPress    bar
    OilTemp     °C
    FuelFlow    L/hr
    Vibration   g RMS
    AltVoltage  V
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional

# Expected healthy residual std-devs (σ_i) — engineering baseline.
# These represent the combined uncertainty of the DT model + sensor noise
# under nominal healthy engine operation.
SIGMA: Dict[str, float] = {
    "RPM":        25.0,    # RPM    — model + sensor uncertainty
    "EGT":        15.0,    # °C
    "CHT":         4.0,    # °C
    "OilPress":    0.15,   # bar
    "OilTemp":     3.0,    # °C
    "FuelFlow":    0.8,    # L/hr
    "Vibration":   0.05,   # g RMS
    "AltVoltage":  0.10,   # V
}

# Z-score severity thresholds
Z_CAUTION = 2.0
Z_WARNING  = 3.0


@dataclass(frozen=True)
class ResidualFrame:
    """
    One timestep's residual vector: raw + normalized, with severity labels.
    """
    raw:       Dict[str, float]   # r_i = measured − predicted
    normalized: Dict[str, float]  # z_i = r_i / σ_i
    severity:  Dict[str, str]     # "NORMAL" | "CAUTION" | "WARNING" per channel
    max_z:     float              # maximum |z| across all channels
    flagged:   bool               # True if any channel is WARNING

    def to_dict(self) -> dict:
        return {
            "raw":        {k: round(v, 4) for k, v in self.raw.items()},
            "normalized": {k: round(v, 4) for k, v in self.normalized.items()},
            "severity":   self.severity,
            "max_z":      round(self.max_z, 3),
            "flagged":    self.flagged,
        }


def compute_residuals(
    measured:  Dict[str, float],
    predicted: Dict[str, float],
    sigma:     Optional[Dict[str, float]] = None,
) -> ResidualFrame:
    """
    Compute the residual vector for all channels present in both dicts.

    Parameters
    ----------
    measured : dict
        Real engine sensor readings keyed by channel name.
    predicted : dict
        DT Core predictions for the same channels.
    sigma : dict, optional
        Per-channel σ values. Defaults to SIGMA.

    Returns
    -------
    ResidualFrame
    """
    if sigma is None:
        sigma = SIGMA

    raw:        Dict[str, float] = {}
    normalized: Dict[str, float] = {}
    severity:   Dict[str, str]   = {}
    max_z = 0.0

    channels = set(measured) & set(predicted)
    for ch in channels:
        r_i = measured[ch] - predicted[ch]
        sig = sigma.get(ch, 1.0)
        z_i = r_i / sig if sig > 0 else 0.0

        raw[ch]        = r_i
        normalized[ch] = z_i
        abs_z = abs(z_i)
        if abs_z >= Z_WARNING:
            severity[ch] = "WARNING"
        elif abs_z >= Z_CAUTION:
            severity[ch] = "CAUTION"
        else:
            severity[ch] = "NORMAL"

        max_z = max(max_z, abs_z)

    return ResidualFrame(
        raw       = raw,
        normalized = normalized,
        severity  = severity,
        max_z     = max_z,
        flagged   = max_z >= Z_WARNING,
    )


def residual_summary(frame: ResidualFrame) -> Dict[str, int]:
    """Count channels at each severity level."""
    counts: Dict[str, int] = {"NORMAL": 0, "CAUTION": 0, "WARNING": 0}
    for sev in frame.severity.values():
        counts[sev] = counts.get(sev, 0) + 1
    return counts
