"""
AeroTwin-4 Healthy Baseline Model.

Characterizes healthy operating behavior, standard deviations, and reference scales
conditioned on (operating_mode, throttle, RPM).
"""

import math
from typing import Dict, Any, Tuple, Optional


class HealthyBaselineModel:
    """
    Operating-condition-aware healthy baseline model providing channel-level
    reference scales and condition-dependent standard deviations for residual normalization.
    """

    # Physical output channel reference scales (SI & presentation units)
    DEFAULT_REFERENCE_SCALES: Dict[str, float] = {
        "rpm": 3500.0,            # Max rated RPM
        "crank_angle": 720.0,      # Cycle angle
        "mean_torque": 150.0,      # Max rated torque (N*m)
        "instant_torque": 250.0,   # Peak instantaneous torque (N*m)
        "load_torque": 150.0,      # Max load torque (N*m)
        "friction_torque": 50.0,   # Max friction torque (N*m)
        "net_torque": 100.0,       # Net acceleration torque (N*m)
        "cylinder_1_torque": 200.0,
        "cylinder_2_torque": 200.0,
        "cylinder_3_torque": 200.0,
        "cylinder_4_torque": 200.0,
        "cht": 200.0,              # Max operating CHT (°C)
        "egt": 850.0,              # Max operating EGT (°C)
        "oil_temperature": 120.0,  # Max oil temp (°C)
        "oil_pressure": 500000.0,  # Max oil pressure (Pa)
        "oil_pressure_psi": 75.0,  # Max oil pressure (PSI)
        "fuel_flow": 0.010,        # Max mass fuel flow (kg/s)
        "fuel_flow_lph": 40.0,     # Max volume fuel flow (L/h)
        "fuel_pressure": 400000.0, # Regulated fuel pressure (Pa)
        "vibration": 1.0,          # Vibration scale (g)
    }

    # Healthy standard deviation baseline estimates per channel
    DEFAULT_HEALTHY_STD: Dict[str, float] = {
        "rpm": 15.0,
        "mean_torque": 2.0,
        "instant_torque": 5.0,
        "load_torque": 2.0,
        "friction_torque": 1.0,
        "net_torque": 3.0,
        "cylinder_1_torque": 5.0,
        "cylinder_2_torque": 5.0,
        "cylinder_3_torque": 5.0,
        "cylinder_4_torque": 5.0,
        "cht": 2.0,
        "egt": 5.0,
        "oil_temperature": 1.0,
        "oil_pressure": 5000.0,
        "oil_pressure_psi": 0.8,
        "fuel_flow": 0.0002,
        "fuel_flow_lph": 0.8,
        "fuel_pressure": 5000.0,
        "vibration": 0.02,
    }

    def __init__(self, custom_scales: Optional[Dict[str, float]] = None, custom_stds: Optional[Dict[str, float]] = None):
        self.scales = custom_scales or self.DEFAULT_REFERENCE_SCALES.copy()
        self.stds = custom_stds or self.DEFAULT_HEALTHY_STD.copy()

    def get_reference_scale(self, channel: str) -> float:
        """
        Get reference scale for a physical output channel.
        """
        return self.scales.get(channel, 1.0)

    def get_conditioned_std(self, channel: str, operating_mode: str = "CRUISE", throttle: float = 0.6, rpm: float = 2500.0) -> float:
        """
        Get operating-condition-aware healthy standard deviation.
        Accounts for higher variance during high-power modes (TAKEOFF/CLIMB vs CRUISE/IDLE).
        """
        base_std = self.stds.get(channel, 1.0)
        mode_upper = operating_mode.upper()

        if mode_upper == "TAKEOFF":
            mult = 1.5
        elif mode_upper == "CLIMB":
            mult = 1.3
        elif mode_upper == "IDLE":
            mult = 0.8
        else:
            mult = 1.0

        # Adjust for throttle power level
        throttle_mult = 0.8 + 0.4 * max(0.0, min(1.0, throttle))
        return base_std * mult * throttle_mult
