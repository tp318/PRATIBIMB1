"""
AeroTwin-4 Residual Engine.

Calculates raw signed residuals (observed - expected), absolute residuals,
and normalized residuals for physical output telemetry channels.

Strict Rule:
Does NOT calculate residuals for metadata fields (timestamp, simulation_time, engine_id, operating_mode).
"""

from typing import Union, Dict, Any, Optional
from simulator.telemetry.schema import EngineTelemetry
from .models import ExpectedState, ResidualState, OperatingState
from .baseline import HealthyBaselineModel


class ResidualGenerator:
    """
    Computes signed, absolute, and normalized physical residuals.
    """

    # Physical output channels subject to residual calculation
    PHYSICAL_OUTPUT_CHANNELS = [
        "rpm", "mean_torque", "instant_torque", "load_torque", "friction_torque", "net_torque",
        "cylinder_1_torque", "cylinder_2_torque", "cylinder_3_torque", "cylinder_4_torque",
        "cht", "egt", "oil_temperature", "oil_pressure", "oil_pressure_psi",
        "fuel_flow", "fuel_flow_lph", "fuel_pressure", "vibration"
    ]

    def __init__(self, baseline_model: Optional[HealthyBaselineModel] = None):
        self.baseline = baseline_model or HealthyBaselineModel()

    def generate(
        self,
        observed: Union[EngineTelemetry, Dict[str, Any]],
        expected: ExpectedState,
        operating_state: Optional[OperatingState] = None,
        use_std_normalization: bool = True
    ) -> ResidualState:
        """
        Calculate signed, absolute, and normalized residuals across all physical channels.
        """
        obs_dict = observed.to_dict() if isinstance(observed, EngineTelemetry) else observed
        exp_dict = expected.to_dict()

        raw_signed = {}
        absolute = {}
        normalized = {}

        mode = operating_state.operating_mode if operating_state else "CRUISE"
        throttle = operating_state.throttle if operating_state else 0.6
        rpm = operating_state.rpm if operating_state else 2500.0

        for ch in self.PHYSICAL_OUTPUT_CHANNELS:
            if ch in obs_dict and ch in exp_dict:
                val_obs = float(obs_dict[ch])
                val_exp = float(exp_dict[ch])

                res_signed = val_obs - val_exp
                raw_signed[ch] = res_signed
                absolute[ch] = abs(res_signed)

                if use_std_normalization:
                    denom = self.baseline.get_conditioned_std(ch, operating_mode=mode, throttle=throttle, rpm=rpm)
                else:
                    denom = self.baseline.get_reference_scale(ch)

                if abs(denom) < 1e-9:
                    denom = 1.0

                normalized[ch] = res_signed / denom

        return ResidualState(
            raw_signed=raw_signed,
            absolute=absolute,
            normalized=normalized,
        )
