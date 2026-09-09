"""
AeroTwin-4 Digital Twin State Engine.

Composite engine orchestrating OperatingState extraction, Healthy Twin estimation,
Residual calculation, and Residual Indicator processing.
"""

from typing import Union, Dict, Any, Optional
from simulator.telemetry.schema import EngineTelemetry
from .models import DigitalTwinFrame, OperatingState, ExpectedState, ResidualState, ResidualIndicators
from .operating_state import OperatingStateExtractor
from .healthy_state import HealthyStateModel
from .baseline import HealthyBaselineModel
from .residuals import ResidualGenerator
from .indicators import ResidualIndicatorEngine


class DigitalTwinStateEngine:
    """
    Primary interface for Phase 4 Digital Twin State Estimation.
    """

    def __init__(
        self,
        dt: float = 0.01,
        seed: Optional[int] = 42,
        mode: str = "COUNTERFACTUAL",  # "COUNTERFACTUAL" (Mode A) or "REFERENCE" (Mode B)
        engine_parameters: Optional[Dict[str, Any]] = None,
    ):
        self.dt = dt
        self.seed = seed
        self.mode = mode.upper()
        self.engine_parameters = engine_parameters
        self.healthy_model = HealthyStateModel(
            dt=self.dt, seed=self.seed, engine_parameters=engine_parameters
        )
        self.baseline_model = HealthyBaselineModel()
        self.residual_generator = ResidualGenerator(baseline_model=self.baseline_model)
        self.indicator_engine = ResidualIndicatorEngine()

    def reset(self, seed: Optional[int] = None):
        """
        Reset counterfactual twin state.
        """
        self.healthy_model.reset_counterfactual(seed=seed)

    def process_telemetry(self, telemetry: Union[EngineTelemetry, Dict[str, Any]]) -> DigitalTwinFrame:
        """
        Process single telemetry frame:
        1. Extract OperatingState (inputs only, no ground truth).
        2. Calculate ExpectedState from HealthyStateModel.
        3. Calculate ResidualState (signed, absolute, normalized).
        4. Calculate ResidualIndicators.
        5. Construct composite DigitalTwinFrame.
        """
        obs_dict = telemetry.to_dict() if isinstance(telemetry, EngineTelemetry) else telemetry

        # 1. Operating State Extraction
        op_state = OperatingStateExtractor.extract(obs_dict)

        # 2. Healthy State Prediction
        if self.mode == "COUNTERFACTUAL":
            expected_state = self.healthy_model.predict_counterfactual_step(op_state)
        else:
            expected_state = self.healthy_model.predict_reference_point(op_state)

        # Extract observed physical outputs
        obs_outputs = {
            ch: float(obs_dict[ch])
            for ch in self.residual_generator.PHYSICAL_OUTPUT_CHANNELS
            if ch in obs_dict
        }

        # 3. Residual Generation
        residuals = self.residual_generator.generate(
            observed=obs_outputs,
            expected=expected_state,
            operating_state=op_state,
        )

        # 4. Residual Indicators
        obs_cyl_torques = [
            obs_outputs.get("cylinder_1_torque", 0.0),
            obs_outputs.get("cylinder_2_torque", 0.0),
            obs_outputs.get("cylinder_3_torque", 0.0),
            obs_outputs.get("cylinder_4_torque", 0.0),
        ]
        indicators = self.indicator_engine.process_frame(residuals, observed_torques=obs_cyl_torques)

        # 5. Composite Digital Twin Frame Construction
        op_inputs = {
            "throttle": op_state.throttle,
            "rpm": op_state.rpm,
            "ambient_temperature": op_state.ambient_temperature,
        }
        exp_outputs = expected_state.to_dict()

        return DigitalTwinFrame(
            timestamp=op_state.timestamp,
            simulation_time=op_state.simulation_time,
            engine_id=op_state.engine_id,
            operating_mode=op_state.operating_mode,
            operating_inputs=op_inputs,
            observed_outputs=obs_outputs,
            expected_outputs=exp_outputs,
            residuals=residuals,
            indicators=indicators,
        )
