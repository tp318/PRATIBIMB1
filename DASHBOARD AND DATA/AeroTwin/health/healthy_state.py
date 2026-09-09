"""
AeroTwin-4 Healthy State Model (Digital Twin Core).

Implements expected healthy state prediction supporting two execution modes:
- Mode A (Synchronized Counterfactual Twin): Runs a synchronized healthy simulation
  (degradation=OFF) under identical initial conditions, timestep, seed, and profile trajectory.
- Mode B (Pointwise Reference Prediction): Reference model mapping operating inputs to expected healthy outputs.
"""

from typing import Optional, Dict, Any, List, Tuple
from simulator.runner import EngineRunner
from simulator.scenarios.profiles import FlightProfile
from .models import OperatingState, ExpectedState


class HealthyStateModel:
    """
    Healthy Digital Twin model providing counterfactual healthy state estimations.
    """

    def __init__(
        self,
        dt: float = 0.01,
        seed: Optional[int] = 42,
        profile: Optional[FlightProfile] = None,
        engine_parameters: Optional[Dict[str, Any]] = None,
    ):
        self.dt = dt
        self.seed = seed
        self.profile = profile or FlightProfile()
        # The counterfactual must be a twin of THIS engine unit in THESE ambient
        # conditions. Left at library defaults it answers "what would a nominal
        # engine do at 20 degC", and the resulting residual is dominated by ambient
        # and build mismatch rather than by degradation - which silently destroys
        # thermal fault detectability.
        self.engine_parameters = engine_parameters
        self.healthy_runner = EngineRunner(
            dt=self.dt,
            seed=self.seed,
            flight_profile=self.profile,
            engine_parameters=self.engine_parameters,
        )

    def reset_counterfactual(self, seed: Optional[int] = None):
        """
        Reset counterfactual healthy runner state.
        """
        use_seed = seed if seed is not None else self.seed
        self.healthy_runner = EngineRunner(
            dt=self.dt,
            seed=use_seed,
            flight_profile=self.profile,
            engine_parameters=self.engine_parameters,
        )

    def predict_counterfactual_step(self, operating_state: OperatingState) -> ExpectedState:
        """
        Mode A: Advance synchronized counterfactual healthy twin by one timestep.
        Uses exact throttle and operating mode from operating_state with degradation OFF.
        """
        # Set manual overrides to match exact synchronized operating inputs
        self.healthy_runner.set_throttle(operating_state.throttle)
        self.healthy_runner.set_operating_mode(operating_state.operating_mode)

        # Advance healthy physics by dt
        phys_state = self.healthy_runner.engine.update(
            throttle=operating_state.throttle,
            dt=self.dt,
            combustion_efficiencies={1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0},
            bearing_friction_multiplier=1.0,
            cooling_efficiency=1.0,
            lubrication_efficiency=1.0,
        )

        self.healthy_runner.clock.step()
        
        # Calculate expected cylinder torque pulses synchronized at observed crank angle for pulse alignment
        cyl_torques, _ = self.healthy_runner.engine.cylinder_model.calculate_torques(
            operating_state.crank_angle,
            phys_state["mean_engine_torque"]
        )

        return ExpectedState(
            rpm=phys_state["rpm"],
            crank_angle=phys_state["crank_angle"],
            mean_torque=phys_state["mean_engine_torque"],
            instant_torque=phys_state["engine_torque"],
            load_torque=phys_state["load_torque"],
            friction_torque=phys_state["friction_torque"],
            net_torque=phys_state["net_torque"],
            cylinder_1_torque=cyl_torques.get(1, 0.0),
            cylinder_2_torque=cyl_torques.get(2, 0.0),
            cylinder_3_torque=cyl_torques.get(3, 0.0),
            cylinder_4_torque=cyl_torques.get(4, 0.0),
            cht=phys_state["cht"],
            egt=phys_state["egt"],
            oil_temperature=phys_state["oil_temperature"],
            oil_pressure=phys_state["oil_pressure"],
            oil_pressure_psi=phys_state["oil_pressure_psi"],
            fuel_flow=phys_state["fuel_flow"],
            fuel_flow_lph=phys_state["fuel_flow_lph"],
            fuel_pressure=phys_state["fuel_pressure"],
            vibration=phys_state["vibration"],
        )

    def predict_reference_point(self, operating_state: OperatingState) -> ExpectedState:
        """
        Mode B: Pointwise reference state prediction based on current operating inputs.
        """
        # Execute single healthy physics evaluation without mutating counterfactual trajectory state
        temp_engine = self.healthy_runner.engine.__class__(parameters=self.healthy_runner.engine.params)
        temp_engine.rpm = operating_state.rpm  # query state at observed speed
        phys_state = temp_engine.update(
            throttle=operating_state.throttle,
            dt=self.dt,
            combustion_efficiencies={1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0},
            bearing_friction_multiplier=1.0,
            cooling_efficiency=1.0,
            lubrication_efficiency=1.0,
        )
        cyl_torques = phys_state.get("cylinder_torques", {})

        return ExpectedState(
            rpm=phys_state["rpm"],
            crank_angle=phys_state["crank_angle"],
            mean_torque=phys_state["mean_engine_torque"],
            instant_torque=phys_state["engine_torque"],
            load_torque=phys_state["load_torque"],
            friction_torque=phys_state["friction_torque"],
            net_torque=phys_state["net_torque"],
            cylinder_1_torque=cyl_torques.get(1, 0.0),
            cylinder_2_torque=cyl_torques.get(2, 0.0),
            cylinder_3_torque=cyl_torques.get(3, 0.0),
            cylinder_4_torque=cyl_torques.get(4, 0.0),
            cht=phys_state["cht"],
            egt=phys_state["egt"],
            oil_temperature=phys_state["oil_temperature"],
            oil_pressure=phys_state["oil_pressure"],
            oil_pressure_psi=phys_state["oil_pressure_psi"],
            fuel_flow=phys_state["fuel_flow"],
            fuel_flow_lph=phys_state["fuel_flow_lph"],
            fuel_pressure=phys_state["fuel_pressure"],
            vibration=phys_state["vibration"],
        )
