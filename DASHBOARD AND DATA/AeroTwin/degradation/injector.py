"""
AeroTwin-4 Degradation Injector.

Wraps EngineRunner to physically evaluate and inject degradation severity S(t)
at EVERY simulation time step into the underlying physics subsystems.

Produces paired outputs: (EngineTelemetry, SampleGroundTruth).
"""

import math
import random
from typing import Optional, Tuple, List, Dict, Any

from simulator.runner import EngineRunner, RunnerState
from simulator.telemetry.schema import EngineTelemetry
from .config import DegradationConfig, DegradationType, ComponentID
from .mechanisms import PhysicalDegradationMapper
from .trajectory import DegradationTrajectoryCalculator
from .ground_truth import RunGroundTruth, SampleGroundTruth
from .conditions import RunConditions


class SensorNoiseModel:
    """
    Measurement noise model applied at the sensor/observation layer.

    Draws from a private seeded RNG so a run is reproducible without depending on
    global interpreter random state. Disabled by default for deterministic
    unit-test fixtures; dataset generation enables it.
    """

    def __init__(
        self,
        enabled: bool = False,
        noise_std_dict: Optional[Dict[str, float]] = None,
        seed: Optional[int] = None,
    ):
        self.enabled = enabled
        # Own RNG stream: global random state is shared with everything else in the
        # process, so seeding it gives no reproducibility guarantee whatsoever.
        self._rng = random.Random(seed)
        self.noise_std = noise_std_dict or {
            "rpm": 2.5,
            "cht": 0.3,
            "egt": 1.5,
            "oil_pressure": 1.0,
            "oil_temperature": 0.2,
            "vibration": 0.01,
        }

    def apply(self, telemetry: EngineTelemetry) -> EngineTelemetry:
        """
        Apply Gaussian measurement noise to sensor readings if enabled.
        """
        if not self.enabled:
            return telemetry

        data = telemetry.to_dict()
        for field, std in self.noise_std.items():
            if field in data and isinstance(data[field], (int, float)):
                data[field] += self._rng.gauss(0.0, std)

        # Enforce non-negative bounds ONLY for channels physically constrained to be non-negative
        for non_neg_field in ["rpm", "cht", "egt", "oil_pressure", "oil_pressure_psi", "fuel_flow", "fuel_flow_lph", "vibration"]:
            if non_neg_field in data:
                data[non_neg_field] = max(0.0, data[non_neg_field])

        return EngineTelemetry.from_dict(data)


class DegradationInjector:
    """
    Physically injects degradation into EngineRunner simulation steps and returns
    paired (EngineTelemetry, SampleGroundTruth) tuples.
    """

    def __init__(
        self,
        config: Optional[DegradationConfig] = None,
        runner: Optional[EngineRunner] = None,
        run_id: str = "RUN_001",
        noise_enabled: bool = False,
        conditions: Optional[RunConditions] = None,
    ):
        self.config = config or DegradationConfig.healthy()
        self.runner = runner or EngineRunner(dt=0.01)
        self.run_id = run_id
        self.conditions = conditions
        # Derive the noise stream from the run seed so two runs of the same engine
        # differ, while re-running one run reproduces it exactly.
        self.noise_model = SensorNoiseModel(
            enabled=noise_enabled, seed=self.runner.seed if self.runner.seed is not None else 42
        )

        # Primary component metadata for run-level ground truth
        primary_deg = self.config.degradation_list[0] if self.config.degradation_list else None
        self.run_ground_truth = RunGroundTruth(
            run_id=self.run_id,
            degradation_type=primary_deg.degradation_type.value if primary_deg else "NONE",
            target_component=primary_deg.component_id.value if primary_deg else "NONE",
            max_severity=primary_deg.severity if primary_deg else 0.0,
            trajectory_type=self.config.trajectory_type.value,
            seed=self.runner.seed or 42,
            operating_profile="DEFAULT_MISSION",
            sample_rate_hz=1.0 / self.runner.clock.dt,
            altitude_m=conditions.altitude_m if conditions else 0.0,
            ambient_temperature_c=(
                conditions.ambient_temperature_c if conditions else 20.0
            ),
        )

    def step(self) -> Tuple[EngineTelemetry, SampleGroundTruth]:
        """
        Advance simulation by one step:
        1. Calculate active trajectory severity S(t) for every component.
        2. Map severities to physical subsystem parameters.
        3. Execute EngineDynamics update with injected parameters.
        4. Generate paired SampleGroundTruth label.
        """
        sim_time = self.runner.clock.simulation_time
        dt = self.runner.clock.dt

        # 1. Compute active severities at current simulation_time
        active_components = []
        max_active_severity = 0.0
        primary_deg = self.config.degradation_list[0] if self.config.degradation_list else None

        for deg in self.config.degradation_list:
            active_s = DegradationTrajectoryCalculator.calculate_active_severity(
                self.config, deg, sim_time
            )
            active_components.append(
                deg.__class__(
                    degradation_type=deg.degradation_type,
                    component_id=deg.component_id,
                    severity=active_s,
                )
            )
            if active_s > max_active_severity:
                max_active_severity = active_s

        # 2. Map severities to low-level physical parameters
        phys_params = PhysicalDegradationMapper.map_degradations_to_physics_params(active_components)

        # 3. Advance physics simulation step with injected degradation parameters
        profile_state = self.runner.profile.get_state_at(sim_time)
        throttle = (
            self.runner._manual_throttle
            if self.runner._manual_throttle is not None
            else profile_state["throttle"]
        )
        op_mode = (
            self.runner._manual_operating_mode
            if self.runner._manual_operating_mode is not None
            else profile_state["operating_mode"]
        )

        phys_state = self.runner.engine.update(
            throttle=throttle,
            dt=dt,
            combustion_efficiencies=phys_params["combustion_efficiencies"],
            bearing_friction_multiplier=phys_params["bearing_friction_multiplier"],
            cooling_efficiency=phys_params["cooling_efficiency"],
            lubrication_efficiency=phys_params["lubrication_efficiency"],
        )

        # Construct raw telemetry object
        cyl_torques = phys_state.get("cylinder_torques", {})
        telemetry = EngineTelemetry(
            timestamp=self.runner.clock.simulation_time,  # deterministic simulation timestamp
            simulation_time=sim_time,
            engine_id=self.runner.engine_id,
            operating_mode=op_mode,
            throttle=phys_state["throttle"],
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

        # Apply measurement noise if enabled
        noisy_telemetry = self.noise_model.apply(telemetry)

        # 4. Construct paired SampleGroundTruth
        sample_gt = SampleGroundTruth(
            timestamp=noisy_telemetry.timestamp,
            simulation_time=sim_time,
            degradation_type=primary_deg.degradation_type.value if primary_deg else "NONE",
            target_component=primary_deg.component_id.value if primary_deg else "NONE",
            active_severity=max_active_severity,
            current_health=max(0.0, 1.0 - max_active_severity),
            is_degraded=(max_active_severity > 0.001),
        )

        # Buffer telemetry in runner history
        self.runner.clock.step()
        self.runner._history.append(noisy_telemetry)

        return noisy_telemetry, sample_gt

    def run_simulation(self, duration_seconds: float) -> Tuple[List[EngineTelemetry], List[SampleGroundTruth]]:
        """
        Run simulation for specified duration in simulation seconds.
        """
        self.runner.start()
        start_t = self.runner.clock.simulation_time
        target_end = start_t + duration_seconds

        telemetry_list = []
        gt_list = []

        while self.runner.clock.simulation_time <= target_end and self.runner.state == RunnerState.RUNNING:
            telem, gt = self.step()
            telemetry_list.append(telem)
            gt_list.append(gt)

        return telemetry_list, gt_list
