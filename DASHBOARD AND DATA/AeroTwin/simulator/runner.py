"""
AeroTwin-4 Engine Runner.

The main Phase-2 component managing the real-time simulation loop lifecycle,
state transitions (STOPPED, RUNNING, PAUSED), scenario execution, and
canonical telemetry generation.
"""

import time
import random
from enum import Enum
from typing import Optional, Dict, Any, List

try:
    from engine.dynamics import EngineDynamics
except ImportError:
    from AeroTwin.phase1.engine.dynamics import EngineDynamics

from .clock import SimulationClock
from .scenarios.profiles import FlightProfile, OperatingMode
from .telemetry.schema import EngineTelemetry


class RunnerState(str, Enum):
    STOPPED = "STOPPED"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"


class EngineRunner:
    """
    Manages continuous engine simulation updates, clock progression,
    operating scenarios, dynamic input overrides, real-time pacing, and telemetry output.
    """

    def __init__(
        self,
        engine_id: str = "AEROTWIN-4-001",
        dt: float = 0.01,
        seed: Optional[int] = 42,
        flight_profile: Optional[FlightProfile] = None,
        engine_parameters: Optional[Dict[str, Any]] = None,
    ):
        self.engine_id = engine_id
        self.seed = seed
        self.clock = SimulationClock(dt=dt)
        self.profile = flight_profile or FlightProfile()
        self.engine_parameters = engine_parameters

        # Initialize deterministic seed if specified
        if self.seed is not None:
            random.seed(self.seed)

        # Instantiate Phase-1 core physics engine
        self.engine = EngineDynamics(parameters=self.engine_parameters)

        # Runner lifecycle state
        self._state = RunnerState.STOPPED

        # Dynamic overrides
        self._manual_throttle: Optional[float] = None
        self._manual_operating_mode: Optional[str] = None

        # Telemetry history buffer
        self._history: List[EngineTelemetry] = []

    @property
    def state(self) -> RunnerState:
        return self._state

    @property
    def history(self) -> List[EngineTelemetry]:
        return self._history

    def set_throttle(self, throttle: float):
        """
        Manually override throttle position [0.0, 1.0].
        """
        self._manual_throttle = max(0.0, min(1.0, float(throttle)))

    def set_operating_mode(self, mode: str):
        """
        Manually override operating mode string.
        """
        self._manual_operating_mode = str(mode)

    def clear_overrides(self):
        """
        Clear manual overrides and return to automated FlightProfile scenario.
        """
        self._manual_throttle = None
        self._manual_operating_mode = None

    def start(self):
        """
        Start engine simulation runner.
        """
        if self._state == RunnerState.STOPPED:
            self._state = RunnerState.RUNNING

    def pause(self):
        """
        Pause engine simulation runner.
        """
        if self._state == RunnerState.RUNNING:
            self._state = RunnerState.PAUSED

    def resume(self):
        """
        Resume paused engine simulation runner.
        """
        if self._state == RunnerState.PAUSED:
            self._state = RunnerState.RUNNING

    def stop(self):
        """
        Stop engine simulation runner.
        """
        self._state = RunnerState.STOPPED

    def reset(self, seed: Optional[int] = None):
        """
        Reset runner, simulation clock, physics state, and telemetry buffer.
        """
        if seed is not None:
            self.seed = seed
        if self.seed is not None:
            random.seed(self.seed)

        self.clock.reset()
        self.engine = EngineDynamics(parameters=self.engine_parameters)
        self._state = RunnerState.STOPPED
        self.clear_overrides()
        self._history.clear()

    def step(self) -> Optional[EngineTelemetry]:
        """
        Advance simulation by one time step and return canonical EngineTelemetry payload.
        """
        if self._state == RunnerState.PAUSED:
            # Runner is paused; return last telemetry without advancing clock
            return self._history[-1] if self._history else None

        if self._state == RunnerState.STOPPED:
            # Auto-start if step() is called directly
            self._state = RunnerState.RUNNING

        # 1. Advance simulation clock
        sim_time = self.clock.simulation_time
        dt = self.clock.dt

        # 2. Get scenario inputs or apply manual overrides
        profile_state = self.profile.get_state_at(sim_time)
        throttle = (
            self._manual_throttle
            if self._manual_throttle is not None
            else profile_state["throttle"]
        )
        op_mode = (
            self._manual_operating_mode
            if self._manual_operating_mode is not None
            else profile_state["operating_mode"]
        )

        # 3. Update Phase 1 engine physics
        phys_state = self.engine.update(throttle=throttle, dt=dt)

        # Extract per-cylinder torque contributions
        cyl_torques = phys_state.get("cylinder_torques", {})

        # 4. Construct canonical EngineTelemetry object
        wall_time = time.time()
        telemetry = EngineTelemetry(
            timestamp=wall_time,
            simulation_time=sim_time,
            engine_id=self.engine_id,
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

        # 5. Advance clock for next step
        self.clock.step()

        # Buffer telemetry
        self._history.append(telemetry)

        return telemetry

    def run_for(self, duration_seconds: float) -> List[EngineTelemetry]:
        """
        Run simulation as fast as computationally possible for specified duration (batch mode).

        Returns
        -------
        List[EngineTelemetry]
            List of telemetry items produced during run.
        """
        self.start()
        start_t = self.clock.simulation_time
        target_end = start_t + duration_seconds
        new_telemetry = []

        while self.clock.simulation_time <= target_end and self._state == RunnerState.RUNNING:
            telem = self.step()
            if telem:
                new_telemetry.append(telem)

        return new_telemetry

    def run_realtime(self, duration_seconds: float, playback_speed: float = 1.0) -> List[EngineTelemetry]:
        """
        Run simulation synchronized with wall-clock time at specified playback speed (e.g. 1.0x real-time).

        Parameters
        ----------
        duration_seconds : float
            Simulation duration in seconds.
        playback_speed : float
            Speed multiplier (1.0 = real-time, 2.0 = 2x speed, 0.5 = 0.5x speed).

        Returns
        -------
        List[EngineTelemetry]
            List of telemetry items produced during run.
        """
        self.start()
        start_sim_time = self.clock.simulation_time
        target_end = start_sim_time + duration_seconds
        new_telemetry = []

        wall_start = time.perf_counter()

        while self.clock.simulation_time <= target_end and self._state == RunnerState.RUNNING:
            sim_elapsed = self.clock.simulation_time - start_sim_time
            target_wall_elapsed = sim_elapsed / max(0.1, playback_speed)

            # Sleep if computation is ahead of wall-clock playback schedule
            actual_wall_elapsed = time.perf_counter() - wall_start
            sleep_time = target_wall_elapsed - actual_wall_elapsed
            if sleep_time > 0.001:
                time.sleep(sleep_time)

            telem = self.step()
            if telem:
                new_telemetry.append(telem)

        return new_telemetry
