"""
AeroTwin-4 Degradation Trajectories.

Calculates time-dependent degradation severity S(t) in [0.0, 1.0] across different
progression profiles: CONSTANT, LINEAR, STEP, EXPONENTIAL.
"""

import math
from .config import TrajectoryType, ComponentDegradation, DegradationConfig


class DegradationTrajectoryCalculator:
    """
    Computes instantaneous degradation severity as a function of simulation time.
    """

    @staticmethod
    def calculate_active_severity(
        config: DegradationConfig,
        deg: ComponentDegradation,
        simulation_time: float
    ) -> float:
        """
        Calculate effective severity for a component degradation at simulation_time.

        Parameters
        ----------
        config : DegradationConfig
        deg : ComponentDegradation
        simulation_time : float

        Returns
        -------
        float
            Active severity in range [0.0, 1.0].
        """
        target_s = max(0.0, min(1.0, deg.severity))
        t = max(0.0, simulation_time)
        t_start = config.start_time
        t_ramp = max(0.1, config.ramp_duration)

        if t < t_start or target_s == 0.0:
            return 0.0

        if config.trajectory_type == TrajectoryType.CONSTANT:
            return target_s

        elif config.trajectory_type == TrajectoryType.LINEAR:
            progress = min(1.0, (t - t_start) / t_ramp)
            return target_s * progress

        elif config.trajectory_type == TrajectoryType.STEP:
            return target_s if t >= t_start + t_ramp / 2.0 else 0.0

        elif config.trajectory_type == TrajectoryType.EXPONENTIAL:
            progress = min(1.0, (t - t_start) / t_ramp)
            # Smooth exponential growth curve: (e^(3*p) - 1) / (e^3 - 1)
            scale = (math.exp(3.0 * progress) - 1.0) / (math.exp(3.0) - 1.0)
            return target_s * scale

        return target_s
