"""
AeroTwin-4 Cylinder Model.

Represents a representative 4-cylinder 4-stroke aero piston engine.
Four-stroke cycle completes over 720 degrees of crank angle (4*pi rad).

Firing order: Cylinder 1 -> Cylinder 3 -> Cylinder 4 -> Cylinder 2
Cylinder phase offsets:
    Cylinder 1:   0 degrees
    Cylinder 3: 180 degrees
    Cylinder 4: 360 degrees
    Cylinder 2: 540 degrees
"""

import math


class FourCylinderModel:
    """
    Simulates instantaneous torque generation across 4 cylinders
    over a 720-degree 4-stroke combustion cycle.
    """

    def __init__(self):
        # Firing offsets for 4 cylinders (degrees)
        # Cylinder 1 (0°), Cylinder 3 (180°), Cylinder 4 (360°), Cylinder 2 (540°)
        self.phase_offsets = {
            1: 0.0,
            3: 180.0,
            4: 360.0,
            2: 540.0,
        }

        # Pre-compute cycle mean normalization factor to ensure
        # exact conservation of mean engine torque capability.
        self._mean_normalized_torque = self._compute_cycle_mean_torque()

    @staticmethod
    def single_cylinder_normalized_torque(local_angle_deg):
        """
        Normalized torque profile for a single cylinder over 0-720° crank cycle.

        Strokes:
          0° - 180°: Intake (minor pumping loss)
        180° - 360°: Compression (negative compression torque)
        360° - 540°: Power (combustion gas expansion torque pulse)
        540° - 720°: Exhaust (minor pumping loss)
        """
        angle = local_angle_deg % 720.0

        if 360.0 <= angle < 540.0:
            # Power stroke: smooth gas expansion pulse
            phi = (angle - 360.0) / 180.0  # 0 to 1
            # Gas pressure / torque pulse shape peaking around 30-40° after TDC
            pulse = (math.sin(math.pi * phi) ** 1.5) * (1.0 + 0.5 * math.cos(math.pi * phi))
            return max(0.0, pulse * 4.2)
        elif 180.0 <= angle < 360.0:
            # Compression stroke: negative torque work done on gas
            phi = (angle - 180.0) / 180.0
            return -0.15 * math.sin(math.pi * phi)
        else:
            # Intake / Exhaust pumping losses
            return -0.02

    def _compute_cycle_mean_torque(self):
        """
        Integrates normalized 4-cylinder total torque over 720°
        to compute cycle average factor for exact torque conservation.
        """
        steps = 720
        total_sum = 0.0
        for deg in range(steps):
            angle = float(deg)
            cyl_sum = sum(
                self.single_cylinder_normalized_torque(angle + offset)
                for offset in self.phase_offsets.values()
            )
            total_sum += cyl_sum

        return total_sum / steps

    def calculate_torques(self, crank_angle_deg, mean_torque, combustion_efficiencies=None):
        """
        Calculate instantaneous individual cylinder torques and total torque.

        Parameters
        ----------
        crank_angle_deg : float
            Current crankshaft angle in degrees [0, 720).
        mean_torque : float
            Mean engine torque capability (N*m) from throttle and RPM.
        combustion_efficiencies : Dict[int, float], optional
            Efficiency factor per cylinder {1: e1, 2: e2, 3: e3, 4: e4} in [0.5, 1.0].
            Default is 1.0 (healthy) for all cylinders.

        Returns
        -------
        tuple (dict, float)
            Dictionary of cylinder torques {1: T1, 2: T2, 3: T3, 4: T4}
            and total instantaneous engine torque.
        """
        if combustion_efficiencies is None:
            combustion_efficiencies = {1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0}

        raw_torques = {}
        for cyl_id, offset in self.phase_offsets.items():
            local_angle = (crank_angle_deg + offset) % 720.0
            eff = combustion_efficiencies.get(cyl_id, 1.0)
            raw_torques[cyl_id] = self.single_cylinder_normalized_torque(local_angle) * eff

        # Scale factor so that cycle-averaged total torque equals mean_torque
        scale = mean_torque / self._mean_normalized_torque if self._mean_normalized_torque > 0 else 0.0

        cylinder_torques = {
            cyl_id: raw_t * scale
            for cyl_id, raw_t in raw_torques.items()
        }

        total_instant_torque = sum(cylinder_torques.values())
        # Engine total torque cannot be negative in forward rotation
        total_instant_torque = max(0.0, total_instant_torque)

        return cylinder_torques, total_instant_torque
