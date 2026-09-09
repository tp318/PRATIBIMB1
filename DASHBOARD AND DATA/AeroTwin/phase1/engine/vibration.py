"""
AeroTwin-4 Mechanical Vibration Subsystem Model.

Derives mechanical vibration directly from instantaneous 4-cylinder torque fluctuations
|T_instant - T_mean| and crankshaft rotational speed imbalance.
"""

import math


class VibrationModel:
    """
    Simulates engine mechanical vibration amplitude (g) derived from
    4-cylinder torque pulsation and rotational speed dynamics.
    """

    def __init__(self, inertia=0.20, torque_gain=0.0035, rot_gain=0.15):
        self.inertia = inertia
        self.torque_gain = torque_gain
        self.rot_gain = rot_gain

    def update(self, instant_torque, mean_torque, rpm, dt=0.001):
        """
        Calculate mechanical vibration amplitude.

        Parameters
        ----------
        instant_torque : float
            Instantaneous total engine torque in N*m.
        mean_torque : float
            Cycle-mean engine torque in N*m.
        rpm : float
            Current engine speed in RPM.
        dt : float
            Time step in seconds.

        Returns
        -------
        dict
            Vibration state containing vibration (g).
        """
        # Torque fluctuation delta
        torque_fluctuation = abs(instant_torque - mean_torque)

        # Vibration component from 4-cylinder combustion pulse acceleration
        pulse_vib = self.torque_gain * (torque_fluctuation / self.inertia)

        # Vibration component from rotational dynamics imbalance
        rot_vib = self.rot_gain * ((rpm / 3500.0) ** 2)

        # Total RMS vibration level in g
        vibration_g = 0.05 + pulse_vib + rot_vib

        return {
            "vibration": max(0.01, vibration_g)
        }
