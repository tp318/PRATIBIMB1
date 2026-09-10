"""
engine.py
=========
DT CORE — Digital Twin Engine Orchestrator

This is the top-level Digital Twin core as described in Absolutely.docx:

    Environment [h, Ta, Pa]
         ↓
    Air Path → air mass flow, η_v
         ↓
    Fuel Model → fuel flow, AFR, λ
         ↓
    Combustion → torque, heat rates
         ↓
    Crankshaft → RPM
         ↓
    Thermal → CHT, EGT
         ↓
    Oil → oil pressure, oil temperature
         ↓
    Y_predicted = {RPM, EGT, CHT, OilPress, OilTemp, FuelFlow}
         ↓
    Residual = Y_measured − Y_predicted  (z-scored)

Usage:
    from engine import DigitalTwinEngine

    dt = DigitalTwinEngine()
    # On each 0.1s tick:
    result = dt.step(
        throttle   = 0.65,
        altitude_ft = 3000,
        ambient_c  = 15.0,
        measured   = {"RPM": 4500, "EGT": 695, "CHT": 140, ...},  # or None
    )
    print(result.predicted)   # DT predictions
    print(result.residuals)   # r_i = measured - predicted
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from environment import compute_atmosphere, AtmosphericState
from air_path    import compute_air_path,   AirPathState
from fuel_model  import compute_fuel,       FuelState
from combustion  import compute_combustion, CombustionState
from crankshaft  import step_crankshaft,    CrankshaftState
from thermal     import step_thermal,       ThermalState
from oil         import step_oil,           OilState
from residuals   import compute_residuals,  ResidualFrame

DT_STEP_S = 0.1   # 10 Hz — one step per 100 ms


@dataclass
class DTStepResult:
    """Complete output of one DT timestep."""
    timestamp:   float              # wall-clock seconds since sortie start
    mission_time: float             # simulation seconds
    inputs: Dict[str, float]        # throttle, altitude_ft, ambient_c
    predicted: Dict[str, float]     # Y_pred from DT physics chain
    measured:  Optional[Dict[str, float]]  # Y_meas from real engine (if available)
    residuals: Optional[ResidualFrame]     # residuals (None if no measured data)

    # Sub-model outputs (for detailed inspection)
    atmosphere:   Optional[Dict]  = field(default=None, repr=False)
    air_path:     Optional[Dict]  = field(default=None, repr=False)
    fuel:         Optional[Dict]  = field(default=None, repr=False)
    combustion:   Optional[Dict]  = field(default=None, repr=False)
    crankshaft:   Optional[Dict]  = field(default=None, repr=False)
    thermal:      Optional[Dict]  = field(default=None, repr=False)
    oil:          Optional[Dict]  = field(default=None, repr=False)

    def to_dict(self) -> Dict[str, Any]:
        """Flat dict suitable for WebSocket broadcast and CSV logging."""
        d: Dict[str, Any] = {
            "timestamp":    round(self.timestamp,    3),
            "mission_time": round(self.mission_time, 2),
            "throttle":     round(self.inputs.get("throttle",    0.5), 3),
            "altitude_ft":  round(self.inputs.get("altitude_ft", 0.0), 1),
            "ambient_c":    round(self.inputs.get("ambient_c",  15.0), 1),
        }
        # Predicted outputs
        for k, v in (self.predicted or {}).items():
            d[f"dt_{k}"] = round(float(v), 4)

        # Measured outputs (from real engine)
        for k, v in (self.measured or {}).items():
            d[f"real_{k}"] = round(float(v), 4)

        # Residuals
        if self.residuals:
            for k, v in self.residuals.raw.items():
                d[f"r_{k}"] = round(float(v), 4)
            for k, v in self.residuals.normalized.items():
                d[f"z_{k}"] = round(float(v), 4)
            d["max_z"]   = round(self.residuals.max_z, 3)
            d["flagged"] = self.residuals.flagged
        return d

    def to_ws_dict(self) -> Dict[str, Any]:
        """Structured dict for WebSocket broadcast (frontend-friendly)."""
        return {
            "type":         "dt_frame",
            "timestamp":    round(self.timestamp,    3),
            "mission_time": round(self.mission_time, 2),
            "predicted":    {k: round(float(v), 3) for k, v in self.predicted.items()},
            "measured":     {k: round(float(v), 3) for k, v in (self.measured or {}).items()},
            "residuals":    self.residuals.to_dict() if self.residuals else None,
            "inputs":       self.inputs,
        }


class DigitalTwinEngine:
    """
    The DT Core physics engine.

    Maintains all integrator states across timesteps.  Call step() at 10 Hz
    to advance the simulation by DT_STEP_S (0.1 s).
    """

    def __init__(
        self,
        initial_rpm:    float = 800.0,
        initial_cht_c:  float =  70.0,
        initial_egt_c:  float = 300.0,
        initial_oil_bar: float = 2.5,
        initial_oil_t_c: float =  60.0,
        dt: float = DT_STEP_S,
    ):
        self.dt = dt
        self._crank   = CrankshaftState(rpm=initial_rpm,
                                         omega=initial_rpm * (2*math.pi/60))
        self._thermal = ThermalState(cht_c=initial_cht_c, egt_c=initial_egt_c)
        self._oil     = OilState(oil_press_bar=initial_oil_bar,
                                  oil_temp_c=initial_oil_t_c)
        self._mission_time: float = 0.0
        self._start_wall:   float = time.monotonic()

    # ---------------------------------------------------------------- public API

    def reset(self) -> None:
        """Reset all integrator states to initial conditions."""
        self.__init__()

    def step(
        self,
        throttle:    float,
        altitude_ft: float = 0.0,
        ambient_c:   float = 15.0,
        measured:    Optional[Dict[str, float]] = None,
    ) -> DTStepResult:
        """
        Advance the DT by one timestep and return the full result.

        Parameters
        ----------
        throttle : float
            Normalised throttle, 0 – 1.
        altitude_ft : float
            Pressure altitude in feet.
        ambient_c : float
            Outside air temperature in °C.
        measured : dict, optional
            Real engine sensor readings keyed by channel name.
            If provided, residuals are computed.
            Keys: RPM, EGT, CHT, OilPress, OilTemp, FuelFlow, Vibration, AltVoltage

        Returns
        -------
        DTStepResult
        """
        throttle    = max(0.0, min(1.0, throttle))
        altitude_ft = max(0.0, min(25000.0, altitude_ft))

        # 1. Environment
        atm = compute_atmosphere(altitude_ft, ambient_c)

        # 2. Air path
        air = compute_air_path(throttle, self._crank.rpm, atm)

        # 3. Fuel model
        fuel = compute_fuel(throttle, air)

        # 4. Combustion
        comb = compute_combustion(self._crank.rpm, fuel, air)

        # 5. Crankshaft dynamics
        step_crankshaft(self._crank, comb.torque_nm, dt=self.dt)

        # 6. Thermal model
        step_thermal(self._thermal, comb, air, ambient_c, dt=self.dt)

        # 7. Oil model
        step_oil(self._oil, self._crank.rpm, self._thermal, dt=self.dt)

        # Build Y_predicted
        predicted: Dict[str, float] = {
            "RPM":        self._crank.rpm,
            "EGT":        self._thermal.egt_c,
            "CHT":        self._thermal.cht_c,
            "OilPress":   self._oil.oil_press_bar,
            "OilTemp":    self._oil.oil_temp_c,
            "FuelFlow":   fuel.fuel_flow_lph,
        }

        # Residuals (only when measured data is available)
        residuals: Optional[ResidualFrame] = None
        if measured:
            residuals = compute_residuals(measured, predicted)

        # Advance mission time
        t_now = self._mission_time
        self._mission_time += self.dt

        return DTStepResult(
            timestamp    = time.monotonic() - self._start_wall,
            mission_time = t_now,
            inputs       = {"throttle": throttle, "altitude_ft": altitude_ft, "ambient_c": ambient_c},
            predicted    = predicted,
            measured     = measured,
            residuals    = residuals,
            atmosphere   = atm.to_dict(),
            air_path     = air.to_dict(),
            fuel         = fuel.to_dict(),
            combustion   = comb.to_dict(),
            crankshaft   = self._crank.to_dict(),
            thermal      = self._thermal.to_dict(),
            oil          = self._oil.to_dict(),
        )
