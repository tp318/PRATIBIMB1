"""
AeroTwin-4 Run Condition Sampler.

Real engines are never identical, and never fly in identical air. Two sources of
legitimate run-to-run variability are modelled here, both strictly independent of
any degradation state:

1. BUILD VARIATION  - unit-to-unit manufacturing tolerance of a single engine
                      (bearing preload, cooling fin area, pump clearance, ...).
                      Fixed for the life of an engine unit.
2. ENVIRONMENT      - ambient conditions of the sortie (altitude, outside air
                      temperature). Fixed for the duration of one run.

Without this, every healthy run is bit-identical, healthy channel variance
collapses to ~0, and any anomaly threshold calibrated on it is meaningless.
"""

import random
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from phase1.engine.parameters import ENGINE


# Fractional (1-sigma) manufacturing tolerance per parameter.
# Values are deliberately small: these are build tolerances, not faults.
BUILD_TOLERANCE = {
    "friction_coefficient": 0.040,
    "cht_thermal_mass": 0.030,
    "oil_thermal_mass": 0.030,
    "cooling_coefficient": 0.045,
    "oil_pressure_max": 0.025,
    "oil_pressure_idle": 0.025,
    "bsfc": 0.020,
    "vibration_torque_gain": 0.050,
    "vibration_rotational_gain": 0.050,
    "load_coefficient": 0.015,
}

# Tolerances are truncated at +/- this many sigma so a build stays a *healthy*
# engine and never silently imitates a degradation signature.
BUILD_CLIP_SIGMA = 2.0

# ISA lapse rate: ambient air temperature falls ~6.5 degC per 1000 m of altitude.
ISA_LAPSE_RATE_C_PER_M = 0.0065
ISA_SEA_LEVEL_TEMP_C = 15.0


@dataclass
class RunConditions:
    """Resolved operating context for a single simulation run."""

    run_id: str
    seed: int
    altitude_m: float
    ambient_temperature_c: float
    sea_level_temp_c: float
    build_factors: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "seed": self.seed,
            "altitude_m": round(self.altitude_m, 1),
            "ambient_temperature_c": round(self.ambient_temperature_c, 2),
            "sea_level_temp_c": round(self.sea_level_temp_c, 2),
            "build_factors": {k: round(v, 6) for k, v in self.build_factors.items()},
        }


class RunConditionSampler:
    """
    Deterministically samples build variation and environmental conditions for a run.

    The same (run_id, seed) always yields the same conditions, so datasets stay
    reproducible while individual runs stay genuinely distinct.
    """

    def __init__(
        self,
        altitude_range_m: tuple = (0.0, 7600.0),
        sea_level_temp_range_c: tuple = (5.0, 40.0),
    ):
        self.altitude_range_m = altitude_range_m
        self.sea_level_temp_range_c = sea_level_temp_range_c

    def sample(self, run_id: str, seed: int) -> RunConditions:
        rng = random.Random(f"{run_id}:{seed}")

        build_factors = {}
        for name, tol in BUILD_TOLERANCE.items():
            z = max(-BUILD_CLIP_SIGMA, min(BUILD_CLIP_SIGMA, rng.gauss(0.0, 1.0)))
            build_factors[name] = 1.0 + z * tol

        altitude_m = rng.uniform(*self.altitude_range_m)
        sea_level_temp_c = rng.uniform(*self.sea_level_temp_range_c)
        ambient_temperature_c = sea_level_temp_c - ISA_LAPSE_RATE_C_PER_M * altitude_m

        return RunConditions(
            run_id=run_id,
            seed=seed,
            altitude_m=altitude_m,
            ambient_temperature_c=ambient_temperature_c,
            sea_level_temp_c=sea_level_temp_c,
            build_factors=build_factors,
        )

    def build_engine_parameters(
        self, conditions: RunConditions, base: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Produce a fresh ENGINE parameter dict with this run's build variation and
        ambient temperature applied. The base dict is never mutated.
        """
        params = dict(base if base is not None else ENGINE)

        for name, factor in conditions.build_factors.items():
            if name in params:
                params[name] = params[name] * factor

        params["ambient_temperature"] = conditions.ambient_temperature_c
        params["altitude_m"] = conditions.altitude_m
        return params
