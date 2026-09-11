"""
profiles.py
===========
Mission profile and operating-condition generation (Part 1).

A run-to-failure trajectory is not a random walk through the operating envelope.
It is a sequence of SORTIES, each of which has a realistic phase structure
(start, taxi, takeoff, climb, cruise, loiter, transitions, descent, landing) and
is flown in a THEATRE that sets the ambient temperature and typical altitude band.

Two rules matter for the integrity of the dataset:

  1. The theatre and the mission mix are drawn INDEPENDENTLY of the degradation
     mechanism assigned to the run. If hot theatres were preferentially given
     cooling faults, "overheating" would become trivially predictable from
     ambient temperature and the whole exercise would be circular. The
     independence is asserted by a test in scripts/check_dataset.py.

  2. Operating hours only accrue while the engine is running. Ground time
     between sorties is skipped, so `operating_hours` is true engine time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

# --------------------------------------------------------------------------- #
# Theatres: where the fleet operates. Sets the sea-level day temperature and
# the altitude band the aircraft tends to work in.
# --------------------------------------------------------------------------- #

THEATRES = {
    "temperate": {"sl_temp_c": (8.0, 24.0), "alt_bias": 0.0, "weight": 0.30},
    "hot_desert": {"sl_temp_c": (28.0, 43.0), "alt_bias": -0.10, "weight": 0.22},
    "cold_high": {"sl_temp_c": (-12.0, 6.0), "alt_bias": +0.18, "weight": 0.18},
    "maritime": {"sl_temp_c": (12.0, 27.0), "alt_bias": -0.05, "weight": 0.18},
    "tropical_humid": {"sl_temp_c": (24.0, 36.0), "alt_bias": -0.02, "weight": 0.12},
}

# --------------------------------------------------------------------------- #
# Sortie archetypes. `phases` is a list of (phase, nominal_hours, spread).
# --------------------------------------------------------------------------- #

SORTIE_TYPES = {
    "endurance_isr": {
        "weight": 0.30,
        "cruise_alt_ft": (14000.0, 19500.0),
        "loiter_alt_ft": (12000.0, 18000.0),
        "phases": [
            ("start_taxi", 0.30, 0.10),
            ("takeoff", 0.10, 0.03),
            ("climb", 1.10, 0.30),
            ("cruise", 2.40, 0.70),
            ("loiter", 12.00, 4.50),
            ("cruise", 1.80, 0.60),
            ("descent", 0.80, 0.25),
            ("landing_taxi", 0.30, 0.10),
        ],
    },
    "medium_recon": {
        "weight": 0.24,
        "cruise_alt_ft": (9000.0, 15000.0),
        "loiter_alt_ft": (7000.0, 13000.0),
        "phases": [
            ("start_taxi", 0.25, 0.08),
            ("takeoff", 0.10, 0.03),
            ("climb", 0.70, 0.20),
            ("cruise", 1.60, 0.50),
            ("loiter", 4.50, 1.80),
            ("transition", 0.60, 0.25),
            ("cruise", 1.20, 0.40),
            ("descent", 0.55, 0.18),
            ("landing_taxi", 0.25, 0.08),
        ],
    },
    "low_level_tactical": {
        "weight": 0.16,
        "cruise_alt_ft": (1500.0, 6500.0),
        "loiter_alt_ft": (800.0, 5000.0),
        "phases": [
            ("start_taxi", 0.22, 0.07),
            ("takeoff", 0.10, 0.03),
            ("climb", 0.30, 0.10),
            ("transition", 1.40, 0.50),
            ("loiter", 2.20, 0.90),
            ("transition", 1.10, 0.45),
            ("descent", 0.25, 0.10),
            ("landing_taxi", 0.22, 0.07),
        ],
    },
    "transit_ferry": {
        "weight": 0.14,
        "cruise_alt_ft": (11000.0, 17500.0),
        "loiter_alt_ft": (10000.0, 15000.0),
        "phases": [
            ("start_taxi", 0.25, 0.08),
            ("takeoff", 0.10, 0.03),
            ("climb", 1.00, 0.28),
            ("cruise", 6.50, 2.20),
            ("descent", 0.75, 0.22),
            ("landing_taxi", 0.25, 0.08),
        ],
    },
    "training_circuit": {
        "weight": 0.16,
        "cruise_alt_ft": (2000.0, 7000.0),
        "loiter_alt_ft": (1500.0, 5500.0),
        "phases": [
            ("start_taxi", 0.30, 0.10),
            ("takeoff", 0.10, 0.03),
            ("climb", 0.25, 0.08),
            ("transition", 1.30, 0.55),
            ("descent", 0.20, 0.08),
            ("takeoff", 0.10, 0.03),
            ("climb", 0.25, 0.08),
            ("transition", 1.10, 0.50),
            ("descent", 0.22, 0.09),
            ("landing_taxi", 0.28, 0.09),
        ],
    },
}

# Per-phase throttle / load / cooling-airflow character.
#   throttle    : (mean, std)
#   load_factor : propeller loading multiplier (pitch, airspeed)
#   airspeed_f  : ram cooling airflow multiplier
#   alt_frac    : fraction of the sortie's cruise altitude flown in this phase
#   volatility  : within-snapshot throttle activity (drives transient content)
PHASE_CHARACTER = {
    "start_taxi":   {"throttle": (0.16, 0.04), "load": 0.55, "airspeed": 0.30,
                     "alt_frac": 0.00, "volatility": 0.05},
    "takeoff":      {"throttle": (0.97, 0.03), "load": 1.02, "airspeed": 0.85,
                     "alt_frac": 0.04, "volatility": 0.04},
    "climb":        {"throttle": (0.88, 0.05), "load": 0.97, "airspeed": 0.92,
                     "alt_frac": 0.55, "volatility": 0.05},
    "cruise":       {"throttle": (0.74, 0.06), "load": 1.00, "airspeed": 1.05,
                     "alt_frac": 1.00, "volatility": 0.03},
    "loiter":       {"throttle": (0.58, 0.07), "load": 0.94, "airspeed": 0.90,
                     "alt_frac": 0.93, "volatility": 0.04},
    "transition":   {"throttle": (0.66, 0.17), "load": 0.98, "airspeed": 0.95,
                     "alt_frac": 0.70, "volatility": 0.26},
    "descent":      {"throttle": (0.28, 0.09), "load": 1.06, "airspeed": 1.10,
                     "alt_frac": 0.45, "volatility": 0.10},
    "landing_taxi": {"throttle": (0.17, 0.05), "load": 0.55, "airspeed": 0.32,
                     "alt_frac": 0.00, "volatility": 0.07},
}

PHASE_NAMES = list(PHASE_CHARACTER.keys())
PHASE_ID = {name: i for i, name in enumerate(PHASE_NAMES)}


@dataclass
class MissionPlan:
    """Per-run mission character, drawn independently of the fault mechanism."""
    theatre: str
    sortie_mix: Dict[str, float]
    sl_temp_mean_c: float
    sl_temp_season_amp_c: float
    humidity_mean: float
    alt_bias: float
    envelope: str          # "nominal" or "stress"


def sample_mission_plan(rng: np.random.Generator, envelope: str = "nominal") -> MissionPlan:
    """Draw a mission plan. `envelope` selects the nominal training envelope or
    the deliberately harsher stress envelope used for the hard test set."""
    if envelope == "stress":
        theatre = rng.choice(["hot_desert", "tropical_humid"], p=[0.65, 0.35])
    else:
        names = list(THEATRES)
        weights = np.array([THEATRES[n]["weight"] for n in names], dtype=float)
        theatre = str(rng.choice(names, p=weights / weights.sum()))

    lo, hi = THEATRES[theatre]["sl_temp_c"]
    sl_mean = float(rng.uniform(lo, hi))
    if envelope == "stress":
        sl_mean = float(rng.uniform(38.0, 52.0))

    # Sortie mix: a Dirichlet draw around the fleet-average weights, so different
    # airframes genuinely fly different duty cycles.
    types = list(SORTIE_TYPES)
    base = np.array([SORTIE_TYPES[t]["weight"] for t in types], dtype=float)
    mix = rng.dirichlet(base * 9.0)

    return MissionPlan(
        theatre=theatre,
        sortie_mix={t: float(m) for t, m in zip(types, mix)},
        sl_temp_mean_c=sl_mean,
        sl_temp_season_amp_c=float(rng.uniform(2.0, 8.0)),
        humidity_mean=float(np.clip(rng.normal(0.55 if theatre != "hot_desert" else 0.22, 0.12),
                                    0.05, 0.95)),
        alt_bias=float(THEATRES[theatre]["alt_bias"]),
        envelope=envelope,
    )


def generate_operating_points(plan: MissionPlan, n_snapshots: int,
                              snapshot_hours: float,
                              rng: np.random.Generator) -> Dict[str, np.ndarray]:
    """Generate `n_snapshots` consecutive condition-monitoring snapshots.

    Returns arrays of equal length describing the operating point at each
    snapshot: altitude, ambient temperature, throttle, load factor, cooling
    airflow factor, mission phase id, sortie index and humidity.
    """
    alt = np.zeros(n_snapshots)
    oat = np.zeros(n_snapshots)
    thr = np.zeros(n_snapshots)
    load = np.zeros(n_snapshots)
    aspd = np.zeros(n_snapshots)
    volat = np.zeros(n_snapshots)
    phase = np.zeros(n_snapshots, dtype=np.int16)
    sortie = np.zeros(n_snapshots, dtype=np.int32)
    humid = np.zeros(n_snapshots)

    types = list(plan.sortie_mix)
    probs = np.array([plan.sortie_mix[t] for t in types], dtype=float)
    probs = probs / probs.sum()

    stress = plan.envelope == "stress"
    alt_lo, alt_hi = (15000.0, 20000.0) if stress else (0.0, 15000.0)

    i = 0
    sortie_idx = 0
    elapsed_h = 0.0

    while i < n_snapshots:
        stype = str(rng.choice(types, p=probs))
        spec = SORTIE_TYPES[stype]

        # Altitude band for this sortie
        c_lo, c_hi = spec["cruise_alt_ft"]
        cruise_alt = float(rng.uniform(c_lo, c_hi)) * (1.0 + plan.alt_bias)
        cruise_alt = float(np.clip(cruise_alt, alt_lo, alt_hi))
        if stress:
            cruise_alt = float(rng.uniform(15500.0, 19800.0))

        # Day temperature: seasonal cycle + weather scatter
        season = np.sin(2.0 * np.pi * elapsed_h / 2400.0 + rng.uniform(0, 6.28))
        sl_temp = (plan.sl_temp_mean_c
                   + plan.sl_temp_season_amp_c * season
                   + rng.normal(0.0, 2.2))
        sortie_humidity = float(np.clip(plan.humidity_mean + rng.normal(0, 0.08), 0.02, 1.0))

        for phase_name, nom_h, spread_h in spec["phases"]:
            if i >= n_snapshots:
                break
            dur = max(snapshot_hours, rng.normal(nom_h, spread_h))
            n_steps = max(1, int(round(dur / snapshot_hours)))
            ch = PHASE_CHARACTER[phase_name]

            for k in range(n_steps):
                if i >= n_snapshots:
                    break

                # Altitude: climb and descent ramp between ground and cruise.
                frac = (k + 0.5) / n_steps
                if phase_name == "climb":
                    a = cruise_alt * (0.05 + 0.95 * frac)
                elif phase_name == "descent":
                    a = cruise_alt * (0.90 * (1.0 - frac) + 0.03)
                else:
                    a = cruise_alt * ch["alt_frac"]
                a = float(np.clip(a + rng.normal(0.0, 140.0), 0.0, 21000.0))

                # Ambient temperature at altitude: lapse from the day's sea-level
                # temperature, plus local weather noise.
                t_at_alt = sl_temp - 1.98 * (a / 1000.0) + rng.normal(0.0, 1.1)

                t_mean, t_std = ch["throttle"]
                th = float(np.clip(rng.normal(t_mean, t_std), 0.08, 1.0))

                alt[i] = a
                oat[i] = t_at_alt
                thr[i] = th
                load[i] = ch["load"] * float(np.clip(rng.normal(1.0, 0.035), 0.85, 1.15))
                aspd[i] = ch["airspeed"] * float(np.clip(rng.normal(1.0, 0.05), 0.8, 1.25))
                volat[i] = ch["volatility"]
                phase[i] = PHASE_ID[phase_name]
                sortie[i] = sortie_idx
                humid[i] = sortie_humidity
                i += 1
                elapsed_h += snapshot_hours

        sortie_idx += 1

    return {
        "altitude_ft": alt,
        "ambient_temperature_c": oat,
        "throttle": thr,
        "load_factor": load,
        "airspeed_factor": aspd,
        "throttle_volatility": volat,
        "mission_phase": phase,
        "sortie_index": sortie,
        "humidity": humid,
    }
