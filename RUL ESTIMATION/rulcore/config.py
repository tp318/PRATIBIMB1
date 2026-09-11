"""
config.py
=========
Central configuration for the hybrid physics-informed RUL / prognostics pipeline.

Every magic number that affects dataset generation, EOL definition, health-index
construction, UKF tuning or model training lives here so that it can be audited
in one place.

Engine class modelled: Rotax 914-class 4-cylinder turbo-normalised aero piston
engine (1352 cc), representative of MALE UAV propulsion.

TIMESCALE DESIGN (important, see README):
    The pipeline is deliberately two-timescale.

      * FAST scale  (dt = 0.1 s, 10 Hz): within-snapshot engine dynamics used to
        synthesise realistic transients and a vibration window.
      * SLOW scale  (dt = SNAPSHOT_HOURS): condition-monitoring snapshot cadence.
        Degradation, UKF health estimation, GRU trending and RUL all live here.

    Simulating 10 Hz continuously for a 400-flight-hour life would be 1.4e7 steps
    per run, which is intractable for 200 runs. Instead each snapshot runs a short
    fast-time burst and condenses it into one monitoring row. This mirrors how real
    engine health monitoring works (periodic condensed reports, not raw streams)
    and matches the CMAPSS convention of one row per flight cycle.
"""

from __future__ import annotations

import os

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #

_HERE = os.path.dirname(os.path.abspath(__file__))
PKG_ROOT = _HERE                                   # .../RUL ESTIMATION/rulcore
MODULE_ROOT = os.path.dirname(PKG_ROOT)            # .../RUL ESTIMATION

OUTPUT_DIR = os.path.join(MODULE_ROOT, "outputs")
DATA_DIR = os.path.join(OUTPUT_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")            # per-run raw trajectories
FEATURE_DIR = os.path.join(DATA_DIR, "features")   # feature dataset + splits
FIGURE_DIR = os.path.join(OUTPUT_DIR, "figures")
REPORT_DIR = os.path.join(OUTPUT_DIR, "reports")
ARTIFACT_DIR = os.path.join(MODULE_ROOT, "artifacts")

for _d in (OUTPUT_DIR, DATA_DIR, RAW_DIR, FEATURE_DIR, FIGURE_DIR, REPORT_DIR, ARTIFACT_DIR):
    os.makedirs(_d, exist_ok=True)


# --------------------------------------------------------------------------- #
# Timescales
# --------------------------------------------------------------------------- #

FAST_DT_S = 0.1                 # 10 Hz within-snapshot integration step
SNAPSHOT_HOURS = 0.25           # 15 min between condition-monitoring snapshots
SNAPSHOT_BURST_S = 60.0         # seconds of fast-time simulation per snapshot
VIB_FS_HZ = 2000.0              # vibration synthesis sample rate
VIB_WINDOW_S = 1.0              # length of raw vibration window per snapshot


# --------------------------------------------------------------------------- #
# Dataset scale
# --------------------------------------------------------------------------- #
# Rationale for run count is documented in README / DATASET_DESIGN.md.
# The statistically meaningful sample size for a prognostic model is the number
# of INDEPENDENT run-to-failure trajectories, not the number of rows or windows.

N_RUNS_MAIN = 200               # main fleet (nominal operating envelope)
N_RUNS_STRESS = 30              # stress envelope (hot / high) - test only
N_RUNS_CROSS_ENGINE = 20        # off-nominal engine builds - test only
RANDOM_SEED = 20260911


# --------------------------------------------------------------------------- #
# Engine nominal parameters (fleet mean).
# Per-engine builds perturb these; the Digital Twin always uses the fleet mean,
# which is what creates realistic non-zero healthy residuals.
# --------------------------------------------------------------------------- #

ENGINE_NOMINAL = {
    "displacement_m3": 1.352e-3,
    "n_cylinders": 4,
    "eta_v_max": 0.88,          # peak volumetric efficiency
    "eta_v_idle": 0.30,
    "rpm_ve_peak": 5000.0,
    "map_idle_frac": 0.24,      # manifold pressure fraction at closed throttle
    "k_inj": 1.45e-4,           # kg of fuel per unit injection_command per cycle
    "lhv": 43.5e6,              # J/kg AVGAS 100LL
    "eta_therm": 0.340,         # indicated thermal efficiency
    "fric_b0": 2.6,             # N.m constant (boundary) friction
    "fric_b1": 0.021,           # N.m/(rad/s) viscous friction
    "k_prop": 2.92e-4,          # N.m/(rad/s)^2 propeller load coefficient
    "j_rot": 0.12,              # kg.m^2 rotating inertia
    # thermal
    "cht_hA": 132.0,            # W/K cooling conductance at reference airflow
    "cool_mdot_ref": 0.0300,    # kg/s air flow defining the cooling reference
    "egt_loss_mdot": 0.0060,    # kg/s scale of exhaust heat loss to the probe
    "cht_base_c": 22.0,         # deg C offset of head over ambient at zero heat
    "tau_cht_s": 25.0,
    "egt_gain_scale": 1.00,     # trim on the exhaust-enthalpy temperature rise
    "oil_fric_gain": 2.14e-3,   # deg C of oil temp per W of friction power
    "tau_egt_s": 3.0,
    # oil
    "oil_press_idle_bar": 1.80,
    "oil_k_pump": 7.0e-4,       # bar per RPM above idle
    "oil_k_visc": 0.0080,       # bar lost per deg C above reference
    "oil_temp_ref_c": 80.0,
    "oil_temp_base_c": 27.0,
    "oil_cht_coupling": 0.35,
    "tau_oil_p_s": 2.0,
    "tau_oil_t_s": 20.0,
    # vibration
    "vib_base_g": 1.10,         # baseline RMS at reference speed
    "vib_rpm_exp": 1.55,        # RMS scaling exponent with speed
}

# Per-engine build scatter (1-sigma, relative unless stated).
# These represent manufacturing/assembly variation across the fleet.
ENGINE_BUILD_SCATTER = {
    "eta_v_max": 0.020,
    "k_inj": 0.020,
    "eta_therm": 0.020,
    "fric_b0": 0.070,
    "fric_b1": 0.070,
    "k_prop": 0.030,
    "cht_hA": 0.050,
    "egt_gain_scale": 0.025,
    "oil_press_idle_bar": 0.040,
    "oil_k_pump": 0.040,
    "oil_cht_coupling": 0.040,
    "vib_base_g": 0.090,
}

# Cross-engine generalisation holdout uses deliberately wider scatter (Part 21).
CROSS_ENGINE_SCATTER_MULT = 2.5


# --------------------------------------------------------------------------- #
# Health parameters (latent simulation ground truth).
# theta_nominal is the commissioning value; theta_eol is the value at which that
# single mechanism alone is considered to have consumed the engine's life.
# --------------------------------------------------------------------------- #

HEALTH_PARAMS = ["eta_inj", "eta_comb", "h_cool", "friction_mult", "lub_health", "eta_vol"]

HEALTH_NOMINAL = {
    "eta_inj": 1.00,
    "eta_comb": 1.00,
    "h_cool": 1.00,
    "friction_mult": 1.00,
    "lub_health": 1.00,
    "eta_vol": 1.00,
}

# Direction of degradation: -1 means the parameter decreases as health is lost.
HEALTH_DIRECTION = {
    "eta_inj": -1,
    "eta_comb": -1,
    "h_cool": -1,
    "friction_mult": +1,
    "lub_health": -1,
    "eta_vol": -1,
}

# Value of each parameter at which that mechanism alone reaches end of life.
# Used both for the health index and as a mechanism-level guard rail.
# DERIVED, not hand-picked. Each value is the point at which that parameter,
# acting alone, first trips one of the EOL_CRITERIA at the reference condition.
# Regenerate with: python scripts/calibrate_eol.py
# Binding criterion per parameter:
#   eta_inj -> power   eta_comb -> power     h_cool  -> cht_rise
#   friction_mult -> vib   lub_health -> oil_press   eta_vol -> power
HEALTH_EOL = {
    "eta_inj": 0.9260,
    "eta_comb": 0.9380,
    "h_cool": 0.7867,
    "friction_mult": 1.3663,
    "lub_health": 0.6986,
    "eta_vol": 0.8534,
}


# --------------------------------------------------------------------------- #
# END OF LIFE DEFINITION (Part 8)
# --------------------------------------------------------------------------- #
# EOL is NOT a function of a health index. It is defined by engineering limits
# evaluated at a STANDARD REFERENCE CONDITION - a virtual test-cell check flown
# at every snapshot with noise disabled, so that limits are comparable across
# operating conditions. EOL is the first snapshot at which ANY criterion is
# violated and stays violated for EOL_PERSISTENCE_SNAPSHOTS consecutive checks.
#
# Reference condition ("power assurance check"), chosen to sit in the normal
# cruise band of a MALE UAV:
EOL_REF_CONDITION = {
    "altitude_ft": 8000.0,
    "ambient_c": 5.0,           # approximately ISA at 8000 ft
    "throttle": 0.78,
    "injection_command": None,  # resolved to the commissioning value per engine
    "load_factor": 1.0,
}

EOL_PERSISTENCE_SNAPSHOTS = 3   # criterion must hold for 3 x 0.25 h = 45 min

# Criterion limits. Ratios are relative to the SAME ENGINE's commissioning
# (green-run) value, which is how real fleets set limits - it removes build
# scatter from the limit and makes the criterion about degradation, not about
# whether this particular engine was born strong or weak.
EOL_CRITERIA = {
    # 1. Fuel efficiency: brake-specific fuel consumption rise
    "bsfc_rise_frac": 0.10,        # +10% BSFC vs commissioning
    # 2. Power capability loss at the reference condition
    "power_loss_frac": 0.10,       # -10% brake power vs commissioning
    # 3. Thermal limit: absolute CHT ceiling (air-cooled head, deg C)
    "cht_limit_c": 232.0,          # 450 F, classical air-cooled CHT red line
    "cht_rise_c": 40.0,            # or +40 C over commissioning at ref condition
    # 4. Lubrication: minimum oil pressure at reference cruise
    "oil_press_min_bar": 2.60,
    "oil_temp_limit_c": 118.0,
    # 5. Mechanical: vibration growth
    "vib_rms_ratio": 1.70,         # 1.7x commissioning RMS
}


# --------------------------------------------------------------------------- #
# Degradation mechanisms (Part 7)
# --------------------------------------------------------------------------- #
# Each run is assigned a DOMINANT mechanism plus low-level background wear on
# every other parameter. Mechanisms couple physically (see dataset/degradation.py).

MECHANISMS = [
    "INJECTOR_FOULING",
    "LUBRICATION_DEGRADATION",
    "COOLING_DEGRADATION",
    "COMBUSTION_DEGRADATION",
    "MECHANICAL_WEAR",
    "RING_BLOWBY",              # compression / volumetric loss
    "COMBINED",                 # two mechanisms at once (harder case)
]

MECHANISM_WEIGHTS = [0.19, 0.17, 0.16, 0.15, 0.16, 0.09, 0.08]

# Degradation rate archetypes -> multiplier on the nominal wear rate.
RATE_ARCHETYPES = {
    "slow": 0.62,
    "medium": 1.00,
    "fast": 1.62,
    "accelerating": 1.00,       # nonlinear; shape handled separately
}
RATE_WEIGHTS = {"slow": 0.27, "medium": 0.31, "fast": 0.22, "accelerating": 0.20}

LIFE_HOURS_RANGE = (150.0, 620.0)   # target spread of engine lives


# --------------------------------------------------------------------------- #
# Sensor model (Part 6)
# --------------------------------------------------------------------------- #
# White noise 1-sigma in engineering units, applied to the snapshot-mean value.
SENSOR_NOISE = {
    "rpm": 9.0,
    "cht": 1.30,
    "egt": 5.20,
    "oil_pressure": 0.045,
    "oil_temperature": 0.85,
    "fuel_flow": 0.22,
    "battery_voltage": 0.035,
    "alternator_current": 0.30,
    "vibration_rms": 0.016,
    "manifold_pressure": 0.55,
}

# Per-engine fixed calibration bias, 1-sigma (engineering units).
SENSOR_BIAS_SIGMA = {
    "rpm": 7.0,
    "cht": 1.60,
    "egt": 6.50,
    "oil_pressure": 0.055,
    "oil_temperature": 1.10,
    "fuel_flow": 0.20,
    "battery_voltage": 0.030,
    "alternator_current": 0.25,
    "vibration_rms": 0.012,
    "manifold_pressure": 0.60,
}

# Slow correlated (1/f-like) drift: amplitude as a fraction of the white noise
# sigma, with an OU correlation time in operating hours.
SENSOR_LF_NOISE_FRAC = 1.15
SENSOR_LF_TAU_H = 9.0

OUTLIER_PROB = 0.0035           # probability a channel spikes in a snapshot
OUTLIER_SIGMA_MULT = (5.0, 13.0)


# --------------------------------------------------------------------------- #
# UKF configuration (Part 10)
# --------------------------------------------------------------------------- #

UKF_STATE = ["eta_inj", "eta_comb", "h_cool", "friction_mult", "lub_health", "eta_vol"]

UKF_MEAS = ["rpm", "cht", "egt", "oil_pressure", "oil_temperature",
            "fuel_flow", "vibration_rms", "manifold_pressure"]

# Scaled unscented transform parameters.
# Scaled unscented transform. alpha=1, kappa=0 gives lambda = 0, so the mean
# sigma point carries zero weight and every remaining weight is +1/(2n). Small
# alpha (the usual 1e-3 rule of thumb) is wrong here: with n=6 it makes
# W0_mean = -1e4 and the propagated covariance is no longer positive definite
# once the measurement model is nonlinear. A tight sigma-point spread is
# obtained by keeping P small instead, which also keeps the points inside the
# physical bounds of the health parameters.
UKF_ALPHA = 1.0
UKF_BETA = 2.0
UKF_KAPPA = 0.0

# Random-walk process noise on the health parameters, per snapshot.
#
# DERIVED, not hand-tuned. Q is set so that the filter's random walk can just
# traverse a parameter's full nominal->EOL range over a typical engine life:
#
#     q_k = |theta_eol_k - theta_nominal_k| / UKF_Q_LIFE_SNAPSHOTS
#
# Sizing it this way matters more than it looks. Too small and the filter cannot
# follow real wear. Too large and it wanders freely inside the range; combined
# with the monotonicity ratchet that wander becomes a one-way accumulation of
# noise and every engine is declared sicker than it is. UKF_Q_LIFE_SNAPSHOTS is
# 1500 snapshots = 375 operating hours, the median life in the fleet.
UKF_Q_LIFE_SNAPSHOTS = 1500

UKF_Q_DIAG = {k: abs(HEALTH_EOL[k] - HEALTH_NOMINAL[k]) / UKF_Q_LIFE_SNAPSHOTS
              for k in HEALTH_PARAMS}

UKF_P0_DIAG = {
    "eta_inj": 8e-3,
    "eta_comb": 8e-3,
    "h_cool": 1.2e-2,
    "friction_mult": 1.0e-2,
    "lub_health": 1.2e-2,
    "eta_vol": 6e-3,
}

# Health parameter box constraints applied after each UKF update.
# Upper bounds sit just above the commissioning value: an engine cannot become
# healthier than new, and letting a parameter drift to 1.05 lets the filter buy
# a better fit on one channel by inventing impossible health on another.
UKF_BOUNDS = {
    "eta_inj": (0.84, 1.010),
    "eta_comb": (0.86, 1.010),
    "h_cool": (0.68, 1.015),
    "friction_mult": (0.985, 1.80),
    "lub_health": (0.52, 1.015),
    "eta_vol": (0.74, 1.010),
}

# Near-monotonicity constraint. Wear is a one-way physical process, so the
# estimate is not allowed to improve faster than UKF_RECOVERY_FRAC of the
# random-walk step per snapshot. This is a physics prior, not a label: it
# encodes "engines do not heal", which is true of the real world and not
# something read off the ground truth. A burn-in lets the filter settle before
# the ratchet engages, so an unlucky first few samples cannot lock in a bias.
UKF_MONOTONE = True
UKF_RECOVERY_FRAC = 0.35
UKF_MONOTONE_BURNIN = 40


# --------------------------------------------------------------------------- #
# Windowing (Part 16)
# --------------------------------------------------------------------------- #

SEQ_LENGTHS = [30, 60, 120]     # snapshots -> 7.5 h, 15 h, 30 h of history
SEQ_LEN_DEFAULT = 60
WINDOW_STRIDE = 6               # 1.5 h; 90% overlap avoided (see docs)
STRIDE_BY_SPLIT = {"train": 6, "val": 12, "test": 12}


# --------------------------------------------------------------------------- #
# Splits (Part 15)
# --------------------------------------------------------------------------- #

SPLIT_FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}

# Nominal (training) operating envelope.
ENVELOPE_TRAIN = {
    "altitude_ft": (0.0, 15000.0),
    "ambient_c": (-5.0, 35.0),
}
# Stress envelope - deliberately outside training support but physically valid.
ENVELOPE_STRESS = {
    "altitude_ft": (15000.0, 20000.0),
    "ambient_c": (35.0, 45.0),
}


# --------------------------------------------------------------------------- #
# RUL propagation (Part 13)
# --------------------------------------------------------------------------- #

RUL_N_PARTICLES = 600
RUL_HORIZON_H = 900.0
RUL_MAX_H = 800.0               # RUL labels are capped (piecewise-linear target)
RUL_CAP_H = 400.0               # standard prognostics cap on the training label
RUL_QUANTILES = [0.10, 0.50, 0.90]
