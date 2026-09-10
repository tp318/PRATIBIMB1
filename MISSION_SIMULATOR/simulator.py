"""
============================================================================
 MALE UAV Aero Piston Engine - Software-in-the-Loop (SIL) Mission Simulator
============================================================================

Purpose
-------
Generates labeled, high-frequency (10 Hz) time-series telemetry data for a
turbocharged MALE-UAV aero piston engine (Rotax 914/915 iS class), for use
as training data for CNN-LSTM based fault/anomaly detection models in a
Digital Twin framework.

The engine is modeled as a simplified Mean Value Engine Model (MVEM):
first-order lag (ODE) dynamics driven by a commanded throttle/mission
profile, plus physically-motivated coupling between RPM, EGT, CHT, Oil
Pressure/Temp, Fuel Flow, Vibration and Alternator Voltage.

Eight procedural fault modes can be injected at a user-specified onset
time with a user-specified severity, producing perfectly labeled fault
data (Fault_Label column) suitable for supervised residual-based learning.

Author: Digital Twin Hackathon Team
============================================================================
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field


# ============================================================================
# SECTION 1: ENGINE CONSTANTS / STEADY-STATE MAPS
# ============================================================================

DT = 0.1          # Simulation & logging time step -> 10 Hz
IDLE_RPM = 2000.0
MAX_RPM = 5800.0   # Rotax 914/915-class redline
AMBIENT_TEMP_C = 15.0  # baseline ambient (ground level ISA), can vary with altitude

# Fault catalog - used for both the CLI menu and the Fault_Label column
FAULT_CATALOG = {
    0: "Healthy",
    1: "Misfire",
    2: "Cooling_Degradation",
    3: "Lubrication_Issue",
    4: "Sensor_Drift_EGT",
    5: "Combustion_Instability",
    6: "Injector_Abnormality",
    7: "Overheating_Trend",
    8: "Battery_Alternator_Fault",
}


# ============================================================================
# SECTION 2: MISSION PROFILE GENERATOR
# ============================================================================

def generate_mission_profile(duration_min: float, dt: float, variation_seed: int = 0):
    """
    Builds a commanded throttle (0-1) profile over the mission, mimicking a
    realistic MALE UAV flight: taxi/idle -> takeoff climb -> cruise/loiter
    (long endurance ISR leg) -> throttle transients -> descent -> idle.

    A `variation_seed` perturbs waypoint timings/throttle levels slightly so
    that multiple runs of the "same" profile are not identical (needed for
    dataset diversity).
    """
    rng = np.random.default_rng(variation_seed)
    n_samples = int(duration_min * 60 / dt)
    t = np.arange(n_samples) * dt
    total_s = duration_min * 60

    # Fractional waypoints of total mission duration (jittered per-run)
    jitter = lambda base, spread: base + rng.uniform(-spread, spread)
    p_taxi_end   = jitter(0.03, 0.01)
    p_climb_end  = jitter(0.15, 0.03)
    p_cruise_end = jitter(0.75, 0.05)
    p_descend_end= jitter(0.93, 0.02)

    throttle_taxi   = jitter(0.15, 0.03)
    throttle_climb  = jitter(0.90, 0.05)
    throttle_cruise = jitter(0.55, 0.08)
    throttle_desc   = jitter(0.25, 0.04)
    throttle_idle   = jitter(0.12, 0.02)

    wp_t = np.array([0, p_taxi_end, p_climb_end, p_cruise_end, p_descend_end, 1.0]) * total_s
    wp_thr = np.array([throttle_idle, throttle_taxi, throttle_climb,
                        throttle_cruise, throttle_desc, throttle_idle])

    throttle_cmd = np.interp(t, wp_t, wp_thr)

    # Superimpose small ISR-loiter throttle "wobble" during cruise for realism
    cruise_mask = (t > wp_t[2]) & (t < wp_t[3])
    wobble = 0.03 * np.sin(2 * np.pi * t / rng.uniform(40, 90)) * cruise_mask
    throttle_cmd = np.clip(throttle_cmd + wobble, 0.05, 1.0)

    # Random small altitude/ambient-temp profile (affects EGT/CHT baseline)
    altitude_ft = np.interp(t, wp_t, [0, 500, 15000, 18000, 5000, 0]) + rng.normal(0, 30, n_samples)
    altitude_ft = np.clip(altitude_ft, 0, None)
    ambient_c = AMBIENT_TEMP_C - altitude_ft / 1000 * 2.0  # ~2C/1000ft lapse rate

    return t, throttle_cmd, ambient_c


# ============================================================================
# SECTION 3: FAULT INJECTION CONFIGURATION
# ============================================================================

@dataclass
class FaultConfig:
    fault_id: int = 0
    onset_s: float = 0.0
    severity: float = 0.5   # normalized 0-1 (user supplied, clipped)
    label_name: str = "Healthy"


def get_fault_active_mask(t: np.ndarray, cfg: FaultConfig):
    """Boolean mask: True where t >= onset (fault has begun and persists)."""
    if cfg.fault_id == 0:
        return np.zeros_like(t, dtype=bool)
    return t >= cfg.onset_s


# ============================================================================
# SECTION 4: MEAN VALUE ENGINE MODEL (MVEM) - CORE SIMULATION LOOP
# ============================================================================

class EngineState:
    """Holds the mutable engine state variables integrated each time step."""
    def __init__(self):
        self.rpm = IDLE_RPM
        self.cht = 70.0        # deg C
        self.egt = 300.0       # deg C
        self.oil_press = 4.5   # bar
        self.oil_temp = 60.0   # deg C
        self.fuel_flow = 8.0   # L/hr
        self.vib_baseline = 0.02   # g RMS baseline
        self.volt = 13.8       # V
        # internal fault accumulators
        self.cooling_fault_gain = 0.0
        self.lube_fault_gain = 0.0
        self.sensor_drift_bias = 0.0
        self.friction_extra = 0.0


def step_engine(state: EngineState, throttle_cmd: float, ambient_c: float,
                 dt: float, fault_active: bool, cfg: FaultConfig, rng: np.random.Generator):
    """
    Advances the engine state by one time step using first-order lag (ODE)
    dynamics towards throttle-commanded steady-state targets, then applies
    fault-specific modifiers. Returns the (possibly fault-corrupted) sensor
    readings actually "measured" for this step, plus the true underlying
    state (useful later for residual-based ground truth).
    """

    # ---- 1. STEADY-STATE TARGETS FROM THROTTLE (engine performance maps) ----
    rpm_target   = IDLE_RPM + throttle_cmd * (MAX_RPM - IDLE_RPM)
    egt_target   = 300 + throttle_cmd * 550       # ~300-850 C typical range
    cht_target   = 70 + throttle_cmd * 90         # ~70-160 C typical range
    oilT_target  = 60 + throttle_cmd * 50         # ~60-110 C
    oilP_target  = 5.5 - throttle_cmd * 0.8       # oil pressure drops slightly at high RPM/temp
    fuel_target  = 6 + throttle_cmd * 34          # ~6-40 L/hr

    # ---- 2. FIRST-ORDER LAG TIME CONSTANTS (thermal mass vs. fast mech.) ----
    tau_rpm, tau_egt, tau_cht = 1.5, 3.0, 25.0
    tau_oilT, tau_oilP, tau_fuel = 20.0, 2.0, 0.8

    state.rpm       += dt * (rpm_target - state.rpm) / tau_rpm
    state.egt        += dt * (egt_target + ambient_c*0.2 - state.egt) / tau_egt
    state.cht         += dt * (cht_target + ambient_c*0.3 - state.cht) / tau_cht
    state.oil_temp    += dt * (oilT_target - state.oil_temp) / tau_oilT
    state.oil_press   += dt * (oilP_target - state.oil_press) / tau_oilP
    state.fuel_flow    += dt * (fuel_target - state.fuel_flow) / tau_fuel

    # ---- 3. ALTERNATOR VOLTAGE - couples to RPM (belt-driven alternator) ----
    volt_target = 13.2 + 1.2 * min(state.rpm / MAX_RPM, 1.0)
    state.volt += dt * (volt_target - state.volt) / 1.0

    # ---- 4. BASELINE SENSOR NOISE (always present - healthy noise floor) ----
    rpm_noise   = rng.normal(0, 15)
    egt_noise   = rng.normal(0, 4)
    cht_noise   = rng.normal(0, 1.5)
    oilP_noise  = rng.normal(0, 0.05)
    oilT_noise  = rng.normal(0, 0.8)
    fuel_noise  = rng.normal(0, 0.3)
    volt_noise  = rng.normal(0, 0.05)
    vib_noise   = abs(rng.normal(state.vib_baseline, 0.01))

    meas_rpm, meas_egt, meas_cht = state.rpm + rpm_noise, state.egt + egt_noise, state.cht + cht_noise
    meas_oilP, meas_oilT = state.oil_press + oilP_noise, state.oil_temp + oilT_noise
    meas_fuel, meas_volt = state.fuel_flow + fuel_noise, state.volt + volt_noise
    meas_vib = vib_noise

    # ============================================================
    # 5. FAULT-SPECIFIC MODIFIERS (only applied when fault_active)
    # ============================================================
    if fault_active:
        sev = np.clip(cfg.severity, 0.01, 1.0)

        if cfg.fault_id == 1:  # ---- MISFIRE ----
            # Intermittent sharp RPM/EGT drops + vibration spike, ~1-3 Hz random pulses
            if rng.random() < 0.08 * sev * 5:   # probability per 0.1s step
                meas_rpm  -= sev * rng.uniform(150, 500)
                meas_egt  -= sev * rng.uniform(50, 200)
                meas_vib  += sev * rng.uniform(0.15, 0.5)

        elif cfg.fault_id == 2:  # ---- COOLING DEGRADATION ----
            # Gradual exponential rise in CHT over time since onset
            state.cooling_fault_gain += dt * sev * 0.02
            meas_cht += state.cooling_fault_gain * 40 * (1 - np.exp(-state.cooling_fault_gain))
            meas_cht += state.cooling_fault_gain * 60

        elif cfg.fault_id == 3:  # ---- LUBRICATION ISSUE ----
            # Exponential decay in oil pressure + rise in oil temp + friction->RPM droop
            state.lube_fault_gain += dt * sev * 0.03
            decay = 1 - np.exp(-state.lube_fault_gain)
            meas_oilP -= decay * sev * 2.5
            meas_oilT += decay * sev * 35
            state.friction_extra = decay * sev * 250
            meas_rpm -= state.friction_extra

        elif cfg.fault_id == 4:  # ---- SENSOR DRIFT (EGT) ----
            # Random-walk bias added ONLY to the reported EGT sensor value;
            # true engine state is unaffected (classic sensor fault).
            state.sensor_drift_bias += rng.normal(sev * 0.8, sev * 0.3)
            meas_egt += state.sensor_drift_bias

        elif cfg.fault_id == 5:  # ---- COMBUSTION INSTABILITY ----
            # High-frequency oscillation/variance on RPM and vibration
            osc = sev * 300 * np.sin(2 * np.pi * 6.0 * (rng.random()))
            meas_rpm += osc + rng.normal(0, sev * 120)
            meas_vib += abs(rng.normal(0, sev * 0.25))

        elif cfg.fault_id == 6:  # ---- INJECTOR ABNORMALITY ----
            # Erratic fuel flow (rich/lean swings) with associated EGT swings
            fuel_err = rng.normal(0, sev * 6)
            meas_fuel += fuel_err
            meas_egt -= fuel_err * 8  # rich -> cooler EGT, lean -> hotter (approx.)

        elif cfg.fault_id == 7:  # ---- OVERHEATING TREND ----
            # Faster, more severe CHT+EGT rise than "cooling degradation" -
            # represents e.g. blocked cooling duct / cowl flap stuck closed
            ramp = sev * (1 - np.exp(-0.01 * sev * cfg_time_since(cfg, dt)))
            meas_cht += ramp * 70
            meas_egt += ramp * 90

        elif cfg.fault_id == 8:  # ---- BATTERY / ALTERNATOR FAULT ----
            # Voltage sags progressively (alternator underperforming / battery fault)
            meas_volt -= sev * min(1.0, 0.02 * cfg_time_since(cfg, dt)) * 4.0
            meas_volt += rng.normal(0, sev * 0.15)  # added ripple/noise

    measured = dict(
        RPM=meas_rpm, CHT_C=meas_cht, EGT_C=meas_egt,
        Oil_Press_bar=meas_oilP, Oil_Temp_C=meas_oilT,
        Fuel_Flow_Lph=meas_fuel, Vibration_g=meas_vib,
        Alternator_V=meas_volt,
    )
    return measured


# Helper: tracks elapsed time since fault onset (used by ramp-style faults)
_fault_start_cache = {}
def cfg_time_since(cfg: FaultConfig, dt: float):
    key = id(cfg)
    _fault_start_cache[key] = _fault_start_cache.get(key, 0.0) + dt
    return _fault_start_cache[key]


# ============================================================================
# SECTION 5: FULL MISSION SIMULATION (single run -> DataFrame)
# ============================================================================

def simulate_mission(duration_min: float, cfg: FaultConfig, variation_seed: int,
                      run_id: str, dt: float = DT):
    """Runs one complete mission and returns a labeled pandas DataFrame."""
    t, throttle_cmd, ambient_c = generate_mission_profile(duration_min, dt, variation_seed)
    rng = np.random.default_rng(variation_seed * 7919 + cfg.fault_id * 31 + 1)
    state = EngineState()
    fault_mask = get_fault_active_mask(t, cfg)

    rows = []
    for i in range(len(t)):
        meas = step_engine(state, throttle_cmd[i], ambient_c[i], dt,
                            fault_mask[i], cfg, rng)
        row = {
            "Run_ID": run_id,
            "Time_s": round(t[i], 2),
            "Throttle_Cmd": round(throttle_cmd[i], 4),
            "Ambient_C": round(ambient_c[i], 2),
            **{k: round(v, 4) for k, v in meas.items()},
            "Fault_ID": cfg.fault_id if fault_mask[i] else 0,
            "Fault_Label": cfg.label_name if fault_mask[i] else "Healthy",
        }
        rows.append(row)

    # reset the fault-onset timer cache for this cfg object between runs
    _fault_start_cache.pop(id(cfg), None)
    return pd.DataFrame(rows)


# ============================================================================
# SECTION 6: CLI - INTERACTIVE MISSION SETUP
# ============================================================================

def print_fault_menu():
    print("\nAvailable Fault Modes:")
    for fid, name in FAULT_CATALOG.items():
        tag = "(no fault / baseline)" if fid == 0 else ""
        print(f"  {fid}: {name} {tag}")


def prompt_float(msg, default):
    raw = input(f"{msg} [default {default}]: ").strip()
    return float(raw) if raw else default


def interactive_single_run():
    print("=" * 70)
    print(" MALE UAV Aero Piston Engine - SIL Mission Simulator")
    print("=" * 70)

    duration_min = prompt_float("Total mission duration (minutes)", 10)

    print_fault_menu()
    fault_id = int(prompt_float("Select fault ID to inject (0 = healthy)", 0))
    fault_id = fault_id if fault_id in FAULT_CATALOG else 0

    if fault_id == 0:
        cfg = FaultConfig(fault_id=0, onset_s=0, severity=0, label_name="Healthy")
    else:
        onset_min = prompt_float("Fault onset time (minutes into mission)", duration_min * 0.4)
        severity = prompt_float("Fault severity (0.0 - 1.0)", 0.5)
        cfg = FaultConfig(fault_id=fault_id, onset_s=onset_min * 60,
                           severity=np.clip(severity, 0.0, 1.0),
                           label_name=FAULT_CATALOG[fault_id])

    seed = int(prompt_float("Random variation seed", 0))
    df = simulate_mission(duration_min, cfg, seed, run_id=f"manual_{cfg.label_name}_{seed}")

    out_name = "simulated_mission_data.csv"
    df.to_csv(out_name, index=False)
    print(f"\nSaved {len(df)} rows -> {out_name}")
    print(df.head())
    return df


# ============================================================================
# SECTION 7: BATCH DATASET GENERATION (for CNN-LSTM training corpus)
# ============================================================================

def generate_training_dataset(duration_min=10, runs_per_class=8, out_csv="uav_engine_dataset.csv"):
    """
    Generates the full labeled dataset described in the hackathon spec:
      - `runs_per_class` healthy variations
      - `runs_per_class` variations for EACH of the 8 fault classes
        (with randomized onset time and severity per run)
    At 10 Hz x 10 min, each run = 6000 rows; e.g. 8 classes+1 healthy x 8 runs
    x 6000 rows ~ 432,000 rows.
    """
    all_runs = []
    rng = np.random.default_rng(42)
    run_counter = 0

    # --- Healthy runs ---
    for v in range(runs_per_class):
        cfg = FaultConfig(fault_id=0, onset_s=0, severity=0, label_name="Healthy")
        df = simulate_mission(duration_min, cfg, variation_seed=run_counter,
                               run_id=f"healthy_{v}")
        all_runs.append(df)
        run_counter += 1

    # --- Fault runs (fault IDs 1-8), randomized onset + severity per run ---
    for fault_id, name in FAULT_CATALOG.items():
        if fault_id == 0:
            continue
        for v in range(runs_per_class):
            onset_s = rng.uniform(0.2, 0.7) * duration_min * 60   # onset between 20-70% of mission
            severity = rng.uniform(0.3, 1.0)                       # randomized severity
            cfg = FaultConfig(fault_id=fault_id, onset_s=onset_s,
                               severity=severity, label_name=name)
            df = simulate_mission(duration_min, cfg, variation_seed=run_counter,
                                   run_id=f"{name}_{v}")
            all_runs.append(df)
            run_counter += 1
            print(f"  Simulated run {run_counter}: {name} (onset={onset_s:.0f}s, "
                  f"severity={severity:.2f}) -> {len(df)} rows")

    full_df = pd.concat(all_runs, ignore_index=True)
    full_df.to_csv(out_csv, index=False)
    print(f"\nDataset generation complete: {len(full_df)} total rows across "
          f"{run_counter} mission runs -> saved to '{out_csv}'")
    print("\nClass distribution (row-level):")
    print(full_df["Fault_Label"].value_counts())
    return full_df


# ============================================================================
# SECTION 8: MAIN ENTRY POINT
# ============================================================================

def main():
    print("\nSelect mode:")
    print("  1: Interactive single-mission run (manual fault selection, CLI-guided)")
    print("  2: Batch-generate full labeled training dataset (Healthy + 8 fault "
          "classes x N variations each)")
    choice = input("Enter choice [1/2] (default 2): ").strip() or "2"

    if choice == "1":
        interactive_single_run()
    else:
        duration_min = prompt_float("Mission length per run (minutes)", 10)
        runs_per_class = int(prompt_float("Number of run variations per class (Healthy & each fault)", 8))
        out_csv = input("Output CSV filename [default uav_engine_dataset.csv]: ").strip() \
            or "uav_engine_dataset.csv"
        generate_training_dataset(duration_min=duration_min,
                                   runs_per_class=runs_per_class,
                                   out_csv=out_csv)


if __name__ == "__main__":
    main()