"""Sanity-check the MVEM against real Rotax 914-class numbers.

Run:  python scripts/check_physics.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from rulcore.physics.engine_model import steady_state, nominal_params, nominal_health

p = nominal_params()


def show(label, thr, alt, oat, theta=None, load=1.0, aspd=1.0):
    th = theta or nominal_health()
    s = steady_state(thr, alt, oat, th, p, airspeed_factor=aspd, load_factor=load)
    print(f"{label:<28s} rpm={float(s['rpm']):6.0f}  "
          f"pwr={float(s['power_brake_w'])/1000:5.1f}kW  "
          f"trq={float(s['torque_brake']):5.1f}Nm  "
          f"ff={float(s['fuel_flow_lph']):5.2f}L/h  "
          f"AFR={float(s['afr']):5.2f}  "
          f"CHT={float(s['cht']):5.1f}C  EGT={float(s['egt']):6.1f}C  "
          f"oilP={float(s['oil_pressure']):4.2f}bar  oilT={float(s['oil_temperature']):5.1f}C  "
          f"vib={float(s['vibration_rms']):4.2f}g  "
          f"BSFC={float(s['bsfc_kg_per_kwh']):5.3f}")


print("=" * 150)
print("HEALTHY ENGINE ACROSS THE OPERATING ENVELOPE")
print("=" * 150)
show("ground idle SL/15C", 0.15, 0, 15.0, aspd=0.35)
show("taxi SL/15C", 0.30, 0, 15.0, aspd=0.4)
show("takeoff SL/15C", 1.00, 0, 15.0, aspd=1.1)
show("climb 5kft/5C", 0.90, 5000, 5.0, aspd=1.0)
show("cruise 8kft/5C", 0.78, 8000, 5.0)
show("cruise 12kft/-5C", 0.75, 12000, -5.0)
show("loiter 15kft/-15C", 0.60, 15000, -15.0, aspd=0.85)
show("descent 8kft/5C", 0.25, 8000, 5.0, aspd=0.9)
show("hot day SL/45C", 0.85, 0, 45.0, aspd=1.0)
show("hot+high 18kft/40C", 0.80, 18000, 40.0, aspd=0.9)

print()
print("=" * 150)
print("DEGRADATION SIGNATURES AT THE REFERENCE CONDITION (8000 ft / 5C / 78% throttle)")
print("=" * 150)
show("baseline (healthy)", 0.78, 8000, 5.0)
for name, val in [("eta_inj", 0.86), ("eta_comb", 0.89), ("h_cool", 0.75),
                  ("friction_mult", 1.45), ("lub_health", 0.70), ("eta_vol", 0.90)]:
    th = nominal_health()
    th = {k: np.array(v, dtype=float) for k, v in th.items()}
    th[name] = np.array(val)
    show(f"{name}={val}", 0.78, 8000, 5.0, theta=th)

print()
print("=" * 150)
print("SIGNATURE MATRIX: d(measurement) for a 10% loss of each health parameter")
print("=" * 150)
base = steady_state(0.78, 8000, 5.0, nominal_health(), p)
chans = ["rpm", "cht", "egt", "oil_pressure", "oil_temperature", "fuel_flow_lph",
         "vibration_rms", "manifold_pressure_kpa"]
print(f"{'param':<16s}" + "".join(f"{c:>16s}" for c in chans))
for name in ["eta_inj", "eta_comb", "h_cool", "friction_mult", "lub_health", "eta_vol"]:
    th = {k: np.array(v, dtype=float) for k, v in nominal_health().items()}
    th[name] = np.array(1.10 if name == "friction_mult" else 0.90)
    s = steady_state(0.78, 8000, 5.0, th, p)
    row = "".join(f"{float(s[c]) - float(base[c]):>16.3f}" for c in chans)
    print(f"{name:<16s}{row}")
