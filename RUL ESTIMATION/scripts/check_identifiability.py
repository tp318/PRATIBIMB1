"""Stage 3: structural identifiability of the health parameters (Part 11).

    python scripts/check_identifiability.py

Writes outputs/reports/identifiability.json and prints the analysis.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from rulcore.config import FEATURE_DIR, HEALTH_PARAMS, REPORT_DIR, UKF_MEAS
from rulcore.estimation.identifiability import (DEFAULT_CONDITIONS, analyse,
                                                confounding_report,
                                                pooled_jacobian,
                                                recommend_state_vector,
                                                sensitivity_jacobian)
from rulcore.estimation.twin import SigmaModel

sig_path = os.path.join(FEATURE_DIR, "sigma_model.json")
sigma_model = SigmaModel.from_dict(json.load(open(sig_path, encoding="utf-8")))

# Map residual-channel sigmas onto measurement names used by the UKF.
sigma = {}
for ch in UKF_MEAS:
    key = f"{ch}_residual"
    sigma[ch] = sigma_model.global_sigma[key]

print("Healthy residual sigma used as the detectability yardstick:")
for k, v in sigma.items():
    print(f"    {k:22s} {v:9.4f}")

names = list(HEALTH_PARAMS)

# ---------------------------------------------------------------- per point --
print("\n" + "=" * 92)
print("PER-CONDITION DETECTABILITY")
print("(sigmas of measurement change produced by consuming 100% of a parameter's life)")
print("=" * 92)
print(f"{'condition':<14s}" + "".join(f"{n:>15s}" for n in names) + f"{'cond#':>10s}")
per_cond = {}
for c in DEFAULT_CONDITIONS:
    J = sensitivity_jacobian(c, sigma)
    r = analyse(J, names)
    per_cond[c["name"]] = r
    row = "".join(f"{r['column_norms'][n]:15.2f}" for n in names)
    print(f"{c['name']:<14s}{row}{r['condition_number']:10.1f}")

# ------------------------------------------------------------------- pooled --
Jp = pooled_jacobian(sigma)
rp = analyse(Jp, names)
row = "".join(f"{rp['column_norms'][n]:15.2f}" for n in names)
print(f"{'POOLED':<14s}{row}{rp['condition_number']:10.1f}")

print("\nPooled singular values (independent observable directions):")
print("    " + "  ".join(f"{s:8.3f}" for s in rp["singular_values"]))
print(f"    effective rank (>1% of largest): {rp['effective_rank_1pct']} of {len(names)}")
print(f"    directions resolvable above 1 healthy sigma: {rp['n_directions_above_1sigma']}")

print("\nLeast observable parameter combination (right singular vector of the")
print("smallest singular value - this is what the sensor set cannot separate):")
for k, v in sorted(rp["least_observable_direction"].items(), key=lambda x: -abs(x[1])):
    print(f"    {k:16s} {v:+7.3f}")

print("\nPairwise cosine similarity of parameter signatures (pooled).")
print("|cos| near 1 means the two parameters are confusable:")
cos = rp["cosine_similarity"]
print(f"{'':16s}" + "".join(f"{n[:12]:>13s}" for n in names))
for i, n in enumerate(names):
    print(f"{n:16s}" + "".join(f"{cos[i, j]:13.3f}" for j in range(len(names))))

# ------------------------------------------------------------- confounding --
print("\n" + "=" * 92)
print("SPECIFIC CONFOUNDING TESTS")
print("=" * 92)
conf = confounding_report(sigma)
labels = {
    "eta_inj_vs_eta_comb_single_point":
        "injector vs combustion efficiency, ONE operating point",
    "eta_inj_vs_eta_comb_pooled":
        "injector vs combustion efficiency, pooled over the envelope",
    "hot_day_vs_cooling_degradation":
        "hot day resembles cooling degradation (mean |cos| over builds)",
    "high_altitude_vs_cooling_degradation":
        "+9000 ft climb resembles cooling degradation (mean |cos|)",
    "oilp_sensor_drift_vs_lubrication_single":
        "oil-pressure sensor drift vs lubrication loss, one point",
    "oilp_sensor_drift_vs_lubrication_pooled":
        "oil-pressure sensor drift vs lubrication loss, pooled",
}
for k, lab in labels.items():
    print(f"  cos = {conf[k]:+.3f}   {lab}")
print(f"\n  For scale, REAL cooling degradation over full life: "
      f"{conf['cooling_degradation_detectability_sigmas']:.2f} sigma")
print(f"  spurious residual from +20 C hot day  (rms / p95): "
      f"{conf['hot_day_residual_norm_sigmas']:.3f} / "
      f"{conf['hot_day_residual_p95_sigmas']:.3f} sigma")
print(f"  spurious residual from +9000 ft climb (rms / p95): "
      f"{conf['high_altitude_residual_norm_sigmas']:.3f} / "
      f"{conf['high_altitude_residual_p95_sigmas']:.3f} sigma")

# ----------------------------------------------------------- recommendation --
print("\n" + "=" * 92)
print("STATE VECTOR RECOMMENDATION")
print("=" * 92)
rec, hist = recommend_state_vector(sigma)
for step in hist["history"]:
    weakest = min(step["column_norms"], key=step["column_norms"].get)
    print(f"  state={len(step['state'])}  cond#={step['condition_number']:7.1f}  "
          f"weakest={weakest} ({step['column_norms'][weakest]:.2f} sigma)")
print(f"\n  RECOMMENDED UKF STATE: {rec}")

out = {
    "sigma": sigma,
    "per_condition": {k: {"condition_number": v["condition_number"],
                          "column_norms": v["column_norms"],
                          "singular_values": v["singular_values"]}
                      for k, v in per_cond.items()},
    "pooled": {"condition_number": rp["condition_number"],
               "column_norms": rp["column_norms"],
               "singular_values": rp["singular_values"],
               "least_observable_direction": rp["least_observable_direction"],
               "cosine_similarity": rp["cosine_similarity"].tolist(),
               "parameter_order": names},
    "confounding": conf,
    "recommended_state": rec,
}
os.makedirs(REPORT_DIR, exist_ok=True)
with open(os.path.join(REPORT_DIR, "identifiability.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, indent=1)
print(f"\nWrote {os.path.join(REPORT_DIR, 'identifiability.json')}")
