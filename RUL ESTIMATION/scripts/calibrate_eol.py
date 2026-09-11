"""Derive HEALTH_EOL from the EOL criteria and rewrite it into config.py.

Run this whenever EOL_CRITERIA, EOL_REF_CONDITION or the engine model changes,
so that the degradation bookkeeping stays consistent with the EOL definition.

    python scripts/calibrate_eol.py [--write]
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from rulcore.config import HEALTH_NOMINAL, HEALTH_PARAMS, PKG_ROOT
from rulcore.dataset.eol import (CRITERION_NAMES, calibrate_health_eol,
                                 commissioning_baseline, criterion_margins,
                                 reference_performance)
from rulcore.physics.engine_model import nominal_params

p = nominal_params()
base = commissioning_baseline({k: float(v) for k, v in HEALTH_NOMINAL.items()}, p)

print("Commissioning baseline (fleet-nominal engine):")
for k, v in base.items():
    print(f"   {k:18s} {v:10.3f}")

eol = calibrate_health_eol(p)

print("\nDerived per-parameter EOL and the criterion that binds:")
binding = {}
for k in HEALTH_PARAMS:
    th = {kk: np.array(float(x)) for kk, x in HEALTH_NOMINAL.items()}
    th[k] = np.array(eol[k])
    m = criterion_margins(reference_performance(th, p), base)
    vals = {c: float(np.asarray(m[c]).ravel()[0]) for c in CRITERION_NAMES}
    top = sorted(vals.items(), key=lambda x: -x[1])[:3]
    binding[k] = top[0][0]
    print(f"   {k:16s} {eol[k]:7.4f}   " + ", ".join(f"{a}={b:.2f}" for a, b in top))

if "--write" in sys.argv:
    cfg = os.path.join(PKG_ROOT, "config.py")
    src = open(cfg, encoding="utf-8").read()
    body = "\n".join(f'    "{k}": {eol[k]:.4f},' for k in HEALTH_PARAMS)
    new = "HEALTH_EOL = {\n" + body + "\n}"
    src2 = re.sub(r"HEALTH_EOL = \{[^}]*\}", new, src, count=1)
    if src2 == src:
        print("\n!! could not locate HEALTH_EOL block; config not modified")
        sys.exit(1)
    open(cfg, "w", encoding="utf-8").write(src2)
    print(f"\nWrote HEALTH_EOL into {cfg}")
else:
    print("\n(dry run - pass --write to update config.py)")
