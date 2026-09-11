"""Stage 2: splits, Digital Twin residuals, sigma calibration, feature dataset.

    python scripts/build_dataset.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from rulcore.config import DATA_DIR
from rulcore.dataset.build_features import (assign_splits, build_dataset,
                                            save_dataset)

index = pd.read_csv(os.path.join(DATA_DIR, "run_index.csv"))
print(f"Loaded run index: {len(index)} runs")

index = assign_splits(index)
print("\nSplit assignment (runs):")
print(index.groupby(["split", "population"]).size().to_string())

print("\nMechanism coverage per split:")
print(pd.crosstab(index.mechanism, index.split).to_string())

frames, sigma = build_dataset(index)

print("\nWriting feature tables:")
manifest = save_dataset(frames, sigma, index)

print("\nFitted healthy residual sigma (global, robust MAD-based):")
for k, v in sigma.global_sigma.items():
    print(f"    {k:34s} {v:9.4f}")

print("\nFeature availability classes:")
print(manifest.availability.value_counts().to_string())
unc = manifest[manifest.availability == "UNCLASSIFIED"]
if len(unc):
    print("\nUNCLASSIFIED columns (must be resolved before training):")
    print("   " + ", ".join(unc.feature.tolist()))
