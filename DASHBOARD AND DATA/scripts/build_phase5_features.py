"""
AeroTwin-4 Phase 5 Feature Matrix Builder CLI.

Extracts feature matrices (Config A Raw, Config B Residual, Config C Hybrid) from Phase 4 derived datasets
and splits them into Train (Healthy), Validation, and Test partitions.
"""

import os
import sys
import json
import glob
import pandas as pd

_script_dir = os.path.dirname(os.path.abspath(__file__))
_root_dir = os.path.dirname(_script_dir)

if _root_dir not in sys.path:
    sys.path.insert(0, _root_dir)

from AeroTwin.ml.anomaly.features import FeatureExtractor
from AeroTwin.ml.anomaly.splits import RunSplitter


def main():
    print("=" * 60)
    print("AeroTwin-4 Phase 5 Feature Matrix Builder")
    print("=" * 60)

    src_dir = os.path.join(_root_dir, "data", "generated", "phase4", "full")
    out_dir = os.path.join(_root_dir, "data", "generated", "phase5")
    os.makedirs(out_dir, exist_ok=True)

    csv_files = glob.glob(os.path.join(src_dir, "**", "*_derived_residuals.csv"), recursive=True)
    csv_files.sort()
    print(f"Found {len(csv_files)} derived residual run files.")

    dfs = []
    for filepath in csv_files:
        df_run = pd.read_csv(filepath)
        filename = os.path.basename(filepath)
        run_id = filename.replace("_derived_residuals.csv", "")
        df_run["run_id"] = run_id
        dfs.append(df_run)

    for cfg_type in ["RAW", "RESIDUAL", "HYBRID"]:
        print(f"\nBuilding Feature Configuration: {cfg_type}...")
        ext = FeatureExtractor(config_type=cfg_type)

        X_all_list = []
        meta_all_list = []

        for df_run in dfs:
            X_run, meta_run = ext.extract_dataset(df_run, window_size_sec=5.0, stride_sec=1.0)
            X_all_list.append(X_run)
            meta_all_list.append(meta_run)

        X_all = pd.concat(X_all_list, ignore_index=True)
        meta_all = pd.concat(meta_all_list, ignore_index=True)

        splitter = RunSplitter()
        (X_tr, m_tr), (X_va, m_va), (X_te, m_te) = splitter.split_dataset(X_all, meta_all)

        cfg_dir = os.path.join(out_dir, "features", cfg_type.lower())
        os.makedirs(os.path.join(cfg_dir, "train"), exist_ok=True)
        os.makedirs(os.path.join(cfg_dir, "validation"), exist_ok=True)
        os.makedirs(os.path.join(cfg_dir, "test"), exist_ok=True)

        X_tr.to_csv(os.path.join(cfg_dir, "train", "X_train.csv"), index=False)
        m_tr.to_csv(os.path.join(cfg_dir, "train", "meta_train.csv"), index=False)

        X_va.to_csv(os.path.join(cfg_dir, "validation", "X_val.csv"), index=False)
        m_va.to_csv(os.path.join(cfg_dir, "validation", "meta_val.csv"), index=False)

        X_te.to_csv(os.path.join(cfg_dir, "test", "X_test.csv"), index=False)
        m_te.to_csv(os.path.join(cfg_dir, "test", "meta_test.csv"), index=False)

        print(f"  [{cfg_type}] Train shape: {X_tr.shape}, Val shape: {X_va.shape}, Test shape: {X_te.shape}")

    # Write metadata summary
    meta_path = os.path.join(out_dir, "metadata", "feature_summary.json")
    os.makedirs(os.path.dirname(meta_path), exist_ok=True)
    summary = {
        "run_count": len(dfs),
        "train_runs": splitter.train_runs,
        "val_runs": splitter.val_runs,
        "sample_rate_hz": 100.0,
        "window_size_sec": 5.0,
        "stride_sec": 1.0,
    }
    with open(meta_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "-" * 40)
    print("Phase 5 Feature Generation Complete!")
    print("-" * 40)


if __name__ == "__main__":
    main()
