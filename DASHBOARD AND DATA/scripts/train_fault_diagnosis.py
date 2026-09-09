"""
AeroTwin-4 Fault Diagnosis Trainer & Evaluator CLI.

Trains the multi-class fault-family classifier on Digital-Twin residual features
and evaluates it on an UNSEEN SEVERITY (SEV080) held-out partition.

Usage:
  .venv/Scripts/python.exe scripts/train_fault_diagnosis.py
  .venv/Scripts/python.exe scripts/train_fault_diagnosis.py --config hybrid
"""

import argparse
import glob
import json
import os
import sys

import pandas as pd

_script_dir = os.path.dirname(os.path.abspath(__file__))
_root_dir = os.path.dirname(_script_dir)
_aerotwin_dir = os.path.join(_root_dir, "AeroTwin")

for _p in [_aerotwin_dir, _root_dir]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from AeroTwin.ml.anomaly.features import FeatureExtractor
from AeroTwin.ml.anomaly.preprocessing import FeatureScaler
from AeroTwin.ml.diagnosis.classifier import FaultDiagnosisClassifier
from AeroTwin.ml.diagnosis.evaluation import DiagnosisEvaluator
from AeroTwin.ml.diagnosis.splits import DiagnosisSplitter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="residual", choices=["raw", "residual", "hybrid"])
    args = ap.parse_args()

    print("=" * 68)
    print(f"AeroTwin-4 Fault Diagnosis Trainer  [features: {args.config.upper()}]")
    print("=" * 68)

    src_dir = os.path.join(_root_dir, "data", "generated", "phase4", "full")
    files = sorted(glob.glob(os.path.join(src_dir, "**", "*_derived_residuals.csv"), recursive=True))
    if not files:
        raise SystemExit(f"No Phase 4 residual files under {src_dir}. Run generate_phase4_dataset.py --full first.")
    print(f"Found {len(files)} derived residual run files.")

    extractor = FeatureExtractor(config_type=args.config.upper())
    X_parts, meta_parts = [], []
    for path in files:
        df_run = pd.read_csv(path)
        # Phase 4 residual CSVs carry no run_id column; it lives in the filename.
        df_run["run_id"] = os.path.basename(path).replace("_derived_residuals.csv", "")
        Xi, mi = extractor.extract_dataset(df_run, window_size_sec=5.0, stride_sec=1.0)
        X_parts.append(Xi)
        meta_parts.append(mi)
    X = pd.concat(X_parts, ignore_index=True)
    meta = pd.concat(meta_parts, ignore_index=True)
    print(f"Feature matrix: {X.shape[0]} windows x {X.shape[1]} features")

    splitter = DiagnosisSplitter()
    parts = splitter.split(X, meta)

    for name in ("train", "val", "test"):
        Xp, mp, yp = parts[name]
        counts = yp.value_counts().to_dict()
        print(f"  [{name:5s}] {len(Xp):5d} windows  runs={mp['run_id'].nunique():3d}  {counts}")

    X_tr, m_tr, y_tr = parts["train"]
    X_va, m_va, y_va = parts["val"]
    X_te, m_te, y_te = parts["test"]

    # Scale on the training partition only.
    scaler = FeatureScaler()
    X_tr_s = pd.DataFrame(scaler.fit_transform(X_tr), columns=scaler.feature_names)
    X_va_s = pd.DataFrame(scaler.transform(X_va), columns=scaler.feature_names)
    X_te_s = pd.DataFrame(scaler.transform(X_te), columns=scaler.feature_names)

    print("\nTraining Random Forest fault diagnoser...")
    clf = FaultDiagnosisClassifier().fit(X_tr_s, y_tr)
    print(f"  classes: {clf.classes_}")

    ev = DiagnosisEvaluator(classes=clf.classes_)

    for label, Xs, y in (("VALIDATION (SEV060)", X_va_s, y_va), ("HELD-OUT TEST (UNSEEN SEV080)", X_te_s, y_te)):
        pred = clf.predict(Xs)
        res = ev.evaluate(y, pred)
        print(f"\n{'-' * 68}\n{label}\n{'-' * 68}")
        s = res["summary"]
        print(
            f"  accuracy={s['accuracy']:.4f}  balanced_accuracy={s['balanced_accuracy']:.4f}  "
            f"macro_F1={s['macro_f1']:.4f}  n={s['n_samples']}"
        )
        print("\n  Per-class:")
        print(res["per_class"].to_string(index=False))
        print("\n  Confusion matrix:")
        print(res["confusion_matrix"].to_string())

    # Persist artifacts + a machine-readable report.
    out_models = os.path.join(_root_dir, "models", "diagnosis", args.config)
    clf.save(out_models)
    scaler.save(os.path.join(out_models, "scaler.json"))

    pred_te = clf.predict(X_te_s)
    res_te = ev.evaluate(y_te, pred_te)
    diag = clf.diagnose(X_te_s)

    out_data = os.path.join(_root_dir, "data", "generated", "diagnosis")
    os.makedirs(out_data, exist_ok=True)

    report = {
        "feature_config": args.config,
        "n_features": int(X.shape[1]),
        "classes": clf.classes_,
        "train_runs": sorted(m_tr["run_id"].unique().tolist()),
        "val_runs": sorted(m_va["run_id"].unique().tolist()),
        "test_runs": sorted(m_te["run_id"].unique().tolist()),
        "test_summary": res_te["summary"],
        "test_per_class": res_te["per_class"].to_dict(orient="records"),
        "test_confusion_matrix": res_te["confusion_matrix"].to_dict(),
        "top_features": clf.feature_importance(20).to_dict(orient="records"),
    }
    with open(os.path.join(out_data, f"diagnosis_report_{args.config}.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    pd.concat([m_te.reset_index(drop=True), diag, y_te.rename("true_fault")], axis=1).to_csv(
        os.path.join(out_data, f"diagnosis_predictions_{args.config}.csv"), index=False
    )

    print(f"\n{'-' * 68}")
    print("Top 10 features driving the diagnosis:")
    print(clf.feature_importance(10).to_string(index=False))
    print(f"\nSaved model    -> {out_models}")
    print(f"Saved report   -> {out_data}")


if __name__ == "__main__":
    main()
