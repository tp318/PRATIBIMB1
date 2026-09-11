"""Stage 5: train the GRU degradation-rate correction and all RUL baselines.

    python scripts/train_models.py [--seq-len 60] [--epochs 30]

Saves every model and scaler into artifacts/ (Deliverable I).
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from rulcore.config import (ARTIFACT_DIR, FEATURE_DIR, HEALTH_PARAMS,
                            REPORT_DIR, RUL_CAP_H, SEQ_LEN_DEFAULT,
                            STRIDE_BY_SPLIT)
from rulcore.models.baselines import train_xgboost_rul, window_summary
from rulcore.models.gru_correction import (DegradationRateGRU, DirectRULGRU,
                                           count_parameters, model_size_bytes,
                                           rate_loss)
from rulcore.models.windows import (FEATURE_SETS, StandardScaler3D,
                                    available_features, build_sequences)

torch.manual_seed(0)
np.random.seed(0)
DEVICE = "cpu"


def load(split):
    return pd.read_parquet(os.path.join(FEATURE_DIR, f"{split}_est.parquet"))


def train_torch(model, tr, va, epochs, lr, batch, loss_fn, name, patience=6):
    """Generic training loop with early stopping on validation loss."""
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    n = len(tr[0])
    best, best_state, bad = float("inf"), None, 0
    hist = []

    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        tot, nb = 0.0, 0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            loss = loss_fn(model, [t[idx] for t in tr])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            tot += float(loss.item())
            nb += 1
        sched.step()

        model.eval()
        with torch.no_grad():
            vl, vb = 0.0, 0
            for i in range(0, len(va[0]), 4096):
                sl = slice(i, i + 4096)
                vl += float(loss_fn(model, [t[sl] for t in va]).item())
                vb += 1
        vloss = vl / max(vb, 1)
        hist.append({"epoch": ep, "train": tot / max(nb, 1), "val": vloss})
        print(f"    [{name}] epoch {ep:2d}  train {tot/max(nb,1):.5f}  val {vloss:.5f}",
              flush=True)

        if vloss < best - 1e-5:
            best, bad = vloss, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                print(f"    [{name}] early stop at epoch {ep}", flush=True)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"best_val": best, "history": hist}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq-len", type=int, default=SEQ_LEN_DEFAULT)
    ap.add_argument("--epochs", type=int, default=30)
    args = ap.parse_args()
    L = args.seq_len

    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    summary = {"seq_len": L}

    print("Loading estimation tables ...", flush=True)
    tr_df, va_df = load("train"), load("val")
    print(f"  train {tr_df.run_id.nunique()} runs / {len(tr_df):,} rows")
    print(f"  val   {va_df.run_id.nunique()} runs / {len(va_df):,} rows")

    rate_targets = [f"rate_true_{k}" for k in HEALTH_PARAMS]
    rate_phys_cols = [f"rate_phys_{k}" for k in HEALTH_PARAMS]

    # ================================================================= #
    # 1. GRU degradation-rate correction (the proposed learned component)
    # ================================================================= #
    print("\n[1/4] GRU degradation-rate correction", flush=True)
    feats = available_features(tr_df, FEATURE_SETS["rate_correction"])
    tcols = rate_targets + rate_phys_cols

    Xtr, Ytr, _, _ = build_sequences(tr_df, feats, tcols, L, STRIDE_BY_SPLIT["train"])
    Xva, Yva, _, _ = build_sequences(va_df, feats, tcols, L, STRIDE_BY_SPLIT["val"])
    print(f"  features {len(feats)}   train windows {len(Xtr):,}   val {len(Xva):,}")

    scaler = StandardScaler3D().fit(Xtr)
    Xtr_s, Xva_s = scaler.transform(Xtr), scaler.transform(Xva)

    K = len(HEALTH_PARAMS)
    tr_t = [torch.from_numpy(Xtr_s), torch.from_numpy(Ytr[:, :K]),
            torch.from_numpy(Ytr[:, K:])]
    va_t = [torch.from_numpy(Xva_s), torch.from_numpy(Yva[:, :K]),
            torch.from_numpy(Yva[:, K:])]

    rate_model = DegradationRateGRU(n_features=len(feats)).to(DEVICE)

    def rate_loss_fn(model, batch):
        x, y_true, r_phys = batch
        out = model(x, r_phys)
        return rate_loss(out["rate"], y_true)

    t0 = time.time()
    rate_model, rate_hist = train_torch(rate_model, tr_t, va_t, args.epochs,
                                        2e-3, 256, rate_loss_fn, "rate-gru")
    torch.save({"state_dict": rate_model.state_dict(), "features": feats,
                "seq_len": L, "scaler": scaler.to_dict()},
               os.path.join(ARTIFACT_DIR, "gru_rate_correction.pt"))
    summary["rate_gru"] = {
        "features": len(feats), "params": count_parameters(rate_model),
        "size_bytes": model_size_bytes(rate_model),
        "best_val_loss": rate_hist["best_val"], "train_seconds": time.time() - t0,
        "train_windows": int(len(Xtr)),
    }
    print(f"  params {count_parameters(rate_model):,}  "
          f"size {model_size_bytes(rate_model)/1024:.1f} KiB")

    # Calibrate the spread of the learned multiplier on validation, for the MC
    # propagation's model-uncertainty term.
    rate_model.eval()
    with torch.no_grad():
        out = rate_model(va_t[0][:20000], va_t[2][:20000])
        lm = out["mult"].log().numpy()
    summary["rate_gru"]["log_mult_std"] = float(np.std(lm))
    summary["rate_gru"]["log_mult_mean"] = float(np.mean(lm))
    summary["rate_gru"]["saturation_frac"] = float(np.mean(np.abs(lm) > 1.5))
    print(f"  learned log-multiplier: mean {np.mean(lm):+.3f}  std {np.std(lm):.3f}  "
          f"saturated {100*np.mean(np.abs(lm)>1.5):.1f}%")

    del Xtr, Xva, Xtr_s, Xva_s, tr_t, va_t

    # ================================================================= #
    # RUL baselines
    # ================================================================= #
    ycol = "RUL_hours_capped"

    def make(split_df, fset, stride):
        f = available_features(split_df, FEATURE_SETS[fset])
        X, y, runs, ends = build_sequences(split_df, f, [ycol, "life_fraction"],
                                           L, stride)
        return f, X, y, runs, ends

    # ---- B1 XGBoost on telemetry window summaries --------------------- #
    print("\n[2/4] Baseline 1: XGBoost on telemetry window summaries", flush=True)
    f1, Xtr1, ytr1, _, _ = make(tr_df, "telemetry", STRIDE_BY_SPLIT["train"])
    _, Xva1, yva1, _, _ = make(va_df, "telemetry", STRIDE_BY_SPLIT["val"])
    Str, names1 = window_summary(Xtr1, f1)
    Sva, _ = window_summary(Xva1, f1)
    print(f"  tabular features {Str.shape[1]}   train rows {len(Str):,}")
    t0 = time.time()
    xgb_model = train_xgboost_rul(Str, ytr1[:, 0], Sva, yva1[:, 0], names1)
    xgb_model.save_model(os.path.join(ARTIFACT_DIR, "xgb_rul_baseline.json"))
    with open(os.path.join(ARTIFACT_DIR, "xgb_rul_features.json"), "w",
              encoding="utf-8") as f:
        json.dump({"base_features": f1, "summary_features": names1, "seq_len": L}, f)
    summary["xgb_baseline"] = {"features": len(names1),
                               "best_iteration": int(xgb_model.best_iteration),
                               "train_seconds": time.time() - t0}
    print(f"  best iteration {xgb_model.best_iteration}")
    del Xtr1, Xva1

    # ---- B2 GRU on telemetry ------------------------------------------ #
    print("\n[3/4] Baseline 2: GRU on telemetry sequences", flush=True)
    f2, Xtr2, ytr2, _, _ = make(tr_df, "telemetry", STRIDE_BY_SPLIT["train"])
    _, Xva2, yva2, _, _ = make(va_df, "telemetry", STRIDE_BY_SPLIT["val"])
    sc2 = StandardScaler3D().fit(Xtr2)
    tr_t = [torch.from_numpy(sc2.transform(Xtr2)), torch.from_numpy(ytr2[:, 0])]
    va_t = [torch.from_numpy(sc2.transform(Xva2)), torch.from_numpy(yva2[:, 0])]
    m2 = DirectRULGRU(n_features=len(f2)).to(DEVICE)

    def rul_loss_fn(model, batch):
        x, y = batch
        return nn.functional.huber_loss(model(x), y, delta=25.0)

    t0 = time.time()
    m2, h2 = train_torch(m2, tr_t, va_t, args.epochs, 2e-3, 256, rul_loss_fn, "gru-telem")
    torch.save({"state_dict": m2.state_dict(), "features": f2, "seq_len": L,
                "scaler": sc2.to_dict()},
               os.path.join(ARTIFACT_DIR, "gru_rul_telemetry.pt"))
    summary["gru_telemetry"] = {"features": len(f2), "params": count_parameters(m2),
                                "size_bytes": model_size_bytes(m2),
                                "best_val_loss": h2["best_val"],
                                "train_seconds": time.time() - t0}
    print(f"  params {count_parameters(m2):,}")
    del Xtr2, Xva2, tr_t, va_t

    # ---- B3 GRU on estimated health ----------------------------------- #
    print("\n[4/4] Baseline 3: GRU on UKF health estimates", flush=True)
    f3, Xtr3, ytr3, _, _ = make(tr_df, "health_only", STRIDE_BY_SPLIT["train"])
    _, Xva3, yva3, _, _ = make(va_df, "health_only", STRIDE_BY_SPLIT["val"])
    sc3 = StandardScaler3D().fit(Xtr3)
    tr_t = [torch.from_numpy(sc3.transform(Xtr3)), torch.from_numpy(ytr3[:, 0])]
    va_t = [torch.from_numpy(sc3.transform(Xva3)), torch.from_numpy(yva3[:, 0])]
    m3 = DirectRULGRU(n_features=len(f3)).to(DEVICE)
    t0 = time.time()
    m3, h3 = train_torch(m3, tr_t, va_t, args.epochs, 2e-3, 256, rul_loss_fn, "gru-health")
    torch.save({"state_dict": m3.state_dict(), "features": f3, "seq_len": L,
                "scaler": sc3.to_dict()},
               os.path.join(ARTIFACT_DIR, "gru_rul_health.pt"))
    summary["gru_health"] = {"features": len(f3), "params": count_parameters(m3),
                             "size_bytes": model_size_bytes(m3),
                             "best_val_loss": h3["best_val"],
                             "train_seconds": time.time() - t0}
    print(f"  params {count_parameters(m3):,}")

    with open(os.path.join(REPORT_DIR, "training_summary.json"), "w",
              encoding="utf-8") as f:
        json.dump(summary, f, indent=1)
    print(f"\nSaved models to {ARTIFACT_DIR}")


if __name__ == "__main__":
    main()
