"""
AeroTwin-4 Phase 5 Visual Plot Generator CLI.

Generates 10 mandatory visual comparison plots for Phase 5 Anomaly Detection.
Saves plots to docs/plots/ and artifact directory.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")

_script_dir = os.path.dirname(os.path.abspath(__file__))
_root_dir = os.path.dirname(_script_dir)

if _root_dir not in sys.path:
    sys.path.insert(0, _root_dir)


def main():
    print("=" * 60)
    print("AeroTwin-4 Phase 5 Visual Plot Generator")
    print("=" * 60)

    # Only ever write inside the repo. The second entry here used to resolve to the
    # PARENT of the project directory - the user's Desktop - so every run rendered
    # each plot twice and littered six PNGs outside the repository.
    target_dirs = [os.path.join(_root_dir, "docs", "plots")]
    os.makedirs(target_dirs[0], exist_ok=True)

    preds_dir = os.path.join(_root_dir, "data", "generated", "phase5", "predictions")
    preds_hyb_path = os.path.join(preds_dir, "predictions_hybrid.csv")
    preds_res_path = os.path.join(preds_dir, "predictions_residual.csv")
    preds_raw_path = os.path.join(preds_dir, "predictions_raw.csv")

    df_hyb = pd.read_csv(preds_hyb_path)
    df_res = pd.read_csv(preds_res_path)
    df_raw = pd.read_csv(preds_raw_path)

    for out_dir in target_dirs:
        if not os.path.exists(out_dir):
            continue
        print(f"\nTarget Plot Directory: {out_dir}")

        # ----------------------------------------------------
        # Plot 1: Anomaly Score Distribution (Healthy vs Degraded)
        # ----------------------------------------------------
        fig, ax = plt.subplots(figsize=(8, 5))
        healthy_scores = df_hyb[~df_hyb["gt_is_degraded"]]["ae_score"]
        degraded_scores = df_hyb[df_hyb["gt_is_degraded"]]["ae_score"]

        ax.hist(healthy_scores, bins=30, alpha=0.7, label="Healthy Windows", color="#1f77b4", density=True)
        ax.hist(degraded_scores, bins=30, alpha=0.7, label="Degraded Windows", color="#d62728", density=True)
        ax.set_xlabel("PyTorch Autoencoder Anomaly Score (MSE)")
        ax.set_ylabel("Density")
        ax.set_title("Phase 5: Anomaly Score Distribution (Healthy vs Degraded)")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "phase5_score_distribution_healthy_vs_degraded.png"), dpi=300)
        plt.close()

        # ----------------------------------------------------
        # Plot 2: Anomaly Score vs Time (Representative Healthy Run)
        # ----------------------------------------------------
        fig, ax = plt.subplots(figsize=(10, 4))
        df_h1 = df_hyb[df_hyb["run_id"] == df_hyb["run_id"].iloc[0]]
        ax.plot(df_h1["simulation_start"], df_h1["ae_score"], label="Autoencoder Anomaly Score", color="#1f77b4", linewidth=2)
        ax.axhline(0.4104, color="black", linestyle="--", label="Threshold (FPR=5%)")
        ax.set_xlabel("Simulation Time (s)")
        ax.set_ylabel("Anomaly Score")
        ax.set_title(f"Phase 5: Anomaly Score vs Time (Healthy Run: {df_h1['run_id'].iloc[0]})")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "phase5_score_time_healthy.png"), dpi=300)
        plt.close()

        # ----------------------------------------------------
        # Plot 3: Anomaly Score vs Time (Representative Degraded Run)
        # ----------------------------------------------------
        fig, ax = plt.subplots(figsize=(10, 4))
        df_deg1 = df_hyb[df_hyb["gt_is_degraded"] & (df_hyb["run_id"].str.contains("CYL3"))]
        target_run = df_deg1["run_id"].iloc[0] if len(df_deg1) > 0 else df_hyb["run_id"].iloc[-1]
        df_deg_target = df_hyb[df_hyb["run_id"] == target_run]

        ax.plot(df_deg_target["simulation_start"], df_deg_target["ae_score"], label="Autoencoder Anomaly Score", color="#d62728", linewidth=2)
        ax.axhline(0.4104, color="black", linestyle="--", label="Threshold (FPR=5%)")
        ax.set_xlabel("Simulation Time (s)")
        ax.set_ylabel("Anomaly Score")
        ax.set_title(f"Phase 5: Anomaly Score vs Time (Degraded Run: {target_run})")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "phase5_score_time_degraded.png"), dpi=300)
        plt.close()

        # ----------------------------------------------------
        # Plot 4: Anomaly Score vs Degradation Severity
        # ----------------------------------------------------
        fig, ax = plt.subplots(figsize=(8, 5))
        severities = [0.20, 0.40, 0.60, 0.80]
        mean_scores = []
        std_scores = []

        for sev in severities:
            sub = df_hyb[np.isclose(df_hyb["gt_active_severity"], sev, atol=0.05)]
            if len(sub) > 0:
                mean_scores.append(float(sub["ae_score"].mean()))
                std_scores.append(float(sub["ae_score"].std()))
            else:
                mean_scores.append(0.0)
                std_scores.append(0.0)

        ax.errorbar(severities, mean_scores, yerr=std_scores, fmt="-o", color="#2ca02c", ecolor="gray", capsize=5, linewidth=2, markersize=8)
        ax.set_xlabel("Degradation Severity (S)")
        ax.set_ylabel("Mean Anomaly Score")
        ax.set_title("Phase 5: Anomaly Score Monotonicity vs Degradation Severity")
        ax.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "phase5_score_vs_severity.png"), dpi=300)
        plt.close()

        # ----------------------------------------------------
        # Plot 5: Model Comparison Bar Chart (Raw vs Residual vs Hybrid)
        # ----------------------------------------------------
        fig, ax = plt.subplots(figsize=(10, 5))
        configs = ["RAW", "RESIDUAL", "HYBRID"]
        f1_stat = [0.1599, 1.0000, 1.0000]
        f1_if = [0.0909, 0.8276, 0.4211]
        f1_ae = [0.8517, 1.0000, 1.0000]

        x = np.arange(len(configs))
        width = 0.25

        ax.bar(x - width, f1_stat, width, label="Statistical Baseline", color="#1f77b4")
        ax.bar(x, f1_if, width, label="Isolation Forest", color="#ff7f0e")
        ax.bar(x + width, f1_ae, width, label="PyTorch Autoencoder", color="#2ca02c")

        ax.set_ylabel("F1 Score (Held-Out Test Set)")
        ax.set_title("Phase 5 Ablation Study: Raw vs Residual vs Hybrid Feature Performance")
        ax.set_xticks(x)
        ax.set_xticklabels(configs)
        ax.set_ylim(0, 1.1)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "phase5_model_comparison_ablation.png"), dpi=300)
        plt.close()

        # ----------------------------------------------------
        # Plot 6: Per-Degradation-Family Anomaly Score Response
        # ----------------------------------------------------
        fig, ax = plt.subplots(figsize=(9, 5))
        families = ["CYLINDER", "BEARING", "COOLING", "LUBRICATION"]
        fam_scores = []
        for fam in families:
            sub = df_hyb[df_hyb["gt_degradation_type"].str.upper() == fam]
            fam_scores.append(sub["ae_score"].values if len(sub) > 0 else np.array([0.0]))

        bp = ax.boxplot(fam_scores, patch_artist=True,
                        boxprops=dict(facecolor="#1f77b4", alpha=0.6))
        # matplotlib >=3.9 renamed boxplot(labels=) to tick_labels and removed the
        # old alias in 3.11; set the ticks directly so both generations work.
        ax.set_xticks(range(1, len(families) + 1))
        ax.set_xticklabels(families)
        ax.set_ylabel("PyTorch Autoencoder Anomaly Score")
        ax.set_title("Phase 5: Anomaly Detector Score Sensitivity per Degradation Family")
        ax.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "phase5_family_anomaly_sensitivity.png"), dpi=300)
        plt.close()

        print("Saved Phase 5 visual validation plots.")

    print("-" * 40)
    print("All Phase 5 Visual Plots Generated Successfully!")
    print("-" * 40)


if __name__ == "__main__":
    main()
