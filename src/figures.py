import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Dict, Any, List



plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.titlesize": 14,
    "axes.edgecolor": "#333333",
    "axes.linewidth": 0.8,
    "grid.color": "#e0e0e0",
    "grid.linestyle": "--",
    "grid.linewidth": 0.5
})

PALETTE = {
    "M0": "#4A90E2",      
    "M1": "#50E3C2",      
    "M2": "#D0021B",      
    "Oracle": "#414141",  
    "Benign": "#2E7D32",  
    "Suspicious": "#C62828", 
    "Neutral": "#757575"
}


def save_multi_format(fig: plt.Figure, base_path: str):
    os.makedirs(os.path.dirname(base_path), exist_ok=True)
    fig.savefig(f"{base_path}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{base_path}.pdf", format="pdf", bbox_inches="tight")
    fig.savefig(f"{base_path}.svg", format="svg", bbox_inches="tight")
    plt.close(fig)


def generate_figure1_workflow(output_dir: str):
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.axis("off")
    
    boxes = [
        {"x": 0.03, "y": 0.55, "w": 0.16, "h": 0.35, "title": "1. IAM Policy\n& Auth Semantics", 
         "desc": "• Statement counts\n• Wildcards (*)\n• Effective actions\n• Sensitive APIs\n• Versions (S09)", "bg": "#EBF5FB", "edge": "#2980B9"},
        {"x": 0.22, "y": 0.55, "w": 0.16, "h": 0.35, "title": "2. Environment\nTopology", 
         "desc": "• Env tier (Prod/Dev)\n• Privileged roles\n• Boundary presence\n• Trust openness\n• Privesc paths", "bg": "#E8F8F5", "edge": "#16A085"},
        {"x": 0.41, "y": 0.55, "w": 0.18, "h": 0.35, "title": "3. CloudTrail\nBehavioral Context", 
         "desc": "• API event counts\n• Error codes (403)\n• Role chain depth\n• Novel User-Agent\n• Failed IAM activity", "bg": "#FDEDEC", "edge": "#C0392B"},
        {"x": 0.63, "y": 0.55, "w": 0.16, "h": 0.35, "title": "4. Glassbox Model\n(EBM Regressor)", 
         "desc": "• Stratified Group-CV\n• Lineage isolation\n• No gt_* leakage\n• Exact additivity\n• Single fixed config", "bg": "#F4ECF7", "edge": "#8E44AD"},
        {"x": 0.82, "y": 0.55, "w": 0.15, "h": 0.35, "title": "5. Analyst Triage\n& Interpretability", 
         "desc": "• Priority Queue\n• NDCG & Lift\n• Same-policy test\n• Additive terms\n• Audit reports", "bg": "#FEF9E7", "edge": "#F39C12"}
    ]
    
    for b in boxes:
        rect = patches.FancyBboxPatch(
            (b["x"], b["y"]), b["w"], b["h"],
            boxstyle="round,pad=0.02,rounding_size=0.02",
            facecolor=b["bg"], edgecolor=b["edge"], linewidth=1.5
        )
        ax.add_patch(rect)
        ax.text(b["x"] + b["w"] / 2, b["y"] + b["h"] - 0.06, b["title"],
                ha="center", va="top", fontsize=10.5, fontweight="bold", color="#2C3E50")
        ax.text(b["x"] + 0.015, b["y"] + b["h"] - 0.14, b["desc"],
                ha="left", va="top", fontsize=8.5, color="#34495E", linespacing=1.4)
        
    
    arrow_starts = [0.19, 0.38, 0.59, 0.79]
    for x in arrow_starts:
        ax.annotate("", xy=(x + 0.03, 0.725), xytext=(x, 0.725),
                    arrowprops=dict(arrowstyle="->", lw=2, color="#2C3E50"))
        
    
    audit_rect = patches.FancyBboxPatch(
        (0.03, 0.10), 0.94, 0.32,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        facecolor="#F8F9F9", edgecolor="#BDC3C7", linewidth=1.2
    )
    ax.add_patch(audit_rect)
    ax.text(0.50, 0.36, "Rigorous Experimental Controls & Zero-Leakage Architecture",
            ha="center", va="top", fontsize=11, fontweight="bold", color="#2C3E50")
    
    controls_text = (
        "• Grouped Splitting: All policy lineages and paired members isolated strictly within folds (StratifiedGroupKFold, n=5)\n"
        "• Target Independence: Oracle priority score (gt_analyst_priority_score) never recomputed or leaked as input feature\n"
        "• Exact Ablation: Identical samples, folds, missingness handling, and EBM parameters across M0, M1, and M2\n"
        "• Pair Equivalence: Type-A pairs verified for identical structural policy hash, environment, and effective authorization"
    )
    ax.text(0.06, 0.30, controls_text, ha="left", va="top", fontsize=9, color="#566573", linespacing=1.5)
    
    save_multi_format(fig, os.path.join(output_dir, "figure1_workflow"))


def generate_figure2_ablation(output_dir: str):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis("off")
    
    models = [
        {"x": 0.06, "w": 0.26, "name": "M0: Policy-Centric", "tier": "Baseline",
         "color": PALETTE["M0"], "features": "15 Active Features",
         "items": ["• Policy Statement Count", "• Wildcard Actions / Resources", "• Effective Allowed Actions", "• Effective Denied Actions", "• Version Count / Rollback"]},
        {"x": 0.37, "w": 0.26, "name": "M1: Policy + Environment", "tier": "Context Tier 1",
         "color": PALETTE["M1"], "features": "28 Active Features (+13)",
         "items": ["• All M0 Policy Features", "• Environment Type (Prod/Dev)", "• Permissions Boundary Flag", "• Trust Openness Score", "• Reachable Privesc Paths"]},
        {"x": 0.68, "w": 0.26, "name": "M2: Full Context", "tier": "Context Tier 2 (Proposed)",
         "color": PALETTE["M2"], "features": "40 Active Features (+12)",
         "items": ["• All M1 Policy+Env Features", "• CloudTrail Event Counts", "• Error Counts (403 AccessDenied)", "• Role Chain Length", "• Novel User-Agent / Source IP"]}
    ]
    
    for m in models:
        rect = patches.FancyBboxPatch(
            (m["x"], 0.15), m["w"], 0.70,
            boxstyle="round,pad=0.02,rounding_size=0.03",
            facecolor="#FFFFFF", edgecolor=m["color"], linewidth=2
        )
        ax.add_patch(rect)
        
        
        header = patches.FancyBboxPatch(
            (m["x"], 0.70), m["w"], 0.15,
            boxstyle="round,pad=0.01,rounding_size=0.02",
            facecolor=m["color"], edgecolor=m["color"]
        )
        ax.add_patch(header)
        ax.text(m["x"] + m["w"]/2, 0.79, m["name"], ha="center", va="center", color="#FFFFFF", fontweight="bold", fontsize=11)
        ax.text(m["x"] + m["w"]/2, 0.73, m["features"], ha="center", va="center", color="#FFFFFF", fontsize=8.5)
        
        
        y_text = 0.64
        for item in m["items"]:
            ax.text(m["x"] + 0.02, y_text, item, ha="left", va="center", fontsize=8.5, color="#333333")
            y_text -= 0.09
            
    
    ax.annotate("+ Env", xy=(0.36, 0.50), xytext=(0.33, 0.50),
                arrowprops=dict(arrowstyle="->", lw=2, color="#555555"), fontsize=9, fontweight="bold")
    ax.annotate("+ Behavior", xy=(0.67, 0.50), xytext=(0.64, 0.50),
                arrowprops=dict(arrowstyle="->", lw=2, color="#555555"), fontsize=9, fontweight="bold")
    
    save_multi_format(fig, os.path.join(output_dir, "figure2_ablation"))


def generate_figure3_main_results(exp1_metrics: Dict[str, Any], output_dir: str):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    
    models = ["M0", "M1", "M2"]
    colors = [PALETTE["M0"], PALETTE["M1"], PALETTE["M2"]]
    
    
    ax1 = axes[0]
    x = np.arange(2)
    width = 0.25
    
    mae_vals = [exp1_metrics.get("M0_mae", 0), exp1_metrics.get("M1_mae", 0), exp1_metrics.get("M2_mae", 0)]
    rmse_vals = [exp1_metrics.get("M0_rmse", 0), exp1_metrics.get("M1_rmse", 0), exp1_metrics.get("M2_rmse", 0)]
    
    mae_errs = [
        [mae_vals[i] - exp1_metrics.get(f"{m}_mae_ci_lower", mae_vals[i]) for i, m in enumerate(models)],
        [exp1_metrics.get(f"{m}_mae_ci_upper", mae_vals[i]) - mae_vals[i] for i, m in enumerate(models)]
    ]
    rmse_errs = [
        [rmse_vals[i] - exp1_metrics.get(f"{m}_rmse_ci_lower", rmse_vals[i]) for i, m in enumerate(models)],
        [exp1_metrics.get(f"{m}_rmse_ci_upper", rmse_vals[i]) - rmse_vals[i] for i, m in enumerate(models)]
    ]
    
    for i, m in enumerate(models):
        ax1.bar(x[0] + (i - 1) * width, mae_vals[i], width, yerr=[[mae_errs[0][i]], [mae_errs[1][i]]],
                capsize=4, color=colors[i], label=m, edgecolor="#333333", alpha=0.9)
        ax1.bar(x[1] + (i - 1) * width, rmse_vals[i], width, yerr=[[rmse_errs[0][i]], [rmse_errs[1][i]]],
                capsize=4, color=colors[i], edgecolor="#333333", alpha=0.9)
        
    ax1.set_xticks(x)
    ax1.set_xticklabels(["MAE", "RMSE"])
    ax1.set_ylabel("Error (Points, lower is better)")
    ax1.set_title("A. Predictive Error Metrics", fontweight="bold")
    ax1.grid(axis="y")
    ax1.legend(loc="upper right", framealpha=0.8)
    
    
    ax2 = axes[1]
    sp_vals = [exp1_metrics.get("M0_spearman", 0), exp1_metrics.get("M1_spearman", 0), exp1_metrics.get("M2_spearman", 0)]
    kd_vals = [exp1_metrics.get("M0_kendall", 0), exp1_metrics.get("M1_kendall", 0), exp1_metrics.get("M2_kendall", 0)]
    
    sp_errs = [
        [sp_vals[i] - exp1_metrics.get(f"{m}_spearman_ci_lower", sp_vals[i]) for i, m in enumerate(models)],
        [exp1_metrics.get(f"{m}_spearman_ci_upper", sp_vals[i]) - sp_vals[i] for i, m in enumerate(models)]
    ]
    kd_errs = [
        [kd_vals[i] - exp1_metrics.get(f"{m}_kendall_ci_lower", kd_vals[i]) for i, m in enumerate(models)],
        [exp1_metrics.get(f"{m}_kendall_ci_upper", kd_vals[i]) - kd_vals[i] for i, m in enumerate(models)]
    ]
    
    for i, m in enumerate(models):
        ax2.bar(x[0] + (i - 1) * width, sp_vals[i], width, yerr=[[sp_errs[0][i]], [sp_errs[1][i]]],
                capsize=4, color=colors[i], label=m, edgecolor="#333333", alpha=0.9)
        ax2.bar(x[1] + (i - 1) * width, kd_vals[i], width, yerr=[[kd_errs[0][i]], [kd_errs[1][i]]],
                capsize=4, color=colors[i], edgecolor="#333333", alpha=0.9)
        
    ax2.set_xticks(x)
    ax2.set_xticklabels(["Spearman $\\rho$", "Kendall $\\tau$"])
    ax2.set_ylabel("Correlation (higher is better)")
    ax2.set_title("B. Ranking Concordance", fontweight="bold")
    ax2.grid(axis="y")
    
    
    ax3 = axes[2]
    x_k = np.arange(3)
    k_labels = ["NDCG@10", "NDCG@25", "NDCG@50"]
    for i, m in enumerate(models):
        k_vals = [exp1_metrics.get(f"{m}_ndcg@10", 0), exp1_metrics.get(f"{m}_ndcg@25", 0), exp1_metrics.get(f"{m}_ndcg@50", 0)]
        k_err_lo = [k_vals[j] - exp1_metrics.get(f"{m}_ndcg@{k}_ci_lower", k_vals[j]) for j, k in enumerate([10, 25, 50])]
        k_err_hi = [exp1_metrics.get(f"{m}_ndcg@{k}_ci_upper", k_vals[j]) - k_vals[j] for j, k in enumerate([10, 25, 50])]
        ax3.bar(x_k + (i - 1) * width, k_vals, width, yerr=[k_err_lo, k_err_hi],
                capsize=4, color=colors[i], label=m, edgecolor="#333333", alpha=0.9)
        
    ax3.set_xticks(x_k)
    ax3.set_xticklabels(k_labels)
    ax3.set_ylabel("Score (higher is better)")
    ax3.set_title("C. Top-K Analyst Triage Utility", fontweight="bold")
    ax3.grid(axis="y")
    
    plt.tight_layout()
    save_multi_format(fig, os.path.join(output_dir, "figure3_main_results"))


def generate_figure4_context_pairs(pair_gaps_df: pd.DataFrame, output_dir: str):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    
    m0_gaps = pair_gaps_df["m0_gap"].values
    m2_gaps = pair_gaps_df["m2_gap"].values
    gt_gaps = pair_gaps_df["gt_priority_gap"].values
    
    
    bins = np.linspace(min(m0_gaps.min(), m2_gaps.min(), -5), max(m0_gaps.max(), m2_gaps.max(), 30), 35)
    ax1.hist(m0_gaps, bins=bins, alpha=0.6, color=PALETTE["M0"], label=f"M0 Policy-Centric (Mean={np.mean(m0_gaps):.2f})", edgecolor="black")
    ax1.hist(m2_gaps, bins=bins, alpha=0.6, color=PALETTE["M2"], label=f"M2 Full Context (Mean={np.mean(m2_gaps):.2f})", edgecolor="black")
    ax1.axvline(0, color="black", linestyle="--", linewidth=1.2, label="Zero Gap (Context Invariance)")
    ax1.axvline(np.mean(gt_gaps), color=PALETTE["Oracle"], linestyle=":", linewidth=1.8, label=f"Oracle GT Gap Mean ({np.mean(gt_gaps):.1f})")
    
    ax1.set_xlabel("Context Gap: Predicted Priority (Suspicious - Benign)")
    ax1.set_ylabel("Pair Count (out of 207)")
    ax1.set_title("A. Distribution of Context Priority Gaps", fontweight="bold")
    ax1.legend(loc="upper right", framealpha=0.85)
    ax1.grid(True)
    
    
    
    sorted_df = pair_gaps_df.sort_values(by="gt_priority_gap").reset_index(drop=True)
    indices = np.linspace(0, len(sorted_df) - 1, 25, dtype=int)
    sample_pairs = sorted_df.iloc[indices]
    
    y_pos = np.arange(len(sample_pairs))
    ax2.scatter(sample_pairs["m0_gap"], y_pos, color=PALETTE["M0"], s=45, label="M0 Gap (Policy-only)", zorder=3)
    ax2.scatter(sample_pairs["m2_gap"], y_pos, color=PALETTE["M2"], s=45, label="M2 Gap (Context-aware)", zorder=3)
    ax2.scatter(sample_pairs["gt_priority_gap"], y_pos, color=PALETTE["Oracle"], marker="|", s=70, linewidth=2, label="Oracle GT Gap", zorder=4)
    
    for i, idx in enumerate(sample_pairs.index):
        row = sample_pairs.loc[idx]
        ax2.plot([row["m0_gap"], row["m2_gap"]], [y_pos[i], y_pos[i]], color="#999999", linestyle="-", linewidth=1.2, zorder=2)
        
    ax2.axvline(0, color="black", linestyle="--", linewidth=1.0)
    ax2.set_xlabel("Context Gap (Suspicious - Benign)")
    ax2.set_ylabel("Sampled Type-A Pairs (Ordered by Severity)")
    ax2.set_title("B. Paired Shift: M0 vs M2 vs Oracle", fontweight="bold")
    ax2.legend(loc="lower right", framealpha=0.85)
    ax2.grid(axis="x")
    
    plt.tight_layout()
    save_multi_format(fig, os.path.join(output_dir, "figure4_context_pairs"))


def generate_figure5_explanations(summary_df: pd.DataFrame, detailed_df: pd.DataFrame, output_dir: str):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    
    
    
    comp_data = summary_df[summary_df["component"].str.contains("Intercept|Policy|Environment|Cloudtrail|CloudTrail", case=False, regex=True)].copy()
    
    y_idx = np.arange(len(comp_data))
    height = 0.35
    
    b_vals = comp_data["benign_contribution"].values
    s_vals = comp_data["suspicious_contribution"].values
    comp_labels = comp_data["component"].tolist()
    
    ax1.barh(y_idx + height/2, b_vals, height, color=PALETTE["Benign"], label="Benign Context", edgecolor="black", alpha=0.85)
    ax1.barh(y_idx - height/2, s_vals, height, color=PALETTE["Suspicious"], label="Suspicious Context", edgecolor="black", alpha=0.85)
    
    ax1.set_yticks(y_idx)
    ax1.set_yticklabels(comp_labels)
    ax1.set_xlabel("Additive Priority Contribution (Points)")
    ax1.set_title("A. High-Level Component Decomposition", fontweight="bold")
    ax1.axvline(0, color="black", linewidth=0.8)
    ax1.grid(axis="x")
    ax1.legend(loc="lower right", framealpha=0.85)
    
    
    ct_features = detailed_df[detailed_df["feature_group"] == "cloudtrail"].copy()
    top_ct = ct_features.sort_values(by="score_delta_susp_minus_benign", ascending=True).tail(8)
    
    y_top = np.arange(len(top_ct))
    deltas = top_ct["score_delta_susp_minus_benign"].values
    f_names = top_ct["feature"].values
    
    bar_colors = [PALETTE["M2"] if d > 0 else PALETTE["M0"] for d in deltas]
    ax2.barh(y_top, deltas, 0.55, color=bar_colors, edgecolor="black", alpha=0.9)
    ax2.set_yticks(y_top)
    ax2.set_yticklabels(f_names)
    ax2.set_xlabel("Behavioral Score Delta: Suspicious - Benign (Points)")
    ax2.set_title("B. Top CloudTrail Features Driving Priority Elevation", fontweight="bold")
    ax2.axvline(0, color="black", linewidth=0.8)
    ax2.grid(axis="x")
    
    plt.tight_layout()
    save_multi_format(fig, os.path.join(output_dir, "figure5_explanations"))
