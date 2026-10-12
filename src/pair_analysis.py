import os
import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Any, Tuple, List


def validate_type_a_pairs(df: pd.DataFrame, output_csv: str = None) -> Tuple[List[str], pd.DataFrame]:
    
    type_a_df = df[df["pair_type"] == "TYPE_A"].copy()
    pair_ids = sorted(type_a_df["pair_id"].unique())
    
    
    eff_cols = [c for c in df.columns if c.startswith("effective_") and c not in ["effective_auth_source", "effective_auth_hash"]]
    
    records = []
    eligible_pair_ids = []
    
    for pid in pair_ids:
        group = type_a_df[type_a_df["pair_id"] == pid]
        if len(group) != 2:
            records.append({
                "pair_id": pid,
                "policy_equal": False,
                "environment_equal": False,
                "effective_auth_equal": False,
                "context_different": False,
                "eligible_for_exp2": False,
                "reason": f"Expected 2 members, found {len(group)}"
            })
            continue
            
        r1, r2 = group.iloc[0], group.iloc[1]
        
        
        policy_eq = bool(r1["structural_policy_hash"] == r2["structural_policy_hash"])
        
        
        env_eq = bool(r1["environment_id"] == r2["environment_id"])
        
        
        eff_eq = bool((r1[eff_cols] == r2[eff_cols]).all())
        
        
        c_types = {r1["context_type"], r2["context_type"]}
        ctx_diff = bool(len(c_types) == 2 and "benign" in c_types and "suspicious" in c_types)
        
        eligible = bool(policy_eq and env_eq and eff_eq and ctx_diff)
        
        records.append({
            "pair_id": pid,
            "policy_equal": policy_eq,
            "environment_equal": env_eq,
            "effective_auth_equal": eff_eq,
            "context_different": ctx_diff,
            "eligible_for_exp2": eligible,
            "reason": "OK" if eligible else "Failed equivalence"
        })
        
        if eligible:
            eligible_pair_ids.append(pid)
            
    val_df = pd.DataFrame(records)
    
    if output_csv:
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        val_df.to_csv(output_csv, index=False)
        
    return eligible_pair_ids, val_df


def compute_pair_gaps(oof_df: pd.DataFrame, df: pd.DataFrame, eligible_pair_ids: List[str]) -> pd.DataFrame:
    
    
    merged = oof_df.copy()
    if "context_type" not in merged.columns or "gt_security_risk_score" not in merged.columns:
        extra_cols = ["sample_id"]
        if "context_type" not in merged.columns:
            extra_cols.append("context_type")
        if "gt_security_risk_score" not in merged.columns:
            extra_cols.append("gt_security_risk_score")
        merged = merged.merge(df[extra_cols], on="sample_id", how="left")
        
    pair_rows = []
    
    for pid in eligible_pair_ids:
        pair_data = merged[merged["pair_id"] == pid]
        if len(pair_data) != 2:
            continue
            
        susp = pair_data[pair_data["context_type"] == "suspicious"]
        benign = pair_data[pair_data["context_type"] == "benign"]
        
        if len(susp) != 1 or len(benign) != 1:
            continue
            
        s_row = susp.iloc[0]
        b_row = benign.iloc[0]
        
        gt_gap = float(s_row["gt_analyst_priority_score"] - b_row["gt_analyst_priority_score"])
        gt_sec_gap = float(s_row["gt_security_risk_score"] - b_row["gt_security_risk_score"])
        m0_gap = float(s_row["M0_prediction"] - b_row["M0_prediction"])
        m1_gap = float(s_row["M1_prediction"] - b_row["M1_prediction"])
        m2_gap = float(s_row["M2_prediction"] - b_row["M2_prediction"])
        
        pair_rows.append({
            "pair_id": pid,
            "policy_lineage_id": s_row["policy_lineage_id"],
            "scenario_family": s_row["scenario_family"],
            "suspicious_sample_id": s_row["sample_id"],
            "benign_sample_id": b_row["sample_id"],
            "gt_analyst_priority_suspicious": float(s_row["gt_analyst_priority_score"]),
            "gt_analyst_priority_benign": float(b_row["gt_analyst_priority_score"]),
            "gt_priority_gap": gt_gap,
            "gt_security_risk_gap": gt_sec_gap,
            "m0_score_suspicious": float(s_row["M0_prediction"]),
            "m0_score_benign": float(b_row["M0_prediction"]),
            "m0_gap": m0_gap,
            "m1_score_suspicious": float(s_row["M1_prediction"]),
            "m1_score_benign": float(b_row["M1_prediction"]),
            "m1_gap": m1_gap,
            "m2_score_suspicious": float(s_row["M2_prediction"]),
            "m2_score_benign": float(b_row["M2_prediction"]),
            "m2_gap": m2_gap,
            "m0_abs_gap": abs(m0_gap),
            "m2_abs_gap": abs(m2_gap),
            "m0_direction_correct": bool((m0_gap > 0) == (gt_gap > 0)),
            "m2_direction_correct": bool((m2_gap > 0) == (gt_gap > 0)),
            "m0_ranking_correct": bool(m0_gap > 0),
            "m2_ranking_correct": bool(m2_gap > 0)
        })
        
    return pd.DataFrame(pair_rows)


def run_experiment_2(
    oof_df: pd.DataFrame,
    df: pd.DataFrame,
    output_dir: str,
    n_bootstraps: int = 1000,
    seed: int = 42
) -> Tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame]:
    
    os.makedirs(output_dir, exist_ok=True)
    val_csv = os.path.join(output_dir, "pair_validation.csv")
    eligible_pair_ids, val_df = validate_type_a_pairs(df, output_csv=val_csv)
    
    results_val_csv = os.path.abspath(os.path.join(output_dir, "..", "pair_validation.csv"))
    val_df.to_csv(results_val_csv, index=False)
    
    pair_gaps = compute_pair_gaps(oof_df, df, eligible_pair_ids)
    
    
    pair_level_path = os.path.join(output_dir, "experiment2_pair_level_gaps.csv")
    pair_gaps.to_csv(pair_level_path, index=False)
    
    gt_gaps = pair_gaps["gt_priority_gap"].values
    m0_gaps = pair_gaps["m0_gap"].values
    m1_gaps = pair_gaps["m1_gap"].values
    m2_gaps = pair_gaps["m2_gap"].values
    
    
    mean_gt_gap = float(np.mean(gt_gaps))
    median_gt_gap = float(np.median(gt_gaps))
    
    mean_m0_gap = float(np.mean(m0_gaps))
    median_m0_gap = float(np.median(m0_gaps))
    mean_abs_m0_gap = float(np.mean(np.abs(m0_gaps)))
    median_abs_m0_gap = float(np.median(np.abs(m0_gaps)))
    
    mean_m2_gap = float(np.mean(m2_gaps))
    median_m2_gap = float(np.median(m2_gaps))
    mean_abs_m2_gap = float(np.mean(np.abs(m2_gaps)))
    median_abs_m2_gap = float(np.median(np.abs(m2_gaps)))
    
    m0_dir_acc = float(np.mean(pair_gaps["m0_direction_correct"]))
    m2_dir_acc = float(np.mean(pair_gaps["m2_direction_correct"]))
    
    m0_rank_acc = float(np.mean(pair_gaps["m0_ranking_correct"]))
    m2_rank_acc = float(np.mean(pair_gaps["m2_ranking_correct"]))
    
    
    mae_m0_gap = float(np.mean(np.abs(m0_gaps - gt_gaps)))
    mae_m2_gap = float(np.mean(np.abs(m2_gaps - gt_gaps)))
    
    
    if np.std(m0_gaps) > 1e-9:
        pearson_m0 = float(stats.pearsonr(m0_gaps, gt_gaps)[0])
        spearman_m0 = float(stats.spearmanr(m0_gaps, gt_gaps)[0])
    else:
        pearson_m0 = 0.0
        spearman_m0 = 0.0
        
    pearson_m2 = float(stats.pearsonr(m2_gaps, gt_gaps)[0])
    spearman_m2 = float(stats.spearmanr(m2_gaps, gt_gaps)[0])
    
    
    
    diff_m2_m0 = m2_gaps - m0_gaps
    try:
        w_stat, w_pvalue = stats.wilcoxon(diff_m2_m0, alternative="greater")
    except Exception:
        w_stat, w_pvalue = 0.0, 1.0
        
    
    try:
        w_stat_m2_zero, w_pvalue_m2_zero = stats.wilcoxon(m2_gaps, alternative="greater")
    except Exception:
        w_stat_m2_zero, w_pvalue_m2_zero = 0.0, 1.0
        
    
    rng = np.random.RandomState(seed)
    observed_mean_diff = np.mean(diff_m2_m0)
    perm_diffs = []
    for _ in range(5000):
        signs = rng.choice([-1, 1], size=len(diff_m2_m0))
        perm_diffs.append(np.mean(diff_m2_m0 * signs))
    perm_pvalue = float(np.mean(np.array(perm_diffs) >= observed_mean_diff))
    
    
    std_diff = np.std(diff_m2_m0, ddof=1)
    cohens_d = float(observed_mean_diff / std_diff) if std_diff > 1e-9 else 0.0
    
    
    n_pairs = len(pair_gaps)
    boot_stats = {
        "mean_m0_gap": [],
        "mean_m2_gap": [],
        "median_m0_gap": [],
        "median_m2_gap": [],
        "m2_minus_m0_gap": [],
        "m2_dir_acc": [],
        "m2_rank_acc": [],
        "mae_m0_gap": [],
        "mae_m2_gap": [],
        "spearman_m2": []
    }
    
    for _ in range(n_bootstraps):
        boot_idx = rng.choice(n_pairs, size=n_pairs, replace=True)
        b_gt = gt_gaps[boot_idx]
        b_m0 = m0_gaps[boot_idx]
        b_m2 = m2_gaps[boot_idx]
        
        boot_stats["mean_m0_gap"].append(np.mean(b_m0))
        boot_stats["mean_m2_gap"].append(np.mean(b_m2))
        boot_stats["median_m0_gap"].append(np.median(b_m0))
        boot_stats["median_m2_gap"].append(np.median(b_m2))
        boot_stats["m2_minus_m0_gap"].append(np.mean(b_m2 - b_m0))
        boot_stats["m2_dir_acc"].append(np.mean((b_m2 > 0) == (b_gt > 0)))
        boot_stats["m2_rank_acc"].append(np.mean(b_m2 > 0))
        boot_stats["mae_m0_gap"].append(np.mean(np.abs(b_m0 - b_gt)))
        boot_stats["mae_m2_gap"].append(np.mean(np.abs(b_m2 - b_gt)))
        if np.std(b_m2) > 1e-9 and np.std(b_gt) > 1e-9:
            boot_stats["spearman_m2"].append(stats.spearmanr(b_m2, b_gt)[0])
        else:
            boot_stats["spearman_m2"].append(0.0)
            
    ci_dict = {}
    for key, values in boot_stats.items():
        ci_dict[f"{key}_ci_lower"] = float(np.percentile(values, 2.5))
        ci_dict[f"{key}_ci_upper"] = float(np.percentile(values, 97.5))
        
    results = {
        "n_valid_type_a_pairs": int(n_pairs),
        "mean_gt_gap": mean_gt_gap,
        "median_gt_gap": median_gt_gap,
        "mean_m0_gap": mean_m0_gap,
        "median_m0_gap": median_m0_gap,
        "mean_abs_m0_gap": mean_abs_m0_gap,
        "median_abs_m0_gap": median_abs_m0_gap,
        "mean_m2_gap": mean_m2_gap,
        "median_m2_gap": median_m2_gap,
        "mean_abs_m2_gap": mean_abs_m2_gap,
        "median_abs_m2_gap": median_abs_m2_gap,
        "m2_minus_m0_mean_gap": float(mean_m2_gap - mean_m0_gap),
        "m0_direction_accuracy": m0_dir_acc,
        "m2_direction_accuracy": m2_dir_acc,
        "m0_ranking_accuracy": m0_rank_acc,
        "m2_ranking_accuracy": m2_rank_acc,
        "m0_gap_mae": mae_m0_gap,
        "m2_gap_mae": mae_m2_gap,
        "m0_gap_pearson": pearson_m0,
        "m0_gap_spearman": spearman_m0,
        "m2_gap_pearson": pearson_m2,
        "m2_gap_spearman": spearman_m2,
        "wilcoxon_stat_m2_gt_m0": float(w_stat),
        "wilcoxon_pvalue_m2_gt_m0": float(w_pvalue),
        "wilcoxon_stat_m2_gt_zero": float(w_stat_m2_zero),
        "wilcoxon_pvalue_m2_gt_zero": float(w_pvalue_m2_zero),
        "permutation_pvalue_m2_gt_m0": perm_pvalue,
        "cohens_d_m2_minus_m0": cohens_d,
        **ci_dict
    }
    
    
    summary_rows = [
        {"metric": "Number of Valid Type-A Pairs", "M0": f"{n_pairs}", "M2": f"{n_pairs}", "Delta (M2-M0)": "0", "95% CI": "N/A"},
        {"metric": "Mean Pair Gap (Suspicious - Benign)", "M0": f"{mean_m0_gap:.4f}", "M2": f"{mean_m2_gap:.4f}", "Delta (M2-M0)": f"{mean_m2_gap - mean_m0_gap:.4f}", "95% CI": f"[{ci_dict['m2_minus_m0_gap_ci_lower']:.4f}, {ci_dict['m2_minus_m0_gap_ci_upper']:.4f}]"},
        {"metric": "Median Pair Gap", "M0": f"{median_m0_gap:.4f}", "M2": f"{median_m2_gap:.4f}", "Delta (M2-M0)": f"{median_m2_gap - median_m0_gap:.4f}", "95% CI": f"[{ci_dict['median_m2_gap_ci_lower']:.4f}, {ci_dict['median_m2_gap_ci_upper']:.4f}]"},
        {"metric": "Mean Absolute Gap", "M0": f"{mean_abs_m0_gap:.4f}", "M2": f"{mean_abs_m2_gap:.4f}", "Delta (M2-M0)": f"{mean_abs_m2_gap - mean_abs_m0_gap:.4f}", "95% CI": "N/A"},
        {"metric": "Pairwise Direction Accuracy", "M0": f"{m0_dir_acc:.4f}", "M2": f"{m2_dir_acc:.4f}", "Delta (M2-M0)": f"{m2_dir_acc - m0_dir_acc:.4f}", "95% CI": f"[{ci_dict['m2_dir_acc_ci_lower']:.4f}, {ci_dict['m2_dir_acc_ci_upper']:.4f}]"},
        {"metric": "Pairwise Ranking Accuracy (Susp > Benign)", "M0": f"{m0_rank_acc:.4f}", "M2": f"{m2_rank_acc:.4f}", "Delta (M2-M0)": f"{m2_rank_acc - m0_rank_acc:.4f}", "95% CI": f"[{ci_dict['m2_rank_acc_ci_lower']:.4f}, {ci_dict['m2_rank_acc_ci_upper']:.4f}]"},
        {"metric": "Gap MAE (vs Oracle GT Gap)", "M0": f"{mae_m0_gap:.4f}", "M2": f"{mae_m2_gap:.4f}", "Delta (M2-M0)": f"{mae_m2_gap - mae_m0_gap:.4f}", "95% CI": f"[{ci_dict['mae_m2_gap_ci_lower']:.4f}, {ci_dict['mae_m2_gap_ci_upper']:.4f}]"},
        {"metric": "Gap Spearman Correlation", "M0": f"{spearman_m0:.4f}", "M2": f"{spearman_m2:.4f}", "Delta (M2-M0)": f"{spearman_m2 - spearman_m0:.4f}", "95% CI": f"[{ci_dict['spearman_m2_ci_lower']:.4f}, {ci_dict['spearman_m2_ci_upper']:.4f}]"},
        {"metric": "Cohen's d (M2 vs M0 Gap Shift)", "M0": "0.0000", "M2": f"{cohens_d:.4f}", "Delta (M2-M0)": f"{cohens_d:.4f}", "95% CI": "N/A"},
        {"metric": "Wilcoxon Signed-Rank Test p-value", "M0": "N/A", "M2": f"{w_pvalue:.4e}", "Delta (M2-M0)": "N/A", "95% CI": "N/A"},
        {"metric": "Permutation Test p-value", "M0": "N/A", "M2": f"{perm_pvalue:.4e}", "Delta (M2-M0)": "N/A", "95% CI": "N/A"}
    ]
    summary_df = pd.DataFrame(summary_rows)
    summary_csv = os.path.join(output_dir, "experiment2_context_pairs.csv")
    summary_df.to_csv(summary_csv, index=False)
    
    return results, pair_gaps, val_df
