import os
import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Any, List, Tuple


def compute_comprehensive_statistics(
    oof_df: pd.DataFrame,
    exp1_metrics: Dict[str, Any],
    exp2_metrics: Dict[str, Any],
    output_dir: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    
    os.makedirs(output_dir, exist_ok=True)
    
    y_true = oof_df["gt_analyst_priority_score"].values
    m0_pred = oof_df["M0_prediction"].values
    m1_pred = oof_df["M1_prediction"].values
    m2_pred = oof_df["M2_prediction"].values
    
    
    m0_abs_err = np.abs(m0_pred - y_true)
    m1_abs_err = np.abs(m1_pred - y_true)
    m2_abs_err = np.abs(m2_pred - y_true)
    
    m0_sq_err = (m0_pred - y_true) ** 2
    m1_sq_err = (m1_pred - y_true) ** 2
    m2_sq_err = (m2_pred - y_true) ** 2
    
    
    sig_records = []
    
    
    diff_abs_m0_m2 = m0_abs_err - m2_abs_err  
    w_stat_mae, w_pval_mae = stats.wilcoxon(diff_abs_m0_m2, alternative="greater")
    t_stat_mae, t_pval_mae = stats.ttest_rel(m0_abs_err, m2_abs_err, alternative="greater")
    d_mae = float(np.mean(diff_abs_m0_m2) / np.std(diff_abs_m0_m2, ddof=1))
    sig_records.append({
        "comparison": "M2 vs M0 Absolute Error (MAE)",
        "null_hypothesis": "MAE(M2) >= MAE(M0)",
        "test_type": "Wilcoxon Signed-Rank Test",
        "statistic": float(w_stat_mae),
        "p_value": float(w_pval_mae),
        "effect_size_cohens_d": d_mae,
        "significance_at_0_05": bool(w_pval_mae < 0.05),
        "interpretation": "M2 significantly reduces MAE compared to M0" if w_pval_mae < 0.05 else "No significant difference"
    })
    sig_records.append({
        "comparison": "M2 vs M0 Absolute Error (MAE)",
        "null_hypothesis": "MAE(M2) >= MAE(M0)",
        "test_type": "Paired Student t-Test",
        "statistic": float(t_stat_mae),
        "p_value": float(t_pval_mae),
        "effect_size_cohens_d": d_mae,
        "significance_at_0_05": bool(t_pval_mae < 0.05),
        "interpretation": "M2 significantly reduces MAE compared to M0" if t_pval_mae < 0.05 else "No significant difference"
    })
    
    
    diff_abs_m1_m2 = m1_abs_err - m2_abs_err
    w_stat_m1_m2, w_pval_m1_m2 = stats.wilcoxon(diff_abs_m1_m2, alternative="greater")
    d_m1_m2 = float(np.mean(diff_abs_m1_m2) / np.std(diff_abs_m1_m2, ddof=1))
    sig_records.append({
        "comparison": "M2 vs M1 Absolute Error (MAE)",
        "null_hypothesis": "MAE(M2) >= MAE(M1)",
        "test_type": "Wilcoxon Signed-Rank Test",
        "statistic": float(w_stat_m1_m2),
        "p_value": float(w_pval_m1_m2),
        "effect_size_cohens_d": d_m1_m2,
        "significance_at_0_05": bool(w_pval_m1_m2 < 0.05),
        "interpretation": "M2 significantly reduces MAE beyond environment features (M1)" if w_pval_m1_m2 < 0.05 else "No significant difference"
    })
    
    
    diff_abs_m0_m1 = m0_abs_err - m1_abs_err
    w_stat_m0_m1, w_pval_m0_m1 = stats.wilcoxon(diff_abs_m0_m1, alternative="greater")
    d_m0_m1 = float(np.mean(diff_abs_m0_m1) / np.std(diff_abs_m0_m1, ddof=1))
    sig_records.append({
        "comparison": "M1 vs M0 Absolute Error (MAE)",
        "null_hypothesis": "MAE(M1) >= MAE(M0)",
        "test_type": "Wilcoxon Signed-Rank Test",
        "statistic": float(w_stat_m0_m1),
        "p_value": float(w_pval_m0_m1),
        "effect_size_cohens_d": d_m0_m1,
        "significance_at_0_05": bool(w_pval_m0_m1 < 0.05),
        "interpretation": "M1 significantly reduces MAE compared to M0" if w_pval_m0_m1 < 0.05 else "No significant difference"
    })
    
    
    sig_records.append({
        "comparison": "Exp 2: M2 Pair Gap > 0",
        "null_hypothesis": "M2 Gap (Suspicious - Benign) <= 0",
        "test_type": "Wilcoxon Signed-Rank Test",
        "statistic": float(exp2_metrics.get("wilcoxon_stat_m2_gt_zero", 0.0)),
        "p_value": float(exp2_metrics.get("wilcoxon_pvalue_m2_gt_zero", 1.0)),
        "effect_size_cohens_d": float(exp2_metrics.get("cohens_d_m2_minus_m0", 0.0)),
        "significance_at_0_05": bool(exp2_metrics.get("wilcoxon_pvalue_m2_gt_zero", 1.0) < 0.05),
        "interpretation": "M2 significantly elevates suspicious contexts above benign contexts"
    })
    sig_records.append({
        "comparison": "Exp 2: M2 Gap vs M0 Gap",
        "null_hypothesis": "M2 Gap <= M0 Gap",
        "test_type": "Paired Permutation Test (5000 resamples)",
        "statistic": float(exp2_metrics.get("m2_minus_m0_mean_gap", 0.0)),
        "p_value": float(exp2_metrics.get("permutation_pvalue_m2_gt_m0", 1.0)),
        "effect_size_cohens_d": float(exp2_metrics.get("cohens_d_m2_minus_m0", 0.0)),
        "significance_at_0_05": bool(exp2_metrics.get("permutation_pvalue_m2_gt_m0", 1.0) < 0.05),
        "interpretation": "M2 context gap shift is statistically significant under permutation"
    })
    
    sig_df = pd.DataFrame(sig_records)
    sig_df.to_csv(os.path.join(output_dir, "significance_tests.csv"), index=False)
    
    
    ci_records = []
    
    
    ci_mapping_exp1 = [
        ("M0 MAE", "M0_mae", "M0_mae_ci_lower", "M0_mae_ci_upper", "Policy-Centric Baseline"),
        ("M1 MAE", "M1_mae", "M1_mae_ci_lower", "M1_mae_ci_upper", "Policy + Environment"),
        ("M2 MAE", "M2_mae", "M2_mae_ci_lower", "M2_mae_ci_upper", "Full Behavioral Context"),
        ("M2 - M0 MAE Delta", "delta_M2_M0_mae", "delta_M2_M0_mae_ci_lower", "delta_M2_M0_mae_ci_upper", "MAE Improvement (M2 vs M0)"),
        ("M0 RMSE", "M0_rmse", "M0_rmse_ci_lower", "M0_rmse_ci_upper", "Policy-Centric Baseline"),
        ("M1 RMSE", "M1_rmse", "M1_rmse_ci_lower", "M1_rmse_ci_upper", "Policy + Environment"),
        ("M2 RMSE", "M2_rmse", "M2_rmse_ci_lower", "M2_rmse_ci_upper", "Full Behavioral Context"),
        ("M2 - M0 RMSE Delta", "delta_M2_M0_rmse", "delta_M2_M0_rmse_ci_lower", "delta_M2_M0_rmse_ci_upper", "RMSE Improvement (M2 vs M0)"),
        ("M0 Spearman rho", "M0_spearman", "M0_spearman_ci_lower", "M0_spearman_ci_upper", "Policy-Centric Baseline"),
        ("M2 Spearman rho", "M2_spearman", "M2_spearman_ci_lower", "M2_spearman_ci_upper", "Full Behavioral Context"),
        ("M2 - M0 Spearman Delta", "delta_M2_M0_spearman", "delta_M2_M0_spearman_ci_lower", "delta_M2_M0_spearman_ci_upper", "Rank Correlation Improvement"),
        ("M0 Kendall tau", "M0_kendall", "M0_kendall_ci_lower", "M0_kendall_ci_upper", "Policy-Centric Baseline"),
        ("M2 Kendall tau", "M2_kendall", "M2_kendall_ci_lower", "M2_kendall_ci_upper", "Full Behavioral Context"),
        ("M0 NDCG@10", "M0_ndcg@10", "M0_ndcg@10_ci_lower", "M0_ndcg@10_ci_upper", "Policy-Centric Baseline"),
        ("M2 NDCG@10", "M2_ndcg@10", "M2_ndcg@10_ci_lower", "M2_ndcg@10_ci_upper", "Full Behavioral Context"),
        ("M2 - M0 NDCG@10 Delta", "delta_M2_M0_ndcg@10", "delta_M2_M0_ndcg@10_ci_lower", "delta_M2_M0_ndcg@10_ci_upper", "Triage Top-10 Ranking Improvement"),
        ("M0 NDCG@25", "M0_ndcg@25", "M0_ndcg@25_ci_lower", "M0_ndcg@25_ci_upper", "Policy-Centric Baseline"),
        ("M2 NDCG@25", "M2_ndcg@25", "M2_ndcg@25_ci_lower", "M2_ndcg@25_ci_upper", "Full Behavioral Context"),
        ("M2 - M0 NDCG@25 Delta", "delta_M2_M0_ndcg@25", "delta_M2_M0_ndcg@25_ci_lower", "delta_M2_M0_ndcg@25_ci_upper", "Triage Top-25 Ranking Improvement")
    ]
    
    for label, val_k, lo_k, hi_k, desc in ci_mapping_exp1:
        if val_k in exp1_metrics and lo_k in exp1_metrics and hi_k in exp1_metrics:
            ci_records.append({
                "experiment": "Experiment 1: Incremental Ablation",
                "metric": label,
                "point_estimate": round(float(exp1_metrics[val_k]), 4),
                "ci_lower_95": round(float(exp1_metrics[lo_k]), 4),
                "ci_upper_95": round(float(exp1_metrics[hi_k]), 4),
                "resampling_unit": "policy_lineage_id (1000 cluster bootstraps)",
                "description": desc
            })
            
    
    ci_mapping_exp2 = [
        ("M0 Mean Context Gap", "mean_m0_gap", "mean_m0_gap_ci_lower", "mean_m0_gap_ci_upper", "Baseline Context Invariance"),
        ("M2 Mean Context Gap", "mean_m2_gap", "mean_m2_gap_ci_lower", "mean_m2_gap_ci_upper", "Behavioral Sensitivity"),
        ("M2 - M0 Context Gap Shift", "m2_minus_m0_mean_gap", "m2_minus_m0_gap_ci_lower", "m2_minus_m0_gap_ci_upper", "Incremental Behavioral Lift"),
        ("M2 Pairwise Direction Accuracy", "m2_direction_accuracy", "m2_dir_acc_ci_lower", "m2_dir_acc_ci_upper", "Directional Concordance"),
        ("M2 Pairwise Ranking Accuracy", "m2_ranking_accuracy", "m2_rank_acc_ci_lower", "m2_rank_acc_ci_upper", "Ranking Concordance"),
        ("M2 Gap MAE", "m2_gap_mae", "mae_m2_gap_ci_lower", "mae_m2_gap_ci_upper", "Context Gap Calibration"),
        ("M2 Gap Spearman rho", "m2_gap_spearman", "spearman_m2_ci_lower", "spearman_m2_ci_upper", "Gap Order Concordance")
    ]
    
    for label, val_k, lo_k, hi_k, desc in ci_mapping_exp2:
        if val_k in exp2_metrics and lo_k in exp2_metrics and hi_k in exp2_metrics:
            ci_records.append({
                "experiment": "Experiment 2: Paired Context Test",
                "metric": label,
                "point_estimate": round(float(exp2_metrics[val_k]), 4),
                "ci_lower_95": round(float(exp2_metrics[lo_k]), 4),
                "ci_upper_95": round(float(exp2_metrics[hi_k]), 4),
                "resampling_unit": "pair_id (1000 pair-resampled bootstraps)",
                "description": desc
            })
            
    ci_df = pd.DataFrame(ci_records)
    ci_df.to_csv(os.path.join(output_dir, "confidence_intervals.csv"), index=False)
    
    return ci_df, sig_df
