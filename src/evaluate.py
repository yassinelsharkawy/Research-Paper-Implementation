import os
from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, ndcg_score
import logging

logger = logging.getLogger("ContextBench.ML.EvaluateExp1")


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, k_values: List[int] = (10, 25, 50)) -> Dict[str, float]:
    
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    
    
    spearman_res = stats.spearmanr(y_true, y_pred)
    kendall_res = stats.kendalltau(y_true, y_pred)
    
    spearman_rho = float(spearman_res.statistic) if hasattr(spearman_res, "statistic") else float(spearman_res[0])
    kendall_tau = float(kendall_res.statistic) if hasattr(kendall_res, "statistic") else float(kendall_res[0])
    
    
    y_true_2d = np.asarray([y_true])
    y_pred_2d = np.asarray([y_pred])
    
    ndcg_dict = {}
    for k in k_values:
        ndcg_k = float(ndcg_score(y_true_2d, y_pred_2d, k=k))
        ndcg_dict[f"NDCG@{k}"] = ndcg_k

    metrics = {
        "MAE": mae,
        "RMSE": rmse,
        "Spearman_rho": spearman_rho,
        "Kendall_tau": kendall_tau,
    }
    metrics.update(ndcg_dict)
    return metrics


def compute_fold_level_metrics(oof_df: pd.DataFrame, target_col: str = "gt_analyst_priority_score") -> pd.DataFrame:
    
    records = []
    models = ["M0", "M1", "M2"]
    for f_id in sorted(oof_df["fold_id"].unique()):
        fold_sub = oof_df[oof_df["fold_id"] == f_id]
        y_true = fold_sub[target_col].values
        for m in models:
            y_pred = fold_sub[f"{m}_prediction"].values
            m_dict = compute_metrics(y_true, y_pred)
            records.append({
                "fold_id": f_id,
                "model": m,
                "samples": len(fold_sub),
                "MAE": m_dict["MAE"],
                "RMSE": m_dict["RMSE"],
                "Spearman_rho": m_dict["Spearman_rho"],
                "Kendall_tau": m_dict["Kendall_tau"],
                "NDCG@10": m_dict["NDCG@10"],
                "NDCG@25": m_dict["NDCG@25"],
                "NDCG@50": m_dict["NDCG@50"]
            })
    return pd.DataFrame(records)


def bootstrap_cluster_metrics_and_deltas(
    df: pd.DataFrame,
    target_col: str = "gt_analyst_priority_score",
    cluster_col: str = "policy_lineage_id",
    n_iterations: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
    k_values: List[int] = (10, 25, 50),
) -> Dict[str, Tuple[float, float]]:
    
    rng = np.random.default_rng(seed)
    clusters = df[cluster_col].unique()
    n_clusters = len(clusters)
    cluster_map = {c: df[df[cluster_col] == c] for c in clusters}

    metric_names = ["MAE", "RMSE", "Spearman_rho", "Kendall_tau"] + [f"NDCG@{k}" for k in k_values]
    boot_history = {f"{m}_{met}": [] for m in ["M0", "M1", "M2"] for met in metric_names}
    boot_deltas = {f"delta_M2_M0_{met}": [] for met in metric_names}
    boot_deltas.update({f"delta_M1_M0_{met}": [] for met in metric_names})
    boot_deltas.update({f"delta_M2_M1_{met}": [] for met in metric_names})

    for _ in range(n_iterations):
        sample_clusters = rng.choice(clusters, size=n_clusters, replace=True)
        boot_df = pd.concat([cluster_map[c] for c in sample_clusters], ignore_index=True)
        
        y_true = boot_df[target_col].values
        
        try:
            m0_m = compute_metrics(y_true, boot_df["M0_prediction"].values, k_values=k_values)
            m1_m = compute_metrics(y_true, boot_df["M1_prediction"].values, k_values=k_values)
            m2_m = compute_metrics(y_true, boot_df["M2_prediction"].values, k_values=k_values)
            
            for met in metric_names:
                boot_history[f"M0_{met}"].append(m0_m[met])
                boot_history[f"M1_{met}"].append(m1_m[met])
                boot_history[f"M2_{met}"].append(m2_m[met])
                boot_deltas[f"delta_M2_M0_{met}"].append(m2_m[met] - m0_m[met])
                boot_deltas[f"delta_M1_M0_{met}"].append(m1_m[met] - m0_m[met])
                boot_deltas[f"delta_M2_M1_{met}"].append(m2_m[met] - m1_m[met])
        except Exception:
            continue

    ci_dict = {}
    lower_p = (alpha / 2.0) * 100
    upper_p = (1.0 - alpha / 2.0) * 100

    for k, values in {**boot_history, **boot_deltas}.items():
        if values:
            low = float(np.percentile(values, lower_p))
            high = float(np.percentile(values, upper_p))
            ci_dict[k] = (low, high)
        else:
            ci_dict[k] = (np.nan, np.nan)

    return ci_dict


def run_experiment_1(
    oof_df: pd.DataFrame,
    output_dir: str = None,
    target_col: str = "gt_analyst_priority_score",
    cluster_col: str = "policy_lineage_id",
    seed: int = 42,
) -> Tuple[pd.DataFrame, Dict[str, Any], pd.DataFrame]:
    
    logger.info("=== Running Experiment 1: Incremental Context Ablation ===")
    
    y_true = oof_df[target_col].values
    k_vals = [10, 25, 50]
    
    m0_metrics = compute_metrics(y_true, oof_df["M0_prediction"].values, k_values=k_vals)
    m1_metrics = compute_metrics(y_true, oof_df["M1_prediction"].values, k_values=k_vals)
    m2_metrics = compute_metrics(y_true, oof_df["M2_prediction"].values, k_values=k_vals)

    logger.info("Computing 95% cluster bootstrap confidence intervals (1000 iterations)...")
    ci_dict = bootstrap_cluster_metrics_and_deltas(oof_df, target_col, cluster_col, seed=seed, k_values=k_vals)

    metric_names = ["MAE", "RMSE", "Spearman_rho", "Kendall_tau", "NDCG@10", "NDCG@25", "NDCG@50"]
    rows = []
    exp1_raw_dict = {}

    for m in metric_names:
        v0 = m0_metrics[m]
        v1 = m1_metrics[m]
        v2 = m2_metrics[m]
        
        delta_m1_m0 = v1 - v0
        delta_m2_m1 = v2 - v1
        delta_m2_m0 = v2 - v0
        
        
        if m in ("MAE", "RMSE"):
            pct_imp_m2_over_m0 = ((v0 - v2) / v0) * 100.0 if v0 != 0 else 0.0
        else:
            pct_imp_m2_over_m0 = ((v2 - v0) / v0) * 100.0 if v0 != 0 else 0.0

        ci_m0 = f"[{ci_dict[f'M0_{m}'][0]:.4f}, {ci_dict[f'M0_{m}'][1]:.4f}]"
        ci_m1 = f"[{ci_dict[f'M1_{m}'][0]:.4f}, {ci_dict[f'M1_{m}'][1]:.4f}]"
        ci_m2 = f"[{ci_dict[f'M2_{m}'][0]:.4f}, {ci_dict[f'M2_{m}'][1]:.4f}]"
        ci_delta = f"[{ci_dict[f'delta_M2_M0_{m}'][0]:.4f}, {ci_dict[f'delta_M2_M0_{m}'][1]:.4f}]"

        m_key = m.lower().replace("_", "").replace("@", "@")
        if m == "Spearman_rho":
            m_key = "spearman"
        elif m == "Kendall_tau":
            m_key = "kendall"

        exp1_raw_dict[f"M0_{m_key}"] = v0
        exp1_raw_dict[f"M1_{m_key}"] = v1
        exp1_raw_dict[f"M2_{m_key}"] = v2
        exp1_raw_dict[f"M0_{m_key}_ci_lower"] = ci_dict[f"M0_{m}"][0]
        exp1_raw_dict[f"M0_{m_key}_ci_upper"] = ci_dict[f"M0_{m}"][1]
        exp1_raw_dict[f"M1_{m_key}_ci_lower"] = ci_dict[f"M1_{m}"][0]
        exp1_raw_dict[f"M1_{m_key}_ci_upper"] = ci_dict[f"M1_{m}"][1]
        exp1_raw_dict[f"M2_{m_key}_ci_lower"] = ci_dict[f"M2_{m}"][0]
        exp1_raw_dict[f"M2_{m_key}_ci_upper"] = ci_dict[f"M2_{m}"][1]
        exp1_raw_dict[f"delta_M2_M0_{m_key}"] = delta_m2_m0
        exp1_raw_dict[f"delta_M2_M0_{m_key}_ci_lower"] = ci_dict[f"delta_M2_M0_{m}"][0]
        exp1_raw_dict[f"delta_M2_M0_{m_key}_ci_upper"] = ci_dict[f"delta_M2_M0_{m}"][1]

        rows.append({
            "metric": m,
            "M0_policy": round(v0, 4),
            "M0_95_CI": ci_m0,
            "M1_environment": round(v1, 4),
            "M1_95_CI": ci_m1,
            "M2_cloudtrail_context": round(v2, 4),
            "M2_95_CI": ci_m2,
            "delta_M1_minus_M0": round(delta_m1_m0, 4),
            "delta_M2_minus_M1": round(delta_m2_m1, 4),
            "delta_M2_minus_M0": round(delta_m2_m0, 4),
            "delta_M2_M0_95_CI": ci_delta,
            "pct_improvement_M2_over_M0": f"{pct_imp_m2_over_m0:+.2f}%",
        })

    results_df = pd.DataFrame(rows)
    fold_metrics_df = compute_fold_level_metrics(oof_df, target_col=target_col)

    if output_dir:
        metrics_dir = os.path.join(output_dir, "results", "metrics")
        tables_dir = os.path.join(output_dir, "results", "tables")
        os.makedirs(metrics_dir, exist_ok=True)
        os.makedirs(tables_dir, exist_ok=True)

        results_df.to_csv(os.path.join(metrics_dir, "experiment1_ablation.csv"), index=False)
        fold_metrics_df.to_csv(os.path.join(metrics_dir, "experiment1_fold_metrics.csv"), index=False)
        logger.info(f"Experiment 1 results exported successfully.")

    return results_df, exp1_raw_dict, fold_metrics_df
