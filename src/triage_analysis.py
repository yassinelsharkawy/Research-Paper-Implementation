import os
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple


def compute_ndcg_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int = 10) -> float:
    
    order = np.argsort(y_score)[::-1]
    y_true_sorted = np.take(y_true, order[:k])
    
    gains = y_true_sorted
    discounts = np.log2(np.arange(2, k + 2))
    dcg = np.sum(gains / discounts)
    
    ideal_order = np.argsort(y_true)[::-1]
    ideal_gains = np.take(y_true, ideal_order[:k])
    idcg = np.sum(ideal_gains / discounts)
    
    if idcg <= 0:
        return 0.0
    return float(dcg / idcg)


def compute_precision_recall_at_k(
    y_binary: np.ndarray,
    y_score: np.ndarray,
    k: int = 10
) -> Tuple[float, float]:
    
    order = np.argsort(y_score)[::-1]
    top_k_binary = np.take(y_binary, order[:k])
    
    total_positives = np.sum(y_binary)
    retrieved_positives = np.sum(top_k_binary)
    
    precision = float(retrieved_positives / k)
    recall = float(retrieved_positives / total_positives) if total_positives > 0 else 0.0
    
    return precision, recall


def run_experiment_3(
    oof_df: pd.DataFrame,
    df: pd.DataFrame,
    output_dir: str,
    k_list: List[int] = [10, 25, 50]
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    
    os.makedirs(output_dir, exist_ok=True)
    
    
    merged = oof_df.copy()
    if "gt_analyst_priority_label" not in merged.columns:
        merged = merged.merge(
            df[["sample_id", "gt_analyst_priority_label"]],
            on="sample_id",
            how="left"
        )
        
    y_true_score = merged["gt_analyst_priority_score"].values
    
    
    
    y_binary_high = (merged["gt_analyst_priority_label"] != "LOW").astype(int).values
    total_high = int(np.sum(y_binary_high))
    
    models = ["M0", "M1", "M2"]
    score_cols = {
        "M0": "M0_prediction",
        "M1": "M1_prediction",
        "M2": "M2_prediction"
    }
    
    triage_results = {
        "total_samples": len(merged),
        "total_high_priority_events": total_high,
        "high_priority_label_used": "MODERATE (top priority class in gt_analyst_priority_label)"
    }
    
    records = []
    
    
    for k in k_list:
        metrics_k = {}
        for m in models:
            scores = merged[score_cols[m]].values
            ndcg = compute_ndcg_at_k(y_true_score, scores, k=k)
            prec, rec = compute_precision_recall_at_k(y_binary_high, scores, k=k)
            
            metrics_k[f"{m}_ndcg"] = ndcg
            metrics_k[f"{m}_precision"] = prec
            metrics_k[f"{m}_recall"] = rec
            
            triage_results[f"{m}_ndcg@{k}"] = ndcg
            triage_results[f"{m}_precision@{k}"] = prec
            triage_results[f"{m}_recall@{k}"] = rec
            
        
        m0_ndcg = metrics_k["M0_ndcg"]
        m1_ndcg = metrics_k["M1_ndcg"]
        m2_ndcg = metrics_k["M2_ndcg"]
        
        ndcg_lift_m1_m0 = ((m1_ndcg - m0_ndcg) / m0_ndcg * 100) if m0_ndcg > 0 else 0.0
        ndcg_lift_m2_m0 = ((m2_ndcg - m0_ndcg) / m0_ndcg * 100) if m0_ndcg > 0 else 0.0
        ndcg_lift_m2_m1 = ((m2_ndcg - m1_ndcg) / m1_ndcg * 100) if m1_ndcg > 0 else 0.0
        
        
        m0_prec = metrics_k["M0_precision"]
        m1_prec = metrics_k["M1_precision"]
        m2_prec = metrics_k["M2_precision"]
        
        prec_lift_m1_m0 = ((m1_prec - m0_prec) / m0_prec * 100) if m0_prec > 0 else 0.0
        prec_lift_m2_m0 = ((m2_prec - m0_prec) / m0_prec * 100) if m0_prec > 0 else 0.0
        prec_lift_m2_m1 = ((m2_prec - m1_prec) / m1_prec * 100) if m1_prec > 0 else 0.0
        
        
        m0_rec = metrics_k["M0_recall"]
        m1_rec = metrics_k["M1_recall"]
        m2_rec = metrics_k["M2_recall"]
        
        rec_lift_m1_m0 = ((m1_rec - m0_rec) / m0_rec * 100) if m0_rec > 0 else 0.0
        rec_lift_m2_m0 = ((m2_rec - m0_rec) / m0_rec * 100) if m0_rec > 0 else 0.0
        rec_lift_m2_m1 = ((m2_rec - m1_rec) / m1_rec * 100) if m1_rec > 0 else 0.0
        
        triage_results[f"lift_m2_over_m0_ndcg@{k}"] = ndcg_lift_m2_m0
        triage_results[f"lift_m2_over_m1_ndcg@{k}"] = ndcg_lift_m2_m1
        triage_results[f"lift_m1_over_m0_ndcg@{k}"] = ndcg_lift_m1_m0
        triage_results[f"lift_m2_over_m0_recall@{k}"] = rec_lift_m2_m0
        triage_results[f"lift_m2_over_m0_precision@{k}"] = prec_lift_m2_m0
        
        records.append({
            "K": k,
            "metric": "NDCG@K",
            "M0": round(m0_ndcg, 4),
            "M1": round(m1_ndcg, 4),
            "M2": round(m2_ndcg, 4),
            "Delta_M2_M0": round(m2_ndcg - m0_ndcg, 4),
            "Lift_M2_over_M0_pct": round(ndcg_lift_m2_m0, 2),
            "Lift_M2_over_M1_pct": round(ndcg_lift_m2_m1, 2)
        })
        records.append({
            "K": k,
            "metric": "Precision@K",
            "M0": round(m0_prec, 4),
            "M1": round(m1_prec, 4),
            "M2": round(m2_prec, 4),
            "Delta_M2_M0": round(m2_prec - m0_prec, 4),
            "Lift_M2_over_M0_pct": round(prec_lift_m2_m0, 2),
            "Lift_M2_over_M1_pct": round(prec_lift_m2_m1, 2)
        })
        records.append({
            "K": k,
            "metric": "Recall@K",
            "M0": round(m0_rec, 4),
            "M1": round(m1_rec, 4),
            "M2": round(m2_rec, 4),
            "Delta_M2_M0": round(m2_rec - m0_rec, 4),
            "Lift_M2_over_M0_pct": round(rec_lift_m2_m0, 2),
            "Lift_M2_over_M1_pct": round(rec_lift_m2_m1, 2)
        })
        
    triage_df = pd.DataFrame(records)
    csv_path = os.path.join(output_dir, "experiment3_triage.csv")
    triage_df.to_csv(csv_path, index=False)
    
    return triage_results, triage_df
