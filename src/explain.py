import os
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from .models import EBMModelWrapper
from .preprocessing import FoldPreprocessor


def extract_global_explanations(
    ebm_wrapper: EBMModelWrapper,
    feature_names: List[str],
    model_name: str,
    output_dir: str
) -> pd.DataFrame:
    
    os.makedirs(output_dir, exist_ok=True)
    ebm = ebm_wrapper.model
    global_exp = ebm.explain_global()
    data = global_exp.data()
    
    names = data.get("names", feature_names)
    scores = data.get("scores", [])
    
    df = pd.DataFrame({
        "feature": names,
        "importance_score": [float(s) for s in scores]
    }).sort_values(by="importance_score", ascending=False).reset_index(drop=True)
    
    csv_path = os.path.join(output_dir, f"{model_name.lower()}_global_importance.csv")
    df.to_csv(csv_path, index=False)
    return df


def explain_local_sample(
    ebm_wrapper: EBMModelWrapper,
    X_sample_df: pd.DataFrame,
    feature_groups_map: Dict[str, str]
) -> pd.DataFrame:
    
    ebm = ebm_wrapper.model
    terms = ebm.eval_terms(X_sample_df)[0]
    intercept = float(ebm.intercept_)
    
    records = []
    for f_name, term_val in zip(ebm_wrapper.feature_names, terms):
        raw_val = X_sample_df[f_name].iloc[0]
        group = feature_groups_map.get(f_name, "other")
        records.append({
            "feature": f_name,
            "feature_group": group,
            "raw_value": raw_val,
            "score_contribution": float(term_val)
        })
        
    df_terms = pd.DataFrame(records)
    return df_terms, intercept


def explain_paired_case(
    ebm_m2: EBMModelWrapper,
    df: pd.DataFrame,
    preprocessor: FoldPreprocessor,
    pair_id: str,
    feature_groups_map: Dict[str, str],
    output_dir: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    
    os.makedirs(output_dir, exist_ok=True)
    pair_df = df[df["pair_id"] == pair_id]
    if len(pair_df) != 2:
        raise ValueError(f"Pair {pair_id} does not contain exactly 2 samples.")
        
    benign_raw = pair_df[pair_df["context_type"] == "benign"]
    susp_raw = pair_df[pair_df["context_type"] == "suspicious"]
    
    if len(benign_raw) != 1 or len(susp_raw) != 1:
        raise ValueError(f"Pair {pair_id} must have 1 benign and 1 suspicious member.")
        
    
    benign_X = preprocessor.transform(benign_raw)
    susp_X = preprocessor.transform(susp_raw)
    
    terms_b, int_b = explain_local_sample(ebm_m2, benign_X, feature_groups_map)
    terms_s, int_s = explain_local_sample(ebm_m2, susp_X, feature_groups_map)
    
    
    detailed = pd.merge(
        terms_b, terms_s,
        on=["feature", "feature_group"],
        suffixes=("_benign", "_suspicious")
    )
    detailed["score_delta_susp_minus_benign"] = (
        detailed["score_contribution_suspicious"] - detailed["score_contribution_benign"]
    )
    detailed = detailed.sort_values(by="score_delta_susp_minus_benign", ascending=False).reset_index(drop=True)
    
    
    groups = ["policy", "environment", "cloudtrail"]
    summary_rows = []
    
    b_pred = float(ebm_m2.predict(benign_X)[0])
    s_pred = float(ebm_m2.predict(susp_X)[0])
    b_gt = float(benign_raw["gt_analyst_priority_score"].iloc[0])
    s_gt = float(susp_raw["gt_analyst_priority_score"].iloc[0])
    
    summary_rows.append({
        "component": "Baseline Intercept",
        "benign_contribution": int_b,
        "suspicious_contribution": int_s,
        "delta (susp - benign)": 0.0
    })
    
    for grp in groups:
        b_grp_sum = float(terms_b[terms_b["feature_group"] == grp]["score_contribution"].sum())
        s_grp_sum = float(terms_s[terms_s["feature_group"] == grp]["score_contribution"].sum())
        summary_rows.append({
            "component": f"{grp.capitalize()} Features Total",
            "benign_contribution": b_grp_sum,
            "suspicious_contribution": s_grp_sum,
            "delta (susp - benign)": s_grp_sum - b_grp_sum
        })
        
    summary_rows.append({
        "component": "Final Predicted Score",
        "benign_contribution": b_pred,
        "suspicious_contribution": s_pred,
        "delta (susp - benign)": s_pred - b_pred
    })
    summary_rows.append({
        "component": "Oracle Ground Truth Priority",
        "benign_contribution": b_gt,
        "suspicious_contribution": s_gt,
        "delta (susp - benign)": s_gt - b_gt
    })
    
    summary_df = pd.DataFrame(summary_rows)
    
    
    detailed.to_csv(os.path.join(output_dir, f"pair_{pair_id}_detailed_explanation.csv"), index=False)
    summary_df.to_csv(os.path.join(output_dir, f"pair_{pair_id}_group_summary.csv"), index=False)
    
    return detailed, summary_df
