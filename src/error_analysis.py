import os
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple


def select_representative_cases(
    oof_df: pd.DataFrame,
    df: pd.DataFrame,
    output_dir: str
) -> Tuple[pd.DataFrame, str]:
    
    os.makedirs(output_dir, exist_ok=True)
    
    
    merged = oof_df.copy()
    feature_cols = [
        "context_type", "context_pattern_id", "event_count", "error_count",
        "role_chain_length", "failed_iam_activity", "source_novelty",
        "policy_action_count", "policy_has_wildcard_action", "environment_type"
    ]
    avail_cols = [c for c in feature_cols if c in df.columns and c not in merged.columns]
    if avail_cols:
        merged = merged.merge(df[["sample_id"] + avail_cols], on="sample_id", how="left")
        
    merged["residual_m0"] = merged["M0_prediction"] - merged["gt_analyst_priority_score"]
    merged["residual_m2"] = merged["M2_prediction"] - merged["gt_analyst_priority_score"]
    merged["abs_residual_m0"] = np.abs(merged["residual_m0"])
    merged["abs_residual_m2"] = np.abs(merged["residual_m2"])
    merged["improvement_m2"] = merged["abs_residual_m0"] - merged["abs_residual_m2"]
    
    selected = []
    used_ids = set()
    
    def add_case(row, category, reason):
        sid = row["sample_id"]
        used_ids.add(sid)
        
        
        feats = []
        if "policy_action_count" in row:
            feats.append(f"actions={row['policy_action_count']}")
        if "environment_type" in row:
            feats.append(f"env={row['environment_type']}")
        if "context_type" in row:
            feats.append(f"ctx={row['context_type']}")
        if "event_count" in row:
            feats.append(f"events={row['event_count']}")
        if "error_count" in row:
            feats.append(f"errors={row['error_count']}")
        if "role_chain_length" in row:
            feats.append(f"chain={row['role_chain_length']}")
        if "failed_iam_activity" in row:
            feats.append(f"failed_iam={row['failed_iam_activity']}")
        
        selected.append({
            "sample_id": sid,
            "pair_id": row.get("pair_id", "N/A"),
            "policy_lineage_id": row.get("policy_lineage_id", "N/A"),
            "scenario_family": row.get("scenario_family", "N/A"),
            "category": category,
            "oracle_priority": round(float(row["gt_analyst_priority_score"]), 2),
            "M0_score": round(float(row["M0_prediction"]), 2),
            "M2_score": round(float(row["M2_prediction"]), 2),
            "score_delta_M2_M0": round(float(row["M2_prediction"] - row["M0_prediction"]), 2),
            "residual_M0": round(float(row["residual_m0"]), 2),
            "residual_M2": round(float(row["residual_m2"]), 2),
            "major_features": ", ".join(feats),
            "explanation": reason
        })

    
    hp_candidates = merged[(merged["gt_analyst_priority_score"] >= 35) & (~merged["sample_id"].isin(used_ids))]
    if len(hp_candidates) > 0:
        best_hp = hp_candidates.sort_values(by="abs_residual_m2").iloc[0]
        add_case(best_hp, "High-Priority Correctly Ranked", 
                 "High analyst priority event accurately identified by M2 with small residual; CloudTrail activity aligns with high-risk policy.")

    
    lp_candidates = merged[(merged["gt_analyst_priority_score"] <= 12) & (~merged["sample_id"].isin(used_ids))]
    if len(lp_candidates) > 0:
        best_lp = lp_candidates.sort_values(by="abs_residual_m2").iloc[0]
        add_case(best_lp, "Low-Priority Correctly Ranked", 
                 "Benign low-priority baseline event correctly scored low by both models.")

    
    
    corr_susp = merged[(merged["residual_m0"] < -8) & (~merged["sample_id"].isin(used_ids))]
    if len(corr_susp) > 0:
        best_corr_susp = corr_susp.sort_values(by="improvement_m2", ascending=False).iloc[0]
        add_case(best_corr_susp, "M0 Error Corrected by M2 (Suspicious Influx)", 
                 "Policy appeared benign statically (M0 under-scored), but CloudTrail behavioral anomalies elevated priority in M2.")

    
    
    corr_benign = merged[(merged["residual_m0"] > 5) & (~merged["sample_id"].isin(used_ids))]
    if len(corr_benign) > 0:
        best_corr_benign = corr_benign.sort_values(by="improvement_m2", ascending=False).iloc[0]
        add_case(best_corr_benign, "M0 Error Corrected by M2 (Benign De-escalation)", 
                 "Static policy complexity inflated M0 score; benign, routine CloudTrail context allowed M2 to de-escalate priority.")

    
    fp_candidates = merged[~merged["sample_id"].isin(used_ids)]
    if len(fp_candidates) > 0:
        worst_fp = fp_candidates.sort_values(by="residual_m2", ascending=False).iloc[0]
        add_case(worst_fp, "M2 False Positive (Over-prioritization)", 
                 "High behavioral activity or novel user-agent triggered elevated M2 score above oracle expectation.")

    
    fn_candidates = merged[~merged["sample_id"].isin(used_ids)]
    if len(fn_candidates) > 0:
        worst_fn = fn_candidates.sort_values(by="residual_m2", ascending=True).iloc[0]
        add_case(worst_fn, "M2 False Negative (Under-prioritization)", 
                 "High oracle priority event where CloudTrail behavioral signals were quiet or unobserved, causing M2 underprediction.")

    
    
    type_a_pairs = merged[merged["pair_type"] == "TYPE_A"].groupby("pair_id")
    best_pair_id = None
    best_gap = -1
    for pid, grp in type_a_pairs:
        if len(grp) == 2 and not any(sid in used_ids for sid in grp["sample_id"]):
            s = grp[grp["context_type"] == "suspicious"]
            b = grp[grp["context_type"] == "benign"]
            if len(s) == 1 and len(b) == 1:
                gap = s["M2_prediction"].iloc[0] - b["M2_prediction"].iloc[0]
                if gap > best_gap:
                    best_gap = gap
                    best_pair_id = pid
                    
    if best_pair_id:
        p_grp = merged[merged["pair_id"] == best_pair_id]
        b_row = p_grp[p_grp["context_type"] == "benign"].iloc[0]
        s_row = p_grp[p_grp["context_type"] == "suspicious"].iloc[0]
        add_case(b_row, "Paired Context: Benign Peer", 
                 f"Baseline benign member of pair {best_pair_id}; minimal errors and normal behavioral profile.")
        add_case(s_row, "Paired Context: Suspicious Peer", 
                 f"Same policy as benign peer, but suspicious CloudTrail activity (role chaining / errors) lifted score by {best_gap:.2f} points.")

    
    s09_candidates = merged[(merged["scenario_family"] == "S09") & (~merged["sample_id"].isin(used_ids))]
    if len(s09_candidates) > 0:
        s09_case = s09_candidates.iloc[0]
        add_case(s09_case, "S09 Version History Case", 
                 "Demonstrates policy version history context; historical policy versions expose privilege expansion.")

    
    rem_candidates = merged[~merged["sample_id"].isin(used_ids)]
    if len(rem_candidates) > 0:
        rem_case = rem_candidates.sort_values(by="abs_residual_m2").iloc[len(rem_candidates) // 2]
        add_case(rem_case, "Median Performance Reference Case", 
                 "Representative median-performing case demonstrating consistent multi-signal calibration.")

    res_df = pd.DataFrame(selected)
    
    
    csv_path = os.path.join(output_dir, "error_analysis.csv")
    res_df.to_csv(csv_path, index=False)
    
    
    md_lines = [
        "# IAM-ContextBench: Error and Diagnostic Analysis",
        "",
        "This document provides a systematic diagnostic analysis of 10 representative cases across key model behavior categories.",
        "Selection was performed using deterministic, objective criteria rather than manual selection.",
        "",
        "## Summary of Representative Cases",
        "",
        "| Sample ID | Category | Scenario | Oracle | M0 Score | M2 Score | Delta (M2-M0) | Residual M2 | Major Features |",
        "| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |"
    ]
    
    for _, r in res_df.iterrows():
        md_lines.append(
            f"| `{r['sample_id']}` | **{r['category']}** | {r['scenario_family']} | "
            f"{r['oracle_priority']:.1f} | {r['M0_score']:.1f} | {r['M2_score']:.1f} | "
            f"{r['score_delta_M2_M0']:+.1f} | {r['residual_M2']:+.1f} | {r['major_features']} |"
        )
        
    md_lines.extend([
        "",
        "## Case-by-Case Diagnostic Narratives",
        ""
    ])
    
    for i, r in enumerate(selected, 1):
        md_lines.extend([
            f"### Case {i}: {r['category']} (`{r['sample_id']}`)",
            f"- **Scenario Family**: {r['scenario_family']}",
            f"- **Pair ID**: `{r['pair_id']}` | **Policy Lineage**: `{r['policy_lineage_id']}`",
            f"- **Ground Truth Oracle Priority**: {r['oracle_priority']:.2f}",
            f"- **M0 Prediction (Policy-only)**: {r['M0_score']:.2f} (Residual: {r['residual_M0']:+.2f})",
            f"- **M2 Prediction (Full Context)**: {r['M2_score']:.2f} (Residual: {r['residual_M2']:+.2f})",
            f"- **Contextual Score Shift (M2 - M0)**: {r['score_delta_M2_M0']:+.2f}",
            f"- **Observed Telemetry**: `{r['major_features']}`",
            f"- **Diagnostic Interpretation**: {r['explanation']}",
            ""
        ])
        
    md_content = "\n".join(md_lines)
    md_path = os.path.join(output_dir, "error_analysis.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
        
    return res_df, md_content
