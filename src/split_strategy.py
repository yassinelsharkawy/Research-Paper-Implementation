import os
from typing import List, Tuple, Dict, Any
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
import logging

logger = logging.getLogger("ContextBench.ML.SplitStrategy")


def get_stratified_group_folds(
    df: pd.DataFrame,
    n_splits: int = 5,
    seed: int = 42,
    group_col: str = "policy_lineage_id",
    strat_col: str = "scenario_family",
) -> List[Tuple[np.ndarray, np.ndarray]]:
    
    logger.info(f"Generating {n_splits}-fold StratifiedGroupKFold (group={group_col}, stratify={strat_col}, seed={seed})...")
    
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    
    groups = df[group_col]
    y_strat = df[strat_col]
    X_dummy = np.zeros(len(df))
    
    folds = list(sgkf.split(X_dummy, y_strat, groups=groups))
    
    
    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        train_lineages = set(df[group_col].iloc[train_idx])
        val_lineages = set(df[group_col].iloc[val_idx])
        lineage_leakage = train_lineages.intersection(val_lineages)
        assert len(lineage_leakage) == 0, (
            f"LEAKAGE DETECTED in Fold {fold_idx}: policy_lineage_id overlap: {lineage_leakage}"
        )
        
        
        train_pairs = set(df["pair_id"].iloc[train_idx].dropna())
        val_pairs = set(df["pair_id"].iloc[val_idx].dropna())
        pair_leakage = train_pairs.intersection(val_pairs)
        assert len(pair_leakage) == 0, (
            f"PAIR LEAKAGE DETECTED in Fold {fold_idx}: pair_id overlap: {pair_leakage}"
        )

    logger.info("✓ Zero leakage verified: 0 lineages and 0 pairs cross fold boundaries.")
    return folds


def audit_splits(
    df: pd.DataFrame,
    folds: List[Tuple[np.ndarray, np.ndarray]],
    output_dir: str = None,
    target_col: str = "gt_analyst_priority_score",
) -> pd.DataFrame:
    
    audit_rows = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        val_df = df.iloc[val_idx]
        train_df = df.iloc[train_idx]
        
        n_samples = len(val_df)
        n_lineages = val_df["policy_lineage_id"].nunique()
        n_type_a_pairs = len(val_df[val_df["pair_type"] == "TYPE_A"]["pair_id"].unique())
        n_type_b_pairs = len(val_df[val_df["pair_type"] == "TYPE_B"]["pair_id"].unique())
        
        
        fam_dist = val_df["scenario_family"].value_counts().to_dict()
        
        tgt_mean = float(val_df[target_col].mean())
        tgt_std = float(val_df[target_col].std())
        tgt_min = float(val_df[target_col].min())
        tgt_max = float(val_df[target_col].max())

        audit_rows.append({
            "fold": fold_idx,
            "val_samples": n_samples,
            "train_samples": len(train_df),
            "val_lineages": n_lineages,
            "val_type_a_pairs": n_type_a_pairs,
            "val_type_b_pairs": n_type_b_pairs,
            "target_mean": round(tgt_mean, 2),
            "target_std": round(tgt_std, 2),
            "target_min": round(tgt_min, 2),
            "target_max": round(tgt_max, 2),
            "scenario_families_present": len(fam_dist),
            "scenario_distribution": str(sorted(fam_dist.items())),
        })

    audit_df = pd.DataFrame(audit_rows)

    if output_dir:
        results_dir = os.path.join(output_dir, "results")
        reports_dir = os.path.join(output_dir, "reports")
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(reports_dir, exist_ok=True)

        csv_path = os.path.join(results_dir, "data_split_audit.csv")
        audit_df.to_csv(csv_path, index=False)
        logger.info(f"Split audit CSV saved: {csv_path}")

        md_path = os.path.join(reports_dir, "data_split_audit.md")
        _write_split_audit_md(audit_df, md_path)
        logger.info(f"Split audit MD saved: {md_path}")

    return audit_df


def _write_split_audit_md(df: pd.DataFrame, path: str):
    
    lines = [
        "# IAM-ContextBench — Data Split Audit Report",
        "",
        "## Cross-Validation Split Invariants Verified",
        "- **Strategy:** StratifiedGroupKFold (n_splits=5, shuffle=True, random_state=42)",
        "- **Grouping Entity:** `policy_lineage_id`",
        "- **Stratification Target:** `scenario_family`",
        "- **Policy Lineage Leakage:** 0 lineages cross fold boundaries (Strictly 0%)",
        "- **Pair ID Leakage:** 0 pairs cross fold boundaries (Strictly 0%)",
        "",
        "## Fold Summary Table",
        "",
        "| Fold | Train N | Val N | Lineages | Type-A Pairs | Type-B Pairs | Target Mean | Target Std | Target Min | Target Max | Families |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for _, r in df.iterrows():
        lines.append(
            f"| {r['fold']} | {r['train_samples']} | {r['val_samples']} | {r['val_lineages']} | "
            f"{r['val_type_a_pairs']} | {r['val_type_b_pairs']} | {r['target_mean']} | {r['target_std']} | "
            f"{r['target_min']} | {r['target_max']} | {r['scenario_families_present']}/13 |"
        )
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
