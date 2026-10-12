import os
import yaml
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger("ContextBench.ML.FeatureGroups")

CONFIG_FILE = os.path.join(os.path.dirname(__file__), "..", "config", "feature_groups.yaml")


def load_feature_config() -> Dict:
    
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    raise FileNotFoundError(f"Feature config not found at: {CONFIG_FILE}")


def get_feature_groups() -> Tuple[List[str], List[str], List[str], List[str], Dict[str, str]]:
    
    cfg = load_feature_config()
    raw_m0 = cfg["groups"]["M0"]
    raw_m1 = cfg["groups"]["M1"]
    raw_m2 = cfg["groups"]["M2"]
    forbidden = cfg.get("forbidden", [])
    excluded = cfg.get("excluded_zero_variance", {})

    
    m0_features = [f for f in raw_m0 if f not in excluded and f not in forbidden]
    
    
    m1_features = list(m0_features) + [f for f in raw_m1 if f not in excluded and f not in forbidden]
    
    
    m2_features = list(m1_features) + [f for f in raw_m2 if f not in excluded and f not in forbidden]

    return m0_features, m1_features, m2_features, forbidden, excluded


def audit_features(df: pd.DataFrame, output_dir: str = None) -> pd.DataFrame:
    
    m0_features, m1_features, m2_features, forbidden, excluded = get_feature_groups()
    cfg = load_feature_config()
    candidate_all = list(cfg["groups"]["M0"]) + list(cfg["groups"]["M1"]) + list(cfg["groups"]["M2"])

    records = []
    
    
    for col in candidate_all:
        if col not in df.columns:
            records.append({
                "feature": col,
                "group": "UNKNOWN",
                "dtype": "MISSING",
                "missing_rate": 1.0,
                "unique_values": 0,
                "variance": 0.0,
                "included_M0": False,
                "included_M1": False,
                "included_M2": False,
                "reason_if_excluded": "Column missing from dataset",
            })
            continue

        series = df[col]
        dtype_str = str(series.dtype)
        n_missing = series.isna().sum()
        miss_rate = float(n_missing / len(df))
        n_unique = int(series.nunique(dropna=False))
        
        
        if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
            var_val = float(series.astype(float).var())
        else:
            var_val = np.nan

        
        if col in cfg["groups"]["M0"]:
            grp = "M0 (Policy)"
        elif col in cfg["groups"]["M1"]:
            grp = "M1 (Environment)"
        elif col in cfg["groups"]["M2"]:
            grp = "M2 (CloudTrail Context)"
        else:
            grp = "Other"

        inc_m0 = col in m0_features
        inc_m1 = col in m1_features
        inc_m2 = col in m2_features

        if col in excluded:
            reason = excluded[col]
        elif col in forbidden:
            reason = "Forbidden metadata/provenance field"
        elif not inc_m2:
            reason = "Not assigned to active feature set"
        else:
            reason = "Included"

        records.append({
            "feature": col,
            "group": grp,
            "dtype": dtype_str,
            "missing_rate": round(miss_rate, 4),
            "unique_values": n_unique,
            "variance": round(var_val, 4) if pd.notna(var_val) else "N/A",
            "included_M0": inc_m0,
            "included_M1": inc_m1,
            "included_M2": inc_m2,
            "reason_if_excluded": reason,
        })

    
    for col in forbidden:
        if col in df.columns:
            series = df[col]
            records.append({
                "feature": col,
                "group": "FORBIDDEN_METADATA",
                "dtype": str(series.dtype),
                "missing_rate": round(float(series.isna().sum() / len(df)), 4),
                "unique_values": int(series.nunique(dropna=False)),
                "variance": "N/A",
                "included_M0": False,
                "included_M1": False,
                "included_M2": False,
                "reason_if_excluded": "Strictly forbidden metadata/provenance/target leakage",
            })

    audit_df = pd.DataFrame(records)

    if output_dir:
        results_dir = os.path.join(output_dir, "results")
        reports_dir = os.path.join(output_dir, "reports")
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(reports_dir, exist_ok=True)

        csv_path = os.path.join(results_dir, "feature_audit.csv")
        audit_df.to_csv(csv_path, index=False)
        logger.info(f"Feature audit CSV saved: {csv_path}")

        md_path = os.path.join(reports_dir, "feature_audit.md")
        _write_feature_audit_md(audit_df, md_path, len(m0_features), len(m1_features), len(m2_features))
        logger.info(f"Feature audit MD saved: {md_path}")

    return audit_df


def _write_feature_audit_md(df: pd.DataFrame, path: str, n_m0: int, n_m1: int, n_m2: int):
    
    lines = [
        "# IAM-ContextBench — Feature Quality & Leakage Audit Report",
        "",
        f"**Active Feature Set Sizes:**",
        f"- **M0 (Policy-centric):** {n_m0} features",
        f"- **M1 (Policy + Environment):** {n_m1} features (+{n_m1 - n_m0} environment features)",
        f"- **M2 (Policy + Environment + CloudTrail):** {n_m2} features (+{n_m2 - n_m1} behavioral context features)",
        "",
        "## Feature Audit Table",
        "",
        "| Feature | Group | Dtype | Missing Rate | Unique | Variance | In M0 | In M1 | In M2 | Exclusion Reason |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for _, r in df.iterrows():
        lines.append(
            f"| `{r['feature']}` | {r['group']} | {r['dtype']} | {r['missing_rate']} | "
            f"{r['unique_values']} | {r['variance']} | {r['included_M0']} | {r['included_M1']} | "
            f"{r['included_M2']} | {r['reason_if_excluded']} |"
        )
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
