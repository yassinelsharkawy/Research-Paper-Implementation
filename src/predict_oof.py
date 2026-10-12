import os
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np
import logging

from .feature_groups import get_feature_groups
from .split_strategy import get_stratified_group_folds, audit_splits
from .preprocessing import FoldPreprocessor
from .models import EBMModelWrapper

logger = logging.getLogger("ContextBench.ML.PredictOOF")


def run_oof_cross_validation(
    df: pd.DataFrame,
    seed: int = 42,
    output_dir: str = None,
    target_col: str = "gt_analyst_priority_score",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    
    logger.info(f"=== Starting 5-Fold OOF Cross-Validation (Seed: {seed}) ===")
    
    m0_features, m1_features, m2_features, _, _ = get_feature_groups()
    folds = get_stratified_group_folds(df, n_splits=5, seed=seed)
    
    
    if output_dir:
        audit_splits(df, folds, output_dir=output_dir, target_col=target_col)

    oof_records = []
    fitted_models = {"M0": [], "M1": [], "M2": []}
    preprocessors = {"M0": [], "M1": [], "M2": []}

    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        logger.info(f"Training Fold {fold_idx + 1}/5 (Train N={len(train_idx)}, Val N={len(val_idx)})...")
        
        train_df = df.iloc[train_idx].copy()
        val_df = df.iloc[val_idx].copy()
        y_train = train_df[target_col]

        
        prep_m0 = FoldPreprocessor(m0_features).fit(train_df)
        X_train_m0 = prep_m0.transform(train_df)
        X_val_m0 = prep_m0.transform(val_df)
        
        ebm_m0 = EBMModelWrapper(m0_features, random_state=seed)
        ebm_m0.fit(X_train_m0, y_train)
        pred_m0 = ebm_m0.predict(X_val_m0)
        
        fitted_models["M0"].append(ebm_m0)
        preprocessors["M0"].append(prep_m0)

        
        prep_m1 = FoldPreprocessor(m1_features).fit(train_df)
        X_train_m1 = prep_m1.transform(train_df)
        X_val_m1 = prep_m1.transform(val_df)
        
        ebm_m1 = EBMModelWrapper(m1_features, random_state=seed)
        ebm_m1.fit(X_train_m1, y_train)
        pred_m1 = ebm_m1.predict(X_val_m1)
        
        fitted_models["M1"].append(ebm_m1)
        preprocessors["M1"].append(prep_m1)

        
        prep_m2 = FoldPreprocessor(m2_features).fit(train_df)
        X_train_m2 = prep_m2.transform(train_df)
        X_val_m2 = prep_m2.transform(val_df)
        
        ebm_m2 = EBMModelWrapper(m2_features, random_state=seed)
        ebm_m2.fit(X_train_m2, y_train)
        pred_m2 = ebm_m2.predict(X_val_m2)
        
        fitted_models["M2"].append(ebm_m2)
        preprocessors["M2"].append(prep_m2)

        
        for i, idx in enumerate(val_idx):
            row = df.iloc[idx]
            oof_records.append({
                "sample_id": row["sample_id"],
                "pair_id": row["pair_id"],
                "policy_lineage_id": row["policy_lineage_id"],
                "scenario_family": row["scenario_family"],
                "context_type": row["context_type"],
                "pair_type": row["pair_type"],
                "gt_analyst_priority_score": float(row[target_col]),
                "gt_analyst_priority_label": row.get("gt_analyst_priority_label", "LOW"),
                "M0_prediction": float(pred_m0[i]),
                "M1_prediction": float(pred_m1[i]),
                "M2_prediction": float(pred_m2[i]),
                "fold_id": fold_idx,
                "model_seed": seed,
            })

    oof_df = pd.DataFrame(oof_records)
    assert len(oof_df) == len(df), f"Expected {len(df)} OOF predictions, got {len(oof_df)}"
    assert oof_df["sample_id"].is_unique, "Duplicate samples in OOF predictions"

    
    if output_dir:
        pred_dir = os.path.join(output_dir, "results", "predictions")
        os.makedirs(pred_dir, exist_ok=True)
        fname = "oof_predictions.csv" if seed == 42 else f"oof_predictions_seed_{seed}.csv"
        csv_path = os.path.join(pred_dir, fname)
        oof_df.to_csv(csv_path, index=False)
        logger.info(f"OOF predictions saved to: {csv_path}")

    return oof_df, {
        "fitted_models": fitted_models,
        "preprocessors": preprocessors,
        "folds": folds,
        "seed": seed,
    }
