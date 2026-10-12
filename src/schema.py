from typing import Dict, Any, List
import pandas as pd
import logging

logger = logging.getLogger("ContextBench.ML.Schema")

EXPECTED_ROW_COUNT = 534
EXPECTED_TYPE_A_PAIRS = 208
EXPECTED_TYPE_B_PAIRS = 39
EXPECTED_UNPAIRED = 40
EXPECTED_SCENARIOS = 13


def validate_dataset_integrity(df: pd.DataFrame) -> Dict[str, Any]:
    
    logger.info("Running dataset schema and integrity validation...")
    
    
    actual_rows = len(df)
    assert actual_rows == EXPECTED_ROW_COUNT, (
        f"Expected {EXPECTED_ROW_COUNT} samples, found {actual_rows}"
    )
    
    
    assert df["sample_id"].is_unique, "Duplicate sample_id found in dataset"
    assert df["experiment_id"].is_unique, "Duplicate experiment_id found in dataset"
    
    
    assert "gt_analyst_priority_score" in df.columns, "Missing primary target gt_analyst_priority_score"
    assert df["gt_analyst_priority_score"].notna().all(), "Null target values found"
    
    
    type_a_rows = len(df[df["pair_type"] == "TYPE_A"])
    type_b_rows = len(df[df["pair_type"] == "TYPE_B"])
    unpaired_rows = len(df[df["pair_type"] == "UNPAIRED"])
    
    assert type_a_rows == EXPECTED_TYPE_A_PAIRS * 2, (
        f"Expected {EXPECTED_TYPE_A_PAIRS * 2} Type-A rows, got {type_a_rows}"
    )
    assert type_b_rows == EXPECTED_TYPE_B_PAIRS * 2, (
        f"Expected {EXPECTED_TYPE_B_PAIRS * 2} Type-B rows, got {type_b_rows}"
    )
    assert unpaired_rows == EXPECTED_UNPAIRED, (
        f"Expected {EXPECTED_UNPAIRED} unpaired rows, got {unpaired_rows}"
    )
    
    
    families = df["scenario_family"].unique()
    assert len(families) == EXPECTED_SCENARIOS, (
        f"Expected {EXPECTED_SCENARIOS} scenario families, got {len(families)}"
    )
    
    
    num_lineages = df["policy_lineage_id"].nunique()
    assert num_lineages >= 80, f"Expected >= 80 policy lineages, found {num_lineages}"
    
    logger.info("[OK] Dataset schema and integrity verification PASSED.")
    return {
        "total_samples": actual_rows,
        "type_a_pairs": EXPECTED_TYPE_A_PAIRS,
        "n_type_a_pairs": EXPECTED_TYPE_A_PAIRS,
        "type_b_pairs": EXPECTED_TYPE_B_PAIRS,
        "n_type_b_pairs": EXPECTED_TYPE_B_PAIRS,
        "unpaired_samples": unpaired_rows,
        "n_unpaired": unpaired_rows,
        "scenario_families": len(families),
        "n_scenario_families": len(families),
        "policy_lineages": num_lineages,
        "n_policy_lineages": num_lineages,
        "status": "VALID",
    }



validate_dataset_invariants = validate_dataset_integrity

