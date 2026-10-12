import os
from typing import Tuple, Dict, Any
import pandas as pd
import logging
from .schema import validate_dataset_integrity

logger = logging.getLogger("ContextBench.ML.DataLoader")

DEFAULT_DATASET_PATHS = [
    os.path.join(os.path.dirname(__file__), "..", "..", "dataset_generator", "output", "aws_iam_contextbench_real_valid.csv"),
    os.path.join(os.path.dirname(__file__), "..", "dataset_generator", "output", "aws_iam_contextbench_real_valid.csv"),
    r"d:\Research Paper\Research Paper Dataset\dataset_generator\output\aws_iam_contextbench_real_valid.csv",
]


def find_dataset_path(custom_path: str = None) -> str:
    
    if custom_path and os.path.exists(custom_path):
        return custom_path
    for p in DEFAULT_DATASET_PATHS:
        abs_p = os.path.abspath(p)
        if os.path.exists(abs_p):
            return abs_p
    raise FileNotFoundError(
        f"Could not locate aws_iam_contextbench_real_valid.csv in standard locations: {DEFAULT_DATASET_PATHS}"
    )


def load_dataset(dataset_path: str = None) -> Tuple[pd.DataFrame, str]:
    
    path = find_dataset_path(dataset_path)
    logger.info(f"Loading dataset from: {path}")
    
    df = pd.read_csv(path)
    validate_dataset_integrity(df)
    
    return df, path

