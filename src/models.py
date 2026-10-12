import os
import yaml
from typing import Dict, Any, List
import pandas as pd
import numpy as np
from interpret.glassbox import ExplainableBoostingRegressor
import logging

logger = logging.getLogger("ContextBench.ML.Models")

CONFIG_FILE = os.path.join(os.path.dirname(__file__), "..", "config", "model_config.yaml")


def load_model_config() -> Dict:
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {
        "model": {
            "hyperparameters": {
                "max_bins": 32,
                "interactions": 0,
                "outer_bags": 8,
                "learning_rate": 0.03,
                "max_rounds": 2000,
                "random_state": 42,
            }
        }
    }


class EBMModelWrapper:
    

    def __init__(self, feature_names: List[str], random_state: int = 42, **kwargs):
        self.feature_names = list(feature_names)
        self.random_state = random_state
        
        cfg = load_model_config()
        params = dict(cfg["model"]["hyperparameters"])
        params["random_state"] = random_state
        params.update(kwargs)
        
        self.params = params
        self.model = ExplainableBoostingRegressor(
            feature_names=self.feature_names,
            **self.params
        )
        self.is_fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "EBMModelWrapper":
        
        X_in = X[self.feature_names]
        self.model.fit(X_in, y)
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        
        assert self.is_fitted, "Model must be fitted before predict"
        X_in = X[self.feature_names]
        return self.model.predict(X_in)

    def explain_global(self):
        
        assert self.is_fitted, "Model must be fitted before explain_global"
        return self.model.explain_global()

    def explain_local(self, X: pd.DataFrame):
        
        assert self.is_fitted, "Model must be fitted before explain_local"
        X_in = X[self.feature_names]
        return self.model.explain_local(X_in)

    def get_feature_importances(self) -> Dict[str, float]:
        
        exp = self.explain_global()
        data = exp.data()
        names = data.get("names", self.feature_names)
        scores = data.get("scores", [])
        return dict(zip(names, scores))
