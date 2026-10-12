from typing import List, Tuple, Dict
import pandas as pd
import numpy as np


class FoldPreprocessor:
    

    def __init__(self, feature_names: List[str]):
        self.feature_names = list(feature_names)
        self.numeric_cols = []
        self.bool_cols = []
        self.categorical_cols = []
        self.medians = {}

    def fit(self, X_train: pd.DataFrame) -> "FoldPreprocessor":
        
        X_sub = X_train[self.feature_names].copy()
        
        self.numeric_cols = []
        self.bool_cols = []
        self.categorical_cols = []
        self.medians = {}

        for col in self.feature_names:
            series = X_sub[col]
            if pd.api.types.is_bool_dtype(series):
                self.bool_cols.append(col)
            elif pd.api.types.is_numeric_dtype(series):
                self.numeric_cols.append(col)
                med_val = series.dropna().median()
                self.medians[col] = float(med_val) if pd.notna(med_val) else 0.0
            else:
                self.categorical_cols.append(col)

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        
        X_out = X[self.feature_names].copy()

        
        for col in self.bool_cols:
            X_out[col] = X_out[col].fillna(False).astype(int)

        
        for col in self.numeric_cols:
            med_val = self.medians.get(col, 0.0)
            X_out[col] = X_out[col].fillna(med_val).astype(float)

        
        for col in self.categorical_cols:
            X_out[col] = X_out[col].fillna("MISSING").astype(str)

        return X_out

    def fit_transform(self, X_train: pd.DataFrame) -> pd.DataFrame:
        return self.fit(X_train).transform(X_train)
