#!/usr/bin/env python3
"""
XGBoost Forecaster for Phase 9.
Implements the XGBoost Regressor model for time-series forecasting.
Includes native JSON serialization to avoid pickle segmentation faults.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import pandas as pd
import numpy as np
import xgboost as xgb

class XGBoostForecaster:
    """
    Wrapper around xgboost's XGBRegressor with JSON-safe saving and loading.
    """
    def __init__(self, n_estimators: int = 100, max_depth: int = 6, learning_rate: float = 0.05, random_state: int = 42):
        self.model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
            n_jobs=-1
        )
        
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Fits the XGBoost model.
        """
        X_numeric = X.select_dtypes(include=[np.number])
        self.model.fit(X_numeric, y)
        return self
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predicts using the fitted model.
        """
        X_numeric = X.select_dtypes(include=[np.number])
        return self.model.predict(X_numeric)
        
    def save_model(self, filepath: str):
        """
        Saves the underlying booster to a JSON format.
        
        Args:
            filepath: Path to save the model. Should end with '.json'.
        """
        # Ensure it has .json extension
        if not filepath.endswith(".json"):
            filepath = filepath + ".json"
        self.model.save_model(filepath)
        print(f"[+] Saved XGBoost model natively to {filepath}")
        
    def load_model(self, filepath: str):
        """
        Loads the underlying booster from a JSON format.
        
        Args:
            filepath: Path to load from.
        """
        if not filepath.endswith(".json"):
            filepath = filepath + ".json"
        self.model.load_model(filepath)
        print(f"[+] Loaded XGBoost model from {filepath}")
        return self
