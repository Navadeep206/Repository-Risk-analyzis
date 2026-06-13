#!/usr/bin/env python3
"""
Random Forest Forecaster for Phase 9.
Implements the Random Forest Regressor model for time-series forecasting.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

class RandomForestForecaster:
    """
    Wrapper around Scikit-learn's RandomForestRegressor for forecasting repository risk.
    """
    def __init__(self, n_estimators: int = 100, max_depth: int = 8, random_state: int = 42):
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1
        )
        
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Fits the Random Forest model on training features and targets.
        
        Args:
            X: Training features.
            y: Training targets.
            
        Returns:
            Self.
        """
        # Select numeric columns only
        X_numeric = X.select_dtypes(include=[np.number])
        self.model.fit(X_numeric, y)
        return self
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predicts future targets using the fitted model.
        
        Args:
            X: Input features.
            
        Returns:
            An array of predictions.
        """
        X_numeric = X.select_dtypes(include=[np.number])
        return self.model.predict(X_numeric)
