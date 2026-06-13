#!/usr/bin/env python3
"""
Baseline Forecaster for Phase 9.
Implements the Persistence Model where Future Risk = Current Risk.
"""

import numpy as np
import pandas as pd

class PersistenceForecaster:
    """
    Persistence model that predicts future target values using the most recent
    observed metrics from the corresponding past window.
    """
    def __init__(self):
        pass
        
    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        """Persistence model requires no training."""
        return self
        
    def predict(self, df: pd.DataFrame, target_name: str, horizon: int) -> np.ndarray:
        """
        Predicts the future values based on past counterpart values.
        
        Args:
            df: The input features DataFrame.
            target_name: The target variable being predicted.
            horizon: The forecasting horizon (30, 60, or 90).
            
        Returns:
            An array of predictions.
        """
        # Map target type to past feature name
        if "risk" in target_name:
            feature_col = f"risk_score_{horizon}d"
        elif "defect" in target_name:
            feature_col = f"defect_count_{horizon}d"
        elif "modification" in target_name:
            feature_col = f"modification_count_{horizon}d"
        else:
            # Fallback based on text match
            if "30" in target_name:
                feature_col = "risk_score_30d"
            elif "60" in target_name:
                feature_col = "risk_score_60d"
            else:
                feature_col = "risk_score_90d"
                
        if feature_col not in df.columns:
            raise KeyError(f"Counterpart feature column '{feature_col}' not found in DataFrame for target '{target_name}'.")
            
        return df[feature_col].fillna(0.0).values
