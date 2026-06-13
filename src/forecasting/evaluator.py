#!/usr/bin/env python3
"""
Evaluator for Phase 9.
Calculates MAE, RMSE, R2, and MAPE for time-series forecasting regression.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculates Mean Absolute Percentage Error (MAPE) robustly.
    Handles zeros in y_true by filtering them out.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    mask = y_true != 0
    if np.sum(mask) == 0:
        return 0.0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)

def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Evaluates y_true vs y_pred using MAE, RMSE, R2, and MAPE.
    
    Returns:
        A dictionary containing the calculated metrics.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = calculate_mape(y_true, y_pred)
    
    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
        "mape": float(mape)
    }
