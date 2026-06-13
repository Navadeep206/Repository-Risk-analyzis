#!/usr/bin/env python3
"""
Model loading and data extraction helpers for explainability.
Excludes all XGBoost imports and model loading to prevent native segfaults.
"""

import os
import sys
import pickle
import pandas as pd
import numpy as np
from typing import Tuple, Any

# Ensure parent directory is in path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import BASE_DIR
from ml.data_loader import load_all_splits
from ml.preprocessing import CodeRiskPreprocessor

def load_explainability_resources() -> Tuple[CodeRiskPreprocessor, Any, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Loads preprocessor, Random Forest model, and raw splits data.
    
    Returns:
        preprocessor: Fitted CodeRiskPreprocessor.
        rf_model: Loaded Random Forest classifier.
        X_train, y_train, X_val, y_val, X_test, y_test: Data splits.
    """
    preproc_path = os.path.join(BASE_DIR, "models", "preprocessor.pkl")
    rf_path = os.path.join(BASE_DIR, "models", "random_forest.pkl")
    
    if not os.path.exists(preproc_path):
        raise FileNotFoundError(f"Preprocessor not found: {preproc_path}")
    if not os.path.exists(rf_path):
        raise FileNotFoundError(f"Random Forest not found: {rf_path}")

    # Load preprocessor
    preprocessor = CodeRiskPreprocessor.load(preproc_path)
    
    # Load RF
    with open(rf_path, "rb") as f:
        rf_model = pickle.load(f)
        
    # Load splits
    X_train, y_train, X_val, y_val, X_test, y_test = load_all_splits()
    
    return preprocessor, rf_model, X_train, y_train, X_val, y_val, X_test, y_test
