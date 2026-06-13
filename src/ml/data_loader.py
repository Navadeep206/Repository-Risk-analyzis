#!/usr/bin/env python3
"""
Data loader for baseline ML pipeline. Loads Train, Validation, and Test splits from CSV.
"""

import os
import pandas as pd
import sys
from typing import Tuple, Dict

# Ensure parent directory is in path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import FINAL_DIR

# Class mapping: LOW=0, MEDIUM=1, HIGH=2
LABEL_MAP: Dict[str, int] = {
    "LOW": 0,
    "MEDIUM": 1,
    "HIGH": 2
}

INV_LABEL_MAP: Dict[int, str] = {v: k for k, v in LABEL_MAP.items()}

def load_split_data(split_name: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Loads a specific dataset split (train, validation, or test).
    
    Args:
        split_name: Name of the split ('train_v2', 'validation_v2', or 'test_v2').
        
    Returns:
        A tuple of (X, y) representing features and targets.
    """
    file_path = os.path.join(FINAL_DIR, f"{split_name}.csv")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Split file does not exist: {file_path}")
        
    df = pd.read_csv(file_path)
    
    # Target label mapping
    if "historical_risk_label" not in df.columns:
        raise ValueError(f"Target column 'historical_risk_label' is missing from split {split_name}.")
        
    y = df["historical_risk_label"].map(LABEL_MAP)
    
    # Features (exclude metadata columns and target)
    exclude_cols = ["repository_name", "file_path", "historical_risk_label"]
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    X = df[feature_cols].copy()
    
    return X, y

def load_all_splits() -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Loads Train, Validation, and Test splits.
    
    Returns:
        A tuple of (X_train, y_train, X_val, y_val, X_test, y_test).
    """
    X_train, y_train = load_split_data("train_v2")
    X_val, y_val = load_split_data("validation_v2")
    X_test, y_test = load_split_data("test_v2")
    
    return X_train, y_train, X_val, y_val, X_test, y_test

if __name__ == "__main__":
    X_tr, y_tr, X_va, y_va, X_te, y_te = load_all_splits()
    print(f"[+] Data loaded successfully.")
    print(f"Train features: {X_tr.shape}, Target: {y_tr.shape}")
    print(f"Val features: {X_va.shape}, Target: {y_va.shape}")
    print(f"Test features: {X_te.shape}, Target: {y_te.shape}")
