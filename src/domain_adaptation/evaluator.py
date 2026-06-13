#!/usr/bin/env python3
"""
Common evaluator utilities for Phase 12 Domain Adaptation & OOD Robustness.
Provides LORO data splits, dataset alignment, and performance evaluation metrics.
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import BASE_DIR, FINAL_DIR
from ml.data_loader import LABEL_MAP

# Inverse mapping for prediction printouts
INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}

def load_master_dataset() -> pd.DataFrame:
    """
    Loads ml_dataset_v2.csv and returns it.
    """
    path = os.path.join(FINAL_DIR, "ml_dataset_v2.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Master dataset v2 not found: {path}")
    return pd.read_csv(path)

def load_aligned_embeddings() -> pd.DataFrame:
    """
    Loads ml_dataset_v2.csv and embedding_dataset.parquet,
    aligns them by repository_name and file_path, and returns the merged DataFrame.
    """
    csv_path = os.path.join(FINAL_DIR, "ml_dataset_v2.csv")
    pq_path = os.path.join(FINAL_DIR, "embedding_dataset.parquet")
    
    if not os.path.exists(csv_path) or not os.path.exists(pq_path):
        raise FileNotFoundError("Master CSV or Embedding Parquet missing.")
        
    df_csv = pd.read_csv(csv_path)
    df_pq = pd.read_parquet(pq_path)
    
    # Merge on identifier columns
    df_merged = pd.merge(df_csv, df_pq, on=["repository_name", "file_path"], suffixes=("", "_pq"))
    
    # Resolve duplicate risk label if any
    if "historical_risk_label_pq" in df_merged.columns:
        df_merged = df_merged.drop(columns=["historical_risk_label_pq"])
        
    return df_merged

def get_loro_folds(df: pd.DataFrame):
    """
    Yields train/test folds for Leave-One-Repository-Out cross-validation.
    """
    repos = df["repository_name"].dropna().unique().tolist()
    for held_out in repos:
        df_train = df[df["repository_name"] != held_out].copy()
        df_test = df[df["repository_name"] == held_out].copy()
        yield held_out, df_train, df_test

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Computes accuracy, macro precision, recall, f1, and weighted f1.
    """
    acc = accuracy_score(y_true, y_pred)
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    _, _, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    return {
        "accuracy": float(acc),
        "precision_macro": float(prec_macro),
        "recall_macro": float(rec_macro),
        "macro_f1": float(f1_macro),
        "weighted_f1": float(f1_weighted)
    }
