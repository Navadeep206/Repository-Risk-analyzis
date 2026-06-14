#!/usr/bin/env python3
"""
Experiment 3: CORAL Feature Alignment.
Implements CORrelation ALignment (CORAL) to match source and target covariance/mean,
and evaluates Random Forest and XGBoost classifiers under LORO validation.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Add parent directory and package directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from config import BASE_DIR
from ml.data_loader import LABEL_MAP
from evaluator import load_master_dataset, get_loro_folds, compute_metrics

def matrix_power(matrix: np.ndarray, power: float) -> np.ndarray:
    """
    Computes matrix power using eigenvalue decomposition for symmetric positive semi-definite matrix.
    """
    vals, vecs = np.linalg.eigh(matrix)
    # Clip eigenvalues to positive to avoid imaginary components due to rounding errors
    vals = np.clip(vals, 1e-9, None)
    return (vecs * (vals ** power)[np.newaxis, :]) @ vecs.T

def coral_align(X_source: np.ndarray, X_target: np.ndarray, lambda_reg: float = 1e-5) -> np.ndarray:
    """
    Aligns the covariance and mean of X_source with X_target.
    """
    # 1. Standardize using target statistics
    scaler = StandardScaler()
    X_target_scaled = scaler.fit_transform(X_target)
    X_source_scaled = scaler.transform(X_source)
    
    # 2. Centering source and target
    mu_s = np.mean(X_source_scaled, axis=0)
    mu_t = np.mean(X_target_scaled, axis=0)
    
    X_source_centered = X_source_scaled - mu_s
    X_target_centered = X_target_scaled - mu_t
    
    # 3. Covariances
    d = X_source.shape[1]
    cov_s = np.cov(X_source_centered, rowvar=False) + lambda_reg * np.eye(d)
    cov_t = np.cov(X_target_centered, rowvar=False) + lambda_reg * np.eye(d)
    
    # 4. Compute matrices
    cov_s_inv_sqrt = matrix_power(cov_s, -0.5)
    cov_t_sqrt = matrix_power(cov_t, 0.5)
    
    # 5. Transform source features
    X_source_aligned = X_source_centered @ cov_s_inv_sqrt @ cov_t_sqrt
    
    # 6. Restore target mean
    X_source_aligned = X_source_aligned + mu_t
    
    return X_source_aligned, X_target_scaled

def run_coral_experiments():
    print("[*] Running Experiment 3: CORAL Feature Alignment...")
    reports_dir = os.path.join(BASE_DIR, "reports", "domain_adaptation")
    os.makedirs(reports_dir, exist_ok=True)
    
    df_raw = load_master_dataset()
    features = [
        "loc", "complexity", "maintainability_index", "commit_count",
        "modification_count", "contributor_count", "commit_frequency", "repository_age_days"
    ]
    
    rf_f1s = []
    xgb_f1s = []
    
    for held_out, df_train, df_test in get_loro_folds(df_raw):
        y_train = df_train["historical_risk_label"].map(LABEL_MAP).fillna(0).astype(int).values
        y_test = df_test["historical_risk_label"].map(LABEL_MAP).fillna(0).astype(int).values
        
        X_train_raw = df_train[features].fillna(0).values
        X_test_raw = df_test[features].fillna(0).values
        
        # Apply CORAL Alignment
        X_train_coral, X_test_scaled = coral_align(X_train_raw, X_test_raw)
        
        # Random Forest
        rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=1)
        rf.fit(X_train_coral, y_train)
        preds_rf = rf.predict(X_test_scaled)
        metrics_rf = compute_metrics(y_test, preds_rf)
        rf_f1s.append(metrics_rf["macro_f1"])
        
        # XGBoost
        xgb = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.05, random_state=42, n_jobs=1)
        xgb.fit(X_train_coral, y_train)
        preds_xgb = xgb.predict(X_test_scaled)
        metrics_xgb = compute_metrics(y_test, preds_xgb)
        xgb_f1s.append(metrics_xgb["macro_f1"])
        
    avg_rf_f1 = np.mean(rf_f1s)
    avg_xgb_f1 = np.mean(xgb_f1s)
    
    print(f"[+] CORAL RF LORO Avg Macro F1: {avg_rf_f1:.4f}")
    print(f"[+] CORAL XGB LORO Avg Macro F1: {avg_xgb_f1:.4f}")
    
    # Save results to CSV
    df_coral = pd.DataFrame([
        {"model": "CORAL RF", "avg_loro_macro_f1": avg_rf_f1},
        {"model": "CORAL XGBoost", "avg_loro_macro_f1": avg_xgb_f1}
    ])
    results_path = os.path.join(reports_dir, "coral_results.csv")
    df_coral.to_csv(results_path, index=False)
    print(f"[+] Saved CORAL results to {results_path}")

if __name__ == "__main__":
    run_coral_experiments()
