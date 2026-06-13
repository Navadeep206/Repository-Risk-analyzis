#!/usr/bin/env python3
"""
Experiment 1: Relative Feature Engineering.
Converts absolute code size and churn metrics to repository-relative metrics,
and evaluates Random Forest and XGBoost classifiers under LORO validation.
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Add parent directory and package directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from config import BASE_DIR
from ml.preprocessing import CodeRiskPreprocessor
from ml.data_loader import LABEL_MAP
from evaluator import load_master_dataset, get_loro_folds, compute_metrics

def compute_relative_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes relative features by dividing absolute features by the repository-level average.
    """
    df_rel = df.copy()
    
    # Calculate repository-level means
    grouped = df_rel.groupby("repository_name")[["loc", "complexity", "modification_count", "commit_count"]].transform("mean")
    
    # Avoid division by zero
    eps = 1e-5
    df_rel["relative_loc"] = df_rel["loc"] / (grouped["loc"] + eps)
    df_rel["relative_complexity"] = df_rel["complexity"] / (grouped["complexity"] + eps)
    df_rel["relative_modifications"] = df_rel["modification_count"] / (grouped["modification_count"] + eps)
    df_rel["relative_commits"] = df_rel["commit_count"] / (grouped["commit_count"] + eps)
    
    # Drop absolute metrics to prevent leakage
    df_rel = df_rel.drop(columns=["loc", "complexity", "modification_count", "commit_count"])
    return df_rel

def run_relative_experiment():
    print("[*] Running Experiment 1: Relative Feature Engineering...")
    reports_dir = os.path.join(BASE_DIR, "reports", "domain_adaptation")
    os.makedirs(reports_dir, exist_ok=True)
    
    df_raw = load_master_dataset()
    df_rel = compute_relative_features(df_raw)
    
    rf_f1s = []
    xgb_f1s = []
    
    # LORO evaluation loop
    for held_out, df_train, df_test in get_loro_folds(df_rel):
        y_train = df_train["historical_risk_label"].map(LABEL_MAP).fillna(0).astype(int).values
        y_test = df_test["historical_risk_label"].map(LABEL_MAP).fillna(0).astype(int).values
        
        # We manually process features (or preprocess them)
        features = [
            "relative_loc", "relative_complexity", "maintainability_index",
            "relative_commits", "relative_modifications", "contributor_count",
            "commit_frequency", "repository_age_days"
        ]
        
        X_train = df_train[features].fillna(0).values
        X_test = df_test[features].fillna(0).values
        
        # Random Forest
        rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=1)
        rf.fit(X_train, y_train)
        preds_rf = rf.predict(X_test)
        metrics_rf = compute_metrics(y_test, preds_rf)
        rf_f1s.append(metrics_rf["macro_f1"])
        
        # XGBoost
        xgb = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.05, random_state=42, n_jobs=1)
        xgb.fit(X_train, y_train)
        preds_xgb = xgb.predict(X_test)
        metrics_xgb = compute_metrics(y_test, preds_xgb)
        xgb_f1s.append(metrics_xgb["macro_f1"])
        
    avg_rf_f1 = np.mean(rf_f1s)
    avg_xgb_f1 = np.mean(xgb_f1s)
    
    print(f"[+] Relative Features RF LORO Macro F1: {avg_rf_f1:.4f}")
    print(f"[+] Relative Features XGB LORO Macro F1: {avg_xgb_f1:.4f}")
    
    # Save results to summary file
    summary_df = pd.DataFrame([
        {"model": "RF (Relative Features)", "avg_loro_macro_f1": avg_rf_f1},
        {"model": "XGBoost (Relative Features)", "avg_loro_macro_f1": avg_xgb_f1}
    ])
    summary_path = os.path.join(reports_dir, "relative_features_results.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"[+] Saved Relative Features results to {summary_path}")

if __name__ == "__main__":
    run_relative_experiment()
