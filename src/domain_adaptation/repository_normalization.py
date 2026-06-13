#!/usr/bin/env python3
"""
Experiment 2: Repository Normalization.
Compares Global StandardScaler, Repository-specific StandardScaler,
Repository-specific RobustScaler, and Quantile Normalization under LORO.
Outputs normalization_results.csv.
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer
from sklearn.ensemble import RandomForestClassifier

# Add parent directory and package directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from config import BASE_DIR
from ml.data_loader import LABEL_MAP
from evaluator import load_master_dataset, get_loro_folds, compute_metrics

def scale_by_repo(df: pd.DataFrame, features: list, scaler_class) -> pd.DataFrame:
    """
    Applies the given scaler class to each repository individually.
    """
    df_scaled = df.copy()
    for col in features:
        df_scaled[col] = df_scaled[col].astype(float)
    for repo in df_scaled["repository_name"].unique():
        idx = df_scaled["repository_name"] == repo
        if np.sum(idx) > 0:
            scaler = scaler_class()
            df_scaled.loc[idx, features] = scaler.fit_transform(df_scaled.loc[idx, features].fillna(0))
    return df_scaled

def run_normalization_experiments():
    print("[*] Running Experiment 2: Repository Normalization...")
    reports_dir = os.path.join(BASE_DIR, "reports", "domain_adaptation")
    os.makedirs(reports_dir, exist_ok=True)
    
    df_raw = load_master_dataset()
    features = [
        "loc", "complexity", "maintainability_index", "commit_count",
        "modification_count", "contributor_count", "commit_frequency", "repository_age_days"
    ]
    
    methods = ["Global StandardScaler", "Repo-specific StandardScaler", "Repo-specific RobustScaler", "Quantile Normalization"]
    results = []
    
    for method in methods:
        f1s = []
        for held_out, df_train, df_test in get_loro_folds(df_raw):
            y_train = df_train["historical_risk_label"].map(LABEL_MAP).fillna(0).astype(int).values
            y_test = df_test["historical_risk_label"].map(LABEL_MAP).fillna(0).astype(int).values
            
            if method == "Global StandardScaler":
                scaler = StandardScaler()
                X_train = scaler.fit_transform(df_train[features].fillna(0).values)
                X_test = scaler.transform(df_test[features].fillna(0).values)
                
            elif method == "Repo-specific StandardScaler":
                # Scale train repositories individually
                df_tr_scaled = scale_by_repo(df_train, features, StandardScaler)
                # Scale test repository individually
                df_te_scaled = scale_by_repo(df_test, features, StandardScaler)
                X_train = df_tr_scaled[features].values
                X_test = df_te_scaled[features].values
                
            elif method == "Repo-specific RobustScaler":
                # Scale train repositories individually
                df_tr_scaled = scale_by_repo(df_train, features, RobustScaler)
                # Scale test repository individually
                df_te_scaled = scale_by_repo(df_test, features, RobustScaler)
                X_train = df_tr_scaled[features].values
                X_test = df_te_scaled[features].values
                
            elif method == "Quantile Normalization":
                # Helper for local QuantileTransformer
                def local_qt():
                    # We set n_quantiles relative to data size to avoid warnings
                    return QuantileTransformer(n_quantiles=30, random_state=42, output_distribution="normal")
                
                df_tr_scaled = scale_by_repo(df_train, features, local_qt)
                df_te_scaled = scale_by_repo(df_test, features, local_qt)
                X_train = df_tr_scaled[features].values
                X_test = df_te_scaled[features].values
                
            # Train Random Forest
            rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=1)
            rf.fit(X_train, y_train)
            preds = rf.predict(X_test)
            metrics = compute_metrics(y_test, preds)
            f1s.append(metrics["macro_f1"])
            
        avg_f1 = np.mean(f1s)
        print(f"[+] Method: {method} | LORO Avg Macro F1: {avg_f1:.4f}")
        results.append({"method": method, "avg_loro_macro_f1": avg_f1})
        
    df_results = pd.DataFrame(results)
    results_path = os.path.join(reports_dir, "normalization_results.csv")
    df_results.to_csv(results_path, index=False)
    print(f"[+] Saved normalization results to {results_path}")

if __name__ == "__main__":
    run_normalization_experiments()
