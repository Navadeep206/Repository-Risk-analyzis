#!/usr/bin/env python3
"""
Permutation Importance module for Phase 8.
Calculates permutation feature importances for Random Forest on the test set.
"""

import os
import sys
import pandas as pd
from sklearn.inspection import permutation_importance
from typing import Tuple

# Ensure parent directory is in path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import BASE_DIR
from explainability.model_explainer import load_explainability_resources

def run_permutation_importance() -> pd.DataFrame:
    """
    Computes permutation importance for Random Forest on test split
    and saves results to reports/explainability/permutation_importance.csv.
    """
    # Load resources
    preprocessor, rf_model, _, _, _, _, X_test, y_test = load_explainability_resources()
    
    # Preprocess test set features
    X_test_proc = preprocessor.transform(X_test)
    feature_names = preprocessor.feature_names
    
    out_dir = os.path.join(BASE_DIR, "reports", "explainability")
    os.makedirs(out_dir, exist_ok=True)
    
    print("[*] Computing Permutation Importance for Random Forest...")
    rf_perm = permutation_importance(
        rf_model, X_test_proc, y_test, n_repeats=10, random_state=42, n_jobs=-1
    )
    
    # Intrinsic feature importances from the trees
    rf_intrinsic = rf_model.feature_importances_
    
    perm_df = pd.DataFrame({
        "feature_name": feature_names,
        "rf_intrinsic_importance": rf_intrinsic,
        "rf_permutation_mean": rf_perm.importances_mean,
        "rf_permutation_std": rf_perm.importances_std
    }).sort_values(by="rf_permutation_mean", ascending=False)
    
    csv_path = os.path.join(out_dir, "permutation_importance.csv")
    perm_df.to_csv(csv_path, index=False)
    print(f"[+] Saved permutation importance comparison to {csv_path}")
    
    return perm_df

if __name__ == "__main__":
    run_permutation_importance()
