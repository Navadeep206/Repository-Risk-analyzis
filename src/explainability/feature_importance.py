#!/usr/bin/env python3
"""
Feature Importance module for Phase 8.
Extracts Gini feature importances from the trained Random Forest model.
"""

import os
import sys
import pandas as pd
from typing import Tuple

# Ensure parent directory is in path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import BASE_DIR
from explainability.model_explainer import load_explainability_resources

def run_feature_importance() -> pd.DataFrame:
    """
    Extracts Gini feature importance from the Random Forest model and saves it.
    """
    preprocessor, rf_model, _, _, _, _, _, _ = load_explainability_resources()
    feature_names = preprocessor.feature_names
    
    # Intrinsic importances
    importances = rf_model.feature_importances_
    
    df_imp = pd.DataFrame({
        "feature_name": feature_names,
        "rf_intrinsic_importance": importances
    }).sort_values(by="rf_intrinsic_importance", ascending=False)
    
    out_dir = os.path.join(BASE_DIR, "reports", "explainability")
    os.makedirs(out_dir, exist_ok=True)
    
    csv_path = os.path.join(out_dir, "feature_importance.csv")
    df_imp.to_csv(csv_path, index=False)
    print(f"[+] Saved Random Forest intrinsic feature importance to {csv_path}")
    
    return df_imp

if __name__ == "__main__":
    run_feature_importance()
