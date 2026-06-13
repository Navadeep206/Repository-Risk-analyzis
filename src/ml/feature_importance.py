#!/usr/bin/env python3
"""
Extracts feature importances and coefficients from trained baseline models.
Saves rankings to reports/feature_importance.csv.
"""

import os
import pickle
import sys
import pandas as pd
import numpy as np

# Ensure parent directory is in path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import BASE_DIR
from ml.preprocessing import CodeRiskPreprocessor

def extract_feature_importance() -> None:
    """
    Loads preprocessor and models, extracts importances/coefficients,
    and writes to reports/feature_importance.csv.
    """
    models_dir = os.path.join(BASE_DIR, "models")
    reports_dir = os.path.join(BASE_DIR, "reports")
    os.makedirs(reports_dir, exist_ok=True)
    
    # Load preprocessor to get feature names
    preproc_path = os.path.join(models_dir, "preprocessor.pkl")
    if not os.path.exists(preproc_path):
        raise FileNotFoundError(f"Preprocessor not found at {preproc_path}.")
        
    preproc = CodeRiskPreprocessor.load(preproc_path)
    feature_names = preproc.feature_names
    
    # Initialize df
    df_importance = pd.DataFrame({"feature_name": feature_names})
    
    # 1. Logistic Regression Coefficients
    lr_path = os.path.join(models_dir, "logistic_regression.pkl")
    if os.path.exists(lr_path):
        with open(lr_path, "rb") as f:
            lr_model = pickle.load(f)
        # lr_model.coef_ shape is (n_classes, n_features)
        # Classes: 0=LOW, 1=MEDIUM, 2=HIGH
        if len(lr_model.classes_) == 3:
            df_importance["lr_coef_LOW"] = lr_model.coef_[0]
            df_importance["lr_coef_MEDIUM"] = lr_model.coef_[1]
            df_importance["lr_coef_HIGH"] = lr_model.coef_[2]
        else:
            # Fallback for binary or single-class (should not happen here)
            for idx, cls in enumerate(lr_model.classes_):
                df_importance[f"lr_coef_{cls}"] = lr_model.coef_[idx]
    else:
        print("[!] Logistic Regression model not found. Skipping coefficients.")

    # 2. Decision Tree Importance
    dt_path = os.path.join(models_dir, "decision_tree.pkl")
    if os.path.exists(dt_path):
        with open(dt_path, "rb") as f:
            dt_model = pickle.load(f)
        df_importance["decision_tree_importance"] = dt_model.feature_importances_
    else:
        print("[!] Decision Tree model not found. Skipping importance.")

    # 3. Random Forest Importance
    rf_path = os.path.join(models_dir, "random_forest.pkl")
    if os.path.exists(rf_path):
        with open(rf_path, "rb") as f:
            rf_model = pickle.load(f)
        df_importance["random_forest_importance"] = rf_model.feature_importances_
    else:
        print("[!] Random Forest model not found. Skipping importance.")

    # 4. XGBoost Importance
    xgb_path = os.path.join(models_dir, "xgboost.pkl")
    if os.path.exists(xgb_path):
        with open(xgb_path, "rb") as f:
            xgb_model = pickle.load(f)
        df_importance["xgboost_importance"] = xgb_model.feature_importances_
    else:
        print("[!] XGBoost model not found. Skipping importance.")

    # Sort by random forest importance if available, else xgboost, else alphabetically
    if "random_forest_importance" in df_importance.columns:
        df_importance = df_importance.sort_values(by="random_forest_importance", ascending=False)
    elif "xgboost_importance" in df_importance.columns:
        df_importance = df_importance.sort_values(by="xgboost_importance", ascending=False)
    else:
        df_importance = df_importance.sort_values(by="feature_name")
        
    csv_path = os.path.join(reports_dir, "feature_importance.csv")
    df_importance.to_csv(csv_path, index=False)
    print(f"[+] Feature importances saved to {csv_path}")

if __name__ == "__main__":
    extract_feature_importance()
