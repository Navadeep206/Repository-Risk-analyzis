#!/usr/bin/env python3
"""
Master orchestrator script for the baseline ML pipeline.
Sequences all steps: loading, preprocessing, training, evaluation, feature importance, and comparisons.
"""

import os
import sys

# Ensure parent directory is in path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import BASE_DIR, ensure_dirs_exist
from ml.data_loader import load_all_splits
from ml.preprocessing import CodeRiskPreprocessor
from ml.train_logistic_regression import train_logistic_regression
from ml.train_decision_tree import train_decision_tree
from ml.train_random_forest import train_random_forest
from ml.train_xgboost import train_xgboost
from ml.train_lightgbm import train_lightgbm
from ml.train_catboost import train_catboost
from ml.evaluate_models import evaluate_models
from ml.feature_importance import extract_feature_importance
from ml.model_comparison import perform_model_comparison

def run_pipeline() -> None:
    """
    Executes the entire machine learning pipeline.
    """
    print("="*60)
    print("Starting Baseline ML Pipeline Orchestrator")
    print("="*60)
    
    # 1. Ensure directories exist
    ensure_dirs_exist()
    models_dir = os.path.join(BASE_DIR, "models")
    reports_dir = os.path.join(BASE_DIR, "reports")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    
    # 2. Load all splits
    print("\n[*] Loading dataset splits...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_all_splits()
    print(f"[+] Loaded Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    
    # 3. Fit and Save Preprocessor
    print("\n[*] Initializing and fitting preprocessor...")
    preproc = CodeRiskPreprocessor()
    preproc.fit(X_train)
    preproc_path = os.path.join(models_dir, "preprocessor.pkl")
    preproc.save(preproc_path)
    
    # Transform
    X_train_proc = preproc.transform(X_train)
    print(f"[+] Data preprocessed. Transformed shape: {X_train_proc.shape}")
    
    # 4. Train Logistic Regression
    print("\n[*] Training Logistic Regression Baseline...")
    lr_path = os.path.join(models_dir, "logistic_regression.pkl")
    train_logistic_regression(X_train_proc, y_train, lr_path)
    
    # 5. Train Decision Tree
    print("\n[*] Training Decision Tree Baseline...")
    dt_path = os.path.join(models_dir, "decision_tree.pkl")
    train_decision_tree(X_train_proc, y_train, dt_path)
    
    # 6. Train Random Forest
    print("\n[*] Training Random Forest Baseline (with Tuning)...")
    rf_path = os.path.join(models_dir, "random_forest.pkl")
    train_random_forest(X_train_proc, y_train, rf_path)
    
    # 7. Train XGBoost
    print("\n[*] Training XGBoost Baseline (with Tuning)...")
    xgb_path = os.path.join(models_dir, "xgboost.pkl")
    train_xgboost(X_train_proc, y_train, xgb_path)
    
    # 7.5 Train LightGBM
    print("\n[*] Training LightGBM Baseline (with Tuning)...")
    lgb_path = os.path.join(models_dir, "lightgbm.pkl")
    train_lightgbm(X_train_proc, y_train, lgb_path)
    
    # 7.6 Train CatBoost
    print("\n[*] Training CatBoost Baseline (with Tuning)...")
    cb_path = os.path.join(models_dir, "catboost.pkl")
    train_catboost(X_train_proc, y_train, cb_path)
    
    # 8. Evaluate Models
    print("\n[*] Evaluating all trained models...")
    evaluate_models()
    
    # 9. Extract Feature Importance
    print("\n[*] Extracting feature importances...")
    extract_feature_importance()
    
    # 10. Perform Model Comparison & Cross-Validation
    print("\n[*] Performing cross-validation and comparative analysis...")
    perform_model_comparison()
    
    print("\n" + "="*60)
    print("Baseline ML Pipeline Completed Successfully!")
    print("="*60)

if __name__ == "__main__":
    run_pipeline()
