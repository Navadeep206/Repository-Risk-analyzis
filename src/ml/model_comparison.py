#!/usr/bin/env python3
"""
Performs 5-fold cross validation on the training split for all 4 baseline models.
Compiles a model comparison matrix containing cross validation statistics and split evaluation metrics.
Saves comparison to reports/model_comparison.csv and reports/cross_validation_results.csv.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import pickle
import sys
import pandas as pd
import numpy as np
from typing import Dict, Any, List
from sklearn.model_selection import cross_validate
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# Ensure parent directory is in path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import BASE_DIR
from ml.data_loader import load_all_splits
from ml.preprocessing import CodeRiskPreprocessor

def perform_model_comparison() -> None:
    """
    Performs cross validation and split evaluation comparison across the models,
    saving the output datasets in reports/.
    """
    models_dir = os.path.join(BASE_DIR, "models")
    reports_dir = os.path.join(BASE_DIR, "reports")
    os.makedirs(reports_dir, exist_ok=True)
    
    # Load splits
    X_train, y_train, X_val, y_val, X_test, y_test = load_all_splits()
    
    # Load preprocessor
    preproc_path = os.path.join(models_dir, "preprocessor.pkl")
    if not os.path.exists(preproc_path):
        raise FileNotFoundError(f"Preprocessor not found at {preproc_path}.")
        
    preproc = CodeRiskPreprocessor.load(preproc_path)
    X_train_proc = preproc.transform(X_train)
    X_val_proc = preproc.transform(X_val)
    X_test_proc = preproc.transform(X_test)
    
    # We will instantiate the models. If tuned models exist on disk, we load them to inherit their best hyperparameters.
    # Otherwise, we use default/tuned hyperparameters defined in their scripts.
    models: Dict[str, Any] = {}
    
    # 1. Logistic Regression
    lr_path = os.path.join(models_dir, "logistic_regression.pkl")
    if os.path.exists(lr_path):
        with open(lr_path, "rb") as f:
            models["Logistic Regression"] = pickle.load(f)
    else:
        models["Logistic Regression"] = LogisticRegression(
            solver="lbfgs", max_iter=1000, random_state=42
        )
        
    # 2. Decision Tree
    dt_path = os.path.join(models_dir, "decision_tree.pkl")
    if os.path.exists(dt_path):
        with open(dt_path, "rb") as f:
            models["Decision Tree"] = pickle.load(f)
    else:
        models["Decision Tree"] = DecisionTreeClassifier(
            max_depth=5, min_samples_split=4, random_state=42
        )
        
    # 3. Random Forest
    rf_path = os.path.join(models_dir, "random_forest.pkl")
    if os.path.exists(rf_path):
        with open(rf_path, "rb") as f:
            models["Random Forest"] = pickle.load(f)
    else:
        # Defaults if not trained yet
        models["Random Forest"] = RandomForestClassifier(
            n_estimators=100, max_depth=10, min_samples_split=5, random_state=42
        )
        
    # 4. XGBoost
    xgb_path = os.path.join(models_dir, "xgboost.pkl")
    if os.path.exists(xgb_path):
        with open(xgb_path, "rb") as f:
            models["XGBoost"] = pickle.load(f)
    else:
        # Defaults if not trained yet
        models["XGBoost"] = XGBClassifier(
            learning_rate=0.1, max_depth=5, n_estimators=100, random_state=42, eval_metric="mlogloss"
        )
        
    # 5. LightGBM
    lgb_path = os.path.join(models_dir, "lightgbm.pkl")
    if os.path.exists(lgb_path):
        with open(lgb_path, "rb") as f:
            models["LightGBM"] = pickle.load(f)
    else:
        models["LightGBM"] = LGBMClassifier(
            learning_rate=0.05, max_depth=5, n_estimators=100, random_state=42, verbosity=-1
        )
        
    # 6. CatBoost
    cb_path = os.path.join(models_dir, "catboost.pkl")
    if os.path.exists(cb_path):
        with open(cb_path, "rb") as f:
            models["CatBoost"] = pickle.load(f)
    else:
        models["CatBoost"] = CatBoostClassifier(
            learning_rate=0.05, depth=6, iterations=100, random_seed=42, verbose=0
        )
        
    cv_records: List[Dict[str, Any]] = []
    comparison_records: List[Dict[str, Any]] = []
    
    scoring = {
        "accuracy": "accuracy",
        "precision_macro": "precision_macro",
        "recall_macro": "recall_macro",
        "f1_macro": "f1_macro"
    }
    
    for model_name, model in models.items():
        print(f"[*] Running 5-fold Cross-Validation for {model_name}...")
        
        # Fit cross validation
        cv_results = cross_validate(
            model, X_train_proc, y_train, cv=5, scoring=scoring, n_jobs=-1
        )
        
        # Log results per fold
        for fold_idx in range(5):
            cv_records.append({
                "model": model_name,
                "fold": fold_idx + 1,
                "accuracy": cv_results["test_accuracy"][fold_idx],
                "precision_macro": cv_results["test_precision_macro"][fold_idx],
                "recall_macro": cv_results["test_recall_macro"][fold_idx],
                "f1_macro": cv_results["test_f1_macro"][fold_idx]
            })
            
        # Standard fit on all training data
        model.fit(X_train_proc, y_train)
        
        # Predict on validation split
        val_preds = model.predict(X_val_proc)
        val_acc = accuracy_score(y_val, val_preds)
        _, _, val_f1, _ = precision_recall_fscore_support(y_val, val_preds, average="macro")
        
        # Predict on test split
        test_preds = model.predict(X_test_proc)
        test_acc = accuracy_score(y_test, test_preds)
        _, _, test_f1, _ = precision_recall_fscore_support(y_test, test_preds, average="macro")
        
        # Aggregated record
        comparison_records.append({
            "model": model_name,
            "cv_accuracy_mean": np.mean(cv_results["test_accuracy"]),
            "cv_accuracy_std": np.std(cv_results["test_accuracy"]),
            "cv_f1_macro_mean": np.mean(cv_results["test_f1_macro"]),
            "cv_f1_macro_std": np.std(cv_results["test_f1_macro"]),
            "validation_accuracy": val_acc,
            "validation_f1_macro": val_f1,
            "test_accuracy": test_acc,
            "test_f1_macro": test_f1
        })
        
    df_cv = pd.DataFrame(cv_records)
    df_comp = pd.DataFrame(comparison_records)
    
    cv_csv_path = os.path.join(reports_dir, "cross_validation_results.csv")
    comp_csv_path = os.path.join(reports_dir, "model_comparison.csv")
    
    df_cv.to_csv(cv_csv_path, index=False)
    df_comp.to_csv(comp_csv_path, index=False)
    
    print(f"[+] CV results saved to {cv_csv_path}")
    print(f"[+] Model comparisons saved to {comp_csv_path}")

if __name__ == "__main__":
    perform_model_comparison()
