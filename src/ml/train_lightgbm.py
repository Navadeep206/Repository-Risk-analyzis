#!/usr/bin/env python3
"""
Trains and saves the LightGBM model using GridSearchCV.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import pickle
import sys
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Ensure parent directory is in path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import BASE_DIR
from ml.data_loader import load_all_splits
from ml.preprocessing import CodeRiskPreprocessor

def train_lightgbm(X_train, y_train, save_path: str) -> LGBMClassifier:
    """
    Tunes, trains, and saves a LightGBM classifier.
    """
    best_params_path = os.path.join(BASE_DIR, "models", "best_lgb_params.pkl")
    if os.path.exists(best_params_path):
        print("[*] Loading best LightGBM parameters from Optuna tuning...")
        with open(best_params_path, "rb") as f:
            best_params = pickle.load(f)
        best_model = LGBMClassifier(
            **best_params,
            class_weight="balanced",
            random_state=42,
            verbosity=-1,
            n_jobs=-1
        )
        best_model.fit(X_train, y_train)
        print(f"[+] Loaded parameters: {best_params}")
    else:
        param_grid = {
            "learning_rate": [0.01, 0.05, 0.1],
            "max_depth": [3, 5, -1],
            "n_estimators": [50, 100, 150]
        }
        
        lgb = LGBMClassifier(
            random_state=42,
            class_weight="balanced",
            verbosity=-1
        )
        
        print("[*] Tuning and training LightGBM baseline...")
        grid_search = GridSearchCV(
            estimator=lgb,
            param_grid=param_grid,
            cv=5,
            scoring="accuracy",
            n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        print(f"[+] Best LightGBM Parameters: {grid_search.best_params_}")
        
    # Save the model
    with open(save_path, "wb") as f:
        pickle.dump(best_model, f)
    print(f"[+] Saved LightGBM model to {save_path}")
    
    return best_model

if __name__ == "__main__":
    X_train, y_train, X_val, y_val, X_test, y_test = load_all_splits()
    
    # Preprocess
    preproc = CodeRiskPreprocessor()
    preproc.fit(X_train)
    X_train_proc = preproc.transform(X_train)
    X_val_proc = preproc.transform(X_val)
    
    models_dir = os.path.join(BASE_DIR, "models")
    save_path = os.path.join(models_dir, "lightgbm.pkl")
    
    model = train_lightgbm(X_train_proc, y_train, save_path)
    
    # Simple check evaluation
    preds = model.predict(X_val_proc)
    acc = accuracy_score(y_val, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(y_val, preds, average="macro")
    
    print(f"Validation Metrics -> Acc: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
