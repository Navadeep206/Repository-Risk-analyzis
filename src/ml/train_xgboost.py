#!/usr/bin/env python3
"""
Trains and saves the XGBoost model using GridSearchCV.
"""

import os
import pickle
import sys
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Ensure parent directory is in path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import BASE_DIR
from ml.data_loader import load_all_splits
from ml.preprocessing import CodeRiskPreprocessor

def train_xgboost(X_train, y_train, save_path: str) -> XGBClassifier:
    """
    Tunes, trains, and saves an XGBoost classifier.
    """
    param_grid = {
        "learning_rate": [0.01, 0.1, 0.2],
        "max_depth": [3, 5, 7],
        "n_estimators": [50, 100, 150]
    }
    
    xgb = XGBClassifier(
        random_state=42,
        eval_metric="mlogloss"
    )
    
    print("[*] Tuning and training XGBoost baseline...")
    # Use 5-fold CV for grid search on the training split
    grid_search = GridSearchCV(
        estimator=xgb,
        param_grid=param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=-1
    )
    
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    print(f"[+] Best XGBoost Parameters: {grid_search.best_params_}")
    print(f"[+] Best XGBoost CV Accuracy: {grid_search.best_score_:.4f}")
    
    # Save the model
    with open(save_path, "wb") as f:
        pickle.dump(best_model, f)
    print(f"[+] Saved XGBoost model to {save_path}")
    
    return best_model

if __name__ == "__main__":
    X_train, y_train, X_val, y_val, X_test, y_test = load_all_splits()
    
    # Preprocess
    preproc = CodeRiskPreprocessor()
    preproc.fit(X_train)
    X_train_proc = preproc.transform(X_train)
    X_val_proc = preproc.transform(X_val)
    
    models_dir = os.path.join(BASE_DIR, "models")
    save_path = os.path.join(models_dir, "xgboost.pkl")
    
    model = train_xgboost(X_train_proc, y_train, save_path)
    
    # Simple check evaluation
    preds = model.predict(X_val_proc)
    acc = accuracy_score(y_val, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(y_val, preds, average="macro")
    
    print(f"Validation Metrics -> Acc: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
