#!/usr/bin/env python3
"""
Optuna hyperparameter tuning for LightGBM, optimizing directly for Macro F1.
"""

import os
import sys
import pickle
import optuna
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score

# Ensure parent directory is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import BASE_DIR
from ml.data_loader import load_all_splits
from ml.preprocessing import CodeRiskPreprocessor

# Suppress Optuna logging details to keep clean terminal output
optuna.logging.set_verbosity(optuna.logging.WARNING)

def objective(trial) -> float:
    # 1. Load splits
    X_train, y_train, X_val, y_val, _, _ = load_all_splits()
    
    # 2. Preprocess features using the preprocessor
    # We turn on relative scaling to ensure repository-invariant features
    preproc = CodeRiskPreprocessor(relative_scaling=True)
    
    preproc.fit(X_train)
    X_train_proc = preproc.transform(X_train)
    X_val_proc = preproc.transform(X_val)
    
    # 3. Define the hyperparameter search space
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 250),
        "num_leaves": trial.suggest_int("num_leaves", 7, 63),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.15, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 40),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        "class_weight": "balanced",  # Address class imbalance natively
        "random_state": 42,
        "verbosity": -1,
        "n_jobs": -1
    }
    
    # 4. Train LightGBM model
    model = LGBMClassifier(**params)
    model.fit(X_train_proc, y_train)
    
    # 5. Predict and evaluate Validation Macro F1
    preds = model.predict(X_val_proc)
    score = f1_score(y_val, preds, average="macro", zero_division=0)
    
    return score

def run_tuning(n_trials: int = 50):
    print(f"[*] Starting Optuna tuning for LightGBM ({n_trials} trials)...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    
    print("\n" + "="*50)
    print("[+] Tuning Complete!")
    print(f"[+] Best Validation Macro F1: {study.best_value:.4f}")
    print("[+] Best Parameters:")
    for k, v in study.best_params.items():
        print(f"    - {k}: {v}")
    print("="*50)
    
    # Save best parameters
    models_dir = os.path.join(BASE_DIR, "models")
    os.makedirs(models_dir, exist_ok=True)
    best_params_path = os.path.join(models_dir, "best_lgb_params.pkl")
    with open(best_params_path, "wb") as f:
        pickle.dump(study.best_params, f)
        
    print(f"[+] Saved best parameters to {best_params_path}")

if __name__ == "__main__":
    # Ensure optuna is installed
    try:
        import optuna
    except ImportError:
        print("[-] Installing optuna library...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "optuna", "--break-system-packages"])
        import optuna
        
    run_tuning(100)
