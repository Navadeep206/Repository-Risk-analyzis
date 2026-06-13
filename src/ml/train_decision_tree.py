#!/usr/bin/env python3
"""
Trains and saves the Decision Tree baseline model.
"""

import os
import pickle
import sys
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Ensure parent directory is in path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import BASE_DIR
from ml.data_loader import load_all_splits
from ml.preprocessing import CodeRiskPreprocessor

def train_decision_tree(X_train, y_train, save_path: str) -> DecisionTreeClassifier:
    """
    Trains and saves a Decision Tree classifier.
    """
    # Instantiate Decision Tree classifier
    model = DecisionTreeClassifier(
        max_depth=5,
        min_samples_split=4,
        random_state=42
    )
    
    print("[*] Training Decision Tree baseline...")
    model.fit(X_train, y_train)
    
    # Save the model
    with open(save_path, "wb") as f:
        pickle.dump(model, f)
    print(f"[+] Saved Decision Tree model to {save_path}")
    
    return model

if __name__ == "__main__":
    X_train, y_train, X_val, y_val, X_test, y_test = load_all_splits()
    
    # Preprocess
    preproc = CodeRiskPreprocessor()
    preproc.fit(X_train)
    X_train_proc = preproc.transform(X_train)
    X_val_proc = preproc.transform(X_val)
    
    models_dir = os.path.join(BASE_DIR, "models")
    save_path = os.path.join(models_dir, "decision_tree.pkl")
    
    model = train_decision_tree(X_train_proc, y_train, save_path)
    
    # Simple check evaluation
    preds = model.predict(X_val_proc)
    acc = accuracy_score(y_val, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(y_val, preds, average="macro")
    
    print(f"Validation Metrics -> Acc: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
