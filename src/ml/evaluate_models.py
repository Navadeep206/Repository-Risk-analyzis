#!/usr/bin/env python3
"""
Evaluates trained baseline models on validation and test splits.
Generates classification reports, confusion matrices, and writes reports/evaluation_report.md.
"""

import os
import pickle
import sys
import pandas as pd
import numpy as np
from typing import Dict, Any, List
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report

# Ensure parent directory is in path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import BASE_DIR
from ml.data_loader import load_all_splits, INV_LABEL_MAP
from ml.preprocessing import CodeRiskPreprocessor

def evaluate_models() -> None:
    """
    Loads all trained models, evaluates them on validation and test sets,
    and generates reports/evaluation_report.md.
    """
    models_dir = os.path.join(BASE_DIR, "models")
    reports_dir = os.path.join(BASE_DIR, "reports")
    os.makedirs(reports_dir, exist_ok=True)
    
    # Load splits
    X_train, y_train, X_val, y_val, X_test, y_test = load_all_splits()
    
    # Load preprocessor
    preproc_path = os.path.join(models_dir, "preprocessor.pkl")
    if not os.path.exists(preproc_path):
        raise FileNotFoundError(f"Preprocessor not found at {preproc_path}. Run preprocessing first.")
        
    preproc = CodeRiskPreprocessor.load(preproc_path)
    X_val_proc = preproc.transform(X_val)
    X_test_proc = preproc.transform(X_test)
    
    model_files = {
        "Logistic Regression": "logistic_regression.pkl",
        "Decision Tree": "decision_tree.pkl",
        "Random Forest": "random_forest.pkl",
        "XGBoost": "xgboost.pkl",
        "LightGBM": "lightgbm.pkl",
        "CatBoost": "catboost.pkl"
    }
    
    evaluation_results: Dict[str, Dict[str, Any]] = {}
    
    # Evaluate each model
    for model_name, file_name in model_files.items():
        model_path = os.path.join(models_dir, file_name)
        if not os.path.exists(model_path):
            print(f"[!] Warning: Model file {file_name} not found. Skipping evaluation for {model_name}.")
            continue
            
        with open(model_path, "rb") as f:
            model = pickle.load(f)
            
        evaluation_results[model_name] = {}
        
        for split_name, X_proc, y in [("Validation", X_val_proc, y_val), ("Test", X_test_proc, y_test)]:
            preds = model.predict(X_proc)
            
            acc = accuracy_score(y, preds)
            prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(y, preds, average="macro")
            prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(y, preds, average="weighted")
            
            # Confusion matrix
            cm = confusion_matrix(y, preds, labels=[0, 1, 2])
            
            # Detailed class metrics
            clf_rep = classification_report(
                y, preds, target_names=["LOW", "MEDIUM", "HIGH"], output_dict=True, labels=[0, 1, 2]
            )
            
            evaluation_results[model_name][split_name] = {
                "accuracy": acc,
                "precision_macro": prec_macro,
                "recall_macro": rec_macro,
                "f1_macro": f1_macro,
                "precision_weighted": prec_weighted,
                "recall_weighted": rec_weighted,
                "f1_weighted": f1_weighted,
                "confusion_matrix": cm,
                "classification_report_dict": clf_rep,
                "predictions": preds
            }
            
    # Write MD report
    report_path = os.path.join(reports_dir, "evaluation_report.md")
    
    with open(report_path, "w") as f:
        f.write("# Model Evaluation Report - Phase 4 Baselines\n\n")
        f.write("This report summarizes the performance of the 4 baseline classifiers (Logistic Regression, Decision Tree, Random Forest, XGBoost) on the repository-disjoint Validation and Test splits.\n\n")
        
        # 1. Overall Performance Table
        f.write("## 1. Summary Performance Metrics\n\n")
        f.write("| Model | Split | Accuracy | Precision (Macro) | Recall (Macro) | F1-Score (Macro) |\n")
        f.write("|-------|-------|----------|-------------------|----------------|------------------|\n")
        
        for model_name in sorted(evaluation_results.keys()):
            for split in ["Validation", "Test"]:
                res = evaluation_results[model_name][split]
                f.write(f"| {model_name} | {split} | {res['accuracy']:.4f} | {res['precision_macro']:.4f} | {res['recall_macro']:.4f} | {res['f1_macro']:.4f} |\n")
        
        f.write("\n---\n\n")
        
        # 2. Detailed Breakdown per Model
        f.write("## 2. Detailed Model Performance Breakdown\n\n")
        
        for model_name in sorted(evaluation_results.keys()):
            f.write(f"### {model_name}\n\n")
            
            for split in ["Validation", "Test"]:
                res = evaluation_results[model_name][split]
                f.write(f"#### {split} Split Metrics\n\n")
                
                # Class breakdown table
                f.write("| Class | Precision | Recall | F1-Score | Support |\n")
                f.write("|-------|-----------|--------|----------|---------|\n")
                for cls in ["LOW", "MEDIUM", "HIGH"]:
                    cls_metrics = res["classification_report_dict"][cls]
                    f.write(f"| {cls} | {cls_metrics['precision']:.4f} | {cls_metrics['recall']:.4f} | {cls_metrics['f1-score']:.4f} | {int(cls_metrics['support'])} |\n")
                
                # Overall row
                f.write(f"| **Macro Avg** | {res['precision_macro']:.4f} | {res['recall_macro']:.4f} | {res['f1_macro']:.4f} | {len(y_val) if split == 'Validation' else len(y_test)} |\n\n")
                
                # Confusion Matrix
                cm = res["confusion_matrix"]
                f.write("##### Confusion Matrix:\n")
                f.write("```\n")
                f.write("                  Predicted\n")
                f.write("               LOW   MEDIUM   HIGH\n")
                f.write(f"Actual LOW    {cm[0][0]:4d}     {cm[0][1]:4d}   {cm[0][2]:4d}\n")
                f.write(f"       MEDIUM {cm[1][0]:4d}     {cm[1][1]:4d}   {cm[1][2]:4d}\n")
                f.write(f"       HIGH   {cm[2][0]:4d}     {cm[2][1]:4d}   {cm[2][2]:4d}\n")
                f.write("```\n\n")
            
            f.write("---\n\n")
            
    print(f"[+] Evaluation report generated at {report_path}")

if __name__ == "__main__":
    evaluate_models()
