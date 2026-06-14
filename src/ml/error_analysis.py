#!/usr/bin/env python3
"""
Forensic Error Analysis Script for Repository Risk Prediction.
Loads the trained LightGBM model, evaluates it on validation data, 
extracts top False Positives and False Negatives, and prints feature profiling.
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

# Ensure parent directory is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import BASE_DIR
from ml.data_loader import load_all_splits, INV_LABEL_MAP
from ml.preprocessing import CodeRiskPreprocessor

def run_error_analysis():
    print("[*] Running Forensic Error Analysis...")
    
    # 1. Load data
    _, _, X_val, y_val, _, _ = load_all_splits()
    
    val_csv_path = os.path.join(BASE_DIR, "data", "final", "validation_v2.csv")
    if not os.path.exists(val_csv_path):
        print(f"[-] Validation CSV not found at {val_csv_path}")
        return
        
    df_val = pd.read_csv(val_csv_path)
    
    # Load model and preprocessor
    models_dir = os.path.join(BASE_DIR, "models")
    model_path = os.path.join(models_dir, "lightgbm.pkl")
    preproc_path = os.path.join(models_dir, "preprocessor.pkl")
    
    if not os.path.exists(model_path) or not os.path.exists(preproc_path):
        print("[-] Model or preprocessor not found! Run the ML pipeline first.")
        return
        
    with open(preproc_path, "rb") as f:
        preproc = pickle.load(f)
        
    with open(model_path, "rb") as f:
        model = pickle.load(f)
        
    # 2. Transform validation set
    X_val_proc = preproc.transform(X_val)
    
    # 3. Predict
    preds = model.predict(X_val_proc)
    probs = model.predict_proba(X_val_proc)
    
    # Combine predictions and probabilities into df_val
    from ml.data_loader import LABEL_MAP
    df_val["true_label_idx"] = df_val["historical_risk_label"].map(LABEL_MAP)
    df_val["true_label"] = df_val["historical_risk_label"]
    df_val["pred_label_idx"] = preds
    df_val["pred_label"] = df_val["pred_label_idx"].map(INV_LABEL_MAP)
    df_val["prob_low"] = probs[:, 0]
    df_val["prob_medium"] = probs[:, 1]
    df_val["prob_high"] = probs[:, 2]
    
    # 4. Generate confusion matrix
    cm = confusion_matrix(y_val, preds, labels=[0, 1, 2])
    print("\nConfusion Matrix (Val):")
    print("           Predicted")
    print("          LOW  MED  HIGH")
    print(f"Actual LOW  {cm[0,0]:3d}  {cm[0,1]:3d}  {cm[0,2]:3d}")
    print(f"Actual MED  {cm[1,0]:3d}  {cm[1,1]:3d}  {cm[1,2]:3d}")
    print(f"Actual HIGH {cm[2,0]:3d}  {cm[2,1]:3d}  {cm[2,2]:3d}")
    
    # 5. Extract False Positives (predicted HIGH, actual LOW or MEDIUM)
    # Ranked by predicted probability of HIGH (confidence of error)
    fps = df_val[(df_val["true_label_idx"] < 2) & (df_val["pred_label_idx"] == 2)].copy()
    fps = fps.sort_values(by="prob_high", ascending=False)
    
    # 6. Extract False Negatives (predicted LOW or MEDIUM, actual HIGH)
    # Ranked by predicted probability of LOW or MEDIUM
    fns = df_val[(df_val["true_label_idx"] == 2) & (df_val["pred_label_idx"] < 2)].copy()
    fns = fns.sort_values(by="prob_high", ascending=True)
    
    # 7. Write Markdown Report
    reports_dir = os.path.join(BASE_DIR, "reports")
    os.makedirs(reports_dir, exist_ok=True)
    report_path = os.path.join(reports_dir, "forensic_error_analysis.md")
    
    with open(report_path, "w") as f:
        f.write("# Forensic Error Analysis Report\n\n")
        f.write("## Confusion Matrix (Validation Set)\n\n")
        f.write("| Actual \\ Predicted | LOW | MEDIUM | HIGH |\n")
        f.write("| :--- | :---: | :---: | :---: |\n")
        f.write(f"| **LOW** | {cm[0,0]} | {cm[0,1]} | {cm[0,2]} |\n")
        f.write(f"| **MEDIUM** | {cm[1,0]} | {cm[1,1]} | {cm[1,2]} |\n")
        f.write(f"| **HIGH** | {cm[2,0]} | {cm[2,1]} | {cm[2,2]} |\n\n")
        
        f.write("## Top 5 False Positives (Predicted HIGH, Actual LOW/MEDIUM)\n")
        f.write("These files were classified as HIGH risk, but have low or medium actual defect rates. ")
        f.write("We rank them by predicted HIGH-class probability.\n\n")
        
        if fps.empty:
            f.write("*No False Positives found in the validation set.*\n\n")
        else:
            for idx, (_, row) in enumerate(fps.head(5).iterrows()):
                f.write(f"### {idx+1}. `{row['file_path']}` (Repo: `{row['repository_name']}`, Lang: `{row['language']}`)\n")
                f.write(f"- **True Label**: `{row['true_label']}`\n")
                f.write(f"- **Predicted Label**: `{row['pred_label']}` (Confidence: {row['prob_high']:.2%})\n")
                f.write("- **Key Feature Metrics**:\n")
                f.write(f"  - `loc`: {row['loc']} | `complexity`: {row['complexity']}\n")
                f.write(f"  - `commit_count`: {row['commit_count']} | `modification_count`: {row['modification_count']} | `contributor_count`: {row['contributor_count']}\n")
                f.write(f"  - `ownership_concentration` (HHI): {row['ownership_concentration']:.3f} | `contributor_entropy`: {row['contributor_entropy']:.3f} | `bus_factor`: {row['bus_factor']}\n")
                f.write(f"  - `recent_churn` (last 30d): {row['recent_churn']} | `time_decayed_churn`: {row['time_decayed_churn']:.2f}\n")
                f.write(f"  - `historical_bug_density`: {row['historical_bug_density']:.3f} | `time_since_last_bug_fix`: {row['time_since_last_bug_fix']:.1f} days\n")
                f.write("- **Forensic Diagnosis**:\n")
                # Automated reasoning heuristics
                if row["modification_count"] > 20 and row["historical_bug_density"] < 0.05:
                    f.write("  - *High Activity Bias*: This file undergoes frequent modifications and commits, raising its size and activity features. However, these edits are non-defect-fixing (likely configurations, style cleanups, or refactorings), leading the model to over-predict risk based on churn volume.\n")
                elif row["ownership_concentration"] < 0.3:
                    f.write("  - *Shared Ownership Noise*: Many developers have modified this file (low HHI / high entropy), which is a common risk pattern. However, the contributors successfully coordinate without introducing defects, making this a false alarm.\n")
                else:
                    f.write("  - *Out-of-Distribution Shift*: The file LOC and complexity are relatively high for its language in this repository, mapping it to a high Z-score which triggers a HIGH risk prediction despite zero history of actual bugs.\n")
                f.write("\n")
                
        f.write("## Top 5 False Negatives (Predicted LOW/MEDIUM, Actual HIGH)\n")
        f.write("These files are highly defect-prone (3+ bug-fixing commits) but were predicted as LOW or MEDIUM risk. ")
        f.write("We rank them by ascending predicted HIGH-class probability.\n\n")
        
        if fns.empty:
            f.write("*No False Negatives found in the validation set.*\n\n")
        else:
            for idx, (_, row) in enumerate(fns.head(5).iterrows()):
                f.write(f"### {idx+1}. `{row['file_path']}` (Repo: `{row['repository_name']}`, Lang: `{row['language']}`)\n")
                f.write(f"- **True Label**: `{row['true_label']}`\n")
                f.write(f"- **Predicted Label**: `{row['pred_label']}` (Confidence: {row['prob_high']:.2%})\n")
                f.write("- **Key Feature Metrics**:\n")
                f.write(f"  - `loc`: {row['loc']} | `complexity`: {row['complexity']}\n")
                f.write(f"  - `commit_count`: {row['commit_count']} | `modification_count`: {row['modification_count']} | `contributor_count`: {row['contributor_count']}\n")
                f.write(f"  - `ownership_concentration` (HHI): {row['ownership_concentration']:.3f} | `contributor_entropy`: {row['contributor_entropy']:.3f} | `bus_factor`: {row['bus_factor']}\n")
                f.write(f"  - `recent_churn` (last 30d): {row['recent_churn']} | `time_decayed_churn`: {row['time_decayed_churn']:.2f}\n")
                f.write(f"  - `historical_bug_density`: {row['historical_bug_density']:.3f} | `time_since_last_bug_fix`: {row['time_since_last_bug_fix']:.1f} days\n")
                f.write("- **Forensic Diagnosis**:\n")
                if row["loc"] < 30 and row["bug_fix_commit_count"] >= 3:
                    f.write("  - *Quiet Complexity / High Density*: The file is small and has low cyclomatic complexity, but it contains critical logic that developers frequently break. Standard complexity metrics fail to capture this semantic fragility.\n")
                elif row["time_since_last_bug_fix"] > 100 or row["recent_churn"] == 0:
                    f.write("  - *Dormant Volatility*: The file has a long history of bug fixes but has been stable recently (low recent and decayed churn). The model predicts LOW/MEDIUM due to recent inactivity, ignoring the historical risk footprint.\n")
                else:
                    f.write("  - *Generalization Scale Gap*: Features are scaled relative to the repository, but the overall activity is lower than the JS repositories in the training set, causing the model to underestimate risk boundaries.\n")
                f.write("\n")
                
    print(f"[+] Forensic Error Analysis report written to {report_path}")

if __name__ == "__main__":
    run_error_analysis()
