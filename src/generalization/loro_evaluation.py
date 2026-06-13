#!/usr/bin/env python3
"""
Step 3: Leave-One-Repository-Out (LORO) Benchmark.
Trains Random Forest model on 5 repositories and tests on the held-out 6th.
Repeats for all repositories to assess generalizability.
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import BASE_DIR, FINAL_DIR
from ml.preprocessing import CodeRiskPreprocessor
from ml.data_loader import LABEL_MAP

def run_loro_evaluation():
    print("[*] Running Leave-One-Repository-Out (LORO) Benchmark...")
    reports_dir = os.path.join(BASE_DIR, "reports", "generalization")
    os.makedirs(reports_dir, exist_ok=True)
    
    # Load master dataset
    dataset_path = os.path.join(FINAL_DIR, "ml_dataset_v2.csv")
    df = pd.read_csv(dataset_path)
    
    repos = df["repository_name"].dropna().unique().tolist()
    print(f"[*] Repositories identified for LORO: {repos}")
    
    results = []
    
    for held_out in repos:
        print(f"[*] Evaluating fold: Held-out Repository = {held_out}")
        
        # Split data
        df_train = df[df["repository_name"] != held_out].copy()
        df_test = df[df["repository_name"] == held_out].copy()
        
        # Map labels
        y_train = df_train["historical_risk_label"].map(LABEL_MAP).fillna(0).astype(int)
        y_test = df_test["historical_risk_label"].map(LABEL_MAP).fillna(0).astype(int)
        
        # Fit preprocessor on train set
        preproc = CodeRiskPreprocessor()
        preproc.fit(df_train)
        
        X_train_proc = preproc.transform(df_train)
        X_test_proc = preproc.transform(df_test)
        
        # Train Random Forest Classifier
        rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        rf.fit(X_train_proc, y_train)
        
        # Predict on held-out test repository
        preds = rf.predict(X_test_proc)
        
        # Calculate metrics
        acc = accuracy_score(y_test, preds)
        precision, recall, f1_macro, _ = precision_recall_fscore_support(
            y_test, preds, average="macro", zero_division=0
        )
        _, _, f1_weighted, _ = precision_recall_fscore_support(
            y_test, preds, average="weighted", zero_division=0
        )
        
        results.append({
            "held_out_repository": held_out,
            "train_samples": len(df_train),
            "test_samples": len(df_test),
            "accuracy": float(acc),
            "macro_f1": float(f1_macro),
            "weighted_f1": float(f1_weighted)
        })
        
        print(f"    [+] Fold result: Accuracy = {acc:.4f}, Macro F1 = {f1_macro:.4f}")
        
    df_loro = pd.DataFrame(results)
    loro_file = os.path.join(reports_dir, "loro_results.csv")
    df_loro.to_csv(loro_file, index=False)
    print(f"[+] Saved LORO results to {loro_file}")
    
    # Generate Markdown Summary
    write_loro_summary(df_loro, reports_dir)

def write_loro_summary(df_loro: pd.DataFrame, reports_dir: str):
    avg_acc = df_loro["accuracy"].mean()
    avg_macro_f1 = df_loro["macro_f1"].mean()
    avg_weighted_f1 = df_loro["weighted_f1"].mean()
    
    md_content = f"""# Leave-One-Repository-Out (LORO) Benchmark Summary

This summary analyzes the generalizability of the Random Forest risk classifier when tested on a completely unseen repository during training.

## 1. LORO Results Table

| Held-Out Repository | Train Samples | Test Samples | Accuracy | Macro F1 | Weighted F1 |
| --- | --- | --- | --- | --- | --- |
"""
    for _, row in df_loro.iterrows():
        md_content += f"| {row['held_out_repository']} | {row['train_samples']} | {row['test_samples']} | {row['accuracy']:.4f} | {row['macro_f1']:.4f} | {row['weighted_f1']:.4f} |\n"
        
    md_content += f"""| **Average** | **-** | **-** | **{avg_acc:.4f}** | **{avg_macro_f1:.4f}** | **{avg_weighted_f1:.4f}** |
    
---

## 2. Key Insights

1. **Hardest Generalization Targets**: Fold-level results indicate where model performance drops significantly. This reflects severe class distribution shifts and baseline activity scale mismatches.
2. **Easiest Generalization Targets**: Repositories that share class profiles and scale features with the pool of training repositories show higher cross-repo accuracy.
"""
    summary_path = os.path.join(reports_dir, "loro_summary.md")
    with open(summary_path, "w") as f:
        f.write(md_content)
    print(f"[+] Saved LORO summary markdown to {summary_path}")

if __name__ == "__main__":
    run_loro_evaluation()
