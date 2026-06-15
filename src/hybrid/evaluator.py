#!/usr/bin/env python3
"""
Model evaluation module for Phase 7.
Computes test metrics and generates model comparison and ablation reports.
"""

import os
import sys
import json

# MUST be set before importing torch, sklearn or any OpenMP-linked library
# Prevents EXC_BAD_ACCESS (SIGSEGV) from duplicate libomp on macOS ARM64
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import torch
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, List
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support, confusion_matrix

# Ensure parent directory is in path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import BASE_DIR

def evaluate_hybrid_model(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device = torch.device("cpu")
) -> Tuple[Dict[str, float], str, np.ndarray]:
    """
    Evaluates the given model on the test split.
    
    Returns:
        metrics: Dictionary of core metrics (accuracy, precision, recall, macro_f1, weighted_f1).
        text_report: Textual classification report.
        cm: Confusion matrix numpy array.
    """
    model.eval()
    model.to(device)
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for X_tab_batch, X_emb_batch, y_batch in test_loader:
            X_tab_batch = X_tab_batch.to(device)
            X_emb_batch = X_emb_batch.to(device)
            
            logits = model(X_tab_batch, X_emb_batch)
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y_batch.numpy())
            
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    # Calculate metrics
    accuracy = accuracy_score(all_targets, all_preds)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        all_targets, all_preds, average="macro", zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        all_targets, all_preds, average="weighted", zero_division=0
    )
    
    metrics = {
        "accuracy": float(accuracy),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
        "precision_weighted": float(precision_weighted),
        "recall_weighted": float(recall_weighted),
        "f1_weighted": float(f1_weighted)
    }
    
    target_names = ["LOW", "MEDIUM", "HIGH"]
    text_report = classification_report(all_targets, all_preds, target_names=target_names, zero_division=0)
    cm = confusion_matrix(all_targets, all_preds)
    
    return metrics, text_report, cm

def compile_ablation_and_comparison(
    metrics_a: Dict[str, float],
    metrics_b: Dict[str, float],
    metrics_c: Dict[str, float]
) -> None:
    """
    Compiles and writes:
    - reports/hybrid/ablation_study.csv (Model A vs B vs C)
    - reports/hybrid/model_comparison.csv (Baselines vs Hybrid Model C)
    """
    reports_dir = os.path.join(BASE_DIR, "reports", "hybrid")
    os.makedirs(reports_dir, exist_ok=True)
    
    # 1. Ablation Study
    ablation_data = [
        {
            "Model": "Model A (Tabular Only)",
            "Accuracy": metrics_a["accuracy"],
            "Macro F1": metrics_a["f1_macro"],
            "Weighted F1": metrics_a["f1_weighted"]
        },
        {
            "Model": "Model B (Embeddings Only)",
            "Accuracy": metrics_b["accuracy"],
            "Macro F1": metrics_b["f1_macro"],
            "Weighted F1": metrics_b["f1_weighted"]
        },
        {
            "Model": "Model C (Hybrid Fusion)",
            "Accuracy": metrics_c["accuracy"],
            "Macro F1": metrics_c["f1_macro"],
            "Weighted F1": metrics_c["f1_weighted"]
        }
    ]
    df_ablation = pd.DataFrame(ablation_data)
    ablation_path = os.path.join(reports_dir, "ablation_study.csv")
    df_ablation.to_csv(ablation_path, index=False)
    print(f"[+] Saved ablation study to {ablation_path}")
    
    # 2. Model Comparison
    # Baseline metrics from Phase 4
    comparison_data = [
        {"Model": "Logistic Regression", "Accuracy": 0.4762, "Macro F1": 0.3132, "Weighted F1": 0.3842, "Improvement Over RF": -0.3582},
        {"Model": "Decision Tree", "Accuracy": 0.6071, "Macro F1": 0.4995, "Weighted F1": 0.5844, "Improvement Over RF": -0.1719},
        {"Model": "Random Forest", "Accuracy": 0.7500, "Macro F1": 0.6714, "Weighted F1": 0.7502, "Improvement Over RF": 0.0},
        {"Model": "XGBoost", "Accuracy": 0.7024, "Macro F1": 0.6373, "Weighted F1": 0.7001, "Improvement Over RF": -0.0341},
        {"Model": "Model B (Embeddings Only)", "Accuracy": metrics_b["accuracy"], "Macro F1": metrics_b["f1_macro"], "Weighted F1": metrics_b["f1_weighted"], "Improvement Over RF": metrics_b["f1_macro"] - 0.6714},
        {"Model": "Hybrid Model C (Tabular + Embeddings)", "Accuracy": metrics_c["accuracy"], "Macro F1": metrics_c["f1_macro"], "Weighted F1": metrics_c["f1_weighted"], "Improvement Over RF": metrics_c["f1_macro"] - 0.6714}
    ]
    
    df_comparison = pd.DataFrame(comparison_data)
    comparison_path = os.path.join(reports_dir, "model_comparison.csv")
    df_comparison.to_csv(comparison_path, index=False)
    print(f"[+] Saved model comparison to {comparison_path}")
