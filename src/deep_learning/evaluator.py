#!/usr/bin/env python3
"""
Model evaluation module for Phase 6.
Evaluates the best trained deep learning checkpoint on the Test loader and exports metric files.
"""

import os
import json
import sys
import torch
import numpy as np
from typing import Dict, Any, Tuple
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report

# Ensure parent directory is in path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import BASE_DIR
from deep_learning.model import RepositoryRiskPredictor
from deep_learning.dataset_loader import get_dataloaders, INV_LABEL_MAP

def evaluate_network(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device = torch.device("cpu")
) -> Tuple[Dict[str, Any], str, np.ndarray]:
    """
    Evaluates the model on the test loader, returning standard metrics,
    the classification report string, and the confusion matrix.
    """
    model.eval()
    model.to(device)
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            logits = model(X_batch)
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y_batch.numpy())
            
    # Calculate scores
    acc = accuracy_score(all_targets, all_preds)
    p_mac, r_mac, f1_mac, _ = precision_recall_fscore_support(all_targets, all_preds, average="macro", zero_division=0)
    p_wei, r_wei, f1_wei, _ = precision_recall_fscore_support(all_targets, all_preds, average="weighted", zero_division=0)
    
    # Text Report
    text_report = classification_report(
        all_targets,
        all_preds,
        target_names=["LOW", "MEDIUM", "HIGH"],
        zero_division=0,
        labels=[0, 1, 2]
    )
    
    # Confusion Matrix
    cm = confusion_matrix(all_targets, all_preds, labels=[0, 1, 2])
    
    metrics = {
        "test_accuracy": float(acc),
        "test_precision_macro": float(p_mac),
        "test_recall_macro": float(r_mac),
        "test_f1_macro": float(f1_mac),
        "test_precision_weighted": float(p_wei),
        "test_recall_weighted": float(r_wei),
        "test_f1_weighted": float(f1_wei)
    }
    
    return metrics, text_report, cm

def run_evaluation() -> None:
    models_dir = os.path.join(BASE_DIR, "models")
    reports_dir = os.path.join(BASE_DIR, "reports", "deep_learning")
    os.makedirs(reports_dir, exist_ok=True)
    
    model_path = os.path.join(models_dir, "repository_risk_predictor.pt")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}. Run training first.")
        
    _, _, test_loader, _ = get_dataloaders()
    
    # Device check
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        
    model = RepositoryRiskPredictor()
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    metrics, text_report, cm = evaluate_network(model, test_loader, device)
    
    # Save best_model_metrics.json
    json_path = os.path.join(reports_dir, "best_model_metrics.json")
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"[+] Saved metrics json to {json_path}")
    
    # Save classification_report.txt
    txt_path = os.path.join(reports_dir, "classification_report.txt")
    with open(txt_path, "w") as f:
        f.write(text_report)
        f.write("\n\nConfusion Matrix:\n")
        f.write(np.array2string(cm))
    print(f"[+] Saved text classification report to {txt_path}")
    
    # Print metrics summary
    print("\n" + "="*40)
    print("Deep Learning Test Split Performance:")
    print("="*40)
    print(f"Accuracy:        {metrics['test_accuracy']:.4f}")
    print(f"Macro F1-Score:  {metrics['test_f1_macro']:.4f}")
    print(f"Weighted F1:     {metrics['test_f1_weighted']:.4f}")
    print(f"Confusion Matrix:\n{cm}")
    print("="*40 + "\n")

if __name__ == "__main__":
    run_evaluation()
