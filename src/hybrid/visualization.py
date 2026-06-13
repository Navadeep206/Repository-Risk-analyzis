#!/usr/bin/env python3
"""
Visualization generator module for Phase 7.
Renders training loss, accuracy, Macro F1 progression curves, and confusion matrix heatmaps.
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Ensure parent directory is in path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import BASE_DIR

REPORTS_DIR = os.path.join(BASE_DIR, "reports", "hybrid")

def plot_curves(history_csv_path: str) -> None:
    """
    Plots training loss vs validation loss, and validation Macro F1 score over epochs.
    """
    if not os.path.exists(history_csv_path):
        print(f"[!] Warning: Training history CSV not found: {history_csv_path}. Skipping curves plotting.")
        return
        
    df = pd.read_csv(history_csv_path)
    epochs = df["epoch"].values
    
    # 1. Loss Curve
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, df["train_loss"], label="Train Loss", color="royalblue", linewidth=2)
    plt.plot(epochs, df["validation_loss"], label="Val Loss", color="tomato", linewidth=2)
    plt.xlabel("Epochs", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.title("Hybrid Model - Training and Validation Loss", fontsize=14, fontweight="bold")
    plt.legend(fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.6)
    
    loss_path = os.path.join(REPORTS_DIR, "loss_curve.png")
    plt.tight_layout()
    plt.savefig(loss_path, dpi=300)
    plt.close()
    print(f"[+] Saved loss curve plot to {loss_path}")
    
    # 2. Validation F1 Curve
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, df["validation_f1_macro"], label="Val Macro F1", color="forestgreen", linewidth=2)
    plt.plot(epochs, df["validation_accuracy"], label="Val Accuracy", color="orange", linewidth=1.5, linestyle="--")
    plt.xlabel("Epochs", fontsize=12)
    plt.ylabel("Score", fontsize=12)
    plt.title("Hybrid Model - Validation Metrics Progression", fontsize=14, fontweight="bold")
    plt.legend(fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.6)
    
    f1_path = os.path.join(REPORTS_DIR, "f1_curve.png")
    plt.tight_layout()
    plt.savefig(f1_path, dpi=300)
    plt.close()
    print(f"[+] Saved metrics progression curve plot to {f1_path}")

def plot_confusion_matrix_heatmap(model: torch.nn.Module, test_loader: torch.utils.data.DataLoader, device: torch.device) -> None:
    """
    Generates and saves a confusion matrix heatmap on the test set.
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
            
    cm = confusion_matrix(all_targets, all_preds, labels=[0, 1, 2])
    
    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["LOW", "MEDIUM", "HIGH"],
        yticklabels=["LOW", "MEDIUM", "HIGH"],
        cbar=True,
        annot_kws={"size": 13, "weight": "bold"}
    )
    plt.ylabel("Actual Label", fontsize=12, fontweight="bold")
    plt.xlabel("Predicted Label", fontsize=12, fontweight="bold")
    plt.title("Confusion Matrix - Hybrid Risk Predictor", fontsize=13, fontweight="bold", pad=15)
    plt.tight_layout()
    
    cm_path = os.path.join(REPORTS_DIR, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"[+] Saved confusion matrix heatmap to {cm_path}")
