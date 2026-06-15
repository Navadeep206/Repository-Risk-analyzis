#!/usr/bin/env python3
"""
Visualization generator module for Phase 6.
Renders training loss, accuracy, Macro F1 progression curves, and confusion matrix heatmaps.
"""

import os
import sys

# MUST be set before importing torch, sklearn or any OpenMP-linked library
# Prevents EXC_BAD_ACCESS (SIGSEGV) from duplicate libomp on macOS ARM64
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Ensure parent directory is in path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import BASE_DIR
from deep_learning.model import RepositoryRiskPredictor
from deep_learning.dataset_loader import get_dataloaders

REPORTS_DIR = os.path.join(BASE_DIR, "reports", "deep_learning")

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
    plt.title("CodeBERT+MLP Model - Training and Validation Loss", fontsize=14, fontweight="bold")
    plt.legend(fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.6)
    
    loss_path = os.path.join(REPORTS_DIR, "loss_curve.png")
    plt.tight_layout()
    plt.savefig(loss_path, dpi=300)
    plt.close()
    print(f"[+] Saved loss curve plot to {loss_path}")
    
    # 2. Validation F1 Curve
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, df["validation_f1_macro"], label="Val Macro F1", color="emerald" if "emerald" in plt.colormaps else "forestgreen", linewidth=2)
    plt.plot(epochs, df["validation_accuracy"], label="Val Accuracy", color="orange", linewidth=1.5, linestyle="--")
    plt.xlabel("Epochs", fontsize=12)
    plt.ylabel("Score", fontsize=12)
    plt.title("CodeBERT+MLP Model - Validation Metrics Progression", fontsize=14, fontweight="bold")
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
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            logits = model(X_batch)
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
    plt.title("Confusion Matrix - Deep Learning Risk Predictor", fontsize=13, fontweight="bold", pad=15)
    plt.tight_layout()
    
    cm_path = os.path.join(REPORTS_DIR, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"[+] Saved confusion matrix heatmap to {cm_path}")

def run_visualizations() -> None:
    history_csv_path = os.path.join(REPORTS_DIR, "training_metrics.csv")
    model_path = os.path.join(BASE_DIR, "models", "repository_risk_predictor.pt")
    
    # 1. Plot Loss & F1 curves
    plot_curves(history_csv_path)
    
    # 2. Plot Confusion Matrix
    if os.path.exists(model_path):
        _, _, test_loader, _ = get_dataloaders()
        
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
            
        model = RepositoryRiskPredictor()
        model.load_state_dict(torch.load(model_path, map_location=device))
        
        plot_confusion_matrix_heatmap(model, test_loader, device)
    else:
        print("[!] Model checkpoint not found, skipping confusion matrix heatmap.")

if __name__ == "__main__":
    run_visualizations()
