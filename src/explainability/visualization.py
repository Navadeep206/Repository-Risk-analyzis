#!/usr/bin/env python3
"""
Visualization module for Phase 8.
Generates:
- feature_importance.png
- permutation_importance.png
- domain_shift.png
- confusion_matrix.png
- class_distribution.png
Excludes all XGBoost imports and model loading.
"""

import os
import sys
import matplotlib
# Force Agg backend for headless systems
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix

# Ensure parent directory is in path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import BASE_DIR
from explainability.model_explainer import load_explainability_resources
from hybrid.hybrid_dataset import get_hybrid_dataloaders
from hybrid.fusion_model import HybridRiskPredictor

PLOTS_DIR = os.path.join(BASE_DIR, "reports", "explainability", "plots")

def generate_explainability_plots() -> None:
    """
    Renders and saves all explainability and distribution visualizations.
    """
    os.makedirs(PLOTS_DIR, exist_ok=True)
    sns.set_theme(style="whitegrid", context="talk")
    
    preproc, rf_model, X_train, y_train, X_val, y_val, X_test, y_test = load_explainability_resources()
    X_test_proc = preproc.transform(X_test)
    feature_names = preproc.feature_names
    
    # 1. Feature Importance Plot (Intrinsic RF)
    feat_imp_csv = os.path.join(BASE_DIR, "reports", "explainability", "feature_importance.csv")
    if os.path.exists(feat_imp_csv):
        df_imp = pd.read_csv(feat_imp_csv).sort_values(by="rf_intrinsic_importance", ascending=True)
        plt.figure(figsize=(10, 6))
        plt.barh(df_imp["feature_name"], df_imp["rf_intrinsic_importance"], color="royalblue", alpha=0.85)
        plt.xlabel("Gini Importance", fontweight="bold")
        plt.title("Random Forest Intrinsic Feature Importance", fontsize=15, fontweight="bold", pad=15)
        plt.tight_layout()
        
        path = os.path.join(PLOTS_DIR, "feature_importance.png")
        plt.savefig(path, dpi=300)
        plt.close()
        print(f"[+] Saved intrinsic feature importance plot to {path}")
        
    # 2. Permutation Importance Plot
    perm_csv = os.path.join(BASE_DIR, "reports", "explainability", "permutation_importance.csv")
    if os.path.exists(perm_csv):
        df_perm = pd.read_csv(perm_csv).sort_values(by="rf_permutation_mean", ascending=True)
        plt.figure(figsize=(10, 6))
        plt.barh(
            df_perm["feature_name"], 
            df_perm["rf_permutation_mean"], 
            xerr=df_perm["rf_permutation_std"],
            color="forestgreen",
            alpha=0.8,
            error_kw=dict(ecolor="black", lw=1.5, capsize=4)
        )
        plt.xlabel("Accuracy Drop (Permutation Mean)", fontweight="bold")
        plt.title("Random Forest Permutation Importance", fontsize=15, fontweight="bold", pad=15)
        plt.tight_layout()
        
        path = os.path.join(PLOTS_DIR, "permutation_importance.png")
        plt.savefig(path, dpi=300)
        plt.close()
        print(f"[+] Saved permutation importance plot to {path}")
        
    # 3. Domain Shift Plot (Density KDE plots for repository age)
    plt.figure(figsize=(10, 5))
    sns.kdeplot(X_train["repository_age_days"], fill=True, label="Train (click/axios/redux)", color="royalblue", alpha=0.5)
    sns.kdeplot(X_val["repository_age_days"], fill=True, label="Val (express)", color="orange", alpha=0.5)
    sns.kdeplot(X_test["repository_age_days"], fill=True, label="Test (databases/jinja)", color="red", alpha=0.5)
    plt.xlabel("Repository Age (Days)", fontweight="bold")
    plt.ylabel("Density", fontweight="bold")
    plt.title("Repository Age Distribution Shift Across Splits", fontsize=14, fontweight="bold", pad=15)
    plt.legend(fontsize=10)
    plt.tight_layout()
    
    path = os.path.join(PLOTS_DIR, "domain_shift.png")
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"[+] Saved domain shift distribution plot to {path}")
    
    # 4. Confusion Matrix Plot (RF vs. Hybrid)
    rf_preds = rf_model.predict(X_test_proc)
    rf_cm = confusion_matrix(y_test, rf_preds, labels=[0, 1, 2])
    
    # Run hybrid on CPU
    device = torch.device("cpu")
    _, _, test_loader, _ = get_hybrid_dataloaders(batch_size=32)
    hybrid_model_path = os.path.join(BASE_DIR, "models", "hybrid_risk_predictor.pt")
    hybrid_model = HybridRiskPredictor()
    hybrid_model.load_state_dict(torch.load(hybrid_model_path, map_location=device))
    hybrid_model.to(device)
    hybrid_model.eval()
    
    hybrid_preds = []
    with torch.no_grad():
        for X_tab, X_emb, _ in test_loader:
            logits = hybrid_model(X_tab.to(device), X_emb.to(device))
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            hybrid_preds.extend(preds)
    hybrid_preds = np.array(hybrid_preds)
    hybrid_cm = confusion_matrix(y_test, hybrid_preds, labels=[0, 1, 2])
    
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    
    # Plot RF Matrix
    sns.heatmap(
        rf_cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["LOW", "MEDIUM", "HIGH"],
        yticklabels=["LOW", "MEDIUM", "HIGH"],
        cbar=False,
        ax=axes[0],
        annot_kws={"size": 13, "weight": "bold"}
    )
    axes[0].set_ylabel("Actual Label", fontweight="bold")
    axes[0].set_xlabel("Predicted Label", fontweight="bold")
    axes[0].set_title("Random Forest Confusion Matrix", fontsize=12, fontweight="bold")
    
    # Plot Hybrid Matrix
    sns.heatmap(
        hybrid_cm,
        annot=True,
        fmt="d",
        cmap="Oranges",
        xticklabels=["LOW", "MEDIUM", "HIGH"],
        yticklabels=["LOW", "MEDIUM", "HIGH"],
        cbar=False,
        ax=axes[1],
        annot_kws={"size": 13, "weight": "bold"}
    )
    axes[1].set_ylabel("Actual Label", fontweight="bold")
    axes[1].set_xlabel("Predicted Label", fontweight="bold")
    axes[1].set_title("Hybrid Fusion Confusion Matrix", fontsize=12, fontweight="bold")
    
    plt.suptitle("Confusion Matrix Comparison (Test Split)", fontsize=15, fontweight="bold")
    plt.tight_layout()
    
    path = os.path.join(PLOTS_DIR, "confusion_matrix.png")
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"[+] Saved confusion matrix comparison plot to {path}")
    
    # 5. Class Distribution Plot across splits
    plt.figure(figsize=(10, 5))
    
    train_dist = pd.Series(y_train).value_counts(normalize=True).sort_index()
    val_dist = pd.Series(y_val).value_counts(normalize=True).sort_index()
    test_dist = pd.Series(y_test).value_counts(normalize=True).sort_index()
    
    df_dist = pd.DataFrame({
        "LOW": [train_dist.get(0, 0), val_dist.get(0, 0), test_dist.get(0, 0)],
        "MEDIUM": [train_dist.get(1, 0), val_dist.get(1, 0), test_dist.get(1, 0)],
        "HIGH": [train_dist.get(2, 0), val_dist.get(2, 0), test_dist.get(2, 0)],
        "Split": ["Train", "Val", "Test"]
    }).melt(id_vars="Split", var_name="Class", value_name="Proportion")
    
    sns.barplot(data=df_dist, x="Split", y="Proportion", hue="Class", palette="viridis")
    plt.xlabel("Dataset Split", fontweight="bold")
    plt.ylabel("Proportion", fontweight="bold")
    plt.title("Target Label Class Proportions Across Splits", fontsize=14, fontweight="bold", pad=15)
    plt.tight_layout()
    
    path = os.path.join(PLOTS_DIR, "class_distribution.png")
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"[+] Saved class distribution plot to {path}")

if __name__ == "__main__":
    generate_explainability_plots()
