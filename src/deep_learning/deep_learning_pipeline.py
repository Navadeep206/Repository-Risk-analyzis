#!/usr/bin/env python3
"""
Master orchestrator script for the Phase 6 Deep Learning Pipeline.
Sequences dataset splitting, training, validation early stopping, test set evaluation, visualization, and comparison.
"""

import os
import sys

# MUST be set before importing torch, sklearn or any OpenMP-linked library
# Prevents EXC_BAD_ACCESS (SIGSEGV) from duplicate libomp on macOS ARM64
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import torch

# Ensure parent directory is in path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import BASE_DIR, ensure_dirs_exist
from deep_learning.dataset_loader import get_dataloaders
from deep_learning.model import RepositoryRiskPredictor
from deep_learning.trainer import train_network
from deep_learning.evaluator import evaluate_network
from deep_learning.visualization import plot_curves, plot_confusion_matrix_heatmap
from deep_learning.model_comparison import compile_model_comparison

def run_dl_pipeline() -> None:
    """
    Executes the entire deep learning prediction pipeline.
    """
    print("="*60)
    print("Starting Phase 6 Deep Learning Predictor Pipeline")
    print("="*60)
    
    # 0. Folders setup
    ensure_dirs_exist()
    models_dir = os.path.join(BASE_DIR, "models")
    reports_dir = os.path.join(BASE_DIR, "reports", "deep_learning")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    
    # 1. Dataset Loading
    print("\n[STAGE 1/5] Loading datasets and initializing splits...")
    train_loader, val_loader, test_loader, class_weights = get_dataloaders()
    
    # Device setup (MPS GPU check)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        
    # 2. Model Initialization
    print("\n[STAGE 2/5] Initializing Neural Network Classifier (768-D MLP)...")
    model = RepositoryRiskPredictor()
    
    # 3. Model Training
    print("\n[STAGE 3/5] Starting Network Training with Early Stopping...")
    history = train_network(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        class_weights=class_weights,
        epochs=100,
        lr=0.001,
        patience=10,
        device=device
    )
    
    # Save optimal checkpoint
    model_save_path = os.path.join(models_dir, "repository_risk_predictor.pt")
    torch.save(model.state_dict(), model_save_path)
    print(f"[+] Saved best checkpoint weights to {model_save_path}")
    
    # Save training history to CSV
    import pandas as pd
    history_df = pd.DataFrame(history)
    csv_save_path = os.path.join(reports_dir, "training_metrics.csv")
    history_df.to_csv(csv_save_path, index=False)
    print(f"[+] Saved training history metrics to {csv_save_path}")
    
    # 4. Evaluation and Visualizations
    print("\n[STAGE 4/5] Evaluating optimal model on test split & rendering plots...")
    
    # Compute metrics
    metrics, text_report, cm = evaluate_network(model, test_loader, device)
    
    # Save metrics JSON
    json_path = os.path.join(reports_dir, "best_model_metrics.json")
    with open(json_path, "w") as f:
        __import__("json").dump(metrics, f, indent=4)
        
    # Save text report
    txt_path = os.path.join(reports_dir, "classification_report.txt")
    with open(txt_path, "w") as f:
        f.write(text_report)
        f.write("\n\nConfusion Matrix:\n")
        f.write(cm.__repr__())
        
    # Render curves & confusion heats
    plot_curves(csv_save_path)
    plot_confusion_matrix_heatmap(model, test_loader, device)
    
    # 5. Model Comparison Sheet
    print("\n[STAGE 5/5] Compiling final model comparison spreadsheet...")
    compile_model_comparison()
    
    print("\n" + "="*60)
    print("Phase 6 Deep Learning Predictor Completed Successfully!")
    print("="*60)

if __name__ == "__main__":
    run_dl_pipeline()
