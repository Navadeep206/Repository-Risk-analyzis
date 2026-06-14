#!/usr/bin/env python3
"""
Master orchestrator script for the Phase 7 Hybrid Intelligence Pipeline.
Sequences dataset merging, training Model A (Tabular Only), Model B (Embeddings Only), 
and Model C (Hybrid Fusion) with early stopping, test set evaluation, 
ablation compiling, model comparisons, and visualizations.
"""

import os
import sys
import json
import torch
import pandas as pd
import numpy as np

# Ensure parent directory is in path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import BASE_DIR, ensure_dirs_exist
from hybrid.hybrid_dataset import get_hybrid_dataloaders
from hybrid.fusion_model import TabularOnlyPredictor, EmbeddingOnlyPredictor, HybridRiskPredictor
from hybrid.trainer import train_hybrid_model
from hybrid.evaluator import evaluate_hybrid_model, compile_ablation_and_comparison
from hybrid.visualization import plot_curves, plot_confusion_matrix_heatmap

def run_hybrid_pipeline() -> None:
    """
    Executes the Phase 7 hybrid predictor pipeline.
    """
    print("="*60)
    print("Starting Phase 7 Hybrid Intelligence Model Pipeline")
    print("="*60)
    
    # 0. Folders setup
    ensure_dirs_exist()
    models_dir = os.path.join(BASE_DIR, "models")
    reports_dir = os.path.join(BASE_DIR, "reports", "hybrid")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    
    # Device setup (MPS GPU check)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"[+] Using device: {device}")
        
    # 1. Dataset Loading
    print("\n[STAGE 1/4] Loading datasets and initializing joint splits...")
    train_loader, val_loader, test_loader, class_weights = get_hybrid_dataloaders(batch_size=32)
    tabular_dim = train_loader.dataset.X_tabular.shape[1]
    print(f"[+] Detected tabular feature dimension: {tabular_dim}")
    
    # 2. Train Ablation Study Models
    print("\n[STAGE 2/4] Training Ablation Study Models...")
    
    # --- Model A: Tabular Only ---
    print("\n>>> Training Model A: Tabular Only Predictor...")
    model_a = TabularOnlyPredictor(input_dim=tabular_dim)
    history_a = train_hybrid_model(
        model=model_a,
        train_loader=train_loader,
        val_loader=val_loader,
        class_weights=class_weights,
        epochs=100,
        lr=0.0005,
        patience=15,
        device=device
    )
    model_a_path = os.path.join(models_dir, "tabular_only_predictor.pt")
    torch.save(model_a.state_dict(), model_a_path)
    print(f"[+] Saved Model A weights to {model_a_path}")
    
    # --- Model B: Embeddings Only ---
    print("\n>>> Training Model B: Embeddings Only Predictor...")
    model_b = EmbeddingOnlyPredictor()
    history_b = train_hybrid_model(
        model=model_b,
        train_loader=train_loader,
        val_loader=val_loader,
        class_weights=class_weights,
        epochs=100,
        lr=0.0005,
        patience=15,
        device=device
    )
    model_b_path = os.path.join(models_dir, "embedding_only_predictor.pt")
    torch.save(model_b.state_dict(), model_b_path)
    print(f"[+] Saved Model B weights to {model_b_path}")
    
    # --- Model C: Hybrid Fusion ---
    print("\n>>> Training Model C: Hybrid Fusion Predictor...")
    model_c = HybridRiskPredictor(tabular_dim=tabular_dim)
    history_c = train_hybrid_model(
        model=model_c,
        train_loader=train_loader,
        val_loader=val_loader,
        class_weights=class_weights,
        epochs=100,
        lr=0.0005,
        patience=15,
        device=device
    )
    model_c_path = os.path.join(models_dir, "hybrid_risk_predictor.pt")
    torch.save(model_c.state_dict(), model_c_path)
    print(f"[+] Saved Model C weights to {model_c_path}")
    
    # Save training history for Model C (fusion)
    history_df = pd.DataFrame(history_c)
    history_csv_path = os.path.join(reports_dir, "training_metrics.csv")
    history_df.to_csv(history_csv_path, index=False)
    print(f"[+] Saved fusion training metrics to {history_csv_path}")

    # 3. Model Evaluation
    print("\n[STAGE 3/4] Evaluating models on Test split...")
    metrics_a, _, _ = evaluate_hybrid_model(model_a, test_loader, device)
    metrics_b, _, _ = evaluate_hybrid_model(model_b, test_loader, device)
    metrics_c, text_report_c, cm_c = evaluate_hybrid_model(model_c, test_loader, device)
    
    print("\n" + "="*45)
    print("Ablation Study Results (Test Set):")
    print(f"Model A (Tabular Only):   Acc = {metrics_a['accuracy']:.4f} | Macro F1 = {metrics_a['f1_macro']:.4f}")
    print(f"Model B (Embeddings Only): Acc = {metrics_b['accuracy']:.4f} | Macro F1 = {metrics_b['f1_macro']:.4f}")
    print(f"Model C (Hybrid Fusion):   Acc = {metrics_c['accuracy']:.4f} | Macro F1 = {metrics_c['f1_macro']:.4f}")
    print("="*45)
    
    # Compile comparison and ablation spreadsheets
    compile_ablation_and_comparison(metrics_a, metrics_b, metrics_c)
    
    # Save best metrics JSON for Model C
    json_path = os.path.join(reports_dir, "best_model_metrics.json")
    with open(json_path, "w") as f:
        json.dump(metrics_c, f, indent=4)
    print(f"[+] Saved best model metrics JSON to {json_path}")
        
    # Save text classification report for Model C
    txt_path = os.path.join(reports_dir, "classification_report.txt")
    with open(txt_path, "w") as f:
        f.write(text_report_c)
        f.write("\n\nConfusion Matrix:\n")
        f.write(np.array2string(cm_c))
    print(f"[+] Saved classification report to {txt_path}")
    
    # 4. Generate Visualizations for Model C
    print("\n[STAGE 4/4] Rendering curves & confusion heatmap for Hybrid Model...")
    plot_curves(history_csv_path)
    plot_confusion_matrix_heatmap(model_c, test_loader, device)
    
    # Performance check log
    rf_baseline_f1 = 0.6714
    if metrics_c["f1_macro"] > rf_baseline_f1:
        print(f"\n[SUCCESS] Hybrid Fusion Model Test Macro F1 ({metrics_c['f1_macro']:.4f}) successfully outperformed Random Forest baseline ({rf_baseline_f1:.4f})!")
    else:
        print(f"\n[NOTICE] Hybrid Fusion Model Test Macro F1 ({metrics_c['f1_macro']:.4f}) did not beat Random Forest baseline ({rf_baseline_f1:.4f}).")
        print("This is documented in the final report.")
        
    print("\n" + "="*60)
    print("Phase 7 Hybrid Predictor Pipeline Completed Successfully!")
    print("="*60)

if __name__ == "__main__":
    run_hybrid_pipeline()
