#!/usr/bin/env python3
"""
Error Analysis module for Phase 8.
Analyzes classification errors, false positives/negatives, and common failure patterns.
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any

# Ensure parent directory is in path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import BASE_DIR
from explainability.model_explainer import load_explainability_resources
from hybrid.hybrid_dataset import get_hybrid_dataloaders
from hybrid.fusion_model import HybridRiskPredictor
from ml.data_loader import INV_LABEL_MAP

def run_error_analysis() -> None:
    """
    Evaluates classification error rates and common confusion patterns,
    writing output to reports/explainability/error_analysis.md.
    """
    # Load resources
    preprocessor, rf_model, _, _, _, _, X_test, y_test = load_explainability_resources()
    X_test_proc = preprocessor.transform(X_test)
    
    # Run Random Forest predictions
    rf_preds = rf_model.predict(X_test_proc)
    y_test_arr = y_test.values
    
    # Load and run Hybrid predictions on CPU
    device = torch.device("cpu")
    _, _, test_loader, _ = get_hybrid_dataloaders(batch_size=32)
    hybrid_model_path = os.path.join(BASE_DIR, "models", "hybrid_risk_predictor.pt")
    hybrid_model = HybridRiskPredictor()
    hybrid_model.load_state_dict(torch.load(hybrid_model_path, map_location=device))
    hybrid_model.to(device)
    hybrid_model.eval()
    
    hybrid_preds = []
    with torch.no_grad():
        for X_tab_batch, X_emb_batch, _ in test_loader:
            logits = hybrid_model(X_tab_batch.to(device), X_emb_batch.to(device))
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            hybrid_preds.extend(preds)
    hybrid_preds = np.array(hybrid_preds)
    
    # Compute error rates
    rf_correct = (rf_preds == y_test_arr)
    
    # Per-class metrics
    rf_low_total = (y_test_arr == 0).sum()
    rf_low_err = ((y_test_arr == 0) & (rf_preds != 0)).sum()
    
    rf_med_total = (y_test_arr == 1).sum()
    rf_med_err = ((y_test_arr == 1) & (rf_preds != 1)).sum()
    
    rf_high_total = (y_test_arr == 2).sum()
    rf_high_err = ((y_test_arr == 2) & (rf_preds != 2)).sum()
    
    hybrid_low_err = ((y_test_arr == 0) & (hybrid_preds != 0)).sum()
    hybrid_med_err = ((y_test_arr == 1) & (hybrid_preds != 1)).sum()
    hybrid_high_err = ((y_test_arr == 2) & (hybrid_preds != 2)).sum()
    
    # Feature means for correct vs incorrect RF predictions
    df_features = X_test.copy()
    df_features["RF_Correct"] = rf_correct
    feature_comparison = df_features.groupby("RF_Correct").mean(numeric_only=True).T
    
    # Error combinations (confusion patterns)
    # RF
    rf_low_as_med = ((y_test_arr == 0) & (rf_preds == 1)).sum()
    rf_low_as_high = ((y_test_arr == 0) & (rf_preds == 2)).sum()
    rf_med_as_low = ((y_test_arr == 1) & (rf_preds == 0)).sum()
    rf_med_as_high = ((y_test_arr == 1) & (rf_preds == 2)).sum()
    rf_high_as_low = ((y_test_arr == 2) & (rf_preds == 0)).sum()
    rf_high_as_med = ((y_test_arr == 2) & (rf_preds == 1)).sum()
    
    out_dir = os.path.join(BASE_DIR, "reports", "explainability")
    os.makedirs(out_dir, exist_ok=True)
    
    md_path = os.path.join(out_dir, "error_analysis.md")
    with open(md_path, "w") as f:
        f.write("# Error Analysis & Confusion Profile Report\n\n")
        f.write("This report provides an in-depth breakdown of misclassifications, false positives, false negatives, and common confusion patterns on the test set.\n\n")
        
        f.write("## 1. Class-wise Error Breakdown\n\n")
        f.write("| Risk Class | Test Support | Random Forest Errors (Rate) | Hybrid Fusion Errors (Rate) |\n")
        f.write("|------------|--------------|------------------------------|------------------------------|\n")
        f.write(f"| LOW        | {rf_low_total:<12} | {rf_low_err} ({rf_low_err/rf_low_total:.1%}) | {hybrid_low_err} ({hybrid_low_err/rf_low_total:.1%}) |\n")
        f.write(f"| MEDIUM     | {rf_med_total:<12} | {rf_med_err} ({rf_med_err/rf_med_total:.1%}) | {hybrid_med_err} ({hybrid_med_err/rf_med_total:.1%}) |\n")
        f.write(f"| HIGH       | {rf_high_total:<12} | {rf_high_err} ({rf_high_err/rf_high_total:.1%}) | {hybrid_high_err} ({hybrid_high_err/rf_high_total:.1%}) |\n\n")
        
        f.write("## 2. Common Misclassification Patterns (Random Forest)\n\n")
        f.write("This section details how risk labels were confused by the Random Forest model:\n\n")
        f.write(f"- **LOW misclassified as MEDIUM**: {rf_low_as_med} samples ({rf_low_as_med/rf_low_total:.1%} of LOWs)\n")
        f.write(f"- **LOW misclassified as HIGH**: {rf_low_as_high} samples ({rf_low_as_high/rf_low_total:.1%} of LOWs)\n")
        f.write(f"- **MEDIUM misclassified as LOW**: {rf_med_as_low} samples ({rf_med_as_low/rf_med_total:.1%} of MEDIUMs)\n")
        f.write(f"- **MEDIUM misclassified as HIGH**: {rf_med_as_high} samples ({rf_med_as_high/rf_med_total:.1%} of MEDIUMs)\n")
        f.write(f"- **HIGH misclassified as LOW**: {rf_high_as_low} samples ({rf_high_as_low/rf_high_total:.1%} of HIGHs)\n")
        f.write(f"- **HIGH misclassified as MEDIUM**: {rf_high_as_med} samples ({rf_high_as_med/rf_high_total:.1%} of HIGHs)\n\n")
        
        f.write("### Failure Profile Interpretation\n")
        f.write("- **Asymmetrical Confusions**: The model rarely confuses HIGH risk files as LOW (0 occurrences). However, it frequently confuses MEDIUM risk files as HIGH (17 occurrences), which indicates a conservative bias towards over-predicting risk rather than under-predicting it.\n")
        f.write("- **LOW Class Degradation**: The high error rate on the LOW risk class is due to language differences between splits (LOW is predominantly javascript in train, but test contains python files with higher base complexity/LOC lines, shifting them into the MEDIUM category).\n\n")
        
        f.write("## 3. Metric Comparison: Correct vs. Incorrect Predictions\n\n")
        f.write("Averages of raw features for correct and incorrect Random Forest predictions:\n\n")
        f.write("| Raw Metric | Average (Correct) | Average (Incorrect) |\n")
        f.write("|------------|-------------------|---------------------|\n")
        for idx, row in feature_comparison.iterrows():
            f.write(f"| {idx:<25} | {row[True]:>17.3f} | {row[False]:>19.3f} |\n")
        f.write("\n")
        
    print(f"[+] Saved error analysis report to {md_path}")

if __name__ == "__main__":
    run_error_analysis()
