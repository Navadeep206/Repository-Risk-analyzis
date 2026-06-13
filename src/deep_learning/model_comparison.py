#!/usr/bin/env python3
"""
Model comparison aggregator module for Phase 6.
Aggregates baseline performance and deep learning metrics to output reports/deep_learning/model_comparison.csv.
"""

import os
import json
import sys
import pandas as pd

# Ensure parent directory is in path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import BASE_DIR

REPORTS_DIR = os.path.join(BASE_DIR, "reports", "deep_learning")

def compile_model_comparison() -> None:
    """
    Loads baseline model statistics and merges them with the deep learning model's
    test set metrics to create the final comparison sheet.
    """
    dl_metrics_path = os.path.join(REPORTS_DIR, "best_model_metrics.json")
    if not os.path.exists(dl_metrics_path):
        raise FileNotFoundError(f"Deep learning metrics json not found: {dl_metrics_path}. Run evaluator first.")
        
    with open(dl_metrics_path, "r") as f:
        dl_metrics = json.load(f)
        
    # Baseline test metrics from Phase 4
    records = [
        {
            "Model": "Logistic Regression",
            "Accuracy": 0.4762,
            "Macro F1": 0.3132,
            "Weighted F1": 0.3842
        },
        {
            "Model": "Decision Tree",
            "Accuracy": 0.6071,
            "Macro F1": 0.4995,
            "Weighted F1": 0.5844
        },
        {
            "Model": "Random Forest",
            "Accuracy": 0.7500,
            "Macro F1": 0.6714,
            "Weighted F1": 0.7502
        },
        {
            "Model": "XGBoost",
            "Accuracy": 0.7024,
            "Macro F1": 0.6373,
            "Weighted F1": 0.7001
        },
        {
            "Model": "Deep Learning (CodeBERT + MLP)",
            "Accuracy": dl_metrics["test_accuracy"],
            "Macro F1": dl_metrics["test_f1_macro"],
            "Weighted F1": dl_metrics["test_f1_weighted"]
        }
    ]
    
    df_comp = pd.DataFrame(records)
    
    # Calculate improvement over Random Forest (top baseline model)
    rf_macro_f1 = 0.6714
    
    def calc_improvement(row: pd.Series) -> str:
        diff = row["Macro F1"] - rf_macro_f1
        if row["Model"] == "Random Forest":
            return "-"
        return f"{diff:+.4f}"
        
    df_comp["Improvement Over RF"] = df_comp.apply(calc_improvement, axis=1)
    
    output_path = os.path.join(REPORTS_DIR, "model_comparison.csv")
    df_comp.to_csv(output_path, index=False)
    print(f"[+] Saved comparison matrix to {output_path}")
    print("\n" + "="*50)
    print("Model Comparison Summary (Test Split):")
    print("="*50)
    print(df_comp.to_string(index=False))
    print("="*50 + "\n")

if __name__ == "__main__":
    compile_model_comparison()
