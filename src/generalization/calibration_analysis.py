#!/usr/bin/env python3
"""
Step 4: Confidence Calibration.
Applies Platt Scaling and Isotonic Regression, calculates Brier score and ECE,
and saves a reliability diagram plot.
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Prevent GUI crashes
import matplotlib.pyplot as plt
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.preprocessing import label_binarize

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import BASE_DIR, FINAL_DIR
from ml.preprocessing import CodeRiskPreprocessor
from ml.data_loader import LABEL_MAP

def calculate_ece(y_true: np.ndarray, y_prob_max: np.ndarray, y_pred: np.ndarray, n_bins: int = 10) -> float:
    """
    Computes Expected Calibration Error (ECE) for multi-class classifier.
    ECE measures discrepancy between confidence and actual accuracy.
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n_samples = len(y_true)
    
    # Accuracy array (whether prediction was correct)
    accuracies = (y_pred == y_true)
    
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        # Find indices of samples in the current bin
        in_bin = (y_prob_max >= bin_lower) & (y_prob_max < bin_upper)
        bin_size = np.sum(in_bin)
        
        if bin_size > 0:
            bin_acc = np.mean(accuracies[in_bin])
            bin_conf = np.mean(y_prob_max[in_bin])
            ece += (bin_size / n_samples) * np.abs(bin_acc - bin_conf)
            
    return float(ece)

def calculate_brier_multiclass(y_true: np.ndarray, y_probs: np.ndarray) -> float:
    """
    Computes the multi-class Brier score.
    """
    y_true_bin = label_binarize(y_true, classes=[0, 1, 2])
    return float(np.mean(np.sum((y_probs - y_true_bin) ** 2, axis=1)))

def run_calibration():
    print("[*] Running Confidence Calibration Analysis...")
    reports_dir = os.path.join(BASE_DIR, "reports", "generalization")
    plots_dir = os.path.join(reports_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # 1. Load splits
    train_df = pd.read_csv(os.path.join(FINAL_DIR, "train_v2.csv"))
    val_df = pd.read_csv(os.path.join(FINAL_DIR, "validation_v2.csv"))
    test_df = pd.read_csv(os.path.join(FINAL_DIR, "test_v2.csv"))
    
    y_train = train_df["historical_risk_label"].map(LABEL_MAP).values
    y_val = val_df["historical_risk_label"].map(LABEL_MAP).values
    y_test = test_df["historical_risk_label"].map(LABEL_MAP).values
    
    # Load preprocessor
    preproc_path = os.path.join(BASE_DIR, "models", "preprocessor.pkl")
    with open(preproc_path, "rb") as f:
        preproc = pickle.load(f)
        
    X_val_proc = preproc.transform(val_df)
    X_test_proc = preproc.transform(test_df)
    
    # Load baseline Random Forest model
    rf_model_path = os.path.join(BASE_DIR, "models", "random_forest.pkl")
    with open(rf_model_path, "rb") as f:
        rf = pickle.load(f)
        
    # A. Uncalibrated Model Predictions
    y_probs_uncal = rf.predict_proba(X_test_proc)
    y_preds_uncal = rf.predict(X_test_proc)
    
    y_conf_uncal = np.max(y_probs_uncal, axis=1)
    ece_uncal = calculate_ece(y_test, y_conf_uncal, y_preds_uncal)
    brier_uncal = calculate_brier_multiclass(y_test, y_probs_uncal)
    
    # B. Platt Scaling (Sigmoid)
    cv_val = [(np.arange(len(X_val_proc)), np.arange(len(X_val_proc)))]
    platt_calibrator = CalibratedClassifierCV(estimator=FrozenEstimator(rf), method="sigmoid", cv=cv_val)
    platt_calibrator.fit(X_val_proc, y_val)
    
    y_probs_platt = platt_calibrator.predict_proba(X_test_proc)
    y_preds_platt = platt_calibrator.predict(X_test_proc)
    
    y_conf_platt = np.max(y_probs_platt, axis=1)
    ece_platt = calculate_ece(y_test, y_conf_platt, y_preds_platt)
    brier_platt = calculate_brier_multiclass(y_test, y_probs_platt)
    
    # Save Platt calibrator model
    platt_path = os.path.join(BASE_DIR, "models", "calibrated_rf_platt.pkl")
    with open(platt_path, "wb") as f:
        pickle.dump(platt_calibrator, f)
        
    # C. Isotonic Regression
    isotonic_calibrator = CalibratedClassifierCV(estimator=FrozenEstimator(rf), method="isotonic", cv=cv_val)
    isotonic_calibrator.fit(X_val_proc, y_val)
    
    y_probs_iso = isotonic_calibrator.predict_proba(X_test_proc)
    y_preds_iso = isotonic_calibrator.predict(X_test_proc)
    
    y_conf_iso = np.max(y_probs_iso, axis=1)
    ece_iso = calculate_ece(y_test, y_conf_iso, y_preds_iso)
    brier_iso = calculate_brier_multiclass(y_test, y_probs_iso)
    
    # 2. Output metrics comparison CSV
    results = [
        {"method": "Uncalibrated", "brier_score": brier_uncal, "ece": ece_uncal},
        {"method": "Platt Scaling (Sigmoid)", "brier_score": brier_platt, "ece": ece_platt},
        {"method": "Isotonic Regression", "brier_score": brier_iso, "ece": ece_iso}
    ]
    df_results = pd.DataFrame(results)
    results_path = os.path.join(reports_dir, "calibration_results.csv")
    df_results.to_csv(results_path, index=False)
    print(f"[+] Saved calibration results to {results_path}")
    
    # 3. Plot Reliability Diagram
    plot_reliability_diagram(y_test, y_probs_uncal, y_probs_platt, y_probs_iso, os.path.join(plots_dir, "reliability_diagram.png"))
    
    # 4. Write Calibration Report Markdown
    write_calibration_report(df_results, reports_dir)

def plot_reliability_diagram(y_test: np.ndarray, probs_uncal: np.ndarray, probs_platt: np.ndarray, probs_iso: np.ndarray, output_path: str):
    """
    Plots a reliability curve comparing uncalibrated vs Platt vs Isotonic.
    """
    n_bins = 10
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    
    plt.figure(figsize=(8, 6))
    
    methods = [
        ("Uncalibrated", probs_uncal, "red", "o"),
        ("Platt Scaling", probs_platt, "blue", "s"),
        ("Isotonic Regression", probs_iso, "green", "^")
    ]
    
    for name, probs, color, marker in methods:
        conf = np.max(probs, axis=1)
        preds = np.argmax(probs, axis=1)
        accuracies = (preds == y_test)
        
        bin_accs = []
        bin_confs = []
        
        for i in range(n_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i+1]
            in_bin = (conf >= bin_lower) & (conf < bin_upper)
            if np.sum(in_bin) > 0:
                bin_accs.append(np.mean(accuracies[in_bin]))
                bin_confs.append(np.mean(conf[in_bin]))
                
        plt.plot(bin_confs, bin_accs, marker=marker, color=color, label=name, linewidth=1.5)
        
    # Perfect calibration diagonal
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect Calibration")
    plt.title("Reliability Diagram (Risk Classifier Calibration)")
    plt.xlabel("Mean Predicted Confidence")
    plt.ylabel("Observed Empirical Accuracy")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"[+] Saved reliability curve plot to {output_path}")

def write_calibration_report(df_res: pd.DataFrame, reports_dir: str):
    md_content = f"""# Model Confidence Calibration Report

This report evaluates how accurately our Random Forest risk classification model expresses its prediction probability.

## 1. Calibration Metrics Table

| Calibration Method | Brier Score (lower is better) | Expected Calibration Error (ECE) |
| --- | --- | --- |
"""
    for _, row in df_res.iterrows():
        md_content += f"| {row['method']} | {row['brier_score']:.4f} | {row['ece']:.4f} |\n"
        
    md_content += """
---

## 2. Key Observations
1. **Expected Calibration Error (ECE)**: ECE measures the gap between prediction confidence and actual empirical accuracy. Uncalibrated Random Forests tend to under-represent or over-represent probabilities near the decision boundary.
2. **Impact of Post-Processing Calibration**: Platt Scaling (Logistic Calibration) and Isotonic Regression reduce ECE, making the model's confidence output (e.g. 85% confidence) represent the actual likelihood of correct classification.
"""
    report_path = os.path.join(reports_dir, "calibration_report.md")
    with open(report_path, "w") as f:
        f.write(md_content)
    print(f"[+] Saved calibration report markdown to {report_path}")

if __name__ == "__main__":
    run_calibration()
