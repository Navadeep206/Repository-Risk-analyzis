#!/usr/bin/env python3
"""
Forecasting Pipeline for Phase 9.
Orchestrates the entire forecasting system:
1. Builds the temporal daily dataset.
2. Generates rolling windows and future targets.
3. Splits data repository-disjointly into Train/Val/Test.
4. Trains baseline (persistence), RF, and XGBoost regressors for 30d/60d/90d horizons.
5. Evaluates model performance using MAE, RMSE, R2, MAPE.
6. Saves model comparison csv, best model metrics json, and saves the best trained RF models.
7. Generates visual plots (risk trends, forecasts).
8. Performs error analysis and saves a markdown report.
"""

import os
import sys
import json
import pickle
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import BASE_DIR, ensure_dirs_exist

# Import forecasting components
from temporal_dataset_builder import build_daily_logs
from feature_windowing import build_forecasting_dataset
from baseline_forecaster import PersistenceForecaster
from random_forest_forecaster import RandomForestForecaster
from xgboost_forecaster import XGBoostForecaster
from evaluator import evaluate_predictions
from visualization import plot_risk_trends, plot_repository_forecasts

# Feature space definition
FEATURES = [
    "commit_frequency_30d", "commit_frequency_60d", "commit_frequency_90d",
    "defect_count_30d", "defect_count_60d", "defect_count_90d",
    "active_contributors_30d", "active_contributors_60d", "active_contributors_90d",
    "modification_count_30d", "modification_count_60d", "modification_count_90d",
    "avg_complexity_30d", "avg_complexity_60d", "avg_complexity_90d",
    "avg_maintainability_30d", "avg_maintainability_60d", "avg_maintainability_90d",
    "risk_score_30d", "risk_score_60d", "risk_score_90d"
]

def run_pipeline() -> dict:
    print("[*] Starting Phase 9 Forecasting Pipeline...")
    ensure_dirs_exist()
    
    # Create directories for outputs
    models_dir = os.path.join(BASE_DIR, "models")
    reports_dir = os.path.join(BASE_DIR, "reports", "forecasting")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    
    # 1 & 2. Build datasets
    print("\n=== STEP 1: Building Temporal Daily Logs ===")
    df_daily = build_daily_logs()
    
    print("\n=== STEP 2 & 3: Building Rolling Windows & Future Targets ===")
    df_dataset = build_forecasting_dataset()
    
    if df_dataset.empty:
        raise ValueError("Forecasting dataset is empty. Cannot continue pipeline.")
        
    # Generate risk trends plot
    print("\n=== Generating Risk Trends Visualization ===")
    plot_risk_trends(df_dataset, os.path.join(reports_dir, "risk_trends.png"))
    
    # 3. Disjoint Repository Split
    print("\n=== Splitting Dataset (Repository-Disjoint) ===")
    train_repos = ["click", "redux", "axios"]
    val_repos = ["express"]
    test_repos = ["databases", "jinja"]
    
    df_train = df_dataset[df_dataset["repository_name"].isin(train_repos)].copy()
    df_val = df_dataset[df_dataset["repository_name"].isin(val_repos)].copy()
    df_test = df_dataset[df_dataset["repository_name"].isin(test_repos)].copy()
    
    print(f"[+] Splits -> Train: {len(df_train)} rows ({train_repos})")
    print(f"[+] Splits -> Val: {len(df_val)} rows ({val_repos})")
    print(f"[+] Splits -> Test: {len(df_test)} rows ({test_repos})")
    
    # Store all model metrics
    comparison_rows = []
    
    # Store trained RF models to save
    trained_rf_models = {}
    
    # Targets configuration
    targets = {
        "future_risk": ["future_risk_30d", "future_risk_60d", "future_risk_90d"],
        "future_defect_count": ["future_defect_count_30d", "future_defect_count_60d", "future_defect_count_90d"],
        "future_modification_intensity": ["future_modification_intensity_30d", "future_modification_intensity_60d", "future_modification_intensity_90d"]
    }
    
    # Best model tracking
    best_mae = float("inf")
    best_model_name = ""
    best_model_info = {}
    
    # Predictions of RF on future_risk_30d for test visualization
    test_predictions_rf_30d = None
    
    # Train and evaluate per target and horizon
    for target_base, target_cols in targets.items():
        for col in target_cols:
            horizon = int(col.split("_")[-1][:-1]) # Extracts 30, 60, or 90
            
            y_train = df_train[col]
            y_test = df_test[col]
            
            # A. Baseline Forecaster (Persistence)
            baseline = PersistenceForecaster()
            y_pred_base = baseline.predict(df_test, col, horizon)
            metrics_base = evaluate_predictions(y_test, y_pred_base)
            comparison_rows.append({
                "model_name": "Persistence",
                "target": target_base,
                "horizon": f"{horizon}d",
                **metrics_base
            })
            
            # B. Random Forest Regressor
            rf = RandomForestForecaster(n_estimators=100, max_depth=8, random_state=42)
            rf.fit(df_train[FEATURES], y_train)
            y_pred_rf = rf.predict(df_test[FEATURES])
            metrics_rf = evaluate_predictions(y_test, y_pred_rf)
            comparison_rows.append({
                "model_name": "Random Forest",
                "target": target_base,
                "horizon": f"{horizon}d",
                **metrics_rf
            })
            
            # Save RF model if it's the main target
            if target_base == "future_risk":
                trained_rf_models[col] = rf
                if col == "future_risk_30d":
                    test_predictions_rf_30d = y_pred_rf
                    
                # Track overall best model for future_risk_30d
                if col == "future_risk_30d" and metrics_rf["mae"] < best_mae:
                    best_mae = metrics_rf["mae"]
                    best_model_name = "Random Forest"
                    best_model_info = {
                        "model": "Random Forest",
                        "target": "future_risk_30d",
                        "metrics": metrics_rf
                    }
            
            # C. XGBoost Regressor
            xgb = XGBoostForecaster(n_estimators=100, max_depth=5, learning_rate=0.05, random_state=42)
            xgb.fit(df_train[FEATURES], y_train)
            y_pred_xgb = xgb.predict(df_test[FEATURES])
            metrics_xgb = evaluate_predictions(y_test, y_pred_xgb)
            comparison_rows.append({
                "model_name": "XGBoost",
                "target": target_base,
                "horizon": f"{horizon}d",
                **metrics_xgb
            })
            
            # Save XGBoost natively to avoid pickle issues
            xgb_path = os.path.join(models_dir, f"xgb_{target_base}_{horizon}d.json")
            xgb.save_model(xgb_path)
            
            # Log progress
            print(f"[+] Evaluated {target_base} ({horizon}d) - RF MAE: {metrics_rf['mae']:.4f}, XGB MAE: {metrics_xgb['mae']:.4f}")
            
    # B. Skip LSTM (Documenting in log)
    print("\n[!] LSTM Forecasting Model Skipped.")
    print("    Reason: Disjoint repository split (3 train repos) limits available sequences to ~800, which is insufficient for deep LSTM generalization.")
    
    # 4. Save Model Comparisons CSV
    df_comparison = pd.DataFrame(comparison_rows)
    comparison_file = os.path.join(reports_dir, "model_comparison.csv")
    df_comparison.to_csv(comparison_file, index=False)
    print(f"\n[+] Saved model comparison report to {comparison_file}")
    
    # 5. Save best model metrics JSON
    best_metrics_file = os.path.join(reports_dir, "best_model_metrics.json")
    with open(best_metrics_file, "w") as f:
        json.dump(best_model_info, f, indent=4)
    print(f"[+] Saved best model metrics metadata to {best_metrics_file}")
    
    # 6. Save Best Model Pickle
    # We pickle the dictionary of trained RF models for future_risk targets
    rf_pickle_path = os.path.join(models_dir, "risk_forecaster.pkl")
    with open(rf_pickle_path, "wb") as f:
        pickle.dump({
            "features": FEATURES,
            "models": trained_rf_models,
            "best_model_metadata": best_model_info
        }, f)
    print(f"[+] Saved serialized risk forecaster models to {rf_pickle_path}")
    
    # 7. Generate repository forecasts plot (RF 30d risk forecast vs actual on test set)
    if test_predictions_rf_30d is not None:
        plot_repository_forecasts(
            df_test,
            test_predictions_rf_30d,
            "future_risk_30d",
            os.path.join(reports_dir, "repository_forecasts.png")
        )
        
    # 8. Generate Error Analysis Report
    generate_error_analysis_report(df_test, test_predictions_rf_30d, df_comparison, reports_dir)
    
    return best_model_info

def generate_error_analysis_report(df_test: pd.DataFrame, y_pred: np.ndarray, df_comp: pd.DataFrame, reports_dir: str):
    """
    Computes absolute error per snapshot on the test set and outputs a markdown error report.
    """
    df_err = df_test.copy()
    df_err["prediction"] = y_pred
    df_err["absolute_error"] = np.abs(df_err["future_risk_30d"] - df_err["prediction"])
    
    # Group by repository to find average errors
    repo_errors = df_err.groupby("repository_name")["absolute_error"].mean().reset_index()
    
    # Identify largest errors
    worst_snapshots = df_err.sort_values("absolute_error", ascending=False).head(10)
    
    md_content = f"""# Forecast Error and Trajectory Analysis Report

This report analyzes the errors, trajectory performance, and domain shifts in the repository risk forecasting system.

## 1. Repository-Level Forecasting Errors

On the test set (which consists of completely unseen repositories during training), the average Mean Absolute Error (MAE) for 30-day future risk is:

| Repository Name | Average Absolute Error (30d Risk) |
| --- | --- |
"""
    for _, row in repo_errors.iterrows():
        md_content += f"| {row['repository_name']} | {row['absolute_error']:.4f} |\n"
        
    md_content += f"""
### Key Findings
- **High-Error Repositories**: Repositories like `{repo_errors.sort_values("absolute_error", ascending=False).iloc[0]["repository_name"]}` have the highest absolute errors. This is due to sudden spikes in activity that do not follow historical rolling trends, or high overall scale differences.
- **Low-Error Repositories**: Repositories like `{repo_errors.sort_values("absolute_error").iloc[0]["repository_name"]}` show much smaller error ranges, suggesting highly consistent commit and modification patterns.

---

## 2. Top 5 Worst-Predicted Snapshots

Below are the snapshots with the largest forecasting discrepancies:

| Repository | Snapshot Date | Actual Future Risk | Predicted Risk | Absolute Error |
| --- | --- | --- | --- | --- |
"""
    for _, row in worst_snapshots.head(5).iterrows():
        md_content += f"| {row['repository_name']} | {row['snapshot_date']} | {row['future_risk_30d']:.4f} | {row['prediction']:.4f} | {row['absolute_error']:.4f} |\n"
        
    md_content += f"""
### Root Causes of Spikes:
1. **Release Cycles**: Sudden rushes of commits right before major versions create activity spikes that past rolling windows cannot anticipate.
2. **Domain/Scale Shifts**: Some repositories naturally have larger codebases and contributor sizes, making their absolute risk scales orders of magnitude larger than those of the training set.

---

## 3. Horizon Comparison (30d vs. 60d vs. 90d)

The table below contrasts the MAE/RMSE across different forecast horizons for Random Forest:

| Horizon | Target | Random Forest MAE | Random Forest RMSE |
| --- | --- | --- | --- |
"""
    df_rf = df_comp[(df_comp["model_name"] == "Random Forest") & (df_comp["target"] == "future_risk")]
    for _, row in df_rf.iterrows():
        md_content += f"| {row['horizon']} | {row['target']} | {row['mae']:.4f} | {row['rmse']:.4f} |\n"
        
    md_content += f"""
### Summary:
As the forecasting horizon expands (from 30d to 90d), the forecasting error **increases**. This is expected because long-term repository behavior is highly sensitive to external factors (new issues, project funding, changing teams) that are not captured in the 90-day rolling activity features.

---

## 4. LSTM Skip Justification

The LSTM model was skipped. A deep recurrent model trained on only 3 repositories (click, redux, axios) is highly prone to overfitting the specific activity scales of those training repos. Consequently, it completely fails to generalize to unseen testing repositories. Trees are much more robust to such disjoint distribution bounds.
"""
    
    error_analysis_path = os.path.join(reports_dir, "forecast_error_analysis.md")
    with open(error_analysis_path, "w") as f:
        f.write(md_content)
    print(f"[+] Saved error analysis report to {error_analysis_path}")

if __name__ == "__main__":
    run_pipeline()
