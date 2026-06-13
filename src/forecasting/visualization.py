#!/usr/bin/env python3
"""
Visualization component for Phase 9.
Generates risk trend and repository forecasting performance plots.
"""

import os
import sys
import matplotlib
matplotlib.use("Agg")  # Use headless Agg backend to prevent GUI crashes
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def plot_risk_trends(df: pd.DataFrame, output_path: str):
    """
    Plots the historical actual future risk trends for all repositories.
    """
    plt.figure(figsize=(10, 6))
    
    # Modern premium style
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.grid(True, linestyle="--", alpha=0.6)
    
    repos = df["repository_name"].unique()
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(repos)))
    
    for i, repo in enumerate(repos):
        repo_df = df[df["repository_name"] == repo].sort_values("snapshot_date")
        dates = pd.to_datetime(repo_df["snapshot_date"])
        plt.plot(dates, repo_df["future_risk_30d"], label=repo, color=colors[i], linewidth=2.0)
        
    plt.title("Historical 30-Day Future Risk Trajectory by Repository", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Snapshot Date", fontsize=12, labelpad=10)
    plt.ylabel("Future Risk Score (30-day Horizon)", fontsize=12, labelpad=10)
    plt.legend(frameon=True, facecolor="white", edgecolor="none", shadow=True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"[+] Saved risk trends plot to {output_path}")

def plot_repository_forecasts(df_test: pd.DataFrame, predictions: np.ndarray, target_name: str, output_path: str):
    """
    Plots the actual vs predicted future risk trajectories for the test set.
    """
    plt.figure(figsize=(12, 6))
    plt.grid(True, linestyle="--", alpha=0.6)
    
    df_test_copy = df_test.copy()
    df_test_copy["prediction"] = predictions
    df_test_copy["snapshot_date"] = pd.to_datetime(df_test_copy["snapshot_date"])
    
    repos = df_test_copy["repository_name"].unique()
    
    for i, repo in enumerate(repos):
        repo_df = df_test_copy[df_test_copy["repository_name"] == repo].sort_values("snapshot_date")
        dates = repo_df["snapshot_date"]
        
        # Plot actual
        plt.plot(dates, repo_df[target_name], label=f"{repo} (Actual)", linestyle="-", linewidth=2.0, alpha=0.8)
        # Plot predicted
        plt.plot(dates, repo_df["prediction"], label=f"{repo} (Predicted)", linestyle="--", linewidth=2.0, alpha=0.8)
        
    plt.title(f"Forecasting Actual vs. Predicted values: {target_name}", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Snapshot Date", fontsize=12, labelpad=10)
    plt.ylabel("Target Value", fontsize=12, labelpad=10)
    plt.legend(frameon=True, facecolor="white", edgecolor="none", shadow=True, ncol=2)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"[+] Saved repository forecasts plot to {output_path}")
