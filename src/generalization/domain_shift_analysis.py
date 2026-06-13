#!/usr/bin/env python3
"""
Step 2: Domain Shift Quantification.
Measures feature differences between repositories using Z-score shifts, KS-test, and PSI.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Prevent GUI crashes
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ks_2samp

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import BASE_DIR, FINAL_DIR

def calculate_psi(expected: np.ndarray, actual: np.ndarray, num_bins: int = 10) -> float:
    """
    Computes the Population Stability Index (PSI) between two distributions.
    """
    # Create bin edges based on expected (others pool) percentiles
    percentiles = np.linspace(0, 100, num_bins + 1)
    bins = np.percentile(expected, percentiles)
    bins = np.unique(bins) # Remove duplicates if data is highly skewed
    
    if len(bins) < 2:
        return 0.0
        
    expected_counts, _ = np.histogram(expected, bins=bins)
    actual_counts, _ = np.histogram(actual, bins=bins)
    
    # Convert counts to percentages
    expected_pct = expected_counts / len(expected)
    actual_pct = actual_counts / len(actual)
    
    # Clip values to avoid log(0) or division by zero
    expected_pct = np.clip(expected_pct, 1e-4, None)
    actual_pct = np.clip(actual_pct, 1e-4, None)
    
    # Compute PSI
    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(psi)

def run_domain_shift():
    print("[*] Running Domain Shift Quantification...")
    reports_dir = os.path.join(BASE_DIR, "reports", "generalization")
    plots_dir = os.path.join(reports_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    dataset_path = os.path.join(FINAL_DIR, "ml_dataset_v2.csv")
    df = pd.read_csv(dataset_path)
    
    features = [
        "loc", "complexity", "maintainability_index", "commit_count",
        "modification_count", "contributor_count", "commit_frequency", "repository_age_days"
    ]
    
    repos = df["repository_name"].unique()
    shift_results = []
    
    for feat in features:
        for repo in repos:
            # Split repo vs others pool
            repo_values = df[df["repository_name"] == repo][feat].dropna().values
            others_values = df[df["repository_name"] != repo][feat].dropna().values
            
            if len(repo_values) == 0 or len(others_values) == 0:
                continue
                
            # 1. Z-score Shift
            mean_repo = np.mean(repo_values)
            mean_others = np.mean(others_values)
            std_others = np.std(others_values)
            z_shift = (mean_repo - mean_others) / std_others if std_others > 0 else 0.0
            
            # 2. KS Test
            ks_stat, p_val = ks_2samp(repo_values, others_values)
            
            # 3. PSI
            psi = calculate_psi(others_values, repo_values)
            
            shift_results.append({
                "feature_name": feat,
                "repository_name": repo,
                "z_score_shift": float(z_shift),
                "ks_stat": float(ks_stat),
                "ks_pvalue": float(p_val),
                "psi": float(psi)
            })
            
    df_shift = pd.DataFrame(shift_results)
    summary_file = os.path.join(reports_dir, "domain_shift_summary.csv")
    df_shift.to_csv(summary_file, index=False)
    print(f"[+] Saved domain shift summary to {summary_file}")
    
    # Rank features by mean PSI across all repositories
    feature_ranking = df_shift.groupby("feature_name")["psi"].mean().sort_values(ascending=False).reset_index()
    feature_ranking.columns = ["feature_name", "mean_psi"]
    
    print("\n[+] Top Volatile Features ranked by Domain Shift Severity (Mean PSI):")
    for idx, row in feature_ranking.iterrows():
        print(f"    {idx+1}. {row['feature_name']}: PSI = {row['mean_psi']:.4f}")
        
    # Plot histograms of top shifted features (e.g. loc and commit_frequency)
    top_shifted = feature_ranking.head(2)["feature_name"].tolist()
    
    plt.figure(figsize=(12, 5))
    for i, feat in enumerate(top_shifted):
        plt.subplot(1, 2, i+1)
        for repo in repos:
            repo_data = df[df["repository_name"] == repo][feat]
            # Use log scale for skewed count features
            if feat in ["loc", "commit_count", "modification_count", "commit_frequency"]:
                sns.kdeplot(np.log1p(repo_data), label=repo, fill=True, alpha=0.1)
                plt.xlabel(f"log(1 + {feat})")
            else:
                sns.kdeplot(repo_data, label=repo, fill=True, alpha=0.1)
                plt.xlabel(feat)
        plt.title(f"Distribution of {feat} across Repositories")
        plt.ylabel("Density")
        plt.legend()
        
    plt.tight_layout()
    hist_plot_path = os.path.join(plots_dir, "domain_shift_histograms.png")
    plt.savefig(hist_plot_path, dpi=150)
    plt.close()
    print(f"[+] Saved domain shift histograms to {hist_plot_path}")
    
    # Write domain_shift_report.md
    write_markdown_report(df_shift, feature_ranking, reports_dir)

def write_markdown_report(df_shift: pd.DataFrame, rankings: pd.DataFrame, reports_dir: str):
    md_content = f"""# Domain Shift Quantification Report

This report analyzes features' distribution drift across codebases to identify why prediction models experience generalization drops.

## 1. Feature Drift Severity Rankings

We quantify domain shift using the **Population Stability Index (PSI)**. Features with PSI > 0.25 exhibit significant shifts.

| Rank | Feature Name | Mean PSI | Shift Severity |
| --- | --- | --- | --- |
"""
    for idx, row in rankings.iterrows():
        severity = "Significant Shift" if row['mean_psi'] > 0.25 else "Moderate Shift" if row['mean_psi'] > 0.1 else "Stable"
        md_content += f"| {idx+1} | {row['feature_name']} | {row['mean_psi']:.4f} | {severity} |\n"
        
    md_content += """
---

## 2. Deep Dive: Most Volatile Covariates

### Z-score & Kolmogorov-Smirnov statistics by Repository:
"""
    for feat in rankings["feature_name"].head(3):
        md_content += f"\n### Feature: `{feat}`\n"
        md_content += "| Repository Name | Z-Score Shift | KS Statistic | KS p-value | PSI |\n"
        md_content += "| --- | --- | --- | --- | --- |\n"
        
        feat_df = df_shift[df_shift["feature_name"] == feat]
        for _, row in feat_df.iterrows():
            md_content += f"| {row['repository_name']} | {row['z_score_shift']:.4f} | {row['ks_stat']:.4f} | {row['ks_pvalue']:.4g} | {row['psi']:.4f} |\n"
            
    md_content += """
---

## 3. Explaining Generalization Failures

1. **Codebase Size Difference (`loc`)**: Utilities and database backends have a much smaller LOC signature than major frameworks, shifting the scale bounds of tree classifiers.
2. **Process Activity Density (`commit_frequency`)**: Active, collaborative codebases have commit rates multiple orders of magnitude higher than smaller utilities. Standardizing using standard scales over training datasets results in out-of-bounds metrics when testing against smaller/large repositories.
"""
    report_path = os.path.join(reports_dir, "domain_shift_report.md")
    with open(report_path, "w") as f:
        f.write(md_content)
    print(f"[+] Saved domain shift report markdown to {report_path}")

if __name__ == "__main__":
    run_domain_shift()
