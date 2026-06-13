#!/usr/bin/env python3
"""
Master orchestrator script for the Phase 8 Explainability Pipeline.
Sequences all SHAP-free and XGBoost-free feature, permutation, error, domain shift, and visualization stages.
"""

import os
import sys
import pandas as pd

# Ensure parent directory is in path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import BASE_DIR
from explainability.feature_importance import run_feature_importance
from explainability.permutation_importance import run_permutation_importance
from explainability.error_analysis import run_error_analysis
from explainability.domain_shift_analysis import run_domain_shift_analysis
from explainability.trustworthiness_analysis import run_trustworthiness_analysis
from explainability.hybrid_failure_analysis import run_hybrid_failure_analysis
from explainability.visualization import generate_explainability_plots

def compile_global_rankings() -> pd.DataFrame:
    """
    Combines Gini feature importance and permutation importance
    into a single CSV: reports/explainability/global_feature_ranking.csv.
    """
    feat_imp_csv = os.path.join(BASE_DIR, "reports", "explainability", "feature_importance.csv")
    perm_csv = os.path.join(BASE_DIR, "reports", "explainability", "permutation_importance.csv")
    
    if not os.path.exists(feat_imp_csv) or not os.path.exists(perm_csv):
        raise FileNotFoundError("Gini or Permutation CSV is missing. Run those stages first.")
        
    df_imp = pd.read_csv(feat_imp_csv)
    df_perm = pd.read_csv(perm_csv)
    
    # Merge datasets on feature_name and rf_intrinsic_importance to avoid suffixes
    df_merged = pd.merge(df_imp, df_perm, on=["feature_name", "rf_intrinsic_importance"])
    
    # Calculate ranks (lower rank = more important, 1 is best)
    df_merged["rank_intrinsic"] = df_merged["rf_intrinsic_importance"].rank(ascending=False)
    df_merged["rank_permutation"] = df_merged["rf_permutation_mean"].rank(ascending=False)
    
    # Compute average rank score
    df_merged["average_rank"] = df_merged[["rank_intrinsic", "rank_permutation"]].mean(axis=1)
    
    # Sort by average rank
    df_merged = df_merged.sort_values(by="average_rank", ascending=True)
    
    # Save output
    out_csv = os.path.join(BASE_DIR, "reports", "explainability", "global_feature_ranking.csv")
    df_merged.to_csv(out_csv, index=False)
    print(f"[+] Saved global feature rankings to {out_csv}")
    
    return df_merged

def run_explainability_pipeline() -> None:
    """
    Executes the entire Explainability and Error Analysis pipeline sequentially.
    """
    print("="*60)
    print("Starting Phase 8 Explainability & Error Analysis Pipeline")
    print("="*60)
    
    # Create main reports folder
    reports_dir = os.path.join(BASE_DIR, "reports", "explainability")
    os.makedirs(reports_dir, exist_ok=True)
    
    # 1. Gini Importance
    print("\n[STAGE 1/6] Running Intrinsic Feature Importance...")
    run_feature_importance()
    
    # 2. Permutation Importance
    print("\n[STAGE 2/6] Running Permutation Importance...")
    run_permutation_importance()
    
    # 3. Compile rankings
    print("\n[STAGE 3/6] Compiling Global Feature Rankings...")
    compile_global_rankings()
    
    # 4. Error & Trustworthiness Analysis
    print("\n[STAGE 4/6] Performing Error and Trustworthiness Analysis...")
    run_error_analysis()
    run_trustworthiness_analysis()
    
    # 5. Domain Shift & Failure analysis
    print("\n[STAGE 5/6] Performing Domain Shift & Hybrid Failure Analysis...")
    run_domain_shift_analysis()
    run_hybrid_failure_analysis()
    
    # 6. Generate charts and plots
    print("\n[STAGE 6/6] Generating Visualizations & Heatmaps...")
    generate_explainability_plots()
    
    print("\n" + "="*60)
    print("Phase 8 Explainability Pipeline Completed Successfully!")
    print("="*60)

if __name__ == "__main__":
    run_explainability_pipeline()
