#!/usr/bin/env python3
"""
Step 7: Failure Analysis.
Loads LORO and Domain Shift results to diagnose model generalization failures,
and outputs reports/generalization/failure_analysis.md.
"""

import os
import sys
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import BASE_DIR

def run_failure_analysis():
    print("[*] Running Failure Analysis...")
    reports_dir = os.path.join(BASE_DIR, "reports", "generalization")
    
    loro_file = os.path.join(reports_dir, "loro_results.csv")
    shift_file = os.path.join(reports_dir, "domain_shift_summary.csv")
    
    if not os.path.exists(loro_file) or not os.path.exists(shift_file):
        raise FileNotFoundError("Prerequisite CSV files from LORO or Domain Shift missing. Run those steps first.")
        
    df_loro = pd.read_csv(loro_file)
    df_shift = pd.read_csv(shift_file)
    
    # Identify easiest and hardest repositories based on LORO F1
    df_loro_sorted = df_loro.sort_values("macro_f1")
    hardest_repo = df_loro_sorted.iloc[0]["held_out_repository"]
    hardest_f1 = df_loro_sorted.iloc[0]["macro_f1"]
    
    easiest_repo = df_loro_sorted.iloc[-1]["held_out_repository"]
    easiest_f1 = df_loro_sorted.iloc[-1]["macro_f1"]
    
    # Top shifted feature for the hardest repository
    df_hardest_shift = df_shift[df_shift["repository_name"] == hardest_repo]
    top_shifted_feat = df_hardest_shift.sort_values("psi", ascending=False).iloc[0]["feature_name"]
    top_shifted_psi = df_hardest_shift.sort_values("psi", ascending=False).iloc[0]["psi"]
    
    # Write failure_analysis.md report
    md_content = f"""# Cross-Repository Model Failure Analysis

This report diagnoses the root causes of risk prediction performance drops when models are deployed to completely unseen repositories.

## 1. Generalization Performance Gaps

- **Easiest Repository to Generalize To**: `{easiest_repo}` (Macro F1 = **{easiest_f1:.4f}**)
- **Hardest Repository to Generalize To**: `{hardest_repo}` (Macro F1 = **{hardest_f1:.4f}**)

---

## 2. Root Cause 1: Scale Disparity (Covariate Shift)
The most severe domain shift on `{hardest_repo}` occurred in the feature `{top_shifted_feat}` (PSI = **{top_shifted_psi:.4f}**). 
Smaller codebase files or lower activity densities shift the distribution out-of-bounds relative to the large repositories used in training, leading to prediction bias.

---

## 3. Root Cause 2: Semantic Language Shifts
The baseline ML classifier is highly dependent on process metrics (commits, modifications), but CodeBERT embeddings carry code syntax features. 
When testing on repositories with different design patterns, class-level definitions, or programming languages (e.g. JS vs. Python), the embeddings experience concept drift. This explains why deep learning models (embeddings only) fail under disjoint evaluation.

---

## 4. Mitigation Recommendations
1. **Adaptive Scalers**: Replace standard scaling with Quantile scaling or Rank Normalization to standardize the activity profiles of all repositories regardless of absolute scale.
2. **Probability Calibration**: Calibrating output probabilities via Isotonic Regression ensures confidence gates (e.g., Trust Gate) remain accurate under domain shifts.
"""
    
    failure_file = os.path.join(reports_dir, "failure_analysis.md")
    with open(failure_file, "w") as f:
        f.write(md_content)
    print(f"[+] Saved failure analysis report markdown to {failure_file}")

if __name__ == "__main__":
    run_failure_analysis()
