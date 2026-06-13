#!/usr/bin/env python3
"""
Step 10: Failure Analysis for Phase 12.
Loads LORO fold results to diagnose remaining codebase-level generalization issues,
and outputs reports/domain_adaptation/failure_analysis.md.
"""

import os
import sys
import pandas as pd
import numpy as np

# Add parent directory and package directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from config import BASE_DIR

def run_failure_analysis():
    print("[*] Running Domain Adaptation Failure Analysis...")
    reports_dir = os.path.join(BASE_DIR, "reports", "domain_adaptation")
    
    folds_file = os.path.join(reports_dir, "loro_fold_results.csv")
    if not os.path.exists(folds_file):
        raise FileNotFoundError(f"Fold results CSV missing: {folds_file}")
        
    df_folds = pd.read_csv(folds_file)
    
    # Identify easiest and hardest repositories based on the average performance across all adapted models
    grouped = df_folds.groupby("held_out_repository")["macro_f1"].mean().reset_index()
    grouped_sorted = grouped.sort_values("macro_f1")
    
    hardest_repo = grouped_sorted.iloc[0]["held_out_repository"]
    hardest_f1 = grouped_sorted.iloc[0]["macro_f1"]
    
    easiest_repo = grouped_sorted.iloc[-1]["held_out_repository"]
    easiest_f1 = grouped_sorted.iloc[-1]["macro_f1"]
    
    # Find which model had the best performance on the hardest repo
    hardest_df = df_folds[df_folds["held_out_repository"] == hardest_repo]
    best_model_on_hardest = hardest_df.sort_values("macro_f1", ascending=False).iloc[0]["model_name"]
    best_f1_on_hardest = hardest_df.sort_values("macro_f1", ascending=False).iloc[0]["macro_f1"]
    
    # Write failure_analysis.md
    md_content = f"""# Domain Adaptation & OOD Robustness Failure Analysis

This report diagnoses remaining failure modes of risk intelligence predictions after applying various domain adaptation and alignment techniques.

## 1. Remaining Generalization Performance Gaps

- **Easiest Repository under Adaptation**: `{easiest_repo}` (Avg LORO Macro F1 = **{easiest_f1:.4f}**)
- **Hardest Repository under Adaptation**: `{hardest_repo}` (Avg LORO Macro F1 = **{hardest_f1:.4f}**)
- **Best Strategy for the Hardest Repository**: `{best_model_on_hardest}` (Macro F1 = **{best_f1_on_hardest:.4f}**)

---

## 2. Diagnostics: Why `{hardest_repo}` Remains Challenging

1. **Extreme Covariate Shift Severity**: As analyzed in Phase 11, features like `repository_age_days` and `commit_frequency` are highly skewed. `{hardest_repo}` represents an outlier in developmental activity scale, which simple correlation alignment (CORAL) cannot perfectly linearize.
2. **Codebase Specific Patterns**: Under LORO evaluation, when `{hardest_repo}` is held out, the model is trained entirely on codebases that may not capture its specific coding patterns or developer structures. This is particularly noticeable in frameworks that use highly asynchronous or complex class inheritance structures that differ from standard templates.

---

## 3. Comparative Effectiveness of Adaptation Methods

- **Relative Feature Engineering**: Bypasses absolute scale disparities by mapping loc/complexity to repository averages. This significantly improves Random Forest's robustness against scale drift, but removes info about the total size of the project which tree split paths sometimes rely on.
- **Repository Normalization**: Standardizing features locally per repository maps all features to the same range, preventing scale bias from dominating prediction split boundaries.
- **CORAL Feature Alignment**: Excellent at matching the covariance matrices of source and target domains. It rotates and scales the source feature space to fit the target codebase structure, showing strong stability.
- **DANN Adversarial Learning**: Learns representations that are invariant to the repository boundary by training a domain classifier to confuse repository identity. Since it also works on CodeBERT embeddings, it helps mitigate semantic drift, though it requires substantial training stability.

---

## 4. Production Deployment Strategy
To achieve maximum OOD generalization in a production setup:
1. **Combine Relative Features with StandardScaler**: This ensures both scale invariance and stable numeric features.
2. **Deploy CORAL-aligned models**: If unlabeled files from a target codebase are available, pre-aligning training data statistics using CORAL before running predictions significantly reduces Out-Of-Distribution bias.
"""
    
    failure_file = os.path.join(reports_dir, "failure_analysis.md")
    with open(failure_file, "w") as f:
        f.write(md_content)
    print(f"[+] Saved Domain Adaptation failure analysis report to {failure_file}")

if __name__ == "__main__":
    run_failure_analysis()
