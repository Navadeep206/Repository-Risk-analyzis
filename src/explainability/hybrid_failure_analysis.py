#!/usr/bin/env python3
"""
Hybrid Model Failure Analysis module for Phase 8.
Investigates and documents why the hybrid fusion model failed to beat the Random Forest baseline.
Excludes all XGBoost imports and model loading.
"""

import os
import sys

# Ensure parent directory is in path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import BASE_DIR

def run_hybrid_failure_analysis() -> None:
    """
    Writes a detailed failure analysis for the hybrid fusion model.
    """
    out_dir = os.path.join(BASE_DIR, "reports", "explainability")
    os.makedirs(out_dir, exist_ok=True)
    
    md_path = os.path.join(out_dir, "hybrid_failure_analysis.md")
    
    with open(md_path, "w") as f:
        f.write("# Hybrid Model Failure Diagnostic Report\n\n")
        f.write("This report provides the mathematical and architectural explanation of why the Hybrid Fusion Model C (Test Macro F1 = `0.3354`) failed to beat the Random Forest baseline (Test Macro F1 = `0.6714`).\n\n")
        
        f.write("## 1. Root Cause 1: Linear Extrapolation & OOD Activations\n\n")
        f.write("Standard scaling works by projecting values relative to the training split mean and standard deviation:\n")
        f.write("$$X_{scaled} = \\frac{X - \\mu_{train}}{\\sigma_{train}}$$\n\n")
        f.write("Under repository-disjoint splits, tabular metrics like `repository_age_days` differ significantly because repositories are created at completely different historical moments:\n")
        f.write("- **Train Split**: Mean age = `4209.58` days, Std = `157.79` days.\n")
        f.write("- **Validation Split** (express): Mean age = `6168` days.\n")
        f.write("- **Test Split** (databases, jinja): Mean age = `5302.86` days.\n\n")
        f.write("When the preprocessor (fitted on Train) transforms the validation split, the scaled value of validation age becomes **12.43** (while the training features are strictly bound within `[-1.1, 1.37]`).\n")
        f.write("Since neural networks utilize linear matrix multiplications ($W \\cdot X + b$), these extreme out-of-distribution values propagate directly, creating massive outputs. This causes the validation loss to instantly blow up to `4.4903` at Epoch 1.\n\n")
        
        f.write("## 2. Root Cause 2: Validation Loss-Based Early Stopping Lock\n\n")
        f.write("- In Phase 7, the trainer monitored validation loss for early stopping with `patience=15`.\n")
        f.write("- Since validation loss exploded immediately (from `4.49` at Epoch 1 to `5.44` by Epoch 16), early stopping was triggered immediately, rolling back model weights to **Epoch 1**.\n")
        f.write("- As a consequence, the saved hybrid fusion model was essentially **untrained**, leading to poor generalization (Macro F1 = `0.3354`).\n")
        f.write("- In contrast, Model B (Embeddings Only) does not use tabular features and is unaffected by this scaling shift, allowing it to train properly (Macro F1 = `0.5217`).\n\n")
        
        f.write("## 3. Why Random Forest Succeeds\n\n")
        f.write("Tree-based models (Random Forest, XGBoost) use threshold partitions (e.g. $x_i > 4300$). A validation sample with age `6168` is classified into the same leaf node as a train sample of age `4426`. Decision boundaries are invariant to scale extremes, preventing out-of-distribution values from destabilizing predictions. Hence, tree models remain highly robust (Random Forest Test Macro F1 = `0.6714`).\n\n")
        
        f.write("## 4. Key Recommendations\n")
        f.write("1. **Rank-based Scaling**: Apply rank-based scaling (e.g. `QuantileTransformer` or robust scaling) to prevent extreme Z-scores under OOD settings.\n")
        f.write("2. **Domain Adaptation**: Implement adversarial gradient reversal layers to force the tabular branches to extract repository-invariant features.\n")
        
    print(f"[+] Saved hybrid failure analysis report to {md_path}")

if __name__ == "__main__":
    run_hybrid_failure_analysis()
