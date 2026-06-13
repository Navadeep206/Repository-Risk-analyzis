# Cross-Repository Model Failure Analysis

This report diagnoses the root causes of risk prediction performance drops when models are deployed to completely unseen repositories.

## 1. Generalization Performance Gaps

- **Easiest Repository to Generalize To**: `databases` (Macro F1 = **0.7462**)
- **Hardest Repository to Generalize To**: `express` (Macro F1 = **0.3188**)

---

## 2. Root Cause 1: Scale Disparity (Covariate Shift)
The most severe domain shift on `express` occurred in the feature `repository_age_days` (PSI = **7.3968**). 
Smaller codebase files or lower activity densities shift the distribution out-of-bounds relative to the large repositories used in training, leading to prediction bias.

---

## 3. Root Cause 2: Semantic Language Shifts
The baseline ML classifier is highly dependent on process metrics (commits, modifications), but CodeBERT embeddings carry code syntax features. 
When testing on repositories with different design patterns, class-level definitions, or programming languages (e.g. JS vs. Python), the embeddings experience concept drift. This explains why deep learning models (embeddings only) fail under disjoint evaluation.

---

## 4. Mitigation Recommendations
1. **Adaptive Scalers**: Replace standard scaling with Quantile scaling or Rank Normalization to standardize the activity profiles of all repositories regardless of absolute scale.
2. **Probability Calibration**: Calibrating output probabilities via Isotonic Regression ensures confidence gates (e.g., Trust Gate) remain accurate under domain shifts.
