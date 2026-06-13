# Domain Adaptation & OOD Robustness Failure Analysis

This report diagnoses remaining failure modes of risk intelligence predictions after applying various domain adaptation and alignment techniques.

## 1. Remaining Generalization Performance Gaps

- **Easiest Repository under Adaptation**: `jinja` (Avg LORO Macro F1 = **0.5741**)
- **Hardest Repository under Adaptation**: `redux` (Avg LORO Macro F1 = **0.3634**)
- **Best Strategy for the Hardest Repository**: `CORAL XGBoost` (Macro F1 = **0.4317**)

---

## 2. Diagnostics: Why `redux` Remains Challenging

1. **Extreme Covariate Shift Severity**: As analyzed in Phase 11, features like `repository_age_days` and `commit_frequency` are highly skewed. `redux` represents an outlier in developmental activity scale, which simple correlation alignment (CORAL) cannot perfectly linearize.
2. **Codebase Specific Patterns**: Under LORO evaluation, when `redux` is held out, the model is trained entirely on codebases that may not capture its specific coding patterns or developer structures. This is particularly noticeable in frameworks that use highly asynchronous or complex class inheritance structures that differ from standard templates.

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
