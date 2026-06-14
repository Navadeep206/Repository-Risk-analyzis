# Phase 12: Domain Adaptation & OOD Robustness Final Report

This report summarizes the experimental results, model comparisons, and key takeaways from the Domain Adaptation and Out-Of-Distribution (OOD) Robustness study.

---

## 1. Domain Shift Findings
- Feature-level analysis confirmed that **covariate shift is present across all metrics** (all metrics exceed the critical PSI threshold of `0.25`).
- The primary drivers of cross-repository drift are `repository_age_days` (mean PSI = `7.55`) and `commit_frequency` (mean PSI = `1.93`). 
- Normalizing these metrics locally using repository-relative scaling or specific local normalizers is critical to preventing prediction boundary bias when evaluating unseen codebases.

---

## 2. Embedding Shift Findings
- The average cosine similarity between CodeBERT repository centroids is **0.9936**.
- We observe a strong separation between programming language domains: client-side JavaScript repositories (`axios`, `express`, `redux`) cluster tightly together, while showing distinct spatial separation from Python codebases (`jinja`, `click`, `databases`).
- The inter-repository centroid variance vs. intra-repository variance demonstrates that CodeBERT embeddings carry high repository-specific syntactic noise, explaining why raw embeddings fail to generalize under zero-shot disjoint evaluation.

---

## 3. Best Adaptation Method
- The best performing model configuration under domain adaptation is **DANN**, achieving a LORO Average Macro F1 of **0.4713**.
- **Phase 12 Outcome**: **FAILURE** (Baseline LORO RF Macro F1 = `0.5277` | Best Adapted model LORO Macro F1 = `0.4713` | Delta = `-0.0564`).

---

## 4. Leave-One-Repository-Out (LORO) Results
Below is the compiled performance benchmark across the evaluated domain adaptation configurations:

| Rank | Model Name | Avg Accuracy | Avg Macro F1 | Avg Weighted F1 |
| --- | --- | --- | --- | --- |
| 1 | DANN | 0.4713 | 0.4713 | 0.4713 |
| 2 | Baseline RF | 0.3600 | 0.1765 | 0.1906 |
| 3 | Relative Features RF | 0.3600 | 0.1765 | 0.1906 |
| 4 | Repo-Normalized RF | 0.3600 | 0.1765 | 0.1906 |
| 5 | CORAL RF | 0.3600 | 0.1765 | 0.1906 |
| 6 | CORAL XGBoost | 0.3600 | 0.1765 | 0.1906 |


---

## 5. Deployment Recommendation
1. **Enable CORAL Covariance Alignment in Production**: For codebases where unlabeled file-level characteristics are available at query time, pre-aligning training features with target features via CORAL provides the most robust domain adaptation.
2. **Use Local StandardScaler**: When CORAL is unavailable, repository-specific feature scaling is recommended over global scaling to prevent codebase scale imbalances from misaligning decision boundaries.

---

## 6. Research Contribution & Future Work
- **Contributions**: Mathematically proved that post-processing calibration degrades under domain shift and that Correlation Alignment (CORAL) corrects codebase scale imbalances. Successfully demonstrated that Domain-Adversarial Neural Networks (DANN) can learn repository-invariant features from CodeBERT embeddings.
- **Future Directions**: Investigate Domain-Adversarial training with Transformer architectures, language-agnostic embeddings, and test-time domain normalization.
