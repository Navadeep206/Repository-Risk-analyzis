# Phase 11: Cross-Repository Generalization and Domain Adaptation Study Report

This report documents our findings from the Cross-Repository Generalization Study (Phase 11). We investigate why repository risk models fail when deployed to completely unseen repositories, quantify similarity between codebases, measure domain shift, and evaluate mitigation strategies including alternative feature scalers and probability calibration.

---

## 1. Repository Similarity Findings

We computed repository-level centroids across the combined tabular metrics and projected the high-dimensional space to analyze similarity patterns. Pairwise cosine similarities and Euclidean distances show clear repository grouping:

### Similarity & Clustering Results
- **Cluster 0**: `axios`, `express`, `redux`
- **Cluster 1**: `click`, `databases`, `jinja`

### Key Observations:
- **axios & redux**: Exhibit extremely high similarity (Cosine Similarity = `0.8750`, Euclidean Distance = `0.5578`), representing highly modular, client-side JavaScript utility frameworks.
- **databases & click**: Group together (Cosine Similarity = `0.6660`, Euclidean Distance = `2.6381`), representing server/database interface packages with similar developmental cadences.
- **Outliers & Negative Similarities**:
  - `axios` vs. `jinja` has a strong negative similarity (`-0.9274`), and `databases` vs. `express` shows a negative similarity of `-0.7939`.
  - PCA and t-SNE projections verify that client-side JavaScript utility libraries are visually distinct from Python backend/template-engine applications (`jinja`, `click`), creating clear spatial domain gaps.

---

## 2. Domain Shift Findings

We measured covariate distribution shift using the **Population Stability Index (PSI)** and Kolmogorov-Smirnov (KS) test statistics.

### Feature Drift Severity Ranking
| Rank | Feature Name | Mean PSI | Shift Severity | Primary Drivers / Details |
| --- | --- | --- | --- | --- |
| 1 | `repository_age_days` | 7.5495 | Significant Shift | Large age discrepancy between legacy frameworks (`express`) and younger ones (`redux`). |
| 2 | `commit_frequency` | 1.9274 | Significant Shift | Extremely high activity densities on major libraries compared to small utilities. |
| 3 | `maintainability_index` | 1.8152 | Significant Shift | Varied syntactic styles and code density profiles across JavaScript and Python. |
| 4 | `commit_count` | 1.1437 | Significant Shift | Scale mismatch between large monorepos and smaller libraries. |
| 5 | `modification_count` | 1.1437 | Significant Shift | Directly correlated with commit count volume. |
| 6 | `loc` (Lines of Code) | 0.9601 | Significant Shift | Varies from small database utilities to deep framework codebases. |
| 7 | `contributor_count` | 0.7646 | Significant Shift | Large community-driven repositories vs. solo/small-team tools. |
| 8 | `complexity` | 0.7383 | Significant Shift | Structural variation depending on programming paradigms. |

All features evaluated exceed the critical PSI threshold of `0.25`, indicating that **covariate shift is present across all metrics**. The primary drivers of shift are `repository_age_days` and `commit_frequency`.

---

## 3. Best Scaling Method

We evaluated different feature scaling methods to assess if they could mitigate Out-Of-Distribution (OOD) domain shifts:

| Scaler Type | Classification Accuracy | Classification Macro F1 | Forecasting MAE (30d) | Forecasting RMSE (30d) |
| --- | --- | --- | --- | --- |
| **StandardScaler** | **0.6548** | **0.5931** | **19.9616** | **30.8886** |
| RobustScaler | 0.6429 | 0.5666 | 19.9616 | 30.8886 |
| QuantileTransformer | 0.6548 | 0.5868 | 20.1839 | 30.9440 |
| Rank Normalization | 0.6429 | 0.5666 | 20.1839 | 30.9440 |

### Insights:
- **StandardScaler** achieved the highest Classification Macro F1 (`0.5931`) and the lowest Forecasting MAE (`19.9616`).
- **Quantile & Rank Normalization** map features to uniform or normal distributions. While this resolves absolute scale discrepancies, it removes useful signal about the absolute size of the codebase, which is a key predictor of risk.
- Because our primary prediction models (Random Forest, XGBoost) are tree-based models, they are invariant to monotonic transformations of individual features. Consequently, scaling changes have minimal impact on model decisions but can influence calibration and split thresholds.

---

## 4. Leave-One-Repository-Out (LORO) Results

Traditional validation splits (e.g. standard cross-validation) draw train and test samples from the same set of repositories, leading to overly optimistic performance estimates due to repository-specific leakage. LORO validation evaluates generalization to a completely unseen codebase.

### LORO Benchmark Performance
| Held-Out Repository | Train Samples | Test Samples | Accuracy | Macro F1 | Weighted F1 |
| --- | --- | --- | --- | --- | --- |
| `click` | 621 | 63 | 0.6349 | 0.4454 | 0.5818 |
| `databases` | 660 | 24 | 0.7917 | **0.7462** | 0.7803 |
| `axios` | 483 | 201 | 0.6567 | 0.6101 | 0.6576 |
| `express` | 543 | 141 | 0.5035 | *0.3188* | 0.3971 |
| `jinja` | 624 | 60 | 0.7667 | 0.6598 | 0.7669 |
| `redux` | 489 | 195 | 0.4462 | 0.3861 | 0.4153 |
| **Average** | **-** | **-** | **0.6333** | **0.5277** | **0.5998** |

### Insights:
- **Easiest Repository**: `databases` (Macro F1 = `0.7462`), which shares similar features and scales with other database-oriented utilities in the pool.
- **Hardest Repository**: `express` (Macro F1 = `0.3188`), which is highly legacy (old age) and has a distinct web-server architecture that standard process metrics fail to align with cleanly.

---

## 5. Calibration Metrics

Confidence calibration measures how well a model's prediction probability matches its empirical frequency of being correct.

| Calibration Method | Brier Score (lower is better) | Expected Calibration Error (ECE) |
| --- | --- | --- |
| **Uncalibrated (RF)** | **0.3656** | **0.1021** |
| Platt Scaling (Sigmoid) | 0.6159 | 0.2586 |
| Isotonic Regression | 0.5769 | 0.2472 |

### Key Discovery:
- **Post-Processing Calibration Degrades Performance**: Both Platt Scaling and Isotonic Regression significantly increased ECE and Brier Score on the test set.
- **Why it Fails**: Traditional post-processing calibrators assume that the validation set (used to fit calibration parameters) and the test set are independent and identically distributed (i.i.d.). Under strong cross-repository domain shift, the calibration parameters overfit to the validation repository distributions and make highly biased predictions on the test repositories, leading to poor calibration.

---

## 6. Model Robustness Rankings

Comparing model types under both disjoint and LORO evaluation highlights the gap between traditional architectures and deep/multimodal learning:

| Model Name | Disjoint Macro F1 | LORO Avg Macro F1 | Disjoint Accuracy | Generalization Rank |
| --- | --- | --- | --- | --- |
| **XGBoost** | **0.6374** | 0.4921 | **0.7024** | 1 (Best under disjoint) |
| **Random Forest** | 0.6228 | **0.5277** | 0.6786 | 2 (Most stable under LORO) |
| **Calibrated Random Forest** | 0.4015 | 0.5107 | 0.4167 | 3 |
| **CodeBERT Model (Embeddings Only)** | 0.4016 | 0.3850 | 0.5217 | 4 |
| **Hybrid Fusion Model** | 0.3354 | 0.3120 | 0.4500 | 5 (Worst generalization) |

### Why Random Forest/XGBoost Wins:
- Traditional software metrics (LOC, maintainability index, commit frequency) act as strong aggregate indicators of repository risk.
- Random Forest generalizes better because its bootstrapped bagging reduces variance and handles the OOD feature shifts more gracefully than deep neural networks or complex concatenated hybrid models, which overfit to language-specific semantic embeddings.

### Why Hybrid Fusion Failed:
- Concatenating CodeBERT embeddings with process metrics created a high-dimensional space (776 dimensions) where the model overfit to syntactic patterns of the training repositories. When evaluated on unseen repositories, the semantic representation experienced severe domain shift, dragging down the predictive utility of the process metrics.

---

## 7. Deployment Recommendations

Based on the study, we recommend the following deployment design:
1. **Enforce Leave-One-Repository-Out (LORO) Benchmarking**: Never use standard i.i.d. splits for model release. LORO is the only way to obtain an unbiased estimate of real-world generalization performance.
2. **StandardScaler for Generalization**: Standard scaling remains the most robust choice. Do not use rank or quantile transformations as they destroy absolute scale features.
3. **Disable Post-Processing Calibration**: Do not use Platt Scaling or Isotonic Regression unless a domain-invariant calibration method is developed. Uncalibrated Random Forest confidence outputs are more trustworthy.
4. **Use Random Forest as the Production Model**: Random Forest provides the most stable performance across all disjoint repositories.

---

## 8. Future Research Directions

1. **Domain-Adversarial Neural Networks (DANN)**: Train models to extract features that are predictive of risk but invariant to the source repository.
2. **Self-Supervised Adaptation**: Adapt models to the target codebase using unlabeled data before predicting risk labels.
3. **Language-Agnostic Embeddings**: Explore multilingual models that reduce the syntactic domain shift between JavaScript and Python repositories.
