# Phase 4 Audit Report - Independent Principal ML Review

**Author**: Independent Principal ML Engineer & Research Reviewer  
**Status**: **PASS WITH CONDITIONS** (Methodology is sound; data distribution requires hardening before Phase 5)

This report details a critical audit of Phase 4: Baseline Machine Learning, assessing model performance, evaluation correctness, feature validity, data leakage, generalization safety, and readiness for Phase 5 (Code Embeddings / CodeBERT).

---

## 1. File Verification (Step 1)

All required Phase 4 baseline artifacts have been successfully generated and verified on disk:
- `models/logistic_regression.pkl` (Serialized Logistic Regression)
- `models/decision_tree.pkl` (Serialized Decision Tree)
- `models/random_forest.pkl` (Serialized Tuned Random Forest)
- `models/xgboost.pkl` (Serialized Tuned XGBoost)
- `reports/model_comparison.csv` (Aggregated CV & test/val performance)
- `reports/cross_validation_results.csv` (Fold-level CV logs)
- `reports/feature_importance.csv` (Aggregated feature weights)
- `reports/evaluation_report.md` (Comprehensive evaluation metrics & confusion matrices)

---

## 2. Model Comparison Audit (Step 2)

Based on `reports/model_comparison.csv`, the model performances across validation, test, and training cross-validation splits are summarized below:

| Model | CV Acc (Mean ± Std) | Val Accuracy | Val F1 (Macro) | Test Accuracy | Test F1 (Macro) | Test F1 (Weighted) |
|-------|---------------------|--------------|----------------|---------------|-----------------|--------------------|
| **Random Forest** | **0.7320 ± 0.0800** | **0.5035**   | **0.3326**     | **0.7500**    | **0.6714**      | **0.7502**         |
| **XGBoost**       | 0.7035 ± 0.0994 | 0.4894       | 0.3016         | 0.7024        | 0.6373          | 0.7001             |
| **Decision Tree** | 0.6598 ± 0.1129 | 0.4965       | 0.3418         | 0.6071        | 0.4995          | 0.5844             |
| **Logistic Regression** | 0.6665 ± 0.0639 | 0.4255 | 0.1990         | 0.4762        | 0.3132          | 0.3842             |

### Model Rankings (Best to Worst by Test F1 Macro)
1. **Random Forest** (Tuned) — Strongest generalization on Test (F1=0.6714), stable CV (0.7320).
2. **XGBoost** (Tuned) — Comparable performance on Test (F1=0.6373) but slightly higher CV variance (Std=0.0994).
3. **Decision Tree** — Underperforms ensembles due to high variance and lack of regularization (Test F1=0.4995).
4. **Logistic Regression** — Poor fit (Test F1=0.3132), proving linear boundaries fail to separate code risk.

### Overfitting & Covariate Shift Signs
- There is a **severe performance discrepancy** between the Training CV/Test performance and the Validation performance. For example, Random Forest drops from **0.7500 Test Accuracy** and **0.7320 CV Accuracy** to **0.5035 Validation Accuracy**.
- This is a symptom of **covariate shift** (domain mismatch) rather than classic model-parameter overfitting. The validation split represents a different programming language profile and development frequency (JS-only, high-churn `express` repository) compared to the training set.

---

## 3. Best Model Review (Step 3)

The **Tuned Random Forest** is the clear winner:
- **Optimal Hyperparameters**: `max_depth=5`, `min_samples_split=10`, `n_estimators=200` (tuned via grid search).
- **Validation Metrics**: Accuracy: `0.5035`, Macro F1: `0.3326`, Weighted F1: `0.4136`
- **Test Metrics**: Accuracy: `0.7500`, Macro F1: `0.6714`, Weighted F1: `0.7502`
- **Winning Rationale**:
  - Linear models (Logistic Regression) fail because code metrics interact non-linearly (e.g., complexity only triggers risk when combined with large LOC).
  - Single Decision Trees overfit to training thresholds.
  - Random Forest outperforms XGBoost on this small dataset (684 rows) because its bagging mechanism (bootstrap averaging) and random feature subsetting provide superior regularization, preventing it from tracking the high-variance noise in XGBoost gradient boosting.

---

## 4. Confusion Matrix Analysis (Step 4)

Let's inspect the Random Forest predictions:

### Test Split (Python-only) Confusion Matrix
```
                  Predicted
               LOW   MEDIUM   HIGH
Actual LOW       4        6      0
       MEDIUM    3       23      4
       HIGH      0        8     36
```
- **LOW Class (Hardest)**: Recall is low (40%). Out of 10 actual LOW risk files, 6 are predicted as MEDIUM. Distinguishing very low-risk files from moderate-risk files using coarse volume metrics is difficult.
- **MEDIUM Class**: Recall is 76.7% (23/30). A few files spill into LOW (3) and HIGH (4).
- **HIGH Class (Easiest)**: Recall is 81.8% (36/44), and Precision is 90.0% (36/40). High-risk files (3+ historical bug fixes) have very loud, clear process signals (high churn, many modifications, high developer overlap) that make them easy for tree-based models to identify.

### Validation Split (JS-only) Confusion Matrix
```
                  Predicted
               LOW   MEDIUM   HIGH
Actual LOW       0       18      8
       MEDIUM    0       14     41
       HIGH      0        3     57
```
- **LOW Class**: Recall is **0%** (predicted LOW exactly 0 times).
- **HIGH Class**: Recall is 95% (57/60), but precision is low (53.8%) because the model classifies almost all files as HIGH risk.
- **Why?** `express` has an average `commit_count` of 19.3, whereas JS projects in the training set (`redux` and `axios`) average 4.7 and 8.3. Because features are globally scaled, the model views `express` files as having exceptionally high activity, categorizing almost all files into MEDIUM or HIGH risk.

---

## 5. Feature Importance Review (Step 5)

All 11 features utilized in the final models are ranked by Random Forest importance below:

| Rank | Feature Name | RF Importance | XGBoost Importance | Rationale & Assessment |
|------|--------------|---------------|--------------------|------------------------|
| 1 | `commit_frequency` | 0.2013 | 0.2671 | **Valid**: High commit churn correlates directly with regression risk. |
| 2 | `modification_count` | 0.1466 | 0.0000 | **Valid**: Total lines changed represents code instability. |
| 3 | `commit_count` | 0.1380 | 0.0755 | **Valid**: File activity levels are direct exposure metrics. |
| 4 | `contributor_count` | 0.1073 | 0.0425 | **Valid**: Multiple authors increase communication overhead/bugs. |
| 5 | `language_javascript` | 0.0789 | 0.0988 | **Suspicious**: Encodes language. Hampers generalizability. |
| 6 | `language_typescript` | 0.0763 | 0.2532 | **Suspicious**: Dummies cannot generalize to unseen languages. |
| 7 | `repository_age_days` | 0.0755 | 0.1345 | **Valid**: Normalizes absolute time exposure. |
| 8 | `complexity` | 0.0719 | 0.0279 | **Valid**: Standard static metric (Cyclomatic Complexity). |
| 9 | `loc` | 0.0667 | 0.0293 | **Valid**: Basic size metric (Lines of Code). |
| 10 | `maintainability_index`| 0.0259 | 0.0712 | **Suspicious/Defect**: Hardcoded to `-1.0` for JS/TS. |
| 11 | `language_python` | 0.0116 | 0.0000 | **Suspicious**: Dummies cannot generalize. |

### Suspicious Findings
- **Maintainability Index**: Radon calculations were only executed on Python. JS/TS files default to `-1.0`. The trees split on this to identify JS vs Python rather than actual maintainability. This is a static quality analyzer defect.
- **Language Dummy Columns**: Because splits are repository-disjoint, validation is 100% JS and test is 100% Python. Language-specific features cause the model to act as a language classifier rather than general risk predictor.

---

## 6. Leakage Audit (Step 6)

- `historical_risk_label` is derived from `bug_fix_commit_count`.
- `bug_fix_commit_count` is **strictly excluded** from the preprocessor features (`numeric_features` list), preventing target leakage.
- Historic process columns (`risk_score`, `historical_bug_count`) have been completely removed.
- Repository identifiers (`repository_name`, `file_path`) are dropped.
- **Verdict**: There is **no target leakage** in the features.

---

## 7. Generalization Audit (Step 7)

- **Split Disjointness**: Checked. No repository overlaps.
- **Split Composition**:
  - Train: `axios`, `redux`, `click` (JS/TS + Python)
  - Val: `express` (JS only)
  - Test: `jinja`, `databases` (Python only)
- **Domain Shift Issue**: With only 6 repositories, repository-disjoint splits force validation and test sets into language siloes. This causes the validation set to have 0% Python/TS, and the test set to have 0% JS/TS.
- **Real-World Confidence**: **LOW**. The model cannot generalize to new repositories because features are globally scaled. It suffers from repository-level activity bias. For example, high-activity JS files in a small repository look low-risk, whereas normal files in a highly active repository look high-risk.

---

## 8. Baseline Quality Assessment (Step 8)

If presented in a Staff/Senior ML Engineer interview, this methodology would be rated **GOOD**.
- **Strengths**: Repository-aware splitting is a highly mature design choice that avoids the fatal data leakage common in academic datasets. The model preprocessing pipeline is correctly modularized and saved.
- **Weaknesses**: The lack of repository-level normalization (e.g. MinMax scaling features *within* each repository to capture relative file risk instead of absolute global scale) is a major flaw that prevents cross-repository generalization. The hardcoded `-1.0` maintainability index for non-Python repositories represents an unhandled metric inconsistency.

---

## 9. Phase 5 Justification (Step 9)

- **Test F1 (Macro)**: `0.6714` (moderate).
- **Validation F1 (Macro)**: `0.3326` (poor).
- **Assessment**: This falls into **Case B (F1 0.65–0.85 on test, <0.65 on validation)**.
- **Recommendation**: **Phase 5 (CodeBERT / Code Embeddings) is highly justified.**
  - Traditional static features (LOC, Complexity) and metadata counts are easily biased by coding styles and project scale.
  - CodeBERT can extract semantic representations (e.g., finding actual dangerous pattern structures like missing error checks or manual resource leaks) that bypass language-specific metadata scale discrepancies.
  - **Prerequisite**: Before using CodeBERT, the dataset splitting must be expanded to include more repositories to balance the splits, and repository-level normalization should be applied to the baseline features.

---

## Final Report Summary

1. **PASS / FAIL**: **PASS WITH CONDITIONS** (Pipeline is clean, but feature scale discrepancies must be addressed before moving to deep learning).
2. **Best Model**: Tuned Random Forest (`max_depth=5`, `min_samples_split=10`, `n_estimators=200`)
3. **Accuracy (Test)**: **0.7500**
4. **Macro F1 (Test)**: **0.6714**
5. **Weighted F1 (Test)**: **0.7502**
6. **Top Features**: `commit_frequency`, `modification_count`, `commit_count`, `contributor_count`.
7. **Confusion Matrix Analysis**: High-risk files are highly predictable (90% precision, 81.8% recall). Low-risk files are frequently confused as Medium risk.
8. **Leakage Assessment**: Clean. No target leakage is present.
9. **Generalization Assessment**: Poor validation performance (Acc=0.5035, F1=0.3326) highlights a severe domain shift. Features must be normalized per-repo.
10. **Interview Readiness**: Rated **GOOD**. Methodological split-safety is excellent, but lack of per-repo normalization is a visible gaps.
11. **Phase 5 (CodeBERT) Recommendation**: **YES (Approved with warning)**. Traditional metrics are saturated; semantic analysis via CodeBERT is required to capture risk structure.
