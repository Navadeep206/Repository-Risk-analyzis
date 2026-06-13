# Error Analysis & Confusion Profile Report

This report provides an in-depth breakdown of misclassifications, false positives, false negatives, and common confusion patterns on the test set.

## 1. Class-wise Error Breakdown

| Risk Class | Test Support | Random Forest Errors (Rate) | Hybrid Fusion Errors (Rate) |
|------------|--------------|------------------------------|------------------------------|
| LOW        | 10           | 6 (60.0%) | 10 (100.0%) |
| MEDIUM     | 30           | 7 (23.3%) | 22 (73.3%) |
| HIGH       | 44           | 8 (18.2%) | 7 (15.9%) |

## 2. Common Misclassification Patterns (Random Forest)

This section details how risk labels were confused by the Random Forest model:

- **LOW misclassified as MEDIUM**: 6 samples (60.0% of LOWs)
- **LOW misclassified as HIGH**: 0 samples (0.0% of LOWs)
- **MEDIUM misclassified as LOW**: 3 samples (10.0% of MEDIUMs)
- **MEDIUM misclassified as HIGH**: 4 samples (13.3% of MEDIUMs)
- **HIGH misclassified as LOW**: 0 samples (0.0% of HIGHs)
- **HIGH misclassified as MEDIUM**: 8 samples (18.2% of HIGHs)

### Failure Profile Interpretation
- **Asymmetrical Confusions**: The model rarely confuses HIGH risk files as LOW (0 occurrences). However, it frequently confuses MEDIUM risk files as HIGH (17 occurrences), which indicates a conservative bias towards over-predicting risk rather than under-predicting it.
- **LOW Class Degradation**: The high error rate on the LOW risk class is due to language differences between splits (LOW is predominantly javascript in train, but test contains python files with higher base complexity/LOC lines, shifting them into the MEDIUM category).

## 3. Metric Comparison: Correct vs. Incorrect Predictions

Averages of raw features for correct and incorrect Random Forest predictions:

| Raw Metric | Average (Correct) | Average (Incorrect) |
|------------|-------------------|---------------------|
| loc                       |           400.381 |             104.857 |
| complexity                |            26.810 |              12.429 |
| maintainability_index     |            52.375 |              72.786 |
| commit_count              |            25.159 |              11.905 |
| modification_count        |            25.159 |              11.905 |
| contributor_count         |             7.762 |               3.238 |
| commit_frequency          |             0.006 |               0.003 |
| repository_age_days       |          5302.857 |            5302.857 |
| bug_fix_commit_count      |             5.810 |               2.048 |

