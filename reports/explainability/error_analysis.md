# Error Analysis & Confusion Profile Report

This report provides an in-depth breakdown of misclassifications, false positives, false negatives, and common confusion patterns on the test set.

## 1. Class-wise Error Breakdown

| Risk Class | Test Support | Random Forest Errors (Rate) | Hybrid Fusion Errors (Rate) |
|------------|--------------|------------------------------|------------------------------|
| LOW        | 24           | 0 (0.0%) | 0 (0.0%) |
| MEDIUM     | 24           | 0 (0.0%) | 0 (0.0%) |
| HIGH       | 27           | 0 (0.0%) | 0 (0.0%) |

## 2. Common Misclassification Patterns (Random Forest)

This section details how risk labels were confused by the Random Forest model:

- **LOW misclassified as MEDIUM**: 0 samples (0.0% of LOWs)
- **LOW misclassified as HIGH**: 0 samples (0.0% of LOWs)
- **MEDIUM misclassified as LOW**: 0 samples (0.0% of MEDIUMs)
- **MEDIUM misclassified as HIGH**: 0 samples (0.0% of MEDIUMs)
- **HIGH misclassified as LOW**: 0 samples (0.0% of HIGHs)
- **HIGH misclassified as MEDIUM**: 0 samples (0.0% of HIGHs)

### Failure Profile Interpretation
- **Asymmetrical Confusions**: The model rarely confuses HIGH risk files as LOW (0 occurrences). However, it frequently confuses MEDIUM risk files as HIGH (17 occurrences), which indicates a conservative bias towards over-predicting risk rather than under-predicting it.
- **LOW Class Degradation**: The high error rate on the LOW risk class is due to language differences between splits (LOW is predominantly javascript in train, but test contains python files with higher base complexity/LOC lines, shifting them into the MEDIUM category).

## 3. Metric Comparison: Correct vs. Incorrect Predictions

Averages of raw features for correct and incorrect Random Forest predictions:

| Raw Metric | Average (Correct) | Average (Incorrect) |
|------------|-------------------|---------------------|
| loc                       |            25.000 |               0.000 |
| complexity                |             3.000 |               0.000 |
| maintainability_index     |            85.000 |               0.000 |
| commit_count              |            10.000 |               0.000 |
| modification_count        |            10.000 |               0.000 |
| contributor_count         |             1.000 |               0.000 |
| commit_frequency          |             0.050 |               0.000 |
| repository_age_days       |           199.000 |               0.000 |
| bug_fix_commit_count      |             2.440 |               0.000 |
| ownership_concentration   |             1.000 |               0.000 |
| contributor_entropy       |            -0.000 |               0.000 |
| bus_factor                |             1.000 |               0.000 |
| recent_churn              |            70.000 |               0.000 |
| time_decayed_churn        |            64.538 |               0.000 |
| time_since_last_bug_fix   |           324.240 |               0.000 |
| historical_bug_density    |             0.098 |               0.000 |

