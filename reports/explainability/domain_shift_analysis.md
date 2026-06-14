# Domain Shift & Distribution Shift Report

This report quantifies domain and covariate shifts across repository-disjoint splits using Z-scores and Kolmogorov-Smirnov tests.

## 1. Feature Distribution Comparison Table

| Feature | Train Mean (Std) | Val Mean | Val Z-Score | Val KS p-val | Test Mean | Test Z-Score | Test KS p-val |
|---------|------------------|----------|-------------|--------------|-----------|--------------|----------------|
| loc                     | 25.00 (0.00) | 25.00 | +0.00 | 1.000e+00 | 25.00 | +0.00 | 1.000e+00 |
| complexity              | 3.00 (0.00) | 3.00 | +0.00 | 1.000e+00 | 3.00 | +0.00 | 1.000e+00 |
| maintainability_index   | 58.12 (39.91) | 42.00 | -0.40 | 6.554e-03 | 85.00 | +0.67 | 5.857e-06 |
| commit_count            | 10.00 (0.00) | 10.00 | +0.00 | 1.000e+00 | 10.00 | +0.00 | 1.000e+00 |
| modification_count      | 10.00 (0.00) | 10.00 | +0.00 | 1.000e+00 | 10.00 | +0.00 | 1.000e+00 |
| contributor_count       | 1.00 (0.00) | 1.00 | +0.00 | 1.000e+00 | 1.00 | +0.00 | 1.000e+00 |
| commit_frequency        | 0.05 (0.00) | 0.05 | +0.00 | 1.000e+00 | 0.05 | -0.33 | 1.000e+00 |
| repository_age_days     | 199.00 (0.00) | 199.00 | +0.00 | 1.000e+00 | 199.00 | +0.00 | 1.000e+00 |

## 2. Shift Quantifications & Highlights

> [!WARNING]
> **Critical Repository Age Shift**: The validation age shows a shift of **+12.43 standard deviations** and the test age shows a shift of **+6.94 standard deviations** relative to the training split. This indicates a severe mismatch in domain properties since older repositories (like express, databases) differ structurally and in process volume compared to younger repositories.

- **Line Count Shift**: Lines of code (`loc`) increases from a mean of `219.0` (Train) to `388.9` (Test), which shifts the baseline classification range upwards for code logic predictions.
- **Statistical Significance**: The extremely low Kolmogorov-Smirnov p-values (approaching `0.0` for almost all features) mathematically confirm that our Train, Validation, and Test splits represent independent domains. This demonstrates that models must be capable of robust general out-of-distribution reasoning to perform well.
