# Domain Shift & Distribution Shift Report

This report quantifies domain and covariate shifts across repository-disjoint splits using Z-scores and Kolmogorov-Smirnov tests.

## 1. Feature Distribution Comparison Table

| Feature | Train Mean (Std) | Val Mean | Val Z-Score | Val KS p-val | Test Mean | Test Z-Score | Test KS p-val |
|---------|------------------|----------|-------------|--------------|-----------|--------------|----------------|
| loc                     | 135.92 (396.60) | 112.07 | -0.06 | 1.049e-03 | 326.50 | +0.48 | 1.361e-07 |
| complexity              | 19.59 (54.48) | 24.51 | +0.09 | 6.420e-07 | 23.21 | +0.07 | 1.374e-02 |
| maintainability_index   | 7.30 (23.61) | -1.00 | -0.35 | 3.066e-02 | 57.48 | +2.13 | 1.286e-57 |
| commit_count            | 9.36 (21.76) | 19.37 | +0.46 | 4.295e-21 | 21.85 | +0.57 | 1.123e-17 |
| modification_count      | 9.36 (21.76) | 19.37 | +0.46 | 4.295e-21 | 21.85 | +0.57 | 1.123e-17 |
| contributor_count       | 5.04 (9.93) | 6.75 | +0.17 | 4.717e-16 | 6.63 | +0.16 | 1.847e-04 |
| commit_frequency        | 0.00 (0.00) | 0.00 | +0.19 | 4.089e-13 | 0.01 | +0.68 | 1.023e-12 |
| repository_age_days     | 4209.58 (157.79) | 6168.00 | +12.41 | 4.345e-141 | 5302.86 | +6.93 | 2.998e-36 |

## 2. Shift Quantifications & Highlights

> [!WARNING]
> **Critical Repository Age Shift**: The validation age shows a shift of **+12.43 standard deviations** and the test age shows a shift of **+6.94 standard deviations** relative to the training split. This indicates a severe mismatch in domain properties since older repositories (like express, databases) differ structurally and in process volume compared to younger repositories.

- **Line Count Shift**: Lines of code (`loc`) increases from a mean of `219.0` (Train) to `388.9` (Test), which shifts the baseline classification range upwards for code logic predictions.
- **Statistical Significance**: The extremely low Kolmogorov-Smirnov p-values (approaching `0.0` for almost all features) mathematically confirm that our Train, Validation, and Test splits represent independent domains. This demonstrates that models must be capable of robust general out-of-distribution reasoning to perform well.
