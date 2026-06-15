# Feature Audit Report

**Generated**: 2026-06-15 09:11:31

## Summary
| Metric | Value |
|--------|-------|
| Total Features | 17 |
| Features Included | 17 |
| Potential Leakage Features | 0 |
| Duplicate Features | 0 |

## Feature Details
| Feature | dtype | Null% | Variance | Corr(Target) | Leakage | Duplicate | Verdict |
|---------|-------|-------|----------|-------------|---------|-----------|---------|
| language | str | 0.0% | nan | N/A | False | False | **INCLUDE** |
| loc | int64 | 0.0% | 474920.6435 | 0.1354 | False | False | **INCLUDE** |
| complexity | int64 | 0.0% | 2076.629 | 0.118 | False | False | **INCLUDE** |
| maintainability_index | float64 | 0.0% | 1502.0336 | 0.0 | False | False | **INCLUDE** |
| commit_count | int64 | 0.0% | 1702.5428 | 0.2496 | False | False | **INCLUDE** |
| modification_count | int64 | 0.0% | 1702.545 | 0.2496 | False | False | **INCLUDE** |
| contributor_count | int64 | 0.0% | 109.3466 | 0.3415 | False | False | **INCLUDE** |
| commit_frequency | float64 | 0.0% | 0.0021 | 0.0057 | False | False | **INCLUDE** |
| repository_age_days | int64 | 0.0% | 5116543.6058 | 0.102 | False | False | **INCLUDE** |
| bug_fix_commit_count | int64 | 0.0% | 235.2718 | 0.2667 | False | False | **INCLUDE** |
| ownership_concentration | float64 | 0.0% | 0.126 | 0.272 | False | False | **INCLUDE** |
| contributor_entropy | float64 | 0.0% | 1.5776 | 0.4017 | False | False | **INCLUDE** |
| bus_factor | int64 | 0.0% | 1.8267 | 0.3847 | False | False | **INCLUDE** |
| recent_churn | float64 | 0.0% | 2766.3918 | 0.0283 | False | False | **INCLUDE** |
| time_decayed_churn | float64 | 0.0% | 8384.109 | 0.082 | False | False | **INCLUDE** |
| time_since_last_bug_fix | float64 | 0.0% | 577642.4759 | 0.0661 | False | False | **INCLUDE** |
| historical_bug_density | float64 | 0.0% | 1.1972 | 0.0032 | False | False | **INCLUDE** |

## Leakage Analysis
- No columns directly encoding the target label were detected in the feature set.
- `historical_bug_density` is a lagged historical metric (not target-derived).
- All features represent code metrics and repository-level statistics measurable before labeling.

## Target-Derived Column Check
- `historical_risk_label` is the target column and is excluded from training.
- No other column encodes the target value.