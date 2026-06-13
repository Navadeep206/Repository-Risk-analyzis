# Domain Shift Quantification Report

This report analyzes features' distribution drift across codebases to identify why prediction models experience generalization drops.

## 1. Feature Drift Severity Rankings

We quantify domain shift using the **Population Stability Index (PSI)**. Features with PSI > 0.25 exhibit significant shifts.

| Rank | Feature Name | Mean PSI | Shift Severity |
| --- | --- | --- | --- |
| 1 | repository_age_days | 7.5495 | Significant Shift |
| 2 | commit_frequency | 1.9274 | Significant Shift |
| 3 | maintainability_index | 1.8152 | Significant Shift |
| 4 | commit_count | 1.1437 | Significant Shift |
| 5 | modification_count | 1.1437 | Significant Shift |
| 6 | loc | 0.9601 | Significant Shift |
| 7 | contributor_count | 0.7646 | Significant Shift |
| 8 | complexity | 0.7383 | Significant Shift |

---

## 2. Deep Dive: Most Volatile Covariates

### Z-score & Kolmogorov-Smirnov statistics by Repository:

### Feature: `repository_age_days`
| Repository Name | Z-Score Shift | KS Statistic | KS p-value | PSI |
| --- | --- | --- | --- | --- |
| click | -0.3033 | 0.6763 | 4.503e-26 | 6.1354 |
| databases | -3.0301 | 1.0000 | 1.697e-44 | 7.8986 |
| axios | -0.4757 | 0.5466 | 1.266e-39 | 5.2508 |
| express | 1.8727 | 0.8895 | 3.474e-97 | 7.3968 |
| jinja | 2.1483 | 1.0000 | 1.887e-87 | 7.9881 |
| redux | -0.8295 | 0.9509 | 1.803e-144 | 10.6274 |

### Feature: `commit_frequency`
| Repository Name | Z-Score Shift | KS Statistic | KS p-value | PSI |
| --- | --- | --- | --- | --- |
| click | 0.6355 | 0.4053 | 5.598e-09 | 1.7977 |
| databases | 1.6311 | 0.3659 | 0.002787 | 4.1333 |
| axios | -0.1943 | 0.4699 | 6.055e-29 | 1.3006 |
| express | 0.0742 | 0.3065 | 8.21e-10 | 1.3804 |
| jinja | 0.0957 | 0.4138 | 5.714e-09 | 2.1823 |
| redux | -0.3223 | 0.2417 | 1.188e-07 | 0.7700 |

### Feature: `maintainability_index`
| Repository Name | Z-Score Shift | KS Statistic | KS p-value | PSI |
| --- | --- | --- | --- | --- |
| click | 2.2476 | 0.8647 | 9.285e-48 | 2.5690 |
| databases | 2.1603 | 0.8258 | 1.183e-17 | 1.4217 |
| axios | -0.5569 | 0.3043 | 3.978e-12 | 2.2152 |
| express | -0.5165 | 0.2707 | 9.955e-08 | 1.4310 |
| jinja | 1.8872 | 0.8606 | 3.257e-45 | 1.0720 |
| redux | -0.5525 | 0.3006 | 1.231e-11 | 2.1826 |

---

## 3. Explaining Generalization Failures

1. **Codebase Size Difference (`loc`)**: Utilities and database backends have a much smaller LOC signature than major frameworks, shifting the scale bounds of tree classifiers.
2. **Process Activity Density (`commit_frequency`)**: Active, collaborative codebases have commit rates multiple orders of magnitude higher than smaller utilities. Standardizing using standard scales over training datasets results in out-of-bounds metrics when testing against smaller/large repositories.
