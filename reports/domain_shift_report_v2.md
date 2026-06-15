# Domain Shift Report v2

**Generated**: 2026-06-15 09:13:18

## Population Stability Index (PSI)
PSI < 0.1: Low shift | PSI 0.1–0.25: Moderate | PSI > 0.25: High shift

| Repository | Mean PSI | Max PSI | Shift Level |
|------------|---------|---------|------------|
| pytorch | 4.6243 | 10.7871 | HIGH 🚨 |
| svelte | 4.3602 | 10.7011 | HIGH 🚨 |
| express | 4.3548 | 13.5273 | HIGH 🚨 |
| jinja | 4.0977 | 13.4585 | HIGH 🚨 |
| prisma | 4.0522 | 11.2546 | HIGH 🚨 |
| ray | 4.0324 | 11.0498 | HIGH 🚨 |
| databases | 3.8674 | 10.8568 | HIGH 🚨 |
| redux | 3.7897 | 11.0134 | HIGH 🚨 |
| lodash | 3.7779 | 8.3735 | HIGH 🚨 |
| pytest | 3.5158 | 13.6153 | HIGH 🚨 |
| axios | 3.4828 | 8.4205 | HIGH 🚨 |
| pandas | 3.2888 | 13.6445 | HIGH 🚨 |
| prefect | 3.2341 | 11.0273 | HIGH 🚨 |
| great_expectations | 3.1397 | 11.8414 | HIGH 🚨 |
| localstack | 3.1088 | 11.8753 | HIGH 🚨 |
| click | 3.0519 | 8.3832 | HIGH 🚨 |
| fastapi | 3.0407 | 10.5933 | HIGH 🚨 |
| airflow | 3.0358 | 12.217 | HIGH 🚨 |
| requests | 2.5419 | 8.3762 | HIGH 🚨 |
| ansible | 2.4237 | 8.7703 | HIGH 🚨 |
| django | 2.2393 | 11.7227 | HIGH 🚨 |
| scikit-learn | 2.1509 | 8.5544 | HIGH 🚨 |

## KS Statistics (Feature Distribution Distance)
KS statistic near 0 = similar distributions | Near 1 = very different

| Repository | Mean KS | Max KS |
|------------|---------|--------|
| pytorch | 0.6151 | 1.0 |
| express | 0.5454 | 0.9187 |
| ray | 0.5162 | 0.8712 |
| jinja | 0.4968 | 0.9204 |
| lodash | 0.4555 | 0.878 |
| pandas | 0.4529 | 0.9123 |
| pytest | 0.4494 | 0.9261 |
| scikit-learn | 0.4482 | 0.8743 |
| click | 0.4285 | 0.8065 |
| redux | 0.4024 | 0.7668 |
| svelte | 0.4019 | 0.8345 |
| prisma | 0.3816 | 0.816 |
| django | 0.366 | 1.0 |
| requests | 0.3522 | 0.852 |
| axios | 0.3481 | 0.8043 |
| databases | 0.3449 | 0.681 |
| prefect | 0.3074 | 0.7453 |
| airflow | 0.296 | 0.7481 |
| great_expectations | 0.2878 | 0.57 |
| ansible | 0.2833 | 0.8443 |
| fastapi | 0.2719 | 0.6048 |
| localstack | 0.2409 | 0.5671 |

## Repository Similarity (Cosine Similarity of Feature Vectors)
| Metric | Repo A | Repo B | Similarity |
|--------|--------|--------|-----------|
| Most Similar | requests | pytest | 0.9999 |
| Least Similar | axios | pytorch | 0.0626 |

## Interpretation
- Repositories with HIGH PSI represent significant distribution shifts from the pool.
- These repositories are harder for the model to generalize to.
- Low KS statistics indicate feature distributions are similar across repositories.
- LORO performance often correlates with PSI: high PSI → lower LORO F1.