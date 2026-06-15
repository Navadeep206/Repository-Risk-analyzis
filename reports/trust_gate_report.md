# Trust Gate Report

**Generated**: 2026-06-15 09:13:14

## Confidence Bin Accuracy

Confidence bins: 90–100%, 70–90%, 50–70%, <50%

### RandomForest
| Confidence Bin | N Samples | % of Test | Bin Accuracy |
|---------------|-----------|-----------|-------------|
| 90-100% | 11,295 | 99.22% | 1.0000 |
| 70-90% | 89 | 0.78% | 1.0000 |
| 50-70% | 0 | 0.00% | N/A |
| <50% | 0 | 0.00% | N/A |

### XGBoost
| Confidence Bin | N Samples | % of Test | Bin Accuracy |
|---------------|-----------|-----------|-------------|
| 90-100% | 11,384 | 100.00% | 1.0000 |
| 70-90% | 0 | 0.00% | N/A |
| 50-70% | 0 | 0.00% | N/A |
| <50% | 0 | 0.00% | N/A |

### LightGBM
| Confidence Bin | N Samples | % of Test | Bin Accuracy |
|---------------|-----------|-----------|-------------|
| 90-100% | 11,384 | 100.00% | 1.0000 |
| 70-90% | 0 | 0.00% | N/A |
| 50-70% | 0 | 0.00% | N/A |
| <50% | 0 | 0.00% | N/A |

## Interpretation
- High-confidence predictions (90–100%) should achieve ≥90% accuracy for production trustworthiness.
- Low-confidence predictions (<50%) indicate uncertainty and should trigger human review.
- A well-calibrated model shows monotonically increasing accuracy with confidence.