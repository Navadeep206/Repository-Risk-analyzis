# Leave-One-Repository-Out (LORO) Benchmark Summary

This summary analyzes the generalizability of the Random Forest risk classifier when tested on a completely unseen repository during training.

## 1. LORO Results Table

| Held-Out Repository | Train Samples | Test Samples | Accuracy | Macro F1 | Weighted F1 |
| --- | --- | --- | --- | --- | --- |
| click | 621 | 63 | 0.6349 | 0.4454 | 0.5818 |
| databases | 660 | 24 | 0.7917 | 0.7462 | 0.7803 |
| axios | 483 | 201 | 0.6567 | 0.6101 | 0.6576 |
| express | 543 | 141 | 0.5035 | 0.3188 | 0.3971 |
| jinja | 624 | 60 | 0.7667 | 0.6598 | 0.7669 |
| redux | 489 | 195 | 0.4462 | 0.3861 | 0.4153 |
| **Average** | **-** | **-** | **0.6333** | **0.5277** | **0.5998** |
    
---

## 2. Key Insights

1. **Hardest Generalization Targets**: Fold-level results indicate where model performance drops significantly. This reflects severe class distribution shifts and baseline activity scale mismatches.
2. **Easiest Generalization Targets**: Repositories that share class profiles and scale features with the pool of training repositories show higher cross-repo accuracy.
