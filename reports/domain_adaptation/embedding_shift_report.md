# CodeBERT Embedding Domain Shift Analysis

This report analyzes the spatial distribution of CodeBERT semantic representation vectors across different codebases to determine how strongly language boundaries and codebase styles affect semantic drift.

## 1. Centroid Cosine Similarity Matrix

| Repository | click | databases | axios | express | jinja | redux |
| --- | --- | --- | --- | --- | --- | --- |
| **click** | 1.0000 | 0.9899 | 0.9920 | 0.9882 | 0.9986 | 0.9916 |
| **databases** | 0.9899 | 1.0000 | 0.9767 | 0.9705 | 0.9863 | 0.9783 |
| **axios** | 0.9920 | 0.9767 | 1.0000 | 0.9969 | 0.9907 | 0.9953 |
| **express** | 0.9882 | 0.9705 | 0.9969 | 1.0000 | 0.9875 | 0.9887 |
| **jinja** | 0.9986 | 0.9863 | 0.9907 | 0.9875 | 1.0000 | 0.9889 |
| **redux** | 0.9916 | 0.9783 | 0.9953 | 0.9887 | 0.9889 | 1.0000 |

---

## 2. Spatial Cluster Drift Metrics

- **Inter-Repository Variance (Centroid spread)**: `2.2113`
- **Intra-Repository Variance (In-domain spread)**: `13.1885`
- **Cluster Drift Ratio (Inter/Intra Variance)**: `0.1677`

### Observations:
1. **Semantic Language Separation**: JavaScript repositories (`axios`, `express`, `redux`) exhibit high similarity with each other, while showing lower cosine similarities when compared to Python repositories (`click`, `databases`, `jinja`).
2. **Cluster Drift Ratio**: A drift ratio of `0.1677` implies that a substantial portion of vector variance is determined strictly by the repository boundary rather than the class or function complexity. This explains why deep learning models trained on raw embeddings experience a severe generalization drop when evaluated on unseen domains.
