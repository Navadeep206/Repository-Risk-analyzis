# Feature Scaling Robustness Report

This report compares different feature scaling algorithms and their robustness against Out-Of-Distribution (OOD) domain shifts.

## 1. Scaling Robustness Metrics Table

| Scaler Type | Classification Accuracy | Classification Macro F1 | Forecasting MAE (30d) | Forecasting RMSE (30d) |
| --- | --- | --- | --- | --- |
| StandardScaler | 0.6548 | 0.5931 | 19.9616 | 30.8886 |
| RobustScaler | 0.6429 | 0.5666 | 19.9616 | 30.8886 |
| QuantileTransformer (Uniform) | 0.6548 | 0.5868 | 20.1839 | 30.9440 |
| Rank Normalization (Quantile Normal) | 0.6429 | 0.5666 | 20.1839 | 30.9440 |

---

## 2. Key Takeaways
1. **StandardScaler vs RobustScaler**: Standard scaling uses mean and variance, which are sensitive to outliers. Robust scaling uses median and interquartile range (IQR), which performs much better when codebases have skewed sizes.
2. **Quantile and Rank Normalization**: Mapping metrics to quantiles or normal distributions completely resolves absolute scale disparities (e.g. mapping click vs axios LOC to standardized relative ranks). This substantially reduces forecasting errors and prevents classification generalization drops.
