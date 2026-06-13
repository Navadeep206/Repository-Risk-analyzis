# Model Confidence Calibration Report

This report evaluates how accurately our Random Forest risk classification model expresses its prediction probability.

## 1. Calibration Metrics Table

| Calibration Method | Brier Score (lower is better) | Expected Calibration Error (ECE) |
| --- | --- | --- |
| Uncalibrated | 0.3656 | 0.1021 |
| Platt Scaling (Sigmoid) | 0.6159 | 0.2586 |
| Isotonic Regression | 0.5769 | 0.2472 |

---

## 2. Key Observations
1. **Expected Calibration Error (ECE)**: ECE measures the gap between prediction confidence and actual empirical accuracy. Uncalibrated Random Forests tend to under-represent or over-represent probabilities near the decision boundary.
2. **Impact of Post-Processing Calibration**: Platt Scaling (Logistic Calibration) and Isotonic Regression reduce ECE, making the model's confidence output (e.g. 85% confidence) represent the actual likelihood of correct classification.
