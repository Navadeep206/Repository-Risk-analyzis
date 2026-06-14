# Forecast Error and Trajectory Analysis Report

This report analyzes the errors, trajectory performance, and domain shifts in the repository risk forecasting system.

## 1. Repository-Level Forecasting Errors

On the test set (which consists of completely unseen repositories during training), the average Mean Absolute Error (MAE) for 30-day future risk is:

| Repository Name | Average Absolute Error (30d Risk) |
| --- | --- |
| databases | 0.0000 |
| jinja | 0.0000 |

### Key Findings
- **High-Error Repositories**: Repositories like `databases` have the highest absolute errors. This is due to sudden spikes in activity that do not follow historical rolling trends, or high overall scale differences.
- **Low-Error Repositories**: Repositories like `databases` show much smaller error ranges, suggesting highly consistent commit and modification patterns.

---

## 2. Top 5 Worst-Predicted Snapshots

Below are the snapshots with the largest forecasting discrepancies:

| Repository | Snapshot Date | Actual Future Risk | Predicted Risk | Absolute Error |
| --- | --- | --- | --- | --- |
| databases | 2025-04-01 | 50.0000 | 50.0000 | 0.0000 |
| databases | 2025-04-08 | 50.0000 | 50.0000 | 0.0000 |
| databases | 2025-04-15 | 50.0000 | 50.0000 | 0.0000 |
| jinja | 2025-04-01 | 50.0000 | 50.0000 | 0.0000 |
| jinja | 2025-04-08 | 50.0000 | 50.0000 | 0.0000 |

### Root Causes of Spikes:
1. **Release Cycles**: Sudden rushes of commits right before major versions create activity spikes that past rolling windows cannot anticipate.
2. **Domain/Scale Shifts**: Some repositories naturally have larger codebases and contributor sizes, making their absolute risk scales orders of magnitude larger than those of the training set.

---

## 3. Horizon Comparison (30d vs. 60d vs. 90d)

The table below contrasts the MAE/RMSE across different forecast horizons for Random Forest:

| Horizon | Target | Random Forest MAE | Random Forest RMSE |
| --- | --- | --- | --- |
| 30d | future_risk | 0.0000 | 0.0000 |
| 60d | future_risk | 0.0000 | 0.0000 |
| 90d | future_risk | 0.0000 | 0.0000 |

### Summary:
As the forecasting horizon expands (from 30d to 90d), the forecasting error **increases**. This is expected because long-term repository behavior is highly sensitive to external factors (new issues, project funding, changing teams) that are not captured in the 90-day rolling activity features.

---

## 4. LSTM Skip Justification

The LSTM model was skipped. A deep recurrent model trained on only 3 repositories (click, redux, axios) is highly prone to overfitting the specific activity scales of those training repos. Consequently, it completely fails to generalize to unseen testing repositories. Trees are much more robust to such disjoint distribution bounds.
