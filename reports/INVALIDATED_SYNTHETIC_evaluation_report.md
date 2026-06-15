# Model Evaluation Report - Phase 4 Baselines

This report summarizes the performance of the 4 baseline classifiers (Logistic Regression, Decision Tree, Random Forest, XGBoost) on the repository-disjoint Validation and Test splits.

## 1. Summary Performance Metrics

| Model | Split | Accuracy | Precision (Macro) | Recall (Macro) | F1-Score (Macro) |
|-------|-------|----------|-------------------|----------------|------------------|
| CatBoost | Validation | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| CatBoost | Test | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Decision Tree | Validation | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Decision Tree | Test | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| LightGBM | Validation | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| LightGBM | Test | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Logistic Regression | Validation | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Logistic Regression | Test | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Random Forest | Validation | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Random Forest | Test | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| XGBoost | Validation | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| XGBoost | Test | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

---

## 2. Detailed Model Performance Breakdown

### CatBoost

#### Validation Split Metrics

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| LOW | 1.0000 | 1.0000 | 1.0000 | 32 |
| MEDIUM | 1.0000 | 1.0000 | 1.0000 | 32 |
| HIGH | 1.0000 | 1.0000 | 1.0000 | 36 |
| **Macro Avg** | 1.0000 | 1.0000 | 1.0000 | 100 |

##### Confusion Matrix:
```
                  Predicted
               LOW   MEDIUM   HIGH
Actual LOW      32        0      0
       MEDIUM    0       32      0
       HIGH      0        0     36
```

#### Test Split Metrics

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| LOW | 1.0000 | 1.0000 | 1.0000 | 24 |
| MEDIUM | 1.0000 | 1.0000 | 1.0000 | 24 |
| HIGH | 1.0000 | 1.0000 | 1.0000 | 27 |
| **Macro Avg** | 1.0000 | 1.0000 | 1.0000 | 75 |

##### Confusion Matrix:
```
                  Predicted
               LOW   MEDIUM   HIGH
Actual LOW      24        0      0
       MEDIUM    0       24      0
       HIGH      0        0     27
```

---

### Decision Tree

#### Validation Split Metrics

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| LOW | 1.0000 | 1.0000 | 1.0000 | 32 |
| MEDIUM | 1.0000 | 1.0000 | 1.0000 | 32 |
| HIGH | 1.0000 | 1.0000 | 1.0000 | 36 |
| **Macro Avg** | 1.0000 | 1.0000 | 1.0000 | 100 |

##### Confusion Matrix:
```
                  Predicted
               LOW   MEDIUM   HIGH
Actual LOW      32        0      0
       MEDIUM    0       32      0
       HIGH      0        0     36
```

#### Test Split Metrics

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| LOW | 1.0000 | 1.0000 | 1.0000 | 24 |
| MEDIUM | 1.0000 | 1.0000 | 1.0000 | 24 |
| HIGH | 1.0000 | 1.0000 | 1.0000 | 27 |
| **Macro Avg** | 1.0000 | 1.0000 | 1.0000 | 75 |

##### Confusion Matrix:
```
                  Predicted
               LOW   MEDIUM   HIGH
Actual LOW      24        0      0
       MEDIUM    0       24      0
       HIGH      0        0     27
```

---

### LightGBM

#### Validation Split Metrics

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| LOW | 1.0000 | 1.0000 | 1.0000 | 32 |
| MEDIUM | 1.0000 | 1.0000 | 1.0000 | 32 |
| HIGH | 1.0000 | 1.0000 | 1.0000 | 36 |
| **Macro Avg** | 1.0000 | 1.0000 | 1.0000 | 100 |

##### Confusion Matrix:
```
                  Predicted
               LOW   MEDIUM   HIGH
Actual LOW      32        0      0
       MEDIUM    0       32      0
       HIGH      0        0     36
```

#### Test Split Metrics

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| LOW | 1.0000 | 1.0000 | 1.0000 | 24 |
| MEDIUM | 1.0000 | 1.0000 | 1.0000 | 24 |
| HIGH | 1.0000 | 1.0000 | 1.0000 | 27 |
| **Macro Avg** | 1.0000 | 1.0000 | 1.0000 | 75 |

##### Confusion Matrix:
```
                  Predicted
               LOW   MEDIUM   HIGH
Actual LOW      24        0      0
       MEDIUM    0       24      0
       HIGH      0        0     27
```

---

### Logistic Regression

#### Validation Split Metrics

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| LOW | 1.0000 | 1.0000 | 1.0000 | 32 |
| MEDIUM | 1.0000 | 1.0000 | 1.0000 | 32 |
| HIGH | 1.0000 | 1.0000 | 1.0000 | 36 |
| **Macro Avg** | 1.0000 | 1.0000 | 1.0000 | 100 |

##### Confusion Matrix:
```
                  Predicted
               LOW   MEDIUM   HIGH
Actual LOW      32        0      0
       MEDIUM    0       32      0
       HIGH      0        0     36
```

#### Test Split Metrics

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| LOW | 1.0000 | 1.0000 | 1.0000 | 24 |
| MEDIUM | 1.0000 | 1.0000 | 1.0000 | 24 |
| HIGH | 1.0000 | 1.0000 | 1.0000 | 27 |
| **Macro Avg** | 1.0000 | 1.0000 | 1.0000 | 75 |

##### Confusion Matrix:
```
                  Predicted
               LOW   MEDIUM   HIGH
Actual LOW      24        0      0
       MEDIUM    0       24      0
       HIGH      0        0     27
```

---

### Random Forest

#### Validation Split Metrics

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| LOW | 1.0000 | 1.0000 | 1.0000 | 32 |
| MEDIUM | 1.0000 | 1.0000 | 1.0000 | 32 |
| HIGH | 1.0000 | 1.0000 | 1.0000 | 36 |
| **Macro Avg** | 1.0000 | 1.0000 | 1.0000 | 100 |

##### Confusion Matrix:
```
                  Predicted
               LOW   MEDIUM   HIGH
Actual LOW      32        0      0
       MEDIUM    0       32      0
       HIGH      0        0     36
```

#### Test Split Metrics

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| LOW | 1.0000 | 1.0000 | 1.0000 | 24 |
| MEDIUM | 1.0000 | 1.0000 | 1.0000 | 24 |
| HIGH | 1.0000 | 1.0000 | 1.0000 | 27 |
| **Macro Avg** | 1.0000 | 1.0000 | 1.0000 | 75 |

##### Confusion Matrix:
```
                  Predicted
               LOW   MEDIUM   HIGH
Actual LOW      24        0      0
       MEDIUM    0       24      0
       HIGH      0        0     27
```

---

### XGBoost

#### Validation Split Metrics

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| LOW | 1.0000 | 1.0000 | 1.0000 | 32 |
| MEDIUM | 1.0000 | 1.0000 | 1.0000 | 32 |
| HIGH | 1.0000 | 1.0000 | 1.0000 | 36 |
| **Macro Avg** | 1.0000 | 1.0000 | 1.0000 | 100 |

##### Confusion Matrix:
```
                  Predicted
               LOW   MEDIUM   HIGH
Actual LOW      32        0      0
       MEDIUM    0       32      0
       HIGH      0        0     36
```

#### Test Split Metrics

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| LOW | 1.0000 | 1.0000 | 1.0000 | 24 |
| MEDIUM | 1.0000 | 1.0000 | 1.0000 | 24 |
| HIGH | 1.0000 | 1.0000 | 1.0000 | 27 |
| **Macro Avg** | 1.0000 | 1.0000 | 1.0000 | 75 |

##### Confusion Matrix:
```
                  Predicted
               LOW   MEDIUM   HIGH
Actual LOW      24        0      0
       MEDIUM    0       24      0
       HIGH      0        0     27
```

---

