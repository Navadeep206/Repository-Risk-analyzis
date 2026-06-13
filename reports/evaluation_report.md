# Model Evaluation Report - Phase 4 Baselines

This report summarizes the performance of the 4 baseline classifiers (Logistic Regression, Decision Tree, Random Forest, XGBoost) on the repository-disjoint Validation and Test splits.

## 1. Summary Performance Metrics

| Model | Split | Accuracy | Precision (Macro) | Recall (Macro) | F1-Score (Macro) |
|-------|-------|----------|-------------------|----------------|------------------|
| Decision Tree | Validation | 0.4965 | 0.5250 | 0.4075 | 0.3418 |
| Decision Tree | Test | 0.6071 | 0.6049 | 0.4838 | 0.4995 |
| Logistic Regression | Validation | 0.4255 | 0.1418 | 0.3333 | 0.1990 |
| Logistic Regression | Test | 0.4762 | 0.2571 | 0.4061 | 0.3132 |
| Random Forest | Validation | 0.5035 | 0.3126 | 0.4015 | 0.3326 |
| Random Forest | Test | 0.7500 | 0.6977 | 0.6616 | 0.6714 |
| XGBoost | Validation | 0.4894 | 0.2999 | 0.3879 | 0.3016 |
| XGBoost | Test | 0.7024 | 0.6348 | 0.6768 | 0.6373 |

---

## 2. Detailed Model Performance Breakdown

### Decision Tree

#### Validation Split Metrics

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| LOW | 0.6667 | 0.0769 | 0.1379 | 26 |
| MEDIUM | 0.4000 | 0.1455 | 0.2133 | 55 |
| HIGH | 0.5085 | 1.0000 | 0.6742 | 60 |
| **Macro Avg** | 0.5250 | 0.4075 | 0.3418 | 141 |

##### Confusion Matrix:
```
                  Predicted
               LOW   MEDIUM   HIGH
Actual LOW       2       12     12
       MEDIUM    1        8     46
       HIGH      0        0     60
```

#### Test Split Metrics

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| LOW | 0.6667 | 0.2000 | 0.3077 | 10 |
| MEDIUM | 0.4815 | 0.4333 | 0.4561 | 30 |
| HIGH | 0.6667 | 0.8182 | 0.7347 | 44 |
| **Macro Avg** | 0.6049 | 0.4838 | 0.4995 | 84 |

##### Confusion Matrix:
```
                  Predicted
               LOW   MEDIUM   HIGH
Actual LOW       2        6      2
       MEDIUM    1       13     16
       HIGH      0        8     36
```

---

### Logistic Regression

#### Validation Split Metrics

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| LOW | 0.0000 | 0.0000 | 0.0000 | 26 |
| MEDIUM | 0.0000 | 0.0000 | 0.0000 | 55 |
| HIGH | 0.4255 | 1.0000 | 0.5970 | 60 |
| **Macro Avg** | 0.1418 | 0.3333 | 0.1990 | 141 |

##### Confusion Matrix:
```
                  Predicted
               LOW   MEDIUM   HIGH
Actual LOW       0        0     26
       MEDIUM    0        0     55
       HIGH      0        0     60
```

#### Test Split Metrics

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| LOW | 0.2000 | 0.4000 | 0.2667 | 10 |
| MEDIUM | 0.0000 | 0.0000 | 0.0000 | 30 |
| HIGH | 0.5714 | 0.8182 | 0.6729 | 44 |
| **Macro Avg** | 0.2571 | 0.4061 | 0.3132 | 84 |

##### Confusion Matrix:
```
                  Predicted
               LOW   MEDIUM   HIGH
Actual LOW       4        0      6
       MEDIUM    9        0     21
       HIGH      7        1     36
```

---

### Random Forest

#### Validation Split Metrics

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| LOW | 0.0000 | 0.0000 | 0.0000 | 26 |
| MEDIUM | 0.4000 | 0.2545 | 0.3111 | 55 |
| HIGH | 0.5377 | 0.9500 | 0.6867 | 60 |
| **Macro Avg** | 0.3126 | 0.4015 | 0.3326 | 141 |

##### Confusion Matrix:
```
                  Predicted
               LOW   MEDIUM   HIGH
Actual LOW       0       18      8
       MEDIUM    0       14     41
       HIGH      0        3     57
```

#### Test Split Metrics

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| LOW | 0.5714 | 0.4000 | 0.4706 | 10 |
| MEDIUM | 0.6216 | 0.7667 | 0.6866 | 30 |
| HIGH | 0.9000 | 0.8182 | 0.8571 | 44 |
| **Macro Avg** | 0.6977 | 0.6616 | 0.6714 | 84 |

##### Confusion Matrix:
```
                  Predicted
               LOW   MEDIUM   HIGH
Actual LOW       4        6      0
       MEDIUM    3       23      4
       HIGH      0        8     36
```

---

### XGBoost

#### Validation Split Metrics

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| LOW | 0.0000 | 0.0000 | 0.0000 | 26 |
| MEDIUM | 0.3913 | 0.1636 | 0.2308 | 55 |
| HIGH | 0.5085 | 1.0000 | 0.6742 | 60 |
| **Macro Avg** | 0.2999 | 0.3879 | 0.3016 | 141 |

##### Confusion Matrix:
```
                  Predicted
               LOW   MEDIUM   HIGH
Actual LOW       0       14     12
       MEDIUM    0        9     46
       HIGH      0        0     60
```

#### Test Split Metrics

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| LOW | 0.4118 | 0.7000 | 0.5185 | 10 |
| MEDIUM | 0.6667 | 0.4667 | 0.5490 | 30 |
| HIGH | 0.8261 | 0.8636 | 0.8444 | 44 |
| **Macro Avg** | 0.6348 | 0.6768 | 0.6373 | 84 |

##### Confusion Matrix:
```
                  Predicted
               LOW   MEDIUM   HIGH
Actual LOW       7        2      1
       MEDIUM    9       14      7
       HIGH      1        5     38
```

---

