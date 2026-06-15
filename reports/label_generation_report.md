# Label Generation Report — v3

**Generated**: 2026-06-15 09:33:44

## Motivation for Redesign
The previous labeling scheme used `bug_fix_commit_count` as the sole determinant:
```
if bug_fix_commit_count == 0:   LOW
elif bug_fix_commit_count <= 2: MEDIUM
else:                           HIGH
```
This caused **direct feature-to-label leakage** (single-feature Macro F1 = 1.0000).
All 6 classical models achieved Macro F1 = 1.0000 — measuring the labeling rule, not model skill.

## New Labeling Methodology

### Features Removed (Leakage)
| Feature | Reason | Single-feature F1 (old labels) |
|---------|--------|-------------------------------|
| `bug_fix_commit_count` | **IS the old label** — primary leakage | 1.0000 |
| `historical_bug_density` | SpearmanR=0.90 with above — proxy leakage | 0.9325 |
| `time_since_last_bug_fix` | Sentinel 1000 = LOW class perfectly | 0.9189 |

### Features Transformed
| Feature | Transformation | Rationale |
|---------|---------------|-----------|
| `time_since_last_bug_fix` | Sentinel 1000 → NaN | Remove encoding of label |
| (new) `has_bug_fix_history` | Binary flag (0/1) | Preserves whether any bug fix occurred |

### Composite Risk Score Formula
```
risk_score =
  0.20 × code_complexity_component     [log(complexity) + log(loc)]
  0.20 × code_quality_component        [1 - clamp(MI, 0,100) / 100]
  0.20 × change_intensity_component    [0.6×log(modification_count) + 0.4×commit_freq]
  0.15 × churn_component               [log(time_decayed_churn)]
  0.15 × team_concentration_component  [0.5×ownership_conc + 0.5×1/log(bus_factor+e)]
  0.10 × contributor_sparsity_component [1 - contributor_entropy_norm]
  ─────────────────────────────────────────────────────────────
  TOTAL WEIGHT: 1.00
```

**Design Principles:**
- All inputs log-transformed to handle extreme skew (skew 15–35 in raw features)
- Clipped at 99th percentile before normalization (outlier robustness)
- No single component exceeds weight 0.20
- `recent_churn` excluded: 89.1% zero inflation
- `repository_age_days` excluded from label: constant per repo (identity leakage risk)
- `has_bug_fix_history` excluded from label: the binary signal was the old leakage source

### Label Assignment (Tertile Quantiles)
| Label | Score Range | Count | Percentage |
|-------|-------------|-------|------------|
| **LOW** | ≤ 0.3478 | 13,424 | 33.3% |
| **MEDIUM** | 0.3478 – 0.4824 | 13,465 | 33.4% |
| **HIGH** | > 0.4824 | 13,424 | 33.3% |

## Leakage Audit Results (Post-Redesign)
Threshold: no single feature may exceed Macro F1 = 0.85
Decision tree depth: max_depth=3

| Feature | Single-feature F1 | Single-feature Acc | Verdict |
|---------|------------------|-------------------|---------|
| `maintainability_index` | 0.7387 | 0.7435 | ✅ PASS |
| `complexity` | 0.5648 | 0.5705 | ✅ PASS |
| `loc` | 0.5158 | 0.5333 | ✅ PASS |
| `time_decayed_churn` | 0.4777 | 0.5111 | ✅ PASS |
| `time_since_last_bug_fix` | 0.4301 | 0.4470 | ✅ PASS |
| `commit_frequency` | 0.4258 | 0.4544 | ✅ PASS |
| `repository_age_days` | 0.4219 | 0.4910 | ✅ PASS |
| `commit_count` | 0.4198 | 0.4228 | ✅ PASS |
| `modification_count` | 0.4198 | 0.4228 | ✅ PASS |
| `ownership_concentration` | 0.3723 | 0.3986 | ✅ PASS |
| `contributor_entropy` | 0.3630 | 0.3883 | ✅ PASS |
| `contributor_count` | 0.3438 | 0.4342 | ✅ PASS |
| `has_bug_fix_history` | 0.3013 | 0.3763 | ✅ PASS |
| `bus_factor` | 0.3006 | 0.3818 | ✅ PASS |
| `recent_churn` | 0.2949 | 0.4039 | ✅ PASS |
| `language` | 0.1669 | 0.3340 | ✅ PASS |

## Final Feature Set
Total features retained: 16

| # | Feature | Role |
|---|---------|------|
| 1 | `language` | Code type (categorical) |
| 2 | `loc` | Code size |
| 3 | `complexity` | Cyclomatic complexity |
| 4 | `maintainability_index` | Code quality metric |
| 5 | `commit_count` | Historical activity volume |
| 6 | `modification_count` | File change count |
| 7 | `contributor_count` | Team size |
| 8 | `commit_frequency` | Normalized commit rate |
| 9 | `repository_age_days` | Repo maturity |
| 10 | `ownership_concentration` | Single-owner risk |
| 11 | `contributor_entropy` | Team distribution |
| 12 | `bus_factor` | Knowledge concentration |
| 13 | `recent_churn` | Recent change velocity |
| 14 | `time_decayed_churn` | Weighted historical churn |
| 15 | `time_since_last_bug_fix` | Days since last bug (NaN if none) |
| 16 | `has_bug_fix_history` | Binary: any bug fix ever |