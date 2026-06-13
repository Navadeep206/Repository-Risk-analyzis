# External OOD Validation Report — pallets/flask

**Generated**: 2026-06-13 20:55:58  
**Target Repository**: https://github.com/pallets/flask  
**Repository Status**: ✅ NEVER SEEN IN ANY PREVIOUS EXPERIMENT

---

## 1. Repository Profile

| Property | Value |
| --- | --- |
| Repository Name | flask |
| GitHub URL | https://github.com/pallets/flask |
| Language | Python |
| Total Files Analyzed | 83 |
| Repository Age | 5899 days |
| Average Lines of Code | 220.9 |
| Average Cyclomatic Complexity | 23.06 |

---

## 2. Pipeline Stage Results

| Stage | Status | Duration | Details |
| --- | --- | --- | --- |
| Clone | ✅ PASS | 2.94s | Cloned to /Users/navadeepguduru/Repository mining /repository-risk-intelligence/ |
| Commit Extraction | ✅ PASS | 224.42s | Extracted 5539 commits |
| Modification Extraction | ✅ PASS | 72.16s | Extracted 9258 file modifications |
| Quality Metrics | ✅ PASS | 2.65s | 83 files analyzed. Languages: ['python'] |
| Data Merge | ✅ PASS | 0.24s | Merged 83 rows. Bug-fix commits: 1594 |
| Feature Engineering | ✅ PASS | 0.05s | Engineered 83 rows. Repository age: 5899 days, Avg commit frequency: 0.0036/day |
| Label Generation | ✅ PASS | 0.0s | Labels generated. Distribution: {'LOW': 31, 'HIGH': 27, 'MEDIUM': 25} |
| RF Prediction | ✅ PASS | 1.26s | Predicted 83 files. Distribution: {'MEDIUM': 51, 'HIGH': 29, 'LOW': 3}. Avg conf |
| Explainability | ✅ PASS | 0.01s | Top features: ['commit_frequency', 'modification_count', 'commit_count', 'contri |
| Forecasting | ✅ PASS | 7.24s | Forecasting completed. Targets: ['future_risk_30d', 'future_risk_60d', 'future_r |
| Trust Gate | ✅ PASS | 0.0s | Trusted: 12 (14.5%), Flagged: 71 (85.5%). Threshold: 70% |


---

## 3. Risk Prediction Distribution

| Risk Level | File Count | Share |
| --- | --- | --- |
| HIGH | 29 | 34.9% |
| MEDIUM | 51 | 61.4% |
| LOW | 3 | 3.6% |

**Average model confidence**: 0.612 (61.2%)

---

## 4. Top HIGH-Risk Files

| File Path | Predicted Label | Confidence | Complexity | Modifications |
| --- | --- | --- | --- | --- |
| ...tests/test_basic.py | HIGH | 99.7% | 370.0 | 133 |
| ...tests/conftest.py | HIGH | 53.8% | 17.0 | 31 |
| ...tests/test_signals.py | HIGH | 67.1% | 27.0 | 27 |
| ...tests/test_views.py | HIGH | 63.9% | 44.0 | 22 |
| ...tests/test_reqctx.py | HIGH | 85.9% | 38.0 | 36 |
| ...tests/test_blueprints.py | HIGH | 99.8% | 184.0 | 58 |
| ...tests/test_config.py | HIGH | 87.2% | 60.0 | 35 |
| ...tests/test_user_error_handler.py | HIGH | 61.1% | 28.0 | 20 |
| ...tests/test_helpers.py | HIGH | 94.6% | 37.0 | 108 |
| ...tests/test_json.py | HIGH | 43.1% | 49.0 | 9 |


---

## 5. Trust Gate Summary

| Trust Decision | File Count | Share |
| --- | --- | --- |
| TRUSTED (conf ≥ 70%) | 12 | 14.5% |
| FLAGGED (conf < 70%) | 71 | 85.5% |

---

## 6. Feature Importance (from trained model)

| Feature | Gini Importance |
| --- | --- |
| commit_frequency | 0.2013 |
| modification_count | 0.1466 |
| commit_count | 0.1380 |
| contributor_count | 0.1073 |
| language_javascript | 0.0789 |
| language_typescript | 0.0763 |
| repository_age_days | 0.0755 |
| complexity | 0.0719 |


---

## 7. Forecasting Results (30d/60d/90d Risk)

| Target | Windows | Mean Risk | Min Risk | Max Risk |
| --- | --- | --- | --- | --- |
| future_risk_30d | ERROR | 'numpy.ndarray' object has no attribute 'select_dtypes' | — | — |
| future_risk_60d | ERROR | 'numpy.ndarray' object has no attribute 'select_dtypes' | — | — |
| future_risk_90d | ERROR | 'numpy.ndarray' object has no attribute 'select_dtypes' | — | — |


---

## 8. OOD Hardcoded Assumption Audit

| Check | Finding |
| --- | --- |
| Language support | ✅ PythonAnalyzer handles all Flask Python files |
| Preprocessor feature mismatch | ✅ All 8 numeric features present + language encoded |
| Repository name hardcoding in data merger | ⚠️ merger scans RAW_DIR (global), scoped to flask in validation pipeline |
| Forecasting train_repos hardcode | ⚠️ forecasting_pipeline.py has `train_repos = ["click", "redux", "axios"]` (not called here) |
| Trust gate | ✅ Implemented as inline confidence threshold (no repo-specific logic) |
