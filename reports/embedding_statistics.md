# Embedding Analysis Report - Phase 5 CodeBERT

This report provides a statistical breakdown and structural audit of the generated CodeBERT embeddings.

## 1. Summary Statistics

| Metric | Value |
|--------|-------|
| **Total Files Embedded** | 575 |
| **Embedding Dimension** | 768 (Expected: 768) |
| **NumPy Array Disk Size** | 1.68 MB |
| **Mean L2 Norm** | 19.2757 |
| **L2 Norm Std Dev** | 0.2179 |
| **Min L2 Norm** | 18.8319 |
| **Max L2 Norm** | 19.4995 |

---

## 2. Repository Distribution

| Repository | Count | Percentage |
|------------|-------|------------|
| lodash | 25 | 4.35% |
| databases | 25 | 4.35% |
| svelte | 25 | 4.35% |
| pytest | 25 | 4.35% |
| express | 25 | 4.35% |
| jinja | 25 | 4.35% |
| prisma | 25 | 4.35% |
| click | 25 | 4.35% |
| requests | 25 | 4.35% |
| ansible | 25 | 4.35% |
| airflow | 25 | 4.35% |
| fastapi | 25 | 4.35% |
| great_expectations | 25 | 4.35% |
| localstack | 25 | 4.35% |
| prefect | 25 | 4.35% |
| pytorch | 25 | 4.35% |
| django | 25 | 4.35% |
| redux | 25 | 4.35% |
| axios | 25 | 4.35% |
| ray | 25 | 4.35% |
| elasticsearch | 25 | 4.35% |
| scikit-learn | 25 | 4.35% |
| pandas | 25 | 4.35% |

---

## 3. Language Distribution

| Language | Count | Percentage |
|----------|-------|------------|
| python | 425 | 73.91% |
| typescript | 100 | 17.39% |
| javascript | 50 | 8.70% |

---

## 4. Quality Audit & Verification Check

- **Null Values Check**: ✅ PASSED (No NaN values)
- **Infinite Values Check**: ✅ PASSED (No infinite values)
- **Dimensionality Audit**: ✅ PASSED (Confirmed 768-D)
