# Embedding Analysis Report - Phase 5 CodeBERT

This report provides a statistical breakdown and structural audit of the generated CodeBERT embeddings.

## 1. Summary Statistics

| Metric | Value |
|--------|-------|
| **Total Files Embedded** | 723 |
| **Embedding Dimension** | 768 (Expected: 768) |
| **NumPy Array Disk Size** | 2.12 MB |
| **Mean L2 Norm** | 18.1314 |
| **L2 Norm Std Dev** | 0.5298 |
| **Min L2 Norm** | 17.0008 |
| **Max L2 Norm** | 19.9525 |

---

## 2. Repository Distribution

| Repository | Count | Percentage |
|------------|-------|------------|
| axios | 198 | 27.39% |
| redux | 193 | 26.69% |
| express | 141 | 19.50% |
| click | 63 | 8.71% |
| jinja | 60 | 8.30% |
| lodash | 44 | 6.09% |
| databases | 24 | 3.32% |

---

## 3. Language Distribution

| Language | Count | Percentage |
|----------|-------|------------|
| javascript | 500 | 69.16% |
| python | 147 | 20.33% |
| typescript | 76 | 10.51% |

---

## 4. Quality Audit & Verification Check

- **Null Values Check**: ✅ PASSED (No NaN values)
- **Infinite Values Check**: ✅ PASSED (No infinite values)
- **Dimensionality Audit**: ✅ PASSED (Confirmed 768-D)
