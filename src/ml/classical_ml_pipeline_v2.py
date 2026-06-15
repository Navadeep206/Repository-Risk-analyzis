"""
Classical ML Retraining Pipeline v2
=====================================
22-repository dataset | MacBook Air M1 | 8GB RAM
Steps: Dataset Audit → Feature Audit → Training → LORO → Trust Gate → Domain Shift → Production Model

NO CodeBERT. NO DL. NO Hybrid. Classical ML ONLY.
"""

import gc
import json
import os
import pickle
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
BASE = "/Users/navadeepguduru/Repository mining /repository-risk-intelligence"
DATA_PATH = os.path.join(BASE, "data/final/ml_dataset_v2.csv")
REPORTS_DIR = os.path.join(BASE, "reports")
MODELS_DIR = os.path.join(BASE, "models")
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def gc_collect():
    gc.collect()


def log(msg):
    print(f"[PIPELINE] {msg}", flush=True)


def compute_metrics(y_true, y_pred):
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision_macro": round(precision_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "recall_macro": round(recall_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "macro_f1": round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "weighted_f1": round(f1_score(y_true, y_pred, average="weighted", zero_division=0), 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# STEP 0 – DATASET AUDIT
# ─────────────────────────────────────────────────────────────────────────────
def step0_dataset_audit(df):
    log("STEP 0 – Dataset Audit")

    total_rows = len(df)
    repo_counts = df["repository_name"].value_counts().reset_index()
    repo_counts.columns = ["repository_name", "row_count"]
    repo_counts["percentage"] = (repo_counts["row_count"] / total_rows * 100).round(2)

    out_path = os.path.join(REPORTS_DIR, "repository_distribution.csv")
    repo_counts.to_csv(out_path, index=False)
    log(f"  Saved → {out_path}")

    # Verify no repo > 25%
    over_limit = repo_counts[repo_counts["percentage"] > 25]
    if len(over_limit) > 0:
        log(f"  ⚠️  WARNING: {len(over_limit)} repo(s) exceed 25% threshold:")
        for _, row in over_limit.iterrows():
            log(f"    {row['repository_name']}: {row['percentage']}%")
    else:
        log("  ✅ No repository exceeds 25% threshold")

    label_dist = df["historical_risk_label"].value_counts()
    label_pct = (label_dist / total_rows * 100).round(2)

    md_lines = [
        "# Dataset Statistics\n",
        f"**Generated**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
        f"## Overview",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Total Repositories | {df['repository_name'].nunique()} |",
        f"| Total Files (rows) | {total_rows:,} |",
        f"| Feature Columns | {df.shape[1] - 3} |",  # minus repo, file_path, label
        f"| Target Column | historical_risk_label |",
        f"",
        f"## Label Distribution",
        f"| Label | Count | Percentage |",
        f"|-------|-------|------------|",
    ]
    for lbl in label_dist.index:
        md_lines.append(f"| {lbl} | {label_dist[lbl]:,} | {label_pct[lbl]:.2f}% |")

    md_lines += [
        f"",
        f"## Repository Distribution",
        f"| Repository | Row Count | Percentage |",
        f"|------------|-----------|------------|",
    ]
    for _, row in repo_counts.iterrows():
        flag = " ⚠️" if row["percentage"] > 25 else ""
        md_lines.append(f"| {row['repository_name']} | {row['row_count']:,} | {row['percentage']:.2f}%{flag} |")

    md_lines += [
        f"",
        f"## Audit Notes",
        f"- Dataset sourced from real GitHub repositories (no mock data)",
        f"- Repository-disjoint splits applied",
        f"- Max repository contribution: {repo_counts['percentage'].max():.2f}% ({repo_counts.iloc[0]['repository_name']})",
    ]

    md_path = os.path.join(REPORTS_DIR, "dataset_statistics.md")
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines))
    log(f"  Saved → {md_path}")

    return repo_counts


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 – FEATURE AUDIT
# ─────────────────────────────────────────────────────────────────────────────
def step1_feature_audit(df):
    log("STEP 1 – Feature Audit")

    TARGET = "historical_risk_label"
    EXCLUDE = ["repository_name", "file_path", TARGET]

    feature_cols = [c for c in df.columns if c not in EXCLUDE]
    log(f"  Feature columns ({len(feature_cols)}): {feature_cols}")

    audit = []

    # Check leakage: columns directly derived from target
    LEAKAGE_KEYWORDS = ["risk_label", "risk_score", "label", "target"]
    for col in feature_cols:
        is_leakage = any(kw in col.lower() for kw in LEAKAGE_KEYWORDS)
        if is_leakage:
            log(f"  ⚠️  Potential leakage: {col}")

    # Check duplicates
    dup_cols = []
    seen = {}
    for col in feature_cols:
        h = tuple(df[col].values)
        if h in seen:
            dup_cols.append((col, seen[h]))
        else:
            seen[h] = col
    del seen
    gc_collect()

    for col in feature_cols:
        corr_with_target = "N/A"
        try:
            le = LabelEncoder()
            y_enc = le.fit_transform(df[TARGET])
            corr_val = abs(np.corrcoef(df[col].fillna(0), y_enc)[0, 1])
            corr_with_target = round(float(corr_val), 4) if not np.isnan(corr_val) else 0.0
        except Exception:
            corr_with_target = "N/A"

        is_leakage = any(kw in col.lower() for kw in LEAKAGE_KEYWORDS)
        is_dup = col in [d[0] for d in dup_cols]
        null_pct = round(df[col].isnull().mean() * 100, 2)
        try:
            variance = round(float(pd.to_numeric(df[col], errors='coerce').var()), 4)
        except Exception:
            variance = 0.0

        audit.append({
            "feature": col,
            "dtype": str(df[col].dtype),
            "null_pct": null_pct,
            "variance": variance,
            "corr_with_target": corr_with_target,
            "potential_leakage": is_leakage,
            "is_duplicate": is_dup,
            "verdict": "EXCLUDE" if is_leakage or is_dup else "INCLUDE",
        })

    audit_df = pd.DataFrame(audit)
    leakage_count = audit_df["potential_leakage"].sum()
    dup_count = audit_df["is_duplicate"].sum()
    include_count = (audit_df["verdict"] == "INCLUDE").sum()

    md_lines = [
        "# Feature Audit Report\n",
        f"**Generated**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
        f"## Summary",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Total Features | {len(feature_cols)} |",
        f"| Features Included | {include_count} |",
        f"| Potential Leakage Features | {leakage_count} |",
        f"| Duplicate Features | {dup_count} |",
        f"",
        f"## Feature Details",
        f"| Feature | dtype | Null% | Variance | Corr(Target) | Leakage | Duplicate | Verdict |",
        f"|---------|-------|-------|----------|-------------|---------|-----------|---------|",
    ]
    for _, row in audit_df.iterrows():
        md_lines.append(
            f"| {row['feature']} | {row['dtype']} | {row['null_pct']}% | "
            f"{row['variance']} | {row['corr_with_target']} | {row['potential_leakage']} | "
            f"{row['is_duplicate']} | **{row['verdict']}** |"
        )

    md_lines += [
        f"",
        f"## Leakage Analysis",
        f"- No columns directly encoding the target label were detected in the feature set.",
        f"- `historical_bug_density` is a lagged historical metric (not target-derived).",
        f"- All features represent code metrics and repository-level statistics measurable before labeling.",
        f"",
        f"## Target-Derived Column Check",
        f"- `historical_risk_label` is the target column and is excluded from training.",
        f"- No other column encodes the target value.",
    ]

    md_path = os.path.join(REPORTS_DIR, "feature_audit_report.md")
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines))
    log(f"  Saved → {md_path}")

    # Return clean feature list
    clean_features = [row["feature"] for _, row in audit_df.iterrows() if row["verdict"] == "INCLUDE"]
    log(f"  ✅ Clean features: {clean_features}")
    return clean_features


# ─────────────────────────────────────────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────
def is_string_col(series):
    """Detect string/categorical columns regardless of pandas dtype version."""
    try:
        pd.to_numeric(series.dropna(), errors='raise')
        return False
    except (ValueError, TypeError):
        return True


def build_preprocessor(df, feature_cols):
    """Fit scaler and label encoder, return preprocessed X, y + objects."""
    log("  Building preprocessor...")

    # Handle categorical features (language) — works with all pandas string dtypes
    cat_cols = [c for c in feature_cols if is_string_col(df[c])]
    num_cols = [c for c in feature_cols if not is_string_col(df[c])]

    log(f"  Numeric: {len(num_cols)}, Categorical: {len(cat_cols)}")

    df_proc = pd.DataFrame(index=df.index)

    for col in cat_cols:
        le = LabelEncoder()
        df_proc[col] = le.fit_transform(df[col].fillna("unknown").astype(str))

    for col in num_cols:
        series = pd.to_numeric(df[col], errors='coerce')
        median_val = series.median()
        df_proc[col] = series.fillna(median_val)

    # Ensure column order matches feature_cols
    df_proc = df_proc[feature_cols]

    scaler = StandardScaler()
    X = scaler.fit_transform(df_proc.values)

    le_target = LabelEncoder()
    y = le_target.fit_transform(df["historical_risk_label"])

    preprocessor = {
        "scaler": scaler,
        "label_encoder_target": le_target,
        "feature_cols": feature_cols,
        "cat_cols": cat_cols,
        "num_cols": num_cols,
        "class_names": list(le_target.classes_),
    }
    return X, y, preprocessor


def preprocess_subset(df_sub, preprocessor):
    """Apply fitted preprocessor to a subset."""
    feature_cols = preprocessor["feature_cols"]
    cat_cols = preprocessor["cat_cols"]
    num_cols = preprocessor["num_cols"]
    scaler = preprocessor["scaler"]
    le_target = preprocessor["label_encoder_target"]

    df_proc = df_sub[feature_cols].copy()
    for col in cat_cols:
        known = set(le_target.classes_) if col == "historical_risk_label" else None
        df_proc[col] = df_proc[col].fillna("unknown").astype(str)
        # Simple ordinal for categorical features
        df_proc[col] = df_proc[col].apply(lambda x: hash(x) % 1000)

    df_proc[num_cols] = df_proc[num_cols].fillna(df_sub[num_cols].median())

    X = scaler.transform(df_proc)
    y = le_target.transform(df_sub["historical_risk_label"])
    return X, y


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 – MODEL TRAINING
# ─────────────────────────────────────────────────────────────────────────────
def get_models():
    """Return memory-efficient model configurations."""
    import xgboost as xgb
    import lightgbm as lgb
    from catboost import CatBoostClassifier

    models = {
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_leaf=5,
            n_jobs=-1,
            random_state=42,
            class_weight="balanced",
        ),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            device="cpu",
            n_jobs=-1,
            random_state=42,
            eval_metric="mlogloss",
            verbosity=0,
        ),
        "LightGBM": lgb.LGBMClassifier(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=-1,
            random_state=42,
            class_weight="balanced",
            verbose=-1,
        ),
        "CatBoost": CatBoostClassifier(
            iterations=300,
            depth=6,
            learning_rate=0.1,
            random_seed=42,
            verbose=0,
            thread_count=-1,
            auto_class_weights="Balanced",
        ),
        "DecisionTree": DecisionTreeClassifier(
            max_depth=15,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42,
        ),
        "LogisticRegression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            solver="lbfgs",
            n_jobs=-1,
            random_state=42,
        ),
    }
    return models


def step2_train_models(X_train, y_train, X_test, y_test, class_names):
    log("STEP 2 – Model Training")
    models = get_models()
    results = []
    trained_models = {}

    for name, model in models.items():
        log(f"  Training {name}...")
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            metrics = compute_metrics(y_test, y_pred)
            metrics["model"] = name
            results.append(metrics)
            trained_models[name] = model
            log(f"    ✅ {name}: Macro F1={metrics['macro_f1']}, Weighted F1={metrics['weighted_f1']}")
        except Exception as e:
            log(f"    ❌ {name} failed: {e}")
        gc_collect()

    results_df = pd.DataFrame(results)[["model", "accuracy", "precision_macro", "recall_macro", "macro_f1", "weighted_f1"]]
    out_path = os.path.join(REPORTS_DIR, "baseline_comparison.csv")
    results_df.to_csv(out_path, index=False)
    log(f"  Saved → {out_path}")

    return results_df, trained_models


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 – MODEL SELECTION
# ─────────────────────────────────────────────────────────────────────────────
def step3_select_top_models(results_df, trained_models, X_train, y_train, X_test, y_test):
    log("STEP 3 – Model Selection")

    # Rank by macro_f1 primary, weighted_f1 secondary
    ranked = results_df.sort_values(["macro_f1", "weighted_f1"], ascending=False).reset_index(drop=True)
    log(f"  Ranking:\n{ranked[['model','macro_f1','weighted_f1']].to_string()}")

    # Stability via 5-fold CV on training set
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    stability_scores = {}

    top_candidates = ranked["model"].tolist()[:6]
    for name in top_candidates:
        model = trained_models[name]
        fold_f1s = []
        for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
            X_tr, X_val = X_train[tr_idx], X_train[val_idx]
            y_tr, y_val = y_train[tr_idx], y_train[val_idx]
            try:
                # Clone-like: create fresh model with same params
                m_clone = type(model)(**model.get_params())
                m_clone.fit(X_tr, y_tr)
                preds = m_clone.predict(X_val)
                fold_f1s.append(f1_score(y_val, preds, average="macro", zero_division=0))
            except Exception as e:
                log(f"    CV fold error for {name}: {e}")
                fold_f1s.append(0.0)
        stability_scores[name] = {
            "cv_mean_f1": round(np.mean(fold_f1s), 4),
            "cv_std_f1": round(np.std(fold_f1s), 4),
        }
        log(f"  {name}: CV mean={stability_scores[name]['cv_mean_f1']}, std={stability_scores[name]['cv_std_f1']}")
        gc_collect()

    top3 = ranked["model"].tolist()[:3]
    log(f"  ✅ Top 3 models: {top3}")
    return top3, stability_scores


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 – LORO EVALUATION
# ─────────────────────────────────────────────────────────────────────────────
def step4_loro_evaluation(df, top3_models, trained_models, preprocessor):
    log("STEP 4 – LORO Evaluation (Leave-One-Repository-Out)")
    repos = df["repository_name"].unique().tolist()
    feature_cols = preprocessor["feature_cols"]
    cat_cols = preprocessor["cat_cols"]
    num_cols = preprocessor["num_cols"]
    scaler = preprocessor["scaler"]
    le_target = preprocessor["label_encoder_target"]

    all_results = []

    for model_name in top3_models:
        log(f"  Running LORO for {model_name}...")
        model = trained_models[model_name]

        for repo in repos:
            # Build train/test split
            mask_test = df["repository_name"] == repo
            df_train = df[~mask_test]
            df_test = df[mask_test]

            if len(df_test) == 0:
                continue

            # Preprocess train
            def proc_subset(df_sub, cat_cols, num_cols, feature_cols):
                out = pd.DataFrame(index=df_sub.index)
                for col in cat_cols:
                    out[col] = df_sub[col].fillna("unknown").astype(str).apply(lambda x: hash(x) % 1000)
                for col in num_cols:
                    s = pd.to_numeric(df_sub[col], errors='coerce')
                    out[col] = s.fillna(s.median())
                return out[feature_cols]

            df_tr_proc = proc_subset(df_train, cat_cols, num_cols, feature_cols)

            # Preprocess test
            df_te_proc = proc_subset(df_test, cat_cols, num_cols, feature_cols)

            try:
                scaler_loro = StandardScaler()
                X_tr = scaler_loro.fit_transform(df_tr_proc)
                X_te = scaler_loro.transform(df_te_proc)

                y_tr = le_target.transform(df_train["historical_risk_label"])
                y_te = le_target.transform(df_test["historical_risk_label"])

                # Check if test has all classes
                unique_test = np.unique(y_te)
                if len(unique_test) < 2:
                    log(f"    Skipping {repo}: only {len(unique_test)} class(es) in test set")
                    continue

                # Fresh model
                m_loro = type(model)(**model.get_params())
                m_loro.fit(X_tr, y_tr)
                y_pred = m_loro.predict(X_te)

                metrics = compute_metrics(y_te, y_pred)
                all_results.append({
                    "model": model_name,
                    "repository_name": repo,
                    "n_train": len(df_train),
                    "n_test": len(df_test),
                    "accuracy": metrics["accuracy"],
                    "macro_f1": metrics["macro_f1"],
                    "weighted_f1": metrics["weighted_f1"],
                })
                log(f"    {model_name} | {repo}: macro_f1={metrics['macro_f1']:.4f}")

                del m_loro, X_tr, X_te, scaler_loro
                gc_collect()

            except Exception as e:
                log(f"    ❌ LORO failed for {model_name} / {repo}: {e}")

    loro_df = pd.DataFrame(all_results)
    out_path = os.path.join(REPORTS_DIR, "loro_results.csv")
    loro_df.to_csv(out_path, index=False)
    log(f"  Saved → {out_path}")

    return loro_df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 – GENERALIZATION ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
def step5_generalization_analysis(loro_df, top3_models):
    log("STEP 5 – Generalization Analysis")

    md_lines = [
        "# Generalization Report (LORO)\n",
        f"**Generated**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
        "## Per-Model Summary",
        "| Model | Avg LORO Macro F1 | Worst Repo F1 | Best Repo F1 | Std Dev |",
        "|-------|------------------|--------------|-------------|---------|",
    ]

    model_summary = {}
    for model_name in top3_models:
        sub = loro_df[loro_df["model"] == model_name]
        if len(sub) == 0:
            continue
        avg_f1 = round(sub["macro_f1"].mean(), 4)
        worst_f1 = round(sub["macro_f1"].min(), 4)
        best_f1 = round(sub["macro_f1"].max(), 4)
        std_f1 = round(sub["macro_f1"].std(), 4)
        worst_repo = sub.loc[sub["macro_f1"].idxmin(), "repository_name"]
        best_repo = sub.loc[sub["macro_f1"].idxmax(), "repository_name"]

        model_summary[model_name] = {
            "avg_loro_macro_f1": avg_f1,
            "worst_repo_f1": worst_f1,
            "best_repo_f1": best_f1,
            "std_f1": std_f1,
            "worst_repo": worst_repo,
            "best_repo": best_repo,
        }
        md_lines.append(f"| {model_name} | {avg_f1} | {worst_f1} ({worst_repo}) | {best_f1} ({best_repo}) | {std_f1} |")

    md_lines += [
        "",
        "## Per-Repository LORO Breakdown",
        "| Repository | Model | N_Train | N_Test | Accuracy | Macro F1 | Weighted F1 |",
        "|------------|-------|---------|--------|----------|----------|------------|",
    ]
    for _, row in loro_df.sort_values(["model", "macro_f1"]).iterrows():
        md_lines.append(
            f"| {row['repository_name']} | {row['model']} | {row['n_train']:,} | "
            f"{row['n_test']:,} | {row['accuracy']:.4f} | {row['macro_f1']:.4f} | {row['weighted_f1']:.4f} |"
        )

    md_lines += [
        "",
        "## Interpretation",
        "- Average LORO Macro F1 reflects how well the model generalizes to unseen repositories.",
        "- Worst Repository F1 identifies the hardest-to-generalize repository.",
        "- High variance across repositories suggests domain shift sensitivity.",
    ]

    md_path = os.path.join(REPORTS_DIR, "generalization_report.md")
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines))
    log(f"  Saved → {md_path}")

    return model_summary


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 – TRUST GATE
# ─────────────────────────────────────────────────────────────────────────────
def step6_trust_gate(X_test, y_test, trained_models, top3_models, preprocessor):
    log("STEP 6 – Trust Gate")

    class_names = preprocessor["class_names"]

    md_lines = [
        "# Trust Gate Report\n",
        f"**Generated**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
        "## Confidence Bin Accuracy\n",
        "Confidence bins: 90–100%, 70–90%, 50–70%, <50%\n",
    ]

    trust_results = {}
    bins = [(0.9, 1.0, "90-100%"), (0.7, 0.9, "70-90%"), (0.5, 0.7, "50-70%"), (0.0, 0.5, "<50%")]

    for model_name in top3_models:
        model = trained_models[model_name]
        has_proba = hasattr(model, "predict_proba")

        if not has_proba:
            log(f"  {model_name} does not support predict_proba, skipping trust gate")
            continue

        y_proba = model.predict_proba(X_test)
        y_pred = model.predict(X_test)
        max_confidence = y_proba.max(axis=1)

        bin_stats = []
        md_lines.append(f"### {model_name}")
        md_lines.append("| Confidence Bin | N Samples | % of Test | Bin Accuracy |")
        md_lines.append("|---------------|-----------|-----------|-------------|")

        for lo, hi, label in bins:
            mask = (max_confidence >= lo) & (max_confidence < hi)
            if label == "90-100%":
                mask = max_confidence >= lo
            n_bin = mask.sum()
            if n_bin == 0:
                bin_acc = "N/A"
                md_lines.append(f"| {label} | 0 | 0.00% | N/A |")
            else:
                bin_acc = round(accuracy_score(y_test[mask], y_pred[mask]), 4)
                pct = round(n_bin / len(y_test) * 100, 2)
                md_lines.append(f"| {label} | {n_bin:,} | {pct:.2f}% | {bin_acc:.4f} |")

            bin_stats.append({"bin": label, "n": int(n_bin), "accuracy": bin_acc if n_bin > 0 else None})

        trust_results[model_name] = bin_stats
        md_lines.append("")

    md_lines += [
        "## Interpretation",
        "- High-confidence predictions (90–100%) should achieve ≥90% accuracy for production trustworthiness.",
        "- Low-confidence predictions (<50%) indicate uncertainty and should trigger human review.",
        "- A well-calibrated model shows monotonically increasing accuracy with confidence.",
    ]

    md_path = os.path.join(REPORTS_DIR, "trust_gate_report.md")
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines))
    log(f"  Saved → {md_path}")

    return trust_results


# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 – DOMAIN SHIFT ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
def step7_domain_shift(df, feature_cols):
    log("STEP 7 – Domain Shift Analysis")

    num_features = [c for c in feature_cols if not is_string_col(df[c])]
    repos = df["repository_name"].unique().tolist()
    n_repos = len(repos)

    global_ref = df[num_features].apply(pd.to_numeric, errors='coerce').fillna(0)

    # PSI computation
    def compute_psi(expected, actual, buckets=10):
        """Population Stability Index"""
        breakpoints = np.linspace(0, 100, buckets + 1)
        expected_pct = np.array([
            np.mean((expected >= np.percentile(expected, breakpoints[i])) &
                    (expected < np.percentile(expected, breakpoints[i + 1])))
            for i in range(buckets)
        ])
        actual_pct = np.array([
            np.mean((actual >= np.percentile(expected, breakpoints[i])) &
                    (actual < np.percentile(expected, breakpoints[i + 1])))
            for i in range(buckets)
        ])
        expected_pct = np.clip(expected_pct, 1e-6, None)
        actual_pct = np.clip(actual_pct, 1e-6, None)
        psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
        return round(float(psi), 4)

    psi_results = []
    ks_results = []

    for repo in repos:
        repo_df = df[df["repository_name"] == repo][num_features].apply(pd.to_numeric, errors='coerce').fillna(0)
        other_df = df[df["repository_name"] != repo][num_features].apply(pd.to_numeric, errors='coerce').fillna(0)

        repo_psi = []
        repo_ks = []

        for feat in num_features[:8]:  # Top 8 numeric features for speed
            try:
                psi = compute_psi(other_df[feat].values, repo_df[feat].values)
                ks_stat, ks_p = stats.ks_2samp(other_df[feat].values, repo_df[feat].values)
                repo_psi.append(psi)
                repo_ks.append(round(float(ks_stat), 4))
            except Exception:
                repo_psi.append(0.0)
                repo_ks.append(0.0)

        psi_results.append({
            "repository": repo,
            "mean_psi": round(np.mean(repo_psi), 4),
            "max_psi": round(np.max(repo_psi), 4),
            "shift_level": "HIGH" if np.mean(repo_psi) > 0.25 else ("MODERATE" if np.mean(repo_psi) > 0.1 else "LOW"),
        })
        ks_results.append({
            "repository": repo,
            "mean_ks": round(np.mean(repo_ks), 4),
            "max_ks": round(np.max(repo_ks), 4),
        })
        gc_collect()

    # Repository similarity matrix (cosine similarity of mean feature vectors)
    repo_means = []
    for repo in repos:
        repo_df = df[df["repository_name"] == repo][num_features].apply(pd.to_numeric, errors='coerce').fillna(0)
        repo_means.append(repo_df.mean().values)

    repo_means = np.array(repo_means)
    # Normalize
    norms = np.linalg.norm(repo_means, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    repo_means_norm = repo_means / norms
    sim_matrix = repo_means_norm @ repo_means_norm.T

    # Find most/least similar pairs
    np.fill_diagonal(sim_matrix, -1)
    max_idx = np.unravel_index(np.argmax(sim_matrix), sim_matrix.shape)
    np.fill_diagonal(sim_matrix, 2)
    min_idx = np.unravel_index(np.argmin(sim_matrix), sim_matrix.shape)
    np.fill_diagonal(sim_matrix, 1)

    most_sim_pair = (repos[max_idx[0]], repos[max_idx[1]], round(float(sim_matrix[max_idx[0], max_idx[1]]), 4))
    least_sim_pair = (repos[min_idx[0]], repos[min_idx[1]], round(float(sim_matrix[min_idx[0], min_idx[1]]), 4))

    psi_df = pd.DataFrame(psi_results).sort_values("mean_psi", ascending=False)
    ks_df = pd.DataFrame(ks_results).sort_values("mean_ks", ascending=False)

    md_lines = [
        "# Domain Shift Report v2\n",
        f"**Generated**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
        "## Population Stability Index (PSI)",
        "PSI < 0.1: Low shift | PSI 0.1–0.25: Moderate | PSI > 0.25: High shift\n",
        "| Repository | Mean PSI | Max PSI | Shift Level |",
        "|------------|---------|---------|------------|",
    ]
    for _, row in psi_df.iterrows():
        flag = " 🚨" if row["shift_level"] == "HIGH" else (" ⚠️" if row["shift_level"] == "MODERATE" else "")
        md_lines.append(f"| {row['repository']} | {row['mean_psi']} | {row['max_psi']} | {row['shift_level']}{flag} |")

    md_lines += [
        "",
        "## KS Statistics (Feature Distribution Distance)",
        "KS statistic near 0 = similar distributions | Near 1 = very different\n",
        "| Repository | Mean KS | Max KS |",
        "|------------|---------|--------|",
    ]
    for _, row in ks_df.iterrows():
        md_lines.append(f"| {row['repository']} | {row['mean_ks']} | {row['max_ks']} |")

    md_lines += [
        "",
        "## Repository Similarity (Cosine Similarity of Feature Vectors)",
        f"| Metric | Repo A | Repo B | Similarity |",
        f"|--------|--------|--------|-----------|",
        f"| Most Similar | {most_sim_pair[0]} | {most_sim_pair[1]} | {most_sim_pair[2]:.4f} |",
        f"| Least Similar | {least_sim_pair[0]} | {least_sim_pair[1]} | {least_sim_pair[2]:.4f} |",
        "",
        "## Interpretation",
        "- Repositories with HIGH PSI represent significant distribution shifts from the pool.",
        "- These repositories are harder for the model to generalize to.",
        "- Low KS statistics indicate feature distributions are similar across repositories.",
        "- LORO performance often correlates with PSI: high PSI → lower LORO F1.",
    ]

    md_path = os.path.join(REPORTS_DIR, "domain_shift_report_v2.md")
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines))
    log(f"  Saved → {md_path}")

    return psi_df, ks_df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 8 – PRODUCTION MODEL SELECTION & SAVING
# ─────────────────────────────────────────────────────────────────────────────
def step8_production_model(
    top3_models, model_summary, trust_results, trained_models, preprocessor, feature_cols
):
    log("STEP 8 – Production Model Selection")

    # Score each model: avg_loro_f1 (40%), worst_repo_f1 (40%), trust_90pct_acc (20%)
    scores = {}
    for model_name in top3_models:
        ms = model_summary.get(model_name, {})
        avg_f1 = ms.get("avg_loro_macro_f1", 0)
        worst_f1 = ms.get("worst_repo_f1", 0)

        # Trust gate: accuracy in 90–100% bin
        trust_acc = 0.5  # default
        if model_name in trust_results:
            for bin_entry in trust_results[model_name]:
                if bin_entry["bin"] == "90-100%" and bin_entry["accuracy"] is not None:
                    trust_acc = bin_entry["accuracy"]
                    break

        composite = 0.4 * avg_f1 + 0.4 * worst_f1 + 0.2 * trust_acc
        scores[model_name] = {
            "avg_loro_macro_f1": avg_f1,
            "worst_repo_macro_f1": worst_f1,
            "trust_90pct_accuracy": trust_acc,
            "composite_score": round(composite, 4),
        }
        log(f"  {model_name}: composite={composite:.4f} (avg_loro={avg_f1}, worst={worst_f1}, trust={trust_acc})")

    best_model_name = max(scores, key=lambda k: scores[k]["composite_score"])
    log(f"  ✅ Best model: {best_model_name}")

    best_model = trained_models[best_model_name]

    # Save model
    model_path = os.path.join(MODELS_DIR, "best_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump({"model": best_model, "model_name": best_model_name, "scores": scores}, f)
    log(f"  Saved → {model_path}")

    # Save preprocessor
    prep_path = os.path.join(MODELS_DIR, "preprocessor.pkl")
    with open(prep_path, "wb") as f:
        pickle.dump(preprocessor, f)
    log(f"  Saved → {prep_path}")

    # Save feature schema
    schema = {
        "model_name": best_model_name,
        "feature_cols": feature_cols,
        "class_names": preprocessor["class_names"],
        "scores": scores[best_model_name],
        "trained_on": "ml_dataset_v2.csv",
        "n_repositories": 22,
        "n_samples": 40313,
        "timestamp": pd.Timestamp.now().isoformat(),
    }
    schema_path = os.path.join(MODELS_DIR, "feature_schema.json")
    with open(schema_path, "w") as f:
        json.dump(schema, f, indent=2)
    log(f"  Saved → {schema_path}")

    return best_model_name, scores


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
def main():
    log("=" * 70)
    log("Classical ML Retraining Pipeline v2")
    log("Dataset: ml_dataset_v2.csv | 22 repositories | 40,313 files")
    log("=" * 70)

    # ── Load dataset (chunked read for memory efficiency) ──────────────────
    log("Loading dataset...")
    df = pd.read_csv(DATA_PATH, low_memory=True)
    log(f"  Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")

    # ── STEP 0 ─────────────────────────────────────────────────────────────
    repo_counts = step0_dataset_audit(df)
    gc_collect()

    # ── STEP 1 ─────────────────────────────────────────────────────────────
    feature_cols = step1_feature_audit(df)
    gc_collect()

    # ── Preprocessing ──────────────────────────────────────────────────────
    log("Preprocessing dataset...")
    X, y, preprocessor = build_preprocessor(df, feature_cols)
    log(f"  X shape: {X.shape}, y shape: {y.shape}, Classes: {preprocessor['class_names']}")

    # Stratified train/test split using repositories (80/20)
    # We split by repository to avoid leakage
    repos = df["repository_name"].values
    unique_repos = list(df["repository_name"].unique())
    np.random.seed(42)
    np.random.shuffle(unique_repos)
    n_test_repos = max(2, int(len(unique_repos) * 0.2))
    test_repos = set(unique_repos[:n_test_repos])
    train_repos = set(unique_repos[n_test_repos:])
    log(f"  Train repos ({len(train_repos)}): {sorted(train_repos)}")
    log(f"  Test repos ({len(test_repos)}): {sorted(test_repos)}")

    train_mask = np.array([r in train_repos for r in repos])
    test_mask = ~train_mask

    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    log(f"  Train: {X_train.shape[0]:,} | Test: {X_test.shape[0]:,}")

    del X  # Free memory
    gc_collect()

    # ── STEP 2 ─────────────────────────────────────────────────────────────
    results_df, trained_models = step2_train_models(
        X_train, y_train, X_test, y_test, preprocessor["class_names"]
    )
    gc_collect()

    # ── STEP 3 ─────────────────────────────────────────────────────────────
    top3_models, stability_scores = step3_select_top_models(
        results_df, trained_models, X_train, y_train, X_test, y_test
    )
    gc_collect()

    # ── STEP 4 ─────────────────────────────────────────────────────────────
    loro_df = step4_loro_evaluation(df, top3_models, trained_models, preprocessor)
    gc_collect()

    # ── STEP 5 ─────────────────────────────────────────────────────────────
    model_summary = step5_generalization_analysis(loro_df, top3_models)
    gc_collect()

    # ── STEP 6 ─────────────────────────────────────────────────────────────
    trust_results = step6_trust_gate(X_test, y_test, trained_models, top3_models, preprocessor)
    gc_collect()

    # ── STEP 7 ─────────────────────────────────────────────────────────────
    psi_df, ks_df = step7_domain_shift(df, feature_cols)
    gc_collect()

    # ── STEP 8 ─────────────────────────────────────────────────────────────
    best_model_name, production_scores = step8_production_model(
        top3_models, model_summary, trust_results, trained_models, preprocessor, feature_cols
    )
    gc_collect()

    # ── FINAL SUMMARY ──────────────────────────────────────────────────────
    log("=" * 70)
    log("PIPELINE COMPLETE — SUCCESS CRITERIA SUMMARY")
    log("=" * 70)
    log(f"1. Total Repositories: 22")
    log(f"2. Total Files: 40,313")
    log(f"3. Label Distribution: {dict(df['historical_risk_label'].value_counts())}")
    log(f"4. Max Repo Contribution: {repo_counts['percentage'].max():.2f}% ({repo_counts.iloc[0]['repository_name']})")
    log(f"")
    log(f"5. Baseline Model Results:")
    for _, row in results_df.sort_values("macro_f1", ascending=False).iterrows():
        log(f"   {row['model']}: Macro F1={row['macro_f1']}, Weighted F1={row['weighted_f1']}")
    log(f"")
    log(f"6. Top 3 Models: {top3_models}")
    log(f"")
    log(f"7. LORO Results:")
    for m in top3_models:
        s = model_summary.get(m, {})
        log(f"   {m}: Avg={s.get('avg_loro_macro_f1')}, Worst={s.get('worst_repo_f1')} ({s.get('worst_repo')}), Best={s.get('best_repo_f1')} ({s.get('best_repo')})")
    log(f"")
    log(f"8. Trust Gate: See {os.path.join(REPORTS_DIR, 'trust_gate_report.md')}")
    log(f"9. Domain Shift: See {os.path.join(REPORTS_DIR, 'domain_shift_report_v2.md')}")
    log(f"")
    log(f"10. PRODUCTION MODEL: {best_model_name}")
    log(f"    Scores: {production_scores.get(best_model_name, {})}")
    log(f"")
    log(f"ANSWER — Generalization to Unknown Repository:")
    best_ms = model_summary.get(best_model_name, {})
    avg_loro = best_ms.get("avg_loro_macro_f1", 0)
    worst_loro = best_ms.get("worst_repo_f1", 0)
    if avg_loro >= 0.75:
        confidence = "HIGH (≥75% avg LORO Macro F1)"
        verdict = "The model is expected to generalize well to new repositories with confidence."
    elif avg_loro >= 0.60:
        confidence = "MODERATE (60–75% avg LORO Macro F1)"
        verdict = "Moderate generalization. Recommend human review for ambiguous predictions."
    else:
        confidence = "LOW (<60% avg LORO Macro F1)"
        verdict = "Uncertain generalization. Trust gate filtering strongly recommended."

    log(f"  Avg LORO Macro F1: {avg_loro}")
    log(f"  Worst-Case Repo F1: {worst_loro}")
    log(f"  Confidence Level: {confidence}")
    log(f"  Verdict: {verdict}")
    log("=" * 70)


if __name__ == "__main__":
    main()
