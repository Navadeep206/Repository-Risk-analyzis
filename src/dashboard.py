#!/usr/bin/env python3
"""
Repository Risk Intelligence Platform — Premium Dashboard
=========================================================
Single-page Streamlit app. User pastes a GitHub URL, clicks Analyze,
and gets a full production-grade risk intelligence report.
"""

import os, sys, time, shutil, traceback, io, json, warnings, pickle, re
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")

# ── path bootstrap ────────────────────────────────────────────────────────────
BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC  = BASE
sys.path.insert(0, SRC)

from config import RAW_DIR, PROCESSED_DIR, REPOS_DIR, ensure_dirs_exist

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Repository Risk Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ══════════════════════════════════════════════════════════════════════════════
# PREMIUM CSS — Dark enterprise theme (GitHub Security × Vercel × Linear)
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body, [class*="css"], .stApp {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    background: #0a0b0e !important;
    color: #e2e8f0 !important;
}

/* ── top nav bar ── */
.nav-bar {
    display: flex; align-items: center; justify-content: space-between;
    padding: 0 0 1.5rem 0; border-bottom: 1px solid #1e293b;
    margin-bottom: 2rem;
}
.nav-logo {
    display: flex; align-items: center; gap: 0.6rem;
    font-size: 1.15rem; font-weight: 700; color: #f8fafc;
}
.nav-badge {
    background: linear-gradient(135deg,#6366f1,#8b5cf6);
    color: #fff; font-size: 0.65rem; font-weight: 700;
    padding: 2px 8px; border-radius: 999px; letter-spacing: 0.08em;
}

/* ── hero section ── */
.hero { text-align: center; padding: 3rem 0 2.5rem; }
.hero-title {
    font-size: clamp(2.2rem, 5vw, 3.5rem);
    font-weight: 900; line-height: 1.1;
    background: linear-gradient(135deg, #f8fafc 0%, #94a3b8 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 0.75rem;
}
.hero-sub {
    font-size: 1.05rem; color: #64748b; max-width: 560px;
    margin: 0 auto 2rem; line-height: 1.6;
}

/* ── URL input card ── */
.url-card {
    background: #111318; border: 1px solid #1e293b;
    border-radius: 16px; padding: 2rem;
    max-width: 760px; margin: 0 auto;
    box-shadow: 0 0 60px rgba(99,102,241,0.08);
}

/* ── risk banner ── */
.risk-banner {
    border-radius: 16px; padding: 2rem 2.5rem;
    display: flex; align-items: center; justify-content: space-between;
    margin-bottom: 1.5rem;
    position: relative; overflow: hidden;
}
.risk-banner::before {
    content: ''; position: absolute; inset: 0;
    background: inherit; filter: blur(0px);
}
.risk-banner-low    { background: linear-gradient(135deg,#052e16 0%,#14532d 100%); border: 1px solid #16a34a; }
.risk-banner-medium { background: linear-gradient(135deg,#1c1917 0%,#451a03 100%); border: 1px solid #d97706; }
.risk-banner-high   { background: linear-gradient(135deg,#1c0a0a 0%,#450a0a 100%); border: 1px solid #dc2626; }
.risk-banner-critical{ background: linear-gradient(135deg,#1c0a0a 0%,#7f1d1d 100%); border: 1px solid #ef4444; }

.risk-label { font-size: 2rem; font-weight: 800; display: flex; align-items: center; gap: 0.75rem; }
.health-ring { text-align: center; }
.health-score { font-size: 3rem; font-weight: 900; line-height: 1; }
.health-sub { font-size: 0.75rem; color: rgba(255,255,255,0.6); letter-spacing: 0.1em; text-transform: uppercase; }

/* ── metric cards ── */
.metric-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(160px, 1fr)); gap: 1rem; margin-bottom: 1.5rem; }
.metric-card {
    background: #111318; border: 1px solid #1e293b;
    border-radius: 12px; padding: 1.2rem 1rem;
    transition: border-color 0.2s, transform 0.2s;
}
.metric-card:hover { border-color: #6366f1; transform: translateY(-2px); }
.metric-icon { font-size: 1.4rem; margin-bottom: 0.4rem; }
.metric-val { font-size: 1.65rem; font-weight: 800; color: #f8fafc; line-height: 1.1; }
.metric-lbl { font-size: 0.72rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.08em; margin-top: 0.25rem; }

/* ── section header ── */
.section-hdr {
    font-size: 0.7rem; font-weight: 700; color: #6366f1;
    text-transform: uppercase; letter-spacing: 0.15em;
    margin-bottom: 0.75rem; display: flex; align-items: center; gap: 0.4rem;
}

/* ── risk badge ── */
.badge {
    display: inline-block; padding: 2px 10px; border-radius: 999px;
    font-size: 0.72rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em;
}
.badge-critical { background: rgba(239,68,68,0.15); color: #f87171; border: 1px solid rgba(239,68,68,0.3); }
.badge-high     { background: rgba(249,115,22,0.15); color: #fb923c; border: 1px solid rgba(249,115,22,0.3); }
.badge-medium   { background: rgba(234,179,8,0.15);  color: #facc15; border: 1px solid rgba(234,179,8,0.3);  }
.badge-low      { background: rgba(34,197,94,0.15);  color: #4ade80; border: 1px solid rgba(34,197,94,0.3);  }

/* ── panel card ── */
.panel {
    background: #111318; border: 1px solid #1e293b;
    border-radius: 16px; padding: 1.5rem;
    margin-bottom: 1rem; height: 100%;
}
.panel-title { font-size: 1rem; font-weight: 700; color: #f8fafc; margin-bottom: 1rem; }

/* ── critical file row ── */
.crit-row {
    display: flex; align-items: flex-start; gap: 0.75rem;
    padding: 0.9rem; border-radius: 10px;
    background: rgba(239,68,68,0.06); border: 1px solid rgba(239,68,68,0.15);
    margin-bottom: 0.6rem;
}
.crit-score {
    background: #ef4444; color: #fff;
    width: 44px; height: 44px; border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.9rem; font-weight: 800; flex-shrink: 0;
}
.crit-path { font-size: 0.82rem; font-weight: 600; color: #f8fafc; font-family: 'Courier New', monospace; }
.crit-reasons { font-size: 0.72rem; color: #94a3b8; margin-top: 0.2rem; }

/* ── heatmap tree ── */
.tree-file {
    display: flex; align-items: center; gap: 0.5rem;
    padding: 0.35rem 0.6rem; border-radius: 6px;
    font-size: 0.8rem; font-family: 'Courier New', monospace;
    transition: background 0.15s;
}
.tree-file:hover { background: rgba(255,255,255,0.04); }
.tree-folder { font-size: 0.75rem; font-weight: 700; color: #6366f1; padding: 0.5rem 0 0.2rem; letter-spacing: 0.04em; }
.risk-dot { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }
.dot-critical { background: #ef4444; box-shadow: 0 0 6px rgba(239,68,68,0.7); }
.dot-high     { background: #f97316; box-shadow: 0 0 6px rgba(249,115,22,0.7); }
.dot-medium   { background: #eab308; box-shadow: 0 0 6px rgba(234,179,8,0.5);  }
.dot-low      { background: #22c55e; box-shadow: 0 0 6px rgba(34,197,94,0.5);  }

/* ── trust gate ── */
.trust-chip {
    display: inline-flex; align-items: center; gap: 0.35rem;
    padding: 3px 10px; border-radius: 999px;
    font-size: 0.72rem; font-weight: 600;
}
.trust-high { background: rgba(34,197,94,0.12); color: #4ade80; border: 1px solid rgba(34,197,94,0.25); }
.trust-mod  { background: rgba(234,179,8,0.12);  color: #facc15; border: 1px solid rgba(234,179,8,0.25);  }
.trust-low  { background: rgba(239,68,68,0.12); color: #f87171; border: 1px solid rgba(239,68,68,0.25); }

/* ── streamlit overrides ── */
.stButton > button {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: white !important; border: none !important;
    border-radius: 10px !important; font-weight: 700 !important;
    font-size: 1rem !important; padding: 0.75rem 2rem !important;
    width: 100% !important; letter-spacing: 0.02em !important;
    transition: opacity 0.2s, transform 0.2s !important;
    box-shadow: 0 4px 24px rgba(99,102,241,0.35) !important;
}
.stButton > button:hover { opacity: 0.9 !important; transform: translateY(-1px) !important; }

.stTextInput > div > div > input {
    background: #0d1117 !important; border: 1.5px solid #1e293b !important;
    color: #f8fafc !important; border-radius: 10px !important;
    font-size: 0.95rem !important; padding: 0.7rem 1rem !important;
}
.stTextInput > div > div > input:focus { border-color: #6366f1 !important; }

.stDataFrame { background: #111318 !important; border-radius: 12px !important; }
div[data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; }

.stProgress > div > div > div > div { background: linear-gradient(90deg,#6366f1,#8b5cf6) !important; }

section[data-testid="stSidebar"] { background: #0d1117 !important; border-right: 1px solid #1e293b; }

.stAlert { border-radius: 10px !important; }
.stSelectbox > div > div { background: #111318 !important; border-color: #1e293b !important; }
.stMultiSelect > div { background: #111318 !important; }

hr { border-color: #1e293b !important; }

/* step log */
.step-log {
    background: #0d1117; border: 1px solid #1e293b; border-radius: 10px;
    padding: 1rem; font-family: 'Courier New', monospace;
    font-size: 0.8rem; max-height: 280px; overflow-y: auto;
}
.log-ok   { color: #4ade80; }
.log-run  { color: #60a5fa; }
.log-err  { color: #f87171; }
.log-info { color: #94a3b8; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════
INV_LABEL = {0: "LOW", 1: "MEDIUM", 2: "HIGH"}
LABEL_MAP  = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
NUMERIC_FEATURES = [
    "loc", "complexity", "maintainability_index", "commit_count",
    "modification_count", "contributor_count", "commit_frequency", "repository_age_days"
]


def risk_score(row: pd.Series) -> float:
    """
    Compute a 0-100 risk score from model probabilities + complexity signals.
    """
    prob_high = float(row.get("prob_HIGH", row.get("prob_MEDIUM", 0.5)))
    prob_med  = float(row.get("prob_MEDIUM", 0.3))
    base      = prob_high * 70 + prob_med * 30

    # Normalize complexity contribution (0-20 pts)
    cmplx = float(row.get("complexity", 0))
    cmplx_pts = min(20, cmplx * 0.8)

    # Low maintainability adds up to 10 pts
    mi = float(row.get("maintainability_index", 100))
    mi_pts = max(0, (100 - mi) * 0.1)

    score = min(100, base + cmplx_pts + mi_pts)
    return round(score, 1)


def risk_level(score: float) -> str:
    if score >= 90: return "Critical"
    if score >= 75: return "High"
    if score >= 50: return "Medium"
    return "Low"


def risk_emoji(level: str) -> str:
    return {"Critical": "🔴", "High": "🟠", "Medium": "🟡", "Low": "🟢"}.get(level, "⚪")


def badge_html(level: str) -> str:
    cls = {"Critical":"badge-critical","High":"badge-high","Medium":"badge-medium","Low":"badge-low"}.get(level,"badge-low")
    return f'<span class="badge {cls}">{risk_emoji(level)} {level}</span>'


def trust_html(conf: float) -> str:
    pct = conf * 100 if conf <= 1 else conf
    if pct >= 90:
        return f'<span class="trust-chip trust-high">🟢 High ({pct:.0f}%)</span>'
    elif pct >= 70:
        return f'<span class="trust-chip trust-mod">🟡 Moderate ({pct:.0f}%)</span>'
    else:
        return f'<span class="trust-chip trust-low">🔴 Review ({pct:.0f}%)</span>'


def health_score(df: pd.DataFrame) -> int:
    """0-100 repo health score (inverse of average risk)."""
    if df.empty or "risk_score_val" not in df.columns:
        return 50
    avg = df["risk_score_val"].mean()
    return max(0, min(100, int(100 - avg)))


def risk_drivers(row: pd.Series) -> list:
    drivers = []
    if float(row.get("complexity", 0))         > 10:  drivers.append("High Cyclomatic Complexity")
    if float(row.get("maintainability_index",100)) < 50: drivers.append("Low Maintainability Index")
    if float(row.get("modification_count", 0)) > 30:  drivers.append("High Modification Frequency")
    if float(row.get("commit_count", 0))        > 50:  drivers.append("High Commit Churn")
    if float(row.get("contributor_count", 0))   > 10:  drivers.append("High Contributor Turnover")
    if float(row.get("loc", 0))                 > 500: drivers.append("Large File Size")
    if not drivers:
        drivers.append("Multiple marginal risk signals")
    return drivers[:3]


def parse_github_url(url: str):
    """Returns (owner, repo) from a GitHub URL or None."""
    url = url.strip().rstrip("/")
    m = re.match(r"https?://github\.com/([^/]+)/([^/\s]+)", url)
    if m:
        return m.group(1), m.group(2)
    return None, None


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE RUNNER
# ══════════════════════════════════════════════════════════════════════════════
def run_pipeline(owner: str, repo_name: str, log_box) -> Optional[pd.DataFrame]:
    """
    Runs the complete repository risk intelligence pipeline and returns
    a dataframe of file-level predictions, or None on fatal error.
    """
    import importlib, shutil

    REPO_LOCAL = os.path.join(REPOS_DIR, repo_name)
    ensure_dirs_exist()
    os.makedirs(REPOS_DIR, exist_ok=True)

    logs = []
    def log(msg: str, kind: str = "run"):
        css = {"ok":"log-ok","run":"log-run","err":"log-err","info":"log-info"}.get(kind,"log-info")
        ts  = datetime.now().strftime("%H:%M:%S")
        logs.append(f'<span class="{css}">[{ts}] {msg}</span>')
        log_box.markdown(
            f'<div class="step-log">{"<br>".join(logs[-18:])}</div>',
            unsafe_allow_html=True
        )

    # ── STEP 0: Clone ─────────────────────────────────────────────────────
    log(f"🔗 Cloning {owner}/{repo_name}…")
    try:
        if not os.path.isdir(os.path.join(REPO_LOCAL, ".git")):
            from git import Repo
            Repo.clone_from(f"https://github.com/{owner}/{repo_name}", REPO_LOCAL)
        log(f"✅ Repository cloned/found at {REPO_LOCAL}", "ok")
    except Exception as e:
        log(f"❌ Clone failed: {e}", "err"); return None

    # ── STEP 1: Commits ───────────────────────────────────────────────────
    log("📝 Extracting commit history…")
    commits_csv = os.path.join(RAW_DIR, f"{repo_name}_commits.csv")
    try:
        if not os.path.exists(commits_csv):
            from commit_extractor import extract_commits
            extract_commits(REPO_LOCAL, commits_csv)
        df_commits = pd.read_csv(commits_csv)
        log(f"✅ {len(df_commits):,} commits extracted", "ok")
    except Exception as e:
        log(f"⚠️  Commit extraction error: {e}", "info")
        df_commits = pd.DataFrame()

    # ── STEP 2: Modifications ─────────────────────────────────────────────
    log("🔍 Mining file-level modifications…")
    mods_csv = os.path.join(RAW_DIR, f"{repo_name}_modifications.csv")
    try:
        if not os.path.exists(mods_csv):
            from modification_extractor import extract_modifications
            extract_modifications(REPO_LOCAL, mods_csv)
        df_mods = pd.read_csv(mods_csv)
        log(f"✅ {len(df_mods):,} file modifications mined", "ok")
    except Exception as e:
        log(f"⚠️  Modification extraction error: {e}", "info")
        df_mods = pd.DataFrame()

    # ── STEP 3: Quality Metrics ───────────────────────────────────────────
    log("🔬 Running quality metrics pipeline…")
    py_bak  = os.path.join(RAW_DIR, "_bak_py.csv")
    js_bak  = os.path.join(RAW_DIR, "_bak_js.csv")
    ts_bak  = os.path.join(RAW_DIR, "_bak_ts.csv")
    for src, bak in [
        (os.path.join(RAW_DIR, "python_metrics.csv"), py_bak),
        (os.path.join(RAW_DIR, "javascript_metrics.csv"), js_bak),
        (os.path.join(RAW_DIR, "typescript_metrics.csv"), ts_bak),
    ]:
        if os.path.exists(src): shutil.copy2(src, bak)
    try:
        from quality_metrics.quality_pipeline import run_quality_pipeline
        quality_out = os.path.join(PROCESSED_DIR, f"{repo_name}_quality_metrics.csv")
        df_quality = run_quality_pipeline(REPO_LOCAL, quality_out)
        df_quality.to_csv(os.path.join(PROCESSED_DIR, "quality_metrics.csv"), index=False)
        log(f"✅ {len(df_quality)} files analyzed. Languages: {df_quality['language'].unique().tolist()}", "ok")
    except Exception as e:
        log(f"⚠️  Quality pipeline error: {e}", "info")
        df_quality = pd.DataFrame()
    finally:
        for bak, dst in [
            (py_bak, os.path.join(RAW_DIR, "python_metrics.csv")),
            (js_bak, os.path.join(RAW_DIR, "javascript_metrics.csv")),
            (ts_bak, os.path.join(RAW_DIR, "typescript_metrics.csv")),
        ]:
            if os.path.exists(bak): shutil.copy2(bak, dst); os.remove(bak)

    if df_quality.empty:
        log("❌ No source files found or quality pipeline failed", "err"); return None

    # ── STEP 4: Merge & Feature Engineering ──────────────────────────────
    log("⚙️  Building features…")
    try:
        df_q = df_quality.copy()
        if "repository_name" not in df_q.columns or not (df_q["repository_name"] == repo_name).any():
            df_q["repository_name"] = repo_name

        # Bug-fix detection
        bug_hashes: set = set()
        if not df_commits.empty and "message" in df_commits.columns:
            df_commits["message"] = df_commits["message"].fillna("").astype(str)
            kw = ["fix","bug","hotfix","regression","patch","issue"]
            df_commits["is_bug_fix"] = df_commits["message"].str.contains("|".join(kw), case=False)
            bug_hashes = set(df_commits[df_commits["is_bug_fix"]]["commit_hash"])

        if not df_mods.empty:
            df_mods["file_path"] = df_mods["new_path"].fillna(df_mods["old_path"])
            df_mods["is_bug_fix"] = df_mods["commit_hash"].isin(bug_hashes)
            agg = df_mods.groupby("file_path").agg(
                modification_count=("commit_hash","count"),
                commit_count=("commit_hash","nunique"),
                contributor_count=("author_email","nunique"),
                bug_fix_commit_count=("commit_hash", lambda x: int(x[df_mods.loc[x.index,"is_bug_fix"]].nunique()))
            ).reset_index()
            df_q = pd.merge(df_q, agg, on="file_path", how="left")
            for c in ["modification_count","commit_count","contributor_count","bug_fix_commit_count"]:
                df_q[c] = df_q[c].fillna(0).astype(int)
        else:
            for c in ["modification_count","commit_count","contributor_count","bug_fix_commit_count"]:
                df_q[c] = 0

        # Repo age
        repo_age = 1
        if not df_commits.empty and "committer_date" in df_commits.columns:
            dates = pd.to_datetime(df_commits["committer_date"], errors="coerce", utc=True)
            diff = (dates.max() - dates.min()).days
            repo_age = max(1, diff)

        df_q["repository_age_days"] = repo_age
        df_q["commit_frequency"]   = df_q["commit_count"] / repo_age
        for col in NUMERIC_FEATURES:
            if col not in df_q.columns: df_q[col] = 0.0
            else: df_q[col] = pd.to_numeric(df_q[col], errors="coerce").fillna(0.0)

        # Labels from bug-fix counts
        def lbl(n): return "LOW" if n==0 else ("MEDIUM" if n<=2 else "HIGH")
        df_q["historical_risk_label"] = df_q.get("bug_fix_commit_count", pd.Series(0, index=df_q.index)).apply(lbl)

        log(f"✅ Features engineered. Repo age: {repo_age} days, {len(bug_hashes)} bug-fix commits", "ok")
    except Exception as e:
        log(f"❌ Feature engineering failed: {e}", "err")
        traceback.print_exc(); return None

    # ── STEP 5: RF Prediction ─────────────────────────────────────────────
    log("🤖 Running Random Forest risk prediction…")
    try:
        from ml.preprocessing import CodeRiskPreprocessor
        preproc_path = os.path.join(BASE, "models", "preprocessor.pkl")
        rf_path      = os.path.join(BASE, "models", "random_forest.pkl")
        preprocessor = CodeRiskPreprocessor.load(preproc_path)
        with open(rf_path, "rb") as f: rf_model = pickle.load(f)

        if "language" not in df_q.columns: df_q["language"] = "python"
        df_q["language"] = df_q["language"].fillna("python").astype(str)

        X = preprocessor.transform(df_q)
        preds = rf_model.predict(X)
        probs = rf_model.predict_proba(X)

        df_q["predicted_label"] = [INV_LABEL.get(p,"LOW") for p in preds]
        df_q["confidence"]      = np.max(probs, axis=1)
        df_q["prob_LOW"]        = probs[:,0]
        df_q["prob_MEDIUM"]     = probs[:,1]
        df_q["prob_HIGH"]       = probs[:,2]
        df_q["risk_score_val"]  = df_q.apply(risk_score, axis=1)
        df_q["risk_level_name"] = df_q["risk_score_val"].apply(risk_level)

        dist = df_q["predicted_label"].value_counts().to_dict()
        avg_conf = float(df_q["confidence"].mean())
        log(f"✅ Predicted {len(df_q)} files — {dist}. Avg confidence: {avg_conf:.1%}", "ok")
    except Exception as e:
        log(f"❌ Prediction failed: {e}", "err")
        traceback.print_exc(); return None

    # ── STEP 6: Done ─────────────────────────────────────────────────────
    log("📊 Generating report…", "info")
    time.sleep(0.3)
    log("🏆 Analysis complete!", "ok")

    return df_q


# ══════════════════════════════════════════════════════════════════════════════
# CHARTS (Plotly)
# ══════════════════════════════════════════════════════════════════════════════
def render_charts(df: pd.DataFrame):
    try:
        import plotly.graph_objects as go
        import plotly.express as px

        PLOTLY_LAYOUT = dict(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter", color="#94a3b8", size=12),
            margin=dict(l=0, r=0, t=30, b=0),
        )
        GRID = dict(gridcolor="#1e293b", zerolinecolor="#1e293b")

        col1, col2, col3 = st.columns(3)

        # ── Pie ─────────────────────────────────────────────────────────
        with col1:
            cnt = df["risk_level_name"].value_counts()
            colors = {"Critical":"#ef4444","High":"#f97316","Medium":"#eab308","Low":"#22c55e"}
            fig = go.Figure(go.Pie(
                labels=cnt.index, values=cnt.values,
                hole=0.6,
                marker=dict(colors=[colors.get(l,"#6366f1") for l in cnt.index],
                            line=dict(color="#0a0b0e", width=2)),
                textfont=dict(color="#f8fafc"),
            ))
            fig.update_layout(**PLOTLY_LAYOUT, title="Risk Distribution",
                              showlegend=True,
                              legend=dict(font=dict(color="#94a3b8")))
            st.plotly_chart(fig, use_container_width=True)

        # ── Histogram ───────────────────────────────────────────────────
        with col2:
            fig2 = px.histogram(
                df, x="risk_score_val", nbins=20,
                title="Risk Score Histogram",
                color_discrete_sequence=["#6366f1"]
            )
            fig2.update_layout(**PLOTLY_LAYOUT, xaxis=dict(title="Risk Score", **GRID),
                               yaxis=dict(title="Files", **GRID))
            st.plotly_chart(fig2, use_container_width=True)

        # ── Language risk breakdown ──────────────────────────────────────
        with col3:
            if "language" in df.columns:
                lang_risk = df.groupby("language")["risk_score_val"].mean().reset_index()
                lang_risk.columns = ["Language", "Avg Risk Score"]
                fig3 = px.bar(lang_risk, x="Language", y="Avg Risk Score",
                              title="Avg Risk by Language",
                              color="Avg Risk Score",
                              color_continuous_scale=["#22c55e","#eab308","#ef4444"])
                fig3.update_layout(**PLOTLY_LAYOUT,
                                   xaxis=dict(**GRID), yaxis=dict(**GRID))
                st.plotly_chart(fig3, use_container_width=True)

        # ── Scatter: Complexity vs Risk Score ────────────────────────────
        st.markdown('<div class="section-hdr">📈 Complexity vs. Risk Score</div>', unsafe_allow_html=True)
        fig4 = px.scatter(
            df, x="complexity", y="risk_score_val",
            color="risk_level_name",
            size="loc", size_max=22,
            hover_data=["file_path","predicted_label","confidence"] if "file_path" in df.columns else [],
            title="",
            color_discrete_map={"Critical":"#ef4444","High":"#f97316","Medium":"#eab308","Low":"#22c55e"},
            opacity=0.8,
        )
        fig4.update_layout(**PLOTLY_LAYOUT,
                           xaxis=dict(title="Cyclomatic Complexity", **GRID),
                           yaxis=dict(title="Risk Score", **GRID))
        st.plotly_chart(fig4, use_container_width=True)

    except ImportError:
        st.info("Install plotly for interactive charts: `pip install plotly`")


# ══════════════════════════════════════════════════════════════════════════════
# HEATMAP TREE
# ══════════════════════════════════════════════════════════════════════════════
def render_heatmap_tree(df: pd.DataFrame):
    if "file_path" not in df.columns:
        st.info("No file paths available for heatmap."); return

    dot_cls = {"Critical":"dot-critical","High":"dot-high","Medium":"dot-medium","Low":"dot-low"}
    emoji_m = {"Critical":"🔴","High":"🟠","Medium":"🟡","Low":"🟢"}

    df_sorted = df.sort_values("risk_score_val", ascending=False)
    folders: dict = {}
    for _, row in df_sorted.iterrows():
        fp    = str(row.get("file_path",""))
        parts = fp.replace("\\","/").split("/")
        folder = "/".join(parts[:-1]) if len(parts) > 1 else "."
        fname  = parts[-1]
        folders.setdefault(folder, []).append((fname, row))

    html_parts = []
    for folder, files in sorted(folders.items()):
        html_parts.append(f'<div class="tree-folder">📁 {folder}/</div>')
        for fname, row in files[:8]:  # max 8 per folder
            lvl  = row.get("risk_level_name","Low")
            dc   = dot_cls.get(lvl,"dot-low")
            sc   = row.get("risk_score_val", 0)
            conf = float(row.get("confidence",0))
            pct  = conf*100 if conf <= 1 else conf
            html_parts.append(
                f'<div class="tree-file">'
                f'<div class="risk-dot {dc}"></div>'
                f'<span style="color:#cbd5e1;">{fname}</span>'
                f'<span style="color:#475569;margin-left:auto;font-size:0.7rem;">'
                f'{emoji_m.get(lvl,"")} {sc:.0f} &nbsp; {pct:.0f}%</span>'
                f'</div>'
            )

    st.markdown(f'<div class="panel" style="max-height:500px;overflow-y:auto;">{"".join(html_parts)}</div>',
                unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN RENDER
# ══════════════════════════════════════════════════════════════════════════════
def render_dashboard(df: pd.DataFrame, owner: str, repo_name: str, url: str):
    import plotly.graph_objects as go

    hs   = health_score(df)
    n    = len(df)
    n_crit  = int((df["risk_level_name"]=="Critical").sum())
    n_high  = int((df["risk_level_name"]=="High").sum())
    n_med   = int((df["risk_level_name"]=="Medium").sum())
    n_low   = int((df["risk_level_name"]=="Low").sum())
    avg_conf= float(df["confidence"].mean()) * 100
    avg_cmx = float(df["complexity"].mean()) if "complexity" in df.columns else 0
    avg_mi  = float(df["maintainability_index"].mean()) if "maintainability_index" in df.columns else 0
    contribs= int(df["contributor_count"].max()) if "contributor_count" in df.columns else 0
    avg_freq= float(df["commit_frequency"].mean()) if "commit_frequency" in df.columns else 0
    repo_age= int(df["repository_age_days"].iloc[0]) if "repository_age_days" in df.columns else 0
    commits = int(df["commit_count"].sum()) if "commit_count" in df.columns else 0
    lang    = df["language"].mode()[0] if "language" in df.columns and not df["language"].empty else "python"

    dom_level = df["risk_level_name"].mode()[0] if not df.empty else "Medium"
    banner_cls = {
        "Critical":"risk-banner-high",
        "High":"risk-banner-high",
        "Medium":"risk-banner-medium",
        "Low":"risk-banner-low"
    }.get(dom_level, "risk-banner-medium")

    ts = datetime.now().strftime("%Y-%m-%d %H:%M UTC")

    # ── TOP HEADER ────────────────────────────────────────────────────────
    st.markdown(f"""
    <div style="background:#111318;border:1px solid #1e293b;border-radius:16px;padding:1.5rem 2rem;margin-bottom:1.5rem;">
      <div style="display:flex;align-items:center;gap:1rem;flex-wrap:wrap;">
        <div>
          <div style="font-size:1.5rem;font-weight:800;color:#f8fafc;">
            {owner} / {repo_name}
            <a href="{url}" target="_blank" style="font-size:0.85rem;color:#6366f1;margin-left:0.75rem;text-decoration:none;">↗ GitHub</a>
          </div>
          <div style="font-size:0.8rem;color:#64748b;margin-top:0.3rem;">
            🌐 {lang.title()} &nbsp;|&nbsp; 📅 {repo_age} days old &nbsp;|&nbsp; 
            👥 {contribs} contributors &nbsp;|&nbsp; 💾 {commits:,} commits &nbsp;|&nbsp;
            🕐 Analyzed {ts}
          </div>
        </div>
        <div style="margin-left:auto;display:flex;gap:0.75rem;align-items:center;">
          {badge_html(dom_level)}
          <span style="background:#1e293b;color:#94a3b8;padding:4px 12px;border-radius:999px;font-size:0.72rem;font-weight:600;">
            Trust {avg_conf:.0f}%
          </span>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── EXECUTIVE RISK BANNER ─────────────────────────────────────────────
    banner_emoji = {"Critical":"🔴","High":"🟠","Medium":"🟡","Low":"🟢"}.get(dom_level,"⚪")
    risk_color   = {"Critical":"#ef4444","High":"#f97316","Medium":"#eab308","Low":"#22c55e"}.get(dom_level,"#6366f1")

    st.markdown(f"""
    <div class="risk-banner {banner_cls}">
      <div>
        <div style="font-size:0.7rem;font-weight:700;color:rgba(255,255,255,0.5);letter-spacing:0.15em;
                    text-transform:uppercase;margin-bottom:0.5rem;">OVERALL REPOSITORY RISK</div>
        <div class="risk-label" style="color:{risk_color};">{banner_emoji} {dom_level.upper()} RISK</div>
        <div style="font-size:0.85rem;color:rgba(255,255,255,0.55);margin-top:0.5rem;">
          {n_crit} Critical &nbsp;·&nbsp; {n_high} High &nbsp;·&nbsp; {n_med} Medium &nbsp;·&nbsp; {n_low} Low
        </div>
      </div>
      <div style="display:flex;gap:2rem;">
        <div class="health-ring">
          <div class="health-score" style="color:{risk_color};">{hs}</div>
          <div class="health-sub">Health Score<br>/100</div>
        </div>
        <div class="health-ring">
          <div class="health-score" style="color:#6366f1;">{avg_conf:.0f}%</div>
          <div class="health-sub">Trust<br>Score</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── KEY METRICS ────────────────────────────────────────────────────────
    st.markdown('<div class="section-hdr">📊 Key Metrics</div>', unsafe_allow_html=True)
    metrics = [
        ("📁", n,           "Files Analyzed"),
        ("🔴", n_crit,      "Critical Files"),
        ("🟠", n_high,      "High Risk Files"),
        ("🟡", n_med,       "Medium Risk Files"),
        ("🟢", n_low,       "Low Risk Files"),
        ("🔀", f"{avg_cmx:.1f}", "Avg Complexity"),
        ("📐", f"{avg_mi:.0f}", "Avg Maintainability"),
        ("👥", contribs,    "Max Contributors"),
        ("⚡", f"{avg_freq:.3f}", "Avg Commit Freq"),
    ]
    cols = st.columns(len(metrics))
    for col, (icon, val, lbl) in zip(cols, metrics):
        with col:
            st.markdown(f"""
            <div class="metric-card">
              <div class="metric-icon">{icon}</div>
              <div class="metric-val">{val}</div>
              <div class="metric-lbl">{lbl}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── TWO-COLUMN LAYOUT ─────────────────────────────────────────────────
    left, right = st.columns([1.1, 0.9])

    # ── TOP CRITICAL FILES ──────────────────────────────────────────────
    with left:
        st.markdown('<div class="section-hdr">🚨 Top Critical Files</div>', unsafe_allow_html=True)
        top_files = df.sort_values("risk_score_val", ascending=False).head(10)
        for _, row in top_files.iterrows():
            lvl    = row.get("risk_level_name","Low")
            sc     = row.get("risk_score_val",0)
            fp     = str(row.get("file_path",""))
            drivers= risk_drivers(row)
            score_color = {"Critical":"#ef4444","High":"#f97316","Medium":"#eab308","Low":"#22c55e"}.get(lvl,"#6366f1")
            short_path = ("…" + fp[-55:]) if len(fp) > 55 else fp
            st.markdown(f"""
            <div class="crit-row">
              <div class="crit-score" style="background:{score_color};">{sc:.0f}</div>
              <div style="min-width:0;">
                <div class="crit-path">{short_path}</div>
                <div class="crit-reasons">⚠️ {' &nbsp;·&nbsp; '.join(drivers)}</div>
                <div style="margin-top:0.35rem;">{badge_html(lvl)} &nbsp; {trust_html(float(row.get('confidence',0)))}</div>
              </div>
            </div>""", unsafe_allow_html=True)

    # ── HEATMAP TREE ─────────────────────────────────────────────────────
    with right:
        st.markdown('<div class="section-hdr">🗺️ Repository Risk Heatmap</div>', unsafe_allow_html=True)
        render_heatmap_tree(df)

    # ── CHARTS ────────────────────────────────────────────────────────────
    st.markdown('<div class="section-hdr" style="margin-top:1.5rem;">📈 Risk Analytics</div>',
                unsafe_allow_html=True)
    render_charts(df)

    # ── INTERACTIVE FILE RISK TABLE ───────────────────────────────────────
    st.markdown('<div class="section-hdr" style="margin-top:1.5rem;">📋 File Risk Intelligence Center</div>',
                unsafe_allow_html=True)

    # Search / Filter controls
    fc1, fc2, fc3, fc4 = st.columns([2, 1.2, 1, 1])
    with fc1:
        search = st.text_input("🔍 Search files or folders", placeholder="e.g. auth, payment, app.py", label_visibility="collapsed")
    with fc2:
        risk_filter = st.multiselect("Risk Level", ["Critical","High","Medium","Low"],
                                     default=["Critical","High","Medium","Low"], label_visibility="collapsed")
    with fc3:
        lang_opts = sorted(df["language"].unique().tolist()) if "language" in df.columns else []
        lang_filter = st.multiselect("Language", lang_opts, default=lang_opts, label_visibility="collapsed")
    with fc4:
        conf_filter = st.select_slider("Min Confidence", [0,50,60,70,80,90], value=0, label_visibility="collapsed")

    df_filtered = df.copy()
    if search:
        df_filtered = df_filtered[df_filtered["file_path"].str.contains(search, case=False, na=False)]
    if risk_filter:
        df_filtered = df_filtered[df_filtered["risk_level_name"].isin(risk_filter)]
    if lang_filter:
        df_filtered = df_filtered[df_filtered["language"].isin(lang_filter)]
    df_filtered = df_filtered[df_filtered["confidence"] * 100 >= conf_filter]

    display_cols = {
        "file_path": "File Path",
        "language": "Language",
        "risk_score_val": "Risk Score",
        "risk_level_name": "Risk Level",
        "confidence": "Confidence",
        "loc": "LOC",
        "complexity": "Complexity",
        "maintainability_index": "Maintainability",
        "commit_count": "Commits",
        "modification_count": "Modifications",
        "contributor_count": "Contributors",
    }
    available = {k: v for k, v in display_cols.items() if k in df_filtered.columns}
    df_show = df_filtered[list(available.keys())].rename(columns=available)
    df_show = df_show.sort_values("Risk Score", ascending=False).reset_index(drop=True)
    df_show.index += 1
    df_show["Confidence"] = (df_show["Confidence"] * 100).round(1).astype(str) + "%"
    df_show["Risk Score"] = df_show["Risk Score"].round(1)

    st.dataframe(
        df_show,
        use_container_width=True,
        height=420,
    )

    # ── FILE DETAIL PANEL ─────────────────────────────────────────────────
    st.markdown('<div class="section-hdr" style="margin-top:1rem;">🔎 File Detail Inspector</div>',
                unsafe_allow_html=True)
    file_paths = df_filtered["file_path"].tolist() if "file_path" in df_filtered.columns else []
    if file_paths:
        sel = st.selectbox("Select a file to inspect", file_paths, label_visibility="collapsed")
        row = df[df["file_path"] == sel].iloc[0] if not df[df["file_path"] == sel].empty else None
        if row is not None:
            lvl = row.get("risk_level_name","Low")
            sc  = float(row.get("risk_score_val",0))
            conf= float(row.get("confidence",0))
            color = {"Critical":"#ef4444","High":"#f97316","Medium":"#eab308","Low":"#22c55e"}.get(lvl,"#6366f1")
            drivers = risk_drivers(row)

            d1, d2, d3 = st.columns([1.5,1,1])
            with d1:
                st.markdown(f"""
                <div class="panel">
                  <div class="panel-title">📄 {sel.split('/')[-1]}</div>
                  <div style="font-family:monospace;font-size:0.72rem;color:#64748b;word-break:break-all;">{sel}</div>
                  <hr style="margin:0.75rem 0;">
                  <div style="display:flex;justify-content:space-between;margin-bottom:0.5rem;">
                    <span style="color:#94a3b8;font-size:0.8rem;">Risk Score</span>
                    <span style="color:{color};font-weight:800;font-size:1.1rem;">{sc:.0f}/100</span>
                  </div>
                  <div style="display:flex;justify-content:space-between;margin-bottom:0.5rem;">
                    <span style="color:#94a3b8;font-size:0.8rem;">Risk Level</span>
                    {badge_html(lvl)}
                  </div>
                  <div style="display:flex;justify-content:space-between;">
                    <span style="color:#94a3b8;font-size:0.8rem;">Trust Gate</span>
                    {trust_html(conf)}
                  </div>
                </div>""", unsafe_allow_html=True)

            with d2:
                fields = [
                    ("Lines of Code",   f"{int(row.get('loc',0)):,}"),
                    ("Complexity",      f"{float(row.get('complexity',0)):.1f}"),
                    ("Maintainability", f"{float(row.get('maintainability_index',0)):.1f}"),
                    ("Commit Count",    f"{int(row.get('commit_count',0)):,}"),
                    ("Modifications",   f"{int(row.get('modification_count',0)):,}"),
                    ("Contributors",    f"{int(row.get('contributor_count',0)):,}"),
                ]
                html = '<div class="panel"><div class="panel-title">📊 Code Metrics</div>'
                for lbl2, val2 in fields:
                    html += f"""
                    <div style="display:flex;justify-content:space-between;padding:0.35rem 0;
                                border-bottom:1px solid #1e293b;">
                      <span style="color:#64748b;font-size:0.8rem;">{lbl2}</span>
                      <span style="color:#f8fafc;font-weight:600;font-size:0.85rem;">{val2}</span>
                    </div>"""
                html += "</div>"
                st.markdown(html, unsafe_allow_html=True)

            with d3:
                html = '<div class="panel"><div class="panel-title">⚠️ Top Risk Drivers</div>'
                for d in drivers:
                    html += f"""
                    <div style="display:flex;align-items:center;gap:0.5rem;padding:0.5rem 0;
                                border-bottom:1px solid #1e293b;">
                      <span style="color:#ef4444;">▶</span>
                      <span style="color:#e2e8f0;font-size:0.82rem;">{d}</span>
                    </div>"""
                html += "</div>"
                st.markdown(html, unsafe_allow_html=True)

    # ── EXPLAINABILITY ─────────────────────────────────────────────────────
    st.markdown('<div class="section-hdr" style="margin-top:1.5rem;">🧠 Explainability & Feature Importance</div>',
                unsafe_allow_html=True)
    try:
        rf_path = os.path.join(BASE, "models", "random_forest.pkl")
        preproc_path = os.path.join(BASE, "models", "preprocessor.pkl")
        from ml.preprocessing import CodeRiskPreprocessor
        preprocessor = CodeRiskPreprocessor.load(preproc_path)
        with open(rf_path, "rb") as f: rf_model = pickle.load(f)
        feat_names = preprocessor.feature_names
        importances = rf_model.feature_importances_

        df_imp = pd.DataFrame({"Feature": feat_names, "Importance": importances})
        df_imp = df_imp.sort_values("Importance", ascending=True).tail(10)

        import plotly.graph_objects as go
        fig = go.Figure(go.Bar(
            x=df_imp["Importance"], y=df_imp["Feature"],
            orientation="h",
            marker=dict(
                color=df_imp["Importance"],
                colorscale=[[0,"#1e293b"],[0.5,"#6366f1"],[1,"#8b5cf6"]],
                line=dict(width=0)
            ),
            text=[f"{v:.3f}" for v in df_imp["Importance"]],
            textposition="outside",
            textfont=dict(color="#94a3b8"),
        ))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter", color="#94a3b8"),
            margin=dict(l=0,r=60,t=10,b=0),
            height=320,
            xaxis=dict(gridcolor="#1e293b",zerolinecolor="#1e293b",title="Gini Importance"),
            yaxis=dict(gridcolor="#1e293b",tickfont=dict(size=11)),
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception:
        st.info("Explainability chart requires the trained model files.")

    # ── EXECUTIVE SUMMARY ─────────────────────────────────────────────────
    st.markdown('<div class="section-hdr" style="margin-top:1.5rem;">📋 Executive Summary</div>',
                unsafe_allow_html=True)
    top5_crit = df.sort_values("risk_score_val", ascending=False).head(5)
    tech_debt  = df.sort_values("maintainability_index").head(5) if "maintainability_index" in df.columns else df.head(5)
    maint_risk = df.sort_values("complexity", ascending=False).head(5) if "complexity" in df.columns else df.head(5)

    e1, e2, e3 = st.columns(3)
    for col2, title, sub_df, icon in [
        (e1, "Top 5 Critical Files",       top5_crit, "🔴"),
        (e2, "Top 5 Technical Debt",        tech_debt, "🔧"),
        (e3, "Top 5 Maintenance Risks",     maint_risk, "⚠️"),
    ]:
        with col2:
            html = f'<div class="panel"><div class="panel-title">{icon} {title}</div>'
            for i, (_, r) in enumerate(sub_df.iterrows(), 1):
                fp = str(r.get("file_path","")).split("/")[-1]
                sc = float(r.get("risk_score_val",0))
                color = {"Critical":"#ef4444","High":"#f97316","Medium":"#eab308","Low":"#22c55e"}.get(
                    r.get("risk_level_name","Low"),"#6366f1")
                html += f"""
                <div style="display:flex;align-items:center;gap:0.5rem;padding:0.5rem 0;
                            border-bottom:1px solid #1e293b;">
                  <span style="color:{color};font-weight:800;font-size:0.9rem;min-width:1.5rem;">{i}.</span>
                  <div style="min-width:0;">
                    <div style="font-size:0.78rem;color:#e2e8f0;font-family:monospace;
                                overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">{fp}</div>
                    <div style="font-size:0.7rem;color:#64748b;">Score: {sc:.0f}</div>
                  </div>
                </div>"""
            html += "</div>"
            st.markdown(html, unsafe_allow_html=True)

    # ── DOWNLOADS ─────────────────────────────────────────────────────────
    st.markdown('<div class="section-hdr" style="margin-top:1.5rem;">⬇️ Download Reports</div>',
                unsafe_allow_html=True)
    dl1, dl2, dl3, dl4 = st.columns(4)

    # Full CSV
    with dl1:
        csv_full = df.to_csv(index=False).encode()
        st.download_button("📊 Full Prediction CSV", csv_full,
                           file_name=f"{repo_name}_predictions.csv", mime="text/csv",
                           use_container_width=True)

    # High-risk CSV
    with dl2:
        df_hr = df[df["risk_level_name"].isin(["Critical","High"])]
        csv_hr = df_hr.to_csv(index=False).encode()
        st.download_button("🔴 High Risk Files CSV", csv_hr,
                           file_name=f"{repo_name}_high_risk.csv", mime="text/csv",
                           use_container_width=True)

    # Executive summary markdown
    with dl3:
        exec_md = f"""# Repository Risk Intelligence — Executive Summary

Repository: {owner}/{repo_name}
Generated: {ts}

## Health Score: {hs}/100
## Trust Score: {avg_conf:.0f}%
## Overall Risk: {dom_level}

## Risk Distribution
- Critical: {n_crit}
- High: {n_high}
- Medium: {n_med}
- Low: {n_low}

## Top 5 Critical Files
""" + "\n".join([f"{i+1}. {r['file_path']} (Score: {r['risk_score_val']:.0f})"
                  for i, (_, r) in enumerate(top5_crit.iterrows())])
        st.download_button("📄 Executive Summary", exec_md.encode(),
                           file_name=f"{repo_name}_executive_summary.md",
                           mime="text/markdown", use_container_width=True)

    # Trust gate CSV
    with dl4:
        df_trust = df[["file_path","risk_level_name","confidence","risk_score_val"]].copy() if "file_path" in df.columns else df.copy()
        df_trust["trust_decision"] = df_trust["confidence"].apply(
            lambda c: "TRUSTED" if c*100 >= 70 else "FLAGGED")
        csv_trust = df_trust.to_csv(index=False).encode()
        st.download_button("🔒 Trust Gate CSV", csv_trust,
                           file_name=f"{repo_name}_trust_gate.csv", mime="text/csv",
                           use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    # Nav bar
    st.markdown("""
    <div class="nav-bar">
      <div class="nav-logo">🛡️ Repository Risk Intelligence
        <span class="nav-badge">BETA</span>
      </div>
      <div style="font-size:0.75rem;color:#475569;">Powered by Random Forest · Explainability AI · Trust Gate</div>
    </div>
    """, unsafe_allow_html=True)

    # Check if we already have results cached
    if "analysis_result" not in st.session_state:
        st.session_state.analysis_result = None
    if "analysis_repo_url" not in st.session_state:
        st.session_state.analysis_repo_url = ""
    if "analysis_owner" not in st.session_state:
        st.session_state.analysis_owner = ""
    if "analysis_repo_name" not in st.session_state:
        st.session_state.analysis_repo_name = ""

    # ── HERO + URL INPUT ──────────────────────────────────────────────────
    if st.session_state.analysis_result is None:
        st.markdown("""
        <div class="hero">
          <div class="hero-title">Repository Risk Intelligence</div>
          <div class="hero-sub">
            Paste any GitHub repository URL and get instant file-level risk scores,
            explainability reports, and trust assessments — powered by production ML.
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="url-card">', unsafe_allow_html=True)
        col_inp, col_btn = st.columns([3, 1])
        with col_inp:
            url = st.text_input(
                "GitHub Repository URL",
                value=st.session_state.analysis_repo_url or "https://github.com/pallets/flask",
                placeholder="https://github.com/owner/repository",
                label_visibility="collapsed"
            )
        with col_btn:
            analyze_clicked = st.button("🚀 Analyze Repository")

        st.markdown("""
        <div style="margin-top:1rem;display:flex;gap:1.5rem;flex-wrap:wrap;">
          <div style="font-size:0.75rem;color:#475569;">
            ✓ Clone &nbsp; ✓ Mine &nbsp; ✓ Quality Metrics &nbsp; ✓ Feature Engineering
            &nbsp; ✓ RF Prediction &nbsp; ✓ Explainability &nbsp; ✓ Trust Gate
          </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Example repos
        st.markdown('<br>', unsafe_allow_html=True)
        st.markdown('<div style="text-align:center;font-size:0.75rem;color:#475569;margin-bottom:0.5rem;">Try an example repository:</div>', unsafe_allow_html=True)
        ex_cols = st.columns(5)
        examples = [
            ("pallets/flask","https://github.com/pallets/flask"),
            ("pallets/click","https://github.com/pallets/click"),
            ("encode/databases","https://github.com/encode/databases"),
            ("axios/axios","https://github.com/axios/axios"),
            ("reduxjs/redux","https://github.com/reduxjs/redux"),
        ]
        for ec, (name, eurl) in zip(ex_cols, examples):
            with ec:
                if st.button(name, key=f"ex_{name}", use_container_width=True):
                    st.session_state.analysis_repo_url = eurl
                    st.rerun()

        if analyze_clicked:
            if not url.strip():
                st.error("Please enter a GitHub repository URL."); return
            owner, repo_name = parse_github_url(url)
            if not owner:
                st.error("Invalid GitHub URL. Use format: https://github.com/owner/repo"); return

            st.session_state.analysis_repo_url  = url
            st.session_state.analysis_owner     = owner
            st.session_state.analysis_repo_name = repo_name

            # ── PIPELINE EXECUTION ──────────────────────────────────────
            st.markdown(f"""
            <div style="background:#111318;border:1px solid #1e293b;border-radius:12px;padding:1.25rem 1.5rem;margin-top:1.5rem;">
              <div style="font-weight:700;color:#f8fafc;margin-bottom:0.75rem;">
                ⚙️ Analyzing <span style="color:#6366f1;">{owner}/{repo_name}</span>
              </div>
            """, unsafe_allow_html=True)

            prog = st.progress(0)
            status = st.empty()
            log_box = st.empty()

            stages = [
                "Cloning repository…",
                "Extracting commits…",
                "Mining modifications…",
                "Quality metrics pipeline…",
                "Feature engineering…",
                "Running RF predictions…",
                "Building explainability…",
                "Generating report…",
            ]
            for i, s in enumerate(stages):
                status.markdown(f'<div style="font-size:0.85rem;color:#94a3b8;">{s}</div>',
                                unsafe_allow_html=True)
                prog.progress((i+1) / len(stages) * 0.85)
                if i == 0: break  # let pipeline start

            df_result = run_pipeline(owner, repo_name, log_box)
            prog.progress(1.0)
            status.empty()

            st.markdown("</div>", unsafe_allow_html=True)

            if df_result is None or df_result.empty:
                st.error("❌ Pipeline failed. Check the log above for details.")
                return

            st.session_state.analysis_result = df_result
            st.rerun()
    else:
        # ── RESULTS DASHBOARD ──────────────────────────────────────────────
        df   = st.session_state.analysis_result
        url  = st.session_state.analysis_repo_url
        owner     = st.session_state.analysis_owner
        repo_name = st.session_state.analysis_repo_name

        # Reset button
        col_reset = st.columns([6,1])[1]
        with col_reset:
            if st.button("🔄 New Analysis"):
                st.session_state.analysis_result    = None
                st.session_state.analysis_repo_url  = ""
                st.session_state.analysis_owner     = ""
                st.session_state.analysis_repo_name = ""
                st.rerun()

        render_dashboard(df, owner, repo_name, url)


if __name__ == "__main__":
    main()
