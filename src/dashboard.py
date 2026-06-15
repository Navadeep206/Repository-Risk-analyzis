#!/usr/bin/env python3
"""
Repository Risk Intelligence Platform — Premium Dashboard (V3)
=============================================================
Streamlit dashboard for repository cloning, feature extraction, 
and file-level risk inference using the leakage-free XGBoost production model.
"""

import os
import sys
import time
import shutil
import traceback
import io
import json
import warnings
import pickle
import re
import math
from datetime import datetime, timezone
from collections import defaultdict

import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import entropy as scipy_entropy
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")

# ── path bootstrap ────────────────────────────────────────────────────────────
BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, "src"))

from pdf_generator import generate_pdf_report

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
    font-size: 0.75rem; font-weight: 700; color: #6366f1;
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
.crit-path { font-size: 0.82rem; font-weight: 600; color: #f8fafc; font-family: 'Courier New', monospace; word-break: break-all; }
.crit-reasons { font-size: 0.72rem; color: #94a3b8; margin-top: 0.2rem; }

/* ── trust gate ── */
.trust-chip {
    display: inline-flex; align-items: center; gap: 0.35rem;
    padding: 3px 10px; border-radius: 999px;
    font-size: 0.72rem; font-weight: 600;
}
.trust-high { background: rgba(34,197,94,0.12); color: #4ade80; border: 1px solid rgba(34,197,94,0.25); }
.trust-good { background: rgba(59,130,246,0.12); color: #60a5fa; border: 1px solid rgba(59,130,246,0.25); }
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
# STATIC VARIABLES & UTILITIES
# ══════════════════════════════════════════════════════════════════════════════
MODELS_DIR  = os.path.join(BASE, "models")
EXT_DIR     = os.path.join(BASE, "data/external_repos")
TRAIN_DATA  = os.path.join(BASE, "data/final/ml_dataset_v3.csv")
CLONE_DEPTH = 200

NUM_FEATS = ["loc","complexity","maintainability_index","commit_count",
             "modification_count","contributor_count","commit_frequency",
             "repository_age_days","ownership_concentration",
             "contributor_entropy","bus_factor","recent_churn","time_decayed_churn"]

def risk_score(row: pd.Series) -> float:
    """Compute a 0-100 risk score combining model probabilities, complexity & MI."""
    prob_high = float(row.get("prob_HIGH", 0.5))
    prob_med  = float(row.get("prob_MEDIUM", 0.3))
    base      = prob_high * 70 + prob_med * 30

    cmplx = float(row.get("complexity", 0))
    cmplx_pts = min(20.0, cmplx * 0.8)

    mi = float(row.get("maintainability_index", 100.0))
    mi_pts = max(0.0, (100.0 - mi) * 0.1)

    score = min(100.0, base + cmplx_pts + mi_pts)
    return round(score, 1)

def tech_debt_priority(score: float) -> str:
    if score >= 80: return "Critical"
    if score >= 60: return "High"
    if score >= 40: return "Medium"
    return "Low"

def tech_debt_color(priority: str) -> str:
    return {"Critical":"#ef4444","High":"#f97316","Medium":"#eab308","Low":"#22c55e"}.get(priority, "#6366f1")

def risk_emoji(level: str) -> str:
    return {"Critical": "🔴", "High": "🟠", "Medium": "🟡", "Low": "🟢"}.get(level, "⚪")

def badge_html(level: str) -> str:
    cls = {"Critical":"badge-critical","High":"badge-high","Medium":"badge-medium","Low":"badge-low"}.get(level,"badge-low")
    return f'<span class="badge {cls}">{risk_emoji(level)} {level}</span>'

def trust_level(conf: float) -> str:
    pct = conf * 100 if conf <= 1.0 else conf
    if pct >= 90: return "HIGH TRUST"
    if pct >= 80: return "GOOD TRUST"
    if pct >= 70: return "MODERATE TRUST"
    return "MANUAL REVIEW RECOMMENDED"

def trust_html(conf: float) -> str:
    pct = conf * 100 if conf <= 1.0 else conf
    if pct >= 90:
        return f'<span class="trust-chip trust-high">🟢 HIGH TRUST ({pct:.0f}%)</span>'
    elif pct >= 80:
        return f'<span class="trust-chip trust-good">🔵 GOOD TRUST ({pct:.0f}%)</span>'
    elif pct >= 70:
        return f'<span class="trust-chip trust-mod">🟡 MODERATE TRUST ({pct:.0f}%)</span>'
    else:
        return f'<span class="trust-chip trust-low">🔴 REVIEW RECOMMENDED ({pct:.0f}%)</span>'

def health_score(df: pd.DataFrame) -> int:
    """Calculates repository health score between 0 and 100."""
    if df.empty or "risk_score_val" not in df.columns:
        return 100
    # Weighted health score: 100 minus the average risk score
    avg_risk = df["risk_score_val"].mean()
    return max(0, min(100, int(100 - avg_risk)))

def parse_github_url(url: str):
    url = url.strip()
    # Strip query parameters and hashes
    url = url.split("?")[0].split("#")[0].rstrip("/")
    m = re.match(r"https?://github\.com/([^/]+)/([^/\s]+)", url)
    if m:
        return m.group(1), m.group(2)
    return None, None

@st.cache_data
def get_training_centroids():
    """Load and cache centroids of training repositories to avoid disk reading repeatedly."""
    if os.path.exists(TRAIN_DATA):
        try:
            train_df = pd.read_csv(TRAIN_DATA, low_memory=True)
            return train_df.groupby("repository_name")[NUM_FEATS].mean()
        except Exception:
            pass
    return None

# ══════════════════════════════════════════════════════════════════════════════
# FEATURE EXTRACTION ENGINE (Leakage-free V3 identical logic)
# ══════════════════════════════════════════════════════════════════════════════
IGNORED_DIRS = {
    ".venv", "venv", "node_modules", ".git", "__pycache__",
    "dist", "build", "vendor", "third_party", ".next", ".turbo",
    "coverage", ".nyc_output", "out", ".output", "fixtures",
}

def get_source_files(repo_path):
    files = {"python": [], "javascript": [], "typescript": []}
    for root, dirs, fnames in os.walk(repo_path):
        dirs[:] = [d for d in dirs if d not in IGNORED_DIRS]
        for f in fnames:
            ext = os.path.splitext(f)[1].lower()
            ap  = os.path.join(root, f)
            if ext == ".py":              files["python"].append(ap)
            elif ext in (".js", ".jsx"):  files["javascript"].append(ap)
            elif ext in (".ts", ".tsx"):  files["typescript"].append(ap)
    return files

def compute_maintainability(loc, complexity):
    try:
        V   = max(1, loc) * 4
        CC  = max(1, complexity)
        L   = max(1, loc)
        mi  = (171 - 5.2 * math.log(V) - 0.23 * CC - 16.2 * math.log(L)) * 100.0 / 171
        return round(max(-100.0, min(100.0, mi)), 4)
    except Exception:
        return 50.0

def analyze_python_file(path):
    try:
        import ast
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            src = f.read()
        loc = len([l for l in src.splitlines() if l.strip() and not l.strip().startswith("#")])
        try:
            tree = ast.parse(src)
            complexity = 1
            for node in ast.walk(tree):
                if isinstance(node, (ast.If, ast.For, ast.While, ast.ExceptHandler,
                                     ast.With, ast.Assert, ast.comprehension,
                                     ast.BoolOp, ast.IfExp)):
                    complexity += 1
        except SyntaxError:
            complexity = max(1, loc // 20)
        mi = compute_maintainability(loc, complexity)
        return loc, complexity, mi
    except Exception:
        return 0, 1, 50.0

def analyze_js_ts_file(path):
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        code_lines = [l for l in lines if l.strip() and not l.strip().startswith("//")
                      and not l.strip().startswith("*") and not l.strip().startswith("/*")]
        loc = len(code_lines)
        src = "".join(lines)
        keywords = ["if ", "else if", "for ", "while ", "switch ", "catch ",
                    "&&", "||", "? ", "?."]
        complexity = 1 + sum(src.count(kw) for kw in keywords)
        complexity = min(complexity, max(1, loc // 3))
        mi = compute_maintainability(loc, complexity)
        return loc, complexity, mi
    except Exception:
        return 0, 1, 50.0

def extract_quality_metrics(repo_path, repo_name, log_func):
    log_func("🔬 Extracting quality metrics...", "run")
    files_by_lang = get_source_files(repo_path)
    rows = []
    for lang, paths in files_by_lang.items():
        for path in paths:
            rel = os.path.relpath(path, repo_path)
            if lang == "python":
                loc, cc, mi = analyze_python_file(path)
            else:
                loc, cc, mi = analyze_js_ts_file(path)
            rows.append({"file_path": rel, "language": lang,
                         "loc": loc, "complexity": cc, "maintainability_index": mi})
    df = pd.DataFrame(rows)
    langs_found = {k: len(v) for k, v in files_by_lang.items() if len(v) > 0}
    log_func(f"✅ Extracted metrics for {len(df):,} source files {langs_found}", "ok")
    return df

def extract_commit_features(repo_path, repo_name, log_func):
    log_func("📝 Analyzing git commit logs (pydriller)...", "run")
    from pydriller import Repository

    file_commits    = defaultdict(set)
    file_authors    = defaultdict(set)
    file_added      = defaultdict(int)
    file_deleted    = defaultdict(int)
    file_dates      = defaultdict(list)
    author_commits  = defaultdict(int)
    commit_dates    = []

    SUPPORTED_EXT = {".py", ".js", ".jsx", ".ts", ".tsx"}
    total_commits = 0

    for commit in Repository(repo_path).traverse_commits():
        total_commits += 1
        c_date = commit.author_date
        if c_date and hasattr(c_date, "tzinfo") and c_date.tzinfo is None:
            c_date = c_date.replace(tzinfo=timezone.utc)
        if c_date:
            commit_dates.append(c_date)
        author = commit.author.email if commit.author else "unknown"
        author_commits[author] += 1

        try:
            for mod in commit.modified_files:
                fpath = mod.new_path or mod.old_path
                if not fpath:
                    continue
                ext = os.path.splitext(fpath)[1].lower()
                if ext not in SUPPORTED_EXT:
                    continue
                file_commits[fpath].add(commit.hash)
                file_authors[fpath].add(author)
                file_added[fpath]   += (mod.added_lines or 0)
                file_deleted[fpath] += (mod.deleted_lines or 0)
                if c_date:
                    file_dates[fpath].append(c_date)
        except Exception:
            continue

        if total_commits % 200 == 0:
            log_func(f"  Processed {total_commits} commits...", "info")

    log_func(f"✅ Commit log analysis complete. Total commits: {total_commits}", "ok")

    # Repo-level stats
    now = datetime.now(timezone.utc)
    if commit_dates:
        commit_dates_sorted = sorted(commit_dates)
        repo_age_days   = max(1, (commit_dates_sorted[-1] - commit_dates_sorted[0]).days)
    else:
        repo_age_days   = 1

    total_repo_commits = sum(author_commits.values())
    if total_repo_commits > 0:
        top_commits = sorted(author_commits.values(), reverse=True)
        top_share   = top_commits[0] / total_repo_commits
        cumulative, bf = 0, 0
        for c in top_commits:
            cumulative += c / total_repo_commits
            bf += 1
            if cumulative >= 0.5:
                break
        repo_ownership_conc = top_share
        repo_bus_factor     = bf
        all_authors         = list(author_commits.keys())
        shares = np.array([author_commits[a] / total_repo_commits for a in all_authors])
        repo_entropy = float(scipy_entropy(shares, base=2)) if len(shares) > 1 else 0.0
    else:
        repo_ownership_conc = 1.0
        repo_bus_factor     = 1
        repo_entropy        = 0.0

    # Per-file features calculation
    RECENT_WINDOW_DAYS = 90
    DECAY_LAMBDA       = 0.01

    per_file_rows = []
    for fpath in set(file_commits.keys()):
        dates = sorted(file_dates.get(fpath, []))
        commit_count      = len(file_commits[fpath])
        modification_count= commit_count
        contributor_count = len(file_authors[fpath])
        commit_frequency  = commit_count / repo_age_days

        recent_churn = 0
        if dates:
            total_churn = file_added[fpath] + file_deleted[fpath]
            recent_churn = round(total_churn * min(1.0, RECENT_WINDOW_DAYS / max(1, repo_age_days)), 2)

        total_churn_f = file_added[fpath] + file_deleted[fpath]
        if dates and total_churn_f > 0:
            mid_date    = dates[len(dates)//2]
            ref_date    = dates[-1]
            days_back   = max(0, (ref_date - mid_date).days)
            decay_w     = math.exp(-DECAY_LAMBDA * days_back)
            time_decayed_churn = round(total_churn_f * decay_w, 4)
        else:
            time_decayed_churn = 0.0

        if contributor_count <= 1:
            ownership_conc = 1.0
            contributor_ent = 0.0
            bus_f = 1
        else:
            ownership_conc = 1.0 / contributor_count
            shares_f = np.ones(contributor_count) / contributor_count
            contributor_ent = float(scipy_entropy(shares_f, base=2))
            bus_f = min(contributor_count, repo_bus_factor)

        has_bug_fix_history = 0
        time_since_last_bug_fix = np.nan

        per_file_rows.append({
            "file_path":              fpath,
            "commit_count":           commit_count,
            "modification_count":     modification_count,
            "contributor_count":      contributor_count,
            "commit_frequency":       round(commit_frequency, 6),
            "repository_age_days":    repo_age_days,
            "ownership_concentration":round(ownership_conc, 6),
            "contributor_entropy":    round(contributor_ent, 6),
            "bus_factor":             bus_f,
            "recent_churn":           recent_churn,
            "time_decayed_churn":     time_decayed_churn,
            "time_since_last_bug_fix":time_since_last_bug_fix,
            "has_bug_fix_history":    has_bug_fix_history,
        })

    commit_df = pd.DataFrame(per_file_rows)
    meta = {
        "total_commits":      total_commits,
        "repo_age_days":      repo_age_days,
        "contributor_count":  len(author_commits),
        "bus_factor":         repo_bus_factor,
        "ownership_conc":     repo_ownership_conc,
        "entropy":            repo_entropy,
    }
    return commit_df, meta

def build_feature_matrix(quality_df, commit_df, repo_name):
    if commit_df is not None and len(commit_df) > 0:
        merged = quality_df.merge(commit_df, on="file_path", how="left")
    else:
        merged = quality_df.copy()
        for col in ["commit_count","modification_count","contributor_count",
                    "commit_frequency","repository_age_days","ownership_concentration",
                    "contributor_entropy","bus_factor","recent_churn",
                    "time_decayed_churn","time_since_last_bug_fix","has_bug_fix_history"]:
            merged[col] = 0.0

    merged["commit_count"]         = merged.get("commit_count", pd.Series()).fillna(0.0)
    merged["modification_count"]   = merged.get("modification_count", pd.Series()).fillna(0.0)
    merged["contributor_count"]    = merged.get("contributor_count", pd.Series()).fillna(0.0)
    merged["commit_frequency"]     = merged.get("commit_frequency", pd.Series()).fillna(0.0)
    
    if "repository_age_days" in merged.columns:
        age_mode = merged["repository_age_days"].mode()
        merged["repository_age_days"] = merged["repository_age_days"].fillna(
            age_mode.iloc[0] if len(age_mode) > 0 else 1.0
        )
    else:
        merged["repository_age_days"] = 1.0
        
    merged["ownership_concentration"] = merged.get("ownership_concentration", pd.Series()).fillna(1.0)
    merged["contributor_entropy"]  = merged.get("contributor_entropy", pd.Series()).fillna(0.0)
    merged["bus_factor"]           = merged.get("bus_factor", pd.Series()).fillna(1.0)
    merged["recent_churn"]         = merged.get("recent_churn", pd.Series()).fillna(0.0)
    merged["time_decayed_churn"]   = merged.get("time_decayed_churn", pd.Series()).fillna(0.0)
    merged["time_since_last_bug_fix"] = merged.get("time_since_last_bug_fix", pd.Series())
    merged["has_bug_fix_history"]  = merged.get("has_bug_fix_history", pd.Series()).fillna(0.0)
    merged["repository_name"]      = repo_name

    return merged[merged["loc"] > 0].copy()

def is_string_col(series):
    try:
        pd.to_numeric(series.dropna(), errors="raise")
        return False
    except (ValueError, TypeError):
        return True

def run_inference(df, preprocessor, model):
    cat_enc  = preprocessor["cat_encodings"]
    scaler   = preprocessor["scaler"]
    le_label = preprocessor["le_label"]
    feat     = preprocessor["feature_cols"]

    cat_cols = [c for c in feat if is_string_col(df[c]) or c == "language"]
    num_cols = [c for c in feat if c not in cat_cols]

    out = pd.DataFrame(index=df.index)
    for col in cat_cols:
        le = cat_enc.get(col)
        if le is None:
            out[col] = 0
            continue
        seen = set(le.classes_)
        vals = df[col].fillna("unknown").astype(str)
        out[col] = vals.apply(lambda v: int(le.transform([v])[0]) if v in seen else 0)
    for col in num_cols:
        s = pd.to_numeric(df[col], errors="coerce")
        out[col] = s.fillna(s.median() if not s.isna().all() else 0.0)

    X = scaler.transform(out[feat].values)
    proba  = model.predict_proba(X)
    labels = le_label.inverse_transform(np.argmax(proba, axis=1))
    conf   = np.max(proba, axis=1)

    classes = le_label.classes_
    result  = df.copy()
    result["predicted_risk"]    = labels
    result["confidence_score"]  = np.round(conf, 4)
    for i, cls in enumerate(classes):
        result[f"prob_{cls}"] = np.round(proba[:, i], 4)

    # Calculate custom UI scores
    result["risk_score_val"] = result.apply(risk_score, axis=1)
    result["tech_debt_priority"] = result["risk_score_val"].apply(tech_debt_priority)
    return result

# ══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE EXECUTION
# ══════════════════════════════════════════════════════════════════════════════
def run_pipeline(owner: str, repo_name: str, log_box) -> Optional[dict]:
    REPO_LOCAL = os.path.join(EXT_DIR, repo_name)
    os.makedirs(EXT_DIR, exist_ok=True)

    logs = []
    def log(msg: str, kind: str = "run"):
        css = {"ok":"log-ok","run":"log-run","err":"log-err","info":"log-info"}.get(kind,"log-info")
        ts  = datetime.now().strftime("%H:%M:%S")
        logs.append(f'<span class="{css}">[{ts}] {msg}</span>')
        log_box.markdown(
            f'<div class="step-log">{"<br>".join(logs[-10:])}</div>',
            unsafe_allow_html=True
        )

    # ── STEP 1: Clone ─────────────────────────────────────────────────────
    log(f"🔗 Cloning {owner}/{repo_name} (depth={CLONE_DEPTH})…")
    try:
        if not os.path.isdir(os.path.join(REPO_LOCAL, ".git")):
            from git import Repo
            Repo.clone_from(
                f"https://github.com/{owner}/{repo_name}",
                REPO_LOCAL,
                depth=CLONE_DEPTH,
                single_branch=True
            )
            log(f"✅ Repository cloned successfully", "ok")
        else:
            log(f"✓ Repository already exists locally — skipping clone", "ok")
    except Exception as e:
        log(f"❌ Clone failed: {e}", "err")
        return None

    # ── STEP 2: Metrics ───────────────────────────────────────────────────
    try:
        quality_df = extract_quality_metrics(REPO_LOCAL, repo_name, log)
        if len(quality_df) == 0:
            log("❌ No source files found for analysis", "err")
            return None
    except Exception as e:
        log(f"❌ Metrics extraction failed: {e}", "err")
        return None

    # ── STEP 3: Commits ───────────────────────────────────────────────────
    try:
        commit_df, meta = extract_commit_features(REPO_LOCAL, repo_name, log)
    except Exception as e:
        log(f"⚠️ Commit analysis failed: {e}. Proceeding with default features...", "info")
        commit_df = None
        meta = {"total_commits": 0, "repo_age_days": 1, "contributor_count": 1, "bus_factor": 1, "ownership_conc": 1.0, "entropy": 0.0}

    # ── STEP 4: Feature Matrix ────────────────────────────────────────────
    log("⚙️ Engineering feature matrix...", "run")
    try:
        feat_df = build_feature_matrix(quality_df, commit_df, repo_name)
        log(f"✅ Engineered features for {len(feat_df):,} files", "ok")
    except Exception as e:
        log(f"❌ Feature matrix builder failed: {e}", "err")
        return None

    # ── STEP 5: ML Inference ──────────────────────────────────────────────
    log("🤖 Loading production XGBoost (V3) artifacts...", "run")
    try:
        with open(os.path.join(MODELS_DIR, "best_model_v3.pkl"), "rb") as f:
            model_pkg = pickle.load(f)
        with open(os.path.join(MODELS_DIR, "preprocessor_v3.pkl"), "rb") as f:
            preprocessor = pickle.load(f)

        model = model_pkg["model"]
        log(f"🤖 Loaded: XGBoost model", "ok")

        log("🔮 Generating predictions...", "run")
        result_df = run_inference(feat_df, preprocessor, model)
        log("🏆 Risk prediction complete!", "ok")
    except Exception as e:
        log(f"❌ ML Inference failed: {e}", "err")
        traceback.print_exc()
        return None

    return {"df": result_df, "meta": meta}

# ══════════════════════════════════════════════════════════════════════════════
# COMPONENT RENDERING
# ══════════════════════════════════════════════════════════════════════════════
def render_health_dashboard(df: pd.DataFrame, owner: str, repo_name: str, meta: dict, url: str):
    import plotly.graph_objects as go
    import plotly.express as px

    hs = health_score(df)
    n_files = len(df)
    
    # Calculate distributions
    n_high = int((df["predicted_risk"] == "HIGH").sum())
    n_med  = int((df["predicted_risk"] == "MEDIUM").sum())
    n_low  = int((df["predicted_risk"] == "LOW").sum())
    avg_conf = float(df["confidence_score"].mean())
    
    # Domain overall risk
    max_risk_level = "LOW"
    if n_high > 0 or n_med > 0:
        pct_high = n_high / n_files
        pct_med  = n_med / n_files
        if pct_high >= 0.15: max_risk_level = "HIGH"
        elif pct_high > 0 or pct_med >= 0.3: max_risk_level = "MEDIUM"

    banner_cls = {
        "HIGH": "risk-banner-high",
        "MEDIUM": "risk-banner-medium",
        "LOW": "risk-banner-low"
    }.get(max_risk_level, "risk-banner-medium")

    risk_color = {
        "HIGH": "#ef4444",
        "MEDIUM": "#f97316",
        "LOW": "#22c55e"
    }.get(max_risk_level, "#6366f1")

    # Header Card
    primary_lang = df["language"].mode()[0] if "language" in df.columns and len(df) > 0 else "N/A"
    
    st.markdown(f"""
    <div style="background:#111318;border:1px solid #1e293b;border-radius:16px;padding:1.5rem 2rem;margin-bottom:1.5rem;">
      <div style="display:flex;align-items:center;gap:1rem;flex-wrap:wrap;">
        <div>
          <div style="font-size:1.5rem;font-weight:800;color:#f8fafc;">
            🛡️ {owner} / {repo_name}
            <a href="{url}" target="_blank" style="font-size:0.85rem;color:#6366f1;margin-left:0.75rem;text-decoration:none;">↗ View GitHub</a>
          </div>
          <div style="font-size:0.8rem;color:#64748b;margin-top:0.3rem;">
            Primary Language: <b>{primary_lang.upper()}</b> &nbsp;|&nbsp; 
            Age: <b>{meta.get('repo_age_days', 1)} days</b> &nbsp;|&nbsp; 
            Contributors: <b>{meta.get('contributor_count', 1)}</b> &nbsp;|&nbsp; 
            Cloned Commits: <b>{meta.get('total_commits', 0)}</b>
          </div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Risk Banner
    st.markdown(f"""
    <div class="risk-banner {banner_cls}">
      <div>
        <div style="font-size:0.7rem;font-weight:700;color:rgba(255,255,255,0.5);letter-spacing:0.15em;text-transform:uppercase;margin-bottom:0.5rem;">OVERALL REPOSITORY RISK PROFILE</div>
        <div class="risk-label" style="color:{risk_color};">{risk_emoji(max_risk_level == 'HIGH' and 'Critical' or (max_risk_level == 'MEDIUM' and 'High' or 'Low'))} {max_risk_level} RISK</div>
        <div style="font-size:0.85rem;color:rgba(255,255,255,0.55);margin-top:0.5rem;">
          Classified by V3 XGBoost model based on code quality & developer dynamics
        </div>
      </div>
      <div style="display:flex;gap:2rem;">
        <div class="health-ring">
          <div class="health-score" style="color:{risk_color};">{hs}</div>
          <div class="health-sub">Health Score<br>/100</div>
        </div>
        <div class="health-ring">
          <div class="health-score" style="color:#6366f1;">{avg_conf * 100:.0f}%</div>
          <div class="health-sub">Avg Confidence</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Metrics grid
    st.markdown('<div class="section-hdr">📊 Key Indicators</div>', unsafe_allow_html=True)
    avg_loc = df["loc"].mean() if "loc" in df.columns else 0
    avg_cc  = df["complexity"].mean() if "complexity" in df.columns else 0
    avg_mi  = df["maintainability_index"].mean() if "maintainability_index" in df.columns else 50
    bus_factor = meta.get("bus_factor", 1)

    indicators = [
        ("📁", f"{n_files:,}", "Files Analyzed"),
        ("🔴", f"{n_high:,}", "HIGH Risk Files"),
        ("🟡", f"{n_med:,}", "MEDIUM Risk Files"),
        ("🟢", f"{n_low:,}", "LOW Risk Files"),
        ("👥", f"{meta.get('contributor_count', 1)}", "Team Size"),
        ("🚌", f"{bus_factor}", "Bus Factor"),
        ("🔀", f"{avg_cc:.1f}", "Avg Complexity"),
        ("📐", f"{avg_mi:.1f}", "Avg Maintainability"),
    ]
    cols = st.columns(len(indicators))
    for col, (icon, val, lbl) in zip(cols, indicators):
        with col:
            st.markdown(f"""
            <div class="metric-card">
              <div class="metric-icon">{icon}</div>
              <div class="metric-val">{val}</div>
              <div class="metric-lbl">{lbl}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Plotly Visualizations
    col1, col2 = st.columns(2)
    with col1:
        # Pie chart
        cnt = pd.Series({"HIGH": n_high, "MEDIUM": n_med, "LOW": n_low})
        colors = {"HIGH": "#ef4444", "MEDIUM": "#f97316", "LOW": "#22c55e"}
        fig = go.Figure(go.Pie(
            labels=cnt.index, values=cnt.values,
            hole=0.5,
            marker=dict(colors=[colors[k] for k in cnt.index], line=dict(color="#0a0b0e", width=2)),
            textfont=dict(color="#f8fafc")
        ))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter", color="#94a3b8"),
            margin=dict(l=0, r=0, t=30, b=0),
            title="Risk Classification breakdown",
            showlegend=True,
            legend=dict(font=dict(color="#94a3b8"))
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Risk Score distribution bar chart (histogram)
        fig2 = px.histogram(
            df, x="risk_score_val", nbins=20,
            title="File Risk Score Profile",
            color_discrete_sequence=["#6366f1"]
        )
        fig2.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter", color="#94a3b8"),
            margin=dict(l=0, r=0, t=30, b=0),
            xaxis=dict(title="File Risk Score (0-100)", gridcolor="#1e293b", zerolinecolor="#1e293b"),
            yaxis=dict(title="File Count", gridcolor="#1e293b", zerolinecolor="#1e293b")
        )
        st.plotly_chart(fig2, use_container_width=True)

def render_interactive_table(df: pd.DataFrame):
    st.markdown('<div class="section-hdr">📋 Top High-Risk Files</div>', unsafe_allow_html=True)
    
    # Filter panels
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        search = st.text_input("🔍 Search File Path", placeholder="e.g. models/, api.py, routes", label_visibility="collapsed")
    with c2:
        risk_opts = ["HIGH", "MEDIUM", "LOW"]
        risk_sel = st.multiselect("Risk Class", risk_opts, default=risk_opts, label_visibility="collapsed")
    with c3:
        lang_opts = sorted(df["language"].unique().tolist()) if "language" in df.columns else []
        lang_sel = st.multiselect("Language Filter", lang_opts, default=lang_opts, label_visibility="collapsed")

    # Filter dataframe
    df_f = df.copy()
    if search:
        df_f = df_f[df_f["file_path"].str.contains(search, case=False, na=False)]
    if risk_sel:
        df_f = df_f[df_f["predicted_risk"].isin(risk_sel)]
    if lang_sel:
        df_f = df_f[df_f["language"].isin(lang_sel)]

    # Dynamic sort
    df_f = df_f.sort_values(by=["risk_score_val", "confidence_score"], ascending=[False, False])
    
    # Interactive view
    cols_to_show = {
        "file_path": "File Path",
        "predicted_risk": "Risk Class",
        "risk_score_val": "Risk Score",
        "confidence_score": "Confidence",
        "complexity": "Complexity",
        "maintainability_index": "Maintainability",
        "modification_count": "Modifications",
        "loc": "LOC",
        "contributor_count": "Contributors"
    }
    
    available_cols = [c for c in cols_to_show.keys() if c in df_f.columns]
    df_display = df_f[available_cols].copy().rename(columns=cols_to_show)
    df_display["Confidence"] = (df_display["Confidence"] * 100).round(1).astype(str) + "%"
    df_display["Risk Score"] = df_display["Risk Score"].round(1)
    df_display = df_display.reset_index(drop=True)
    df_display.index += 1

    st.dataframe(df_display, use_container_width=True, height=350)
    return df_f

def render_file_explainer(df: pd.DataFrame):
    st.markdown('<div class="section-hdr">🔎 File Risk Explainer</div>', unsafe_allow_html=True)
    
    file_paths = df["file_path"].tolist() if "file_path" in df.columns else []
    if not file_paths:
        st.info("No files found to inspect.")
        return
        
    selected_file = st.selectbox("Select a file to inspect details", file_paths, label_visibility="collapsed")
    
    # Get row details
    row = df[df["file_path"] == selected_file].iloc[0]
    risk_class = row["predicted_risk"]
    conf = row["confidence_score"]
    score = row["risk_score_val"]
    
    color = {"HIGH": "#ef4444", "MEDIUM": "#f97316", "LOW": "#22c55e"}.get(risk_class, "#6366f1")

    # Positives/Negatives Drivers
    pos_drivers = []
    neg_drivers = []

    # LOC Driver
    loc = int(row.get("loc", 0))
    if loc > 500: pos_drivers.append(f"Lines of Code is extremely high ({loc:,} lines)")
    elif loc < 50: neg_drivers.append(f"Compact file footprint ({loc:,} lines)")

    # Complexity Driver
    cc = float(row.get("complexity", 1))
    if cc > 25: pos_drivers.append(f"Cyclomatic complexity is elevated ({cc:.0f})")
    elif cc < 5: neg_drivers.append(f"Simple control flow structures ({cc:.0f})")

    # Maintainability Driver
    mi = float(row.get("maintainability_index", 50))
    if mi < 40: pos_drivers.append(f"Maintainability Index is critically low ({mi:.1f})")
    elif mi > 75: neg_drivers.append(f"Excellent Maintainability score ({mi:.1f})")

    # Modifications Driver
    mods = int(row.get("modification_count", 0))
    if mods > 15: pos_drivers.append(f"High modification frequency ({mods} changes)")
    elif mods < 3: neg_drivers.append(f"Stable codebase addition ({mods} modifications)")

    # Ownership concentration
    own = float(row.get("ownership_concentration", 1.0))
    if own > 0.8: pos_drivers.append(f"High ownership concentration ({own * 100:.0f}%) — single dev dominance")
    elif own < 0.4: neg_drivers.append(f"Distributed module updates (concentration is {own * 100:.0f}%)")

    # Fallbacks if list is empty
    if not pos_drivers: pos_drivers.append("No critical positive risk drivers observed")
    if not neg_drivers: neg_drivers.append("No critical negative risk drivers observed")

    # Build human explanation
    explanation = f"This file is classified as <b>{risk_class}</b> risk because:"
    
    # Render layout
    d1, d2, d3 = st.columns([1.5, 1, 1])
    with d1:
        st.markdown(f"""
        <div class="panel">
          <div class="panel-title">📄 {selected_file.split('/')[-1]}</div>
          <div style="font-family:monospace;font-size:0.75rem;color:#64748b;word-break:break-all;">{selected_file}</div>
          <hr style="margin:0.75rem 0;">
          <div style="display:flex;justify-content:space-between;margin-bottom:0.5rem;">
            <span style="color:#94a3b8;font-size:0.8rem;">Risk Score</span>
            <span style="color:{color};font-weight:800;font-size:1.1rem;">{score:.0f}/100</span>
          </div>
          <div style="display:flex;justify-content:space-between;margin-bottom:0.5rem;">
            <span style="color:#94a3b8;font-size:0.8rem;">Risk Class</span>
            {badge_html(risk_class == 'HIGH' and 'Critical' or (risk_class == 'MEDIUM' and 'High' or 'Low'))}
          </div>
          <div style="display:flex;justify-content:space-between;">
            <span style="color:#94a3b8;font-size:0.8rem;">Trust Gate Level</span>
            {trust_html(conf)}
          </div>
        </div>""", unsafe_allow_html=True)

    with d2:
        fields = [
            ("Lines of Code", f"{int(row.get('loc',0)):,}"),
            ("Complexity", f"{float(row.get('complexity',0)):.1f}"),
            ("Maintainability", f"{float(row.get('maintainability_index',0)):.1f}"),
            ("Commit Count", f"{int(row.get('commit_count',0)):,}"),
            ("Modifications", f"{int(row.get('modification_count',0)):,}"),
            ("Contributors", f"{int(row.get('contributor_count',0)):,}"),
        ]
        html = '<div class="panel"><div class="panel-title">📊 File Code Metrics</div>'
        for lbl2, val2 in fields:
            html += f"""
            <div style="display:flex;justify-content:space-between;padding:0.35rem 0;border-bottom:1px solid #1e293b;">
              <span style="color:#64748b;font-size:0.8rem;">{lbl2}</span>
              <span style="color:#f8fafc;font-weight:600;font-size:0.85rem;">{val2}</span>
            </div>"""
        html += "</div>"
        st.markdown(html, unsafe_allow_html=True)

    with d3:
        html = '<div class="panel"><div class="panel-title">🛡️ Critical Risk Drivers</div>'
        html += '<div style="font-size:0.75rem;color:#ef4444;font-weight:700;margin-bottom:0.3rem;">Positive Drivers (Increases Risk):</div>'
        for d in pos_drivers[:3]:
            html += f'<div style="font-size:0.75rem;color:#cbd5e1;margin-bottom:0.25rem;">• {d}</div>'
        html += '<div style="font-size:0.75rem;color:#22c55e;font-weight:700;margin-top:0.6rem;margin-bottom:0.3rem;">Negative Drivers (Reduces Risk):</div>'
        for d in neg_drivers[:3]:
            html += f'<div style="font-size:0.75rem;color:#cbd5e1;margin-bottom:0.25rem;">• {d}</div>'
        html += "</div>"
        st.markdown(html, unsafe_allow_html=True)

def render_tech_debt_hotspots(df: pd.DataFrame):
    st.markdown('<div class="section-hdr">🔥 Technical Debt Hotspots</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:0.8rem;color:#64748b;margin-bottom:0.75rem;">Top 25 Refactoring Candidates ranked by Risk Score & Priority.</div>', unsafe_allow_html=True)
    
    # Sort and rank top 25 candidate refactoring files
    top25 = df.sort_values(by="risk_score_val", ascending=False).head(25).copy().reset_index(drop=True)
    
    hot_headers = [
        Paragraph("<b>Rank</b>", ParagraphStyle('H', fontSize=9, textColor=colors.HexColor('#94a3b8'))),
        Paragraph("<b>File Path</b>", ParagraphStyle('H', fontSize=9, textColor=colors.HexColor('#94a3b8'))),
        Paragraph("<b>Risk Score</b>", ParagraphStyle('H', fontSize=9, textColor=colors.HexColor('#94a3b8'))),
        Paragraph("<b>Confidence</b>", ParagraphStyle('H', fontSize=9, textColor=colors.HexColor('#94a3b8'))),
        Paragraph("<b>Priority</b>", ParagraphStyle('H', fontSize=9, textColor=colors.HexColor('#94a3b8')))
    ]
    
    # Build dataframe for presentation
    hotspots_df = []
    for idx, (_, row) in enumerate(top25.iterrows(), 1):
        priority = row["tech_debt_priority"]
        hotspots_df.append({
            "Rank": idx,
            "File Path": row["file_path"],
            "Risk Score": row["risk_score_val"],
            "Confidence": f"{row['confidence_score'] * 100:.1f}%",
            "Technical Debt Priority": priority
        })
        
    st.dataframe(pd.DataFrame(hotspots_df).set_index("Rank"), use_container_width=True, height=350)

def render_trust_gate_details(df: pd.DataFrame):
    st.markdown('<div class="section-hdr">🔒 Trust Gate Assessment</div>', unsafe_allow_html=True)
    
    # Calculate file-level trust
    confs = df["confidence_score"] * 100
    n_high = (confs >= 90).sum()
    n_good = ((confs >= 80) & (confs < 90)).sum()
    n_mod  = ((confs >= 70) & (confs < 80)).sum()
    n_rev  = (confs < 70).sum()
    total = len(df)
    
    avg_conf = confs.mean()
    gate_level = trust_level(avg_conf)
    
    t1, t2 = st.columns([1, 1])
    with t1:
        st.markdown(f"""
        <div class="panel">
          <div class="panel-title">Verification Outcome</div>
          <div style="font-size:1.6rem;font-weight:900;color:#f8fafc;margin-bottom:0.5rem;">{gate_level}</div>
          <div style="font-size:0.8rem;color:#64748b;">
            Calculated across all predicted files. Trust represents the model's confidence in identifying file risk structures.
          </div>
          <hr style="margin:1rem 0;">
          <div style="display:flex;justify-content:space-between;margin-bottom:0.4rem;font-size:0.85rem;">
            <span>Average Prediction Confidence:</span>
            <b>{avg_conf:.1f}%</b>
          </div>
        </div>
        """, unsafe_allow_html=True)
        
    with t2:
        trust_table = pd.DataFrame([
            {"Trust Bin": "HIGH TRUST (≥90%)", "Files": f"{n_high:,}", "Pct": f"{n_high/total*100:.1f}%"},
            {"Trust Bin": "GOOD TRUST (80-90%)", "Files": f"{n_good:,}", "Pct": f"{n_good/total*100:.1f}%"},
            {"Trust Bin": "MODERATE TRUST (70-80%)", "Files": f"{n_mod:,}", "Pct": f"{n_mod/total*100:.1f}%"},
            {"Trust Bin": "MANUAL REVIEW (<70%)", "Files": f"{n_rev:,}", "Pct": f"{n_rev/total*100:.1f}%"},
        ])
        st.dataframe(trust_table.set_index("Trust Bin"), use_container_width=True)

def render_repo_similarity(df: pd.DataFrame, repo_name: str):
    st.markdown('<div class="section-hdr">🤝 Repository Similarity Mapping</div>', unsafe_allow_html=True)
    
    centroids = get_training_centroids()
    if centroids is None:
        st.info("Training dataset (`ml_dataset_v3.csv`) not found — similarity mapping unavailable.")
        return
        
    # Compute analyzed repo centroid
    ext_centroid = df[NUM_FEATS].mean().values.reshape(1, -1)
    
    sims = {}
    for t_repo in centroids.index:
        t_vec = centroids.loc[t_repo].values.reshape(1, -1)
        sim = cosine_similarity(ext_centroid, t_vec)[0][0]
        sims[t_repo] = round(float(sim), 4)
        
    top3 = sorted(sims.items(), key=lambda x: x[1], reverse=True)[:3]
    
    s1, s2 = st.columns([1, 1.2])
    with s1:
        st.markdown(f"""
        <div class="panel">
          <div class="panel-title">Similarity Analysis</div>
          <div style="font-size:0.8rem;color:#64748b;margin-bottom:0.75rem;">
            This comparison calculates similarity against the 22 software engineering repositories in the model's training set. 
            Highly similar training profiles indicate prediction patterns are highly aligned with verified baselines.
          </div>
          <hr style="margin:0.75rem 0;">
          <div style="font-size:0.85rem;color:#a0aec0;">
            Prediction Reliability: <b>{"HIGH" if top3[0][1] >= 0.85 else "MODERATE"}</b>
          </div>
        </div>
        """, unsafe_allow_html=True)
        
    with s2:
        sim_table = pd.DataFrame([
            {"Rank": i+1, "Similar Training Repo": name, "Cosine Similarity": f"{val * 100:.1f}%"}
            for i, (name, val) in enumerate(top3)
        ])
        st.dataframe(sim_table.set_index("Rank"), use_container_width=True)
    return pd.DataFrame([
        {"external_repo": repo_name, "rank": i+1, "similar_training_repo": name, "cosine_similarity": val}
        for i, (name, val) in enumerate(top3)
    ])

def render_executive_summary(df: pd.DataFrame, owner: str, repo_name: str, meta: dict, sim_df: pd.DataFrame):
    st.markdown('<div class="section-hdr">📋 Executive Assessment & Recommendations</div>', unsafe_allow_html=True)
    
    hs = health_score(df)
    n_files = len(df)
    n_high = int((df["predicted_risk"] == "HIGH").sum())
    n_med  = int((df["predicted_risk"] == "MEDIUM").sum())
    
    # Model average conf
    avg_conf = float(df["confidence_score"].mean())
    gate_level = trust_level(avg_conf)
    
    # Top critical file
    top1 = df.sort_values(by="risk_score_val", ascending=False).iloc[0] if len(df) > 0 else None
    top1_path = top1["file_path"].split('/')[-1] if top1 is not None else "N/A"
    top1_score = top1["risk_score_val"] if top1 is not None else 0
    
    # Recommendations text
    recommendations = []
    if n_high > 0:
        recommendations.append(f"Refactor the top flagged candidates (specifically <b>{top1_path}</b> with a critical score of {top1_score:.0f}).")
    if meta.get("bus_factor", 1) <= 2:
        recommendations.append("Distribute knowledge: core developer ownership concentration is high, exposing the team to a high bus factor risk.")
    recommendations.append("Continuous checks: Integrate file risk analysis checks in the pull-request pipeline to safeguard new changes.")

    e1, e2, e3 = st.columns(3)
    with e1:
        st.markdown(f"""
        <div class="panel">
          <div class="panel-title">🛡️ Risk Assessment Summary</div>
          <div style="font-size:0.85rem;line-height:1.5;color:#cbd5e1;">
            • Health Score: <b>{hs}/100</b><br>
            • Risk Distribution: <b>{n_high:,} HIGH</b>, <b>{n_med:,} MEDIUM</b> files<br>
            • Trust Classification: <b>{gate_level}</b><br>
            • Codebase Scale: <b>{n_files:,} files analyzed</b>
          </div>
        </div>
        """, unsafe_allow_html=True)
        
    with e2:
        st.markdown(f"""
        <div class="panel">
          <div class="panel-title">🔥 Critical Technical Debt Areas</div>
          <div style="font-size:0.85rem;line-height:1.5;color:#cbd5e1;">
            • Most Critical Refactoring Target:<br>&nbsp;&nbsp;<span style="color:#ef4444;font-family:monospace;font-size:0.75rem;">{top1_path} ({top1_score:.0f} Risk)</span><br>
            • Top similarity reference match:<br>&nbsp;&nbsp;<b>{sim_df.iloc[0]['similar_training_repo'] if sim_df is not None and not sim_df.empty else 'N/A'} ({sim_df.iloc[0]['cosine_similarity']*100:.1f}% Similarity)</b>
          </div>
        </div>
        """, unsafe_allow_html=True)
        
    with e3:
        rec_html = f"""<div class="panel"><div class="panel-title">💡 Strategic Next Steps</div>
        <div style="font-size:0.8rem;line-height:1.4;color:#cbd5e1;">"""
        for r in recommendations:
            rec_html += f"• {r}<br><br>"
        rec_html += "</div></div>"
        st.markdown(rec_html, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# DASHBOARD ENTRYPOINT
# ══════════════════════════════════════════════════════════════════════════════
def main():
    # Navbar
    st.markdown("""
    <div class="nav-bar">
      <div class="nav-logo">🛡️ Repository Risk Intelligence
        <span class="nav-badge">PRODUCTION</span>
      </div>
      <div style="font-size:0.75rem;color:#475569;">V3 Leakage-Free Pipeline · Trust Gate & Explainability</div>
    </div>
    """, unsafe_allow_html=True)

    # Initialize states
    if "analysis_res" not in st.session_state:
        st.session_state.analysis_res = None
    if "analysis_url" not in st.session_state:
        st.session_state.analysis_url = ""
    if "analysis_owner" not in st.session_state:
        st.session_state.analysis_owner = ""
    if "analysis_repo" not in st.session_state:
        st.session_state.analysis_repo = ""

    # Home/Search Input View
    if st.session_state.analysis_res is None:
        st.markdown("""
        <div class="hero">
          <div class="hero-title">Repository Risk Intelligence Platform</div>
          <div class="hero-sub">
            Analyze codebases directly from public GitHub URLs. Predict file-level risk, identify technical debt hot-spots, and generate executive-ready reports.
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="url-card">', unsafe_allow_html=True)
        ci, cb = st.columns([3, 1])
        with ci:
            url = st.text_input(
                "GitHub Repository URL",
                value=st.session_state.analysis_url or "https://github.com/pallets/flask",
                placeholder="https://github.com/owner/repository",
                label_visibility="collapsed"
            )
        with cb:
            click = st.button("🚀 Analyze Repository")
            
        st.markdown("""
        <div style="margin-top:1rem;font-size:0.75rem;color:#475569;text-align:center;">
          ✓ Fully leakage-free validation &nbsp;·&nbsp; ✓ v3 Composite Labels &nbsp;·&nbsp; ✓ XGBoost Engine &nbsp;·&nbsp; ✓ PDF Export Report
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Preloaded examples
        st.markdown('<br>', unsafe_allow_html=True)
        st.markdown('<div style="text-align:center;font-size:0.75rem;color:#475569;margin-bottom:0.5rem;">Pre-cached external validation templates:</div>', unsafe_allow_html=True)
        ex_cols = st.columns(3)
        examples = [
            ("pallets/flask", "https://github.com/pallets/flask"),
            ("streamlit/streamlit", "https://github.com/streamlit/streamlit"),
            ("calcom/cal.com", "https://github.com/calcom/cal.com"),
        ]
        for ec, (name, eurl) in zip(ex_cols, examples):
            with ec:
                if st.button(name, key=f"ex_{name}", use_container_width=True):
                    st.session_state.analysis_url = eurl
                    st.rerun()

        if click:
            if not url.strip():
                st.error("Please enter a GitHub repository URL.")
                return
            owner, repo_name = parse_github_url(url)
            if not owner:
                st.error("Invalid GitHub URL structure. Use format: https://github.com/owner/repo")
                return

            st.session_state.analysis_url  = f"https://github.com/{owner}/{repo_name}"
            st.session_state.analysis_owner     = owner
            st.session_state.analysis_repo = repo_name

            # Run pipeline
            st.markdown(f"""
            <div style="background:#111318;border:1px solid #1e293b;border-radius:12px;padding:1.25rem 1.5rem;margin-top:1.5rem;">
              <div style="font-weight:700;color:#f8fafc;margin-bottom:0.75rem;">
                ⚙️ Executing pipeline for <span style="color:#6366f1;">{owner}/{repo_name}</span>
              </div>
            """, unsafe_allow_html=True)

            prog = st.progress(0)
            status = st.empty()
            log_box = st.empty()

            stages = [
                "Cloning repository...",
                "Running Quality extraction...",
                "Running Commit features analysis...",
                "Engineering feature matrix...",
                "Loading production model...",
                "Generating predictions...",
                "Finalizing summary..."
            ]

            for i, s in enumerate(stages):
                status.markdown(f'<div style="font-size:0.85rem;color:#94a3b8;">{s}</div>', unsafe_allow_html=True)
                prog.progress((i+1) / len(stages) * 0.9)
                if i == 0: break
                
            res = run_pipeline(owner, repo_name, log_box)
            prog.progress(1.0)
            status.empty()

            st.markdown("</div>", unsafe_allow_html=True)

            if res is None:
                st.error("Pipeline run failed. Please check logs.")
                return
                
            st.session_state.analysis_res = res
            st.rerun()

    else:
        # Results View
        res = st.session_state.analysis_res
        df = res["df"]
        meta = res["meta"]
        url = st.session_state.analysis_url
        owner = st.session_state.analysis_owner
        repo_name = st.session_state.analysis_repo

        # Top dashboard panel reset controls
        c_space, c_reset = st.columns([6, 1])
        with c_reset:
            if st.button("🔄 New Analysis"):
                st.session_state.analysis_res  = None
                st.session_state.analysis_url  = ""
                st.session_state.analysis_owner     = ""
                st.session_state.analysis_repo = ""
                st.rerun()

        # Render panels
        render_health_dashboard(df, owner, repo_name, meta, url)
        
        # Interactive table
        filtered_df = render_interactive_table(df)
        
        # File Explainer
        render_file_explainer(filtered_df)
        
        # Technical Debt hotspots
        render_tech_debt_hotspots(df)
        
        # Trust Gate details
        render_trust_gate_details(df)
        
        # Repo Similarity
        sim_df = render_repo_similarity(df, repo_name)
        
        # Executive Summary & Recommendations
        render_executive_summary(df, owner, repo_name, meta, sim_df)
        
        # PDF EXPORT BUTTON
        st.markdown('<div class="section-hdr" style="margin-top:1.5rem;">📥 Export Report</div>', unsafe_allow_html=True)
        pdf_col, _ = st.columns([1, 3])
        with pdf_col:
            try:
                # Calculate required stats for generator
                hs = health_score(df)
                n_high = int((df["predicted_risk"] == "HIGH").sum())
                n_med  = int((df["predicted_risk"] == "MEDIUM").sum())
                n_files = len(df)
                
                overall_risk = "LOW"
                if n_high > 0 or n_med > 0:
                    pct_high = n_high / n_files
                    pct_med  = n_med / n_files
                    if pct_high >= 0.15: overall_risk = "HIGH"
                    elif pct_high > 0 or pct_med >= 0.3: overall_risk = "MEDIUM"
                    
                avg_conf = float(df["confidence_score"].mean())
                
                pdf_data = generate_pdf_report(
                    df=df,
                    owner=owner,
                    repo_name=repo_name,
                    health_score=hs,
                    overall_risk=overall_risk,
                    avg_confidence=avg_conf,
                    sim_df=sim_df
                )
                
                st.download_button(
                    label="📄 Download Executive Assessment PDF",
                    data=pdf_data,
                    file_name=f"{repo_name}_risk_assessment.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
            except Exception as ex:
                st.error(f"Error compiling PDF report: {ex}")
                traceback.print_exc()

if __name__ == "__main__":
    main()
