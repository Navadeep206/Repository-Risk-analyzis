import os
import re
import math
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timezone

HISTORY_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data/runtime_history.csv")

def fetch_github_metadata(owner: str, repo: str) -> dict:
    """
    Fetches repository metadata from the public GitHub API.
    Returns a dictionary of raw metadata or empty dict if the request fails.
    """
    try:
        url = f"https://api.github.com/repos/{owner}/{repo}"
        headers = {"Accept": "application/vnd.github.v3+json"}
        response = requests.get(url, headers=headers, timeout=5)
        if response.status_code == 200:
            return response.json()
    except Exception:
        pass
    return {}

def estimate_repository_stats(meta: dict) -> dict:
    """
    Estimates repository size, stars, forks, language, commit count,
    contributor count, repository age in days, and file count based on GitHub metadata.
    """
    if not meta:
        # Graceful fallback values
        return {
            "size_kb": 10000,
            "stars": 50,
            "forks": 10,
            "language": "python",
            "age_days": 365,
            "est_files": 250,
            "est_commits": 300,
            "est_contributors": 8
        }
        
    size_kb = meta.get("size", 1000)
    stars = meta.get("stargazers_count", 0)
    forks = meta.get("forks_count", 0)
    lang = meta.get("language", "python")
    if not lang:
        lang = "python"
        
    # Calculate age
    created_at_str = meta.get("created_at")
    age_days = 365
    if created_at_str:
        try:
            # Match ISO format e.g. 2019-04-12T17:10:43Z
            created_at_dt = datetime.strptime(created_at_str, "%Y-%m-%dT%H:%M:%SZ")
            age_days = max(1, (datetime.utcnow() - created_at_dt).days)
        except Exception:
            pass

    # Heuristics for estimates (accounting for depth=200 clone for analyzed stats)
    # File count estimate: general scale based on size
    est_files = max(10, min(100000, int(size_kb * 0.05)))
    
    # Total commits estimate in repo (e.g. forks and stars reflect commit activity)
    est_commits = max(10, int(forks * 2.5 + age_days * 0.1 + 20))
    
    # Contributor count estimate
    est_contributors = max(1, int(math.sqrt(stars) * 0.8 + forks * 0.05 + 2))
    
    return {
        "size_kb": size_kb,
        "stars": stars,
        "forks": forks,
        "language": lang.lower(),
        "age_days": age_days,
        "est_files": est_files,
        "est_commits": est_commits,
        "est_contributors": est_contributors
    }

def predict_analysis_runtime(files: int, commits: int, contributors: int) -> float:
    """
    Predicts analysis runtime in seconds using previous runs (regression) or baseline heuristics.
    """
    # Baseline heuristic:
    # - 3.0s base for cloning/setup overhead
    # - 0.015s per file (static analysis and ast parsing)
    # - 0.02s per commit (pydriller traversal is the bottleneck)
    # - 0.05s per contributor (entropy calculation and author grouping)
    baseline = 3.0 + (files * 0.015) + (commits * 0.02) + (contributors * 0.05)
    
    if not os.path.exists(HISTORY_FILE):
        return max(3.0, baseline)
        
    try:
        df = pd.read_csv(HISTORY_FILE)
        if len(df) >= 3:
            # Prepare regression inputs: X = [files, commits, contributors], y = runtime
            X = df[["files", "commits", "contributors"]].values
            y = df["runtime"].values
            
            # Solve normal equations: beta = (X^T X)^(-1) X^T y
            # Add bias column
            X_bias = np.hstack([np.ones((X.shape[0], 1)), X])
            beta, residuals, rank, s = np.linalg.lstsq(X_bias, y, rcond=None)
            
            # Predict
            pred = beta[0] + beta[1] * files + beta[2] * commits + beta[3] * contributors
            # Sanity bound: predicted time should not be less than 30% of baseline and at least 3.0s
            return float(max(3.0, max(baseline * 0.3, pred)))
    except Exception:
        pass
        
    return max(3.0, baseline)

def save_run_history(repo_name: str, files: int, commits: int, contributors: int, runtime: float):
    """
    Saves a successful run metrics to the runtime database.
    """
    os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)
    row = pd.DataFrame([{
        "repository_name": repo_name,
        "files": files,
        "commits": commits,
        "contributors": contributors,
        "runtime": round(runtime, 2),
        "timestamp": datetime.now(timezone.utc).isoformat()
    }])
    
    if not os.path.exists(HISTORY_FILE):
        row.to_csv(HISTORY_FILE, index=False)
    else:
        row.to_csv(HISTORY_FILE, mode="a", header=False, index=False)
