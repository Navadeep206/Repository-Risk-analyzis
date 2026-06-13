#!/usr/bin/env python3
"""
Repository Service for Phase 10.
Manages repository indexing, metric loading, and uploaded codebases.
"""

import os
import sys
import zipfile
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from config import RAW_DIR, PROCESSED_DIR, ensure_dirs_exist

class RepositoryService:
    """
    Manages access to repository metadata, commits, modifications, and quality metrics.
    """
    def __init__(self):
        ensure_dirs_exist()
        self.quality_file = os.path.join(PROCESSED_DIR, "quality_metrics.csv")
        self.repos_metadata = os.path.join(PROCESSED_DIR, "..", "repositories_metadata.csv")
        
    def list_repositories(self) -> List[str]:
        """Returns the list of standard mined repositories."""
        if os.path.exists(self.quality_file):
            df = pd.read_csv(self.quality_file)
            return sorted(df["repository_name"].dropna().unique().tolist())
        return ["click", "databases", "axios", "express", "jinja", "redux"]
        
    def load_metrics(self, repo_name: str) -> pd.DataFrame:
        """Loads quality metrics for files in a repository."""
        if os.path.exists(self.quality_file):
            df = pd.read_csv(self.quality_file)
            return df[df["repository_name"] == repo_name].copy()
        # Fallback if no file exists
        return pd.DataFrame()
        
    def load_summary(self, repo_name: str) -> Dict[str, any]:
        """Loads high-level summary statistics for a repository."""
        # Try metadata file first
        if os.path.exists(self.repos_metadata):
            df_meta = pd.read_csv(self.repos_metadata)
            match = df_meta[df_meta["repository_name"] == repo_name]
            if not match.empty:
                row = match.iloc[0]
                return {
                    "repository_name": repo_name,
                    "language": str(row.get("repository_language", "javascript")),
                    "loc": int(row.get("repository_size", 0)), # size in lines or bytes
                    "commits_count": int(row.get("total_commits", 0)),
                    "contributors_count": int(row.get("total_contributors", 0)),
                    "repository_age_days": int(row.get("repository_age", 1))
                }
                
        # Fallback: compute from raw files
        df_quality = self.load_metrics(repo_name)
        total_loc = df_quality["loc"].sum() if not df_quality.empty and "loc" in df_quality.columns else 0
        
        commits_file = os.path.join(RAW_DIR, f"{repo_name}_commits.csv")
        commits_count = 0
        contributors_count = 0
        age_days = 1
        
        if os.path.exists(commits_file):
            try:
                df_commits = pd.read_csv(commits_file)
                commits_count = len(df_commits)
                contributors_count = df_commits["author_email"].nunique()
                dates = pd.to_datetime(df_commits["committer_date"], errors="coerce")
                if not dates.dropna().empty:
                    age_days = max(1, (dates.max() - dates.min()).days)
            except Exception:
                pass
                
        return {
            "repository_name": repo_name,
            "language": df_quality["language"].iloc[0] if not df_quality.empty else "javascript",
            "loc": total_loc,
            "commits_count": commits_count,
            "contributors_count": contributors_count,
            "repository_age_days": age_days
        }

    def process_uploaded_zip(self, zip_file) -> Tuple[pd.DataFrame, Dict[str, any]]:
        """
        Extracts a zipped repository, analyzes files (LOC, counts),
        and simulates complexity/maintainability scores.
        
        Returns:
            A tuple of (df_quality, summary_dict).
        """
        temp_dir = os.path.join(PROCESSED_DIR, "..", "intermediate", "uploaded_temp")
        os.makedirs(temp_dir, exist_ok=True)
        
        zip_path = os.path.join(temp_dir, "temp_uploaded.zip")
        with open(zip_path, "wb") as f:
            f.write(zip_file.getbuffer())
            
        file_records = []
        languages_count = {}
        
        with zipfile.ZipFile(zip_path, "r") as ref:
            for name in ref.namelist():
                if name.endswith("/") or "__pycache__" in name or ".git" in name:
                    continue
                
                # Check extension for language
                ext = os.path.splitext(name)[1].lower()
                lang = "other"
                if ext in [".py", ".pyw"]:
                    lang = "python"
                elif ext in [".js", ".jsx", ".mjs"]:
                    lang = "javascript"
                elif ext in [".ts", ".tsx"]:
                    lang = "typescript"
                elif ext in [".html", ".htm"]:
                    lang = "html"
                elif ext in [".css", ".scss"]:
                    lang = "css"
                    
                if lang == "other" and ext not in [".md", ".json", ".txt", ".yml", ".yaml"]:
                    # Skip binary or irrelevant files
                    continue
                    
                try:
                    with ref.open(name) as f:
                        content = f.read().decode("utf-8", errors="ignore")
                        lines = content.splitlines()
                        loc = len(lines)
                except Exception:
                    loc = 0
                    
                if loc == 0:
                    continue
                    
                # Simulate complexity and maintainability index based on file length
                # Basic code styling: longer files tend to be more complex
                complexity = max(1, int(np.random.poisson(loc * 0.04) + 1))
                maintainability = float(np.clip(100.0 - (loc * 0.08) - np.random.normal(5, 2), 10.0, 100.0))
                
                file_records.append({
                    "repository_name": "uploaded_repo",
                    "file_path": name,
                    "language": lang,
                    "loc": loc,
                    "complexity": complexity,
                    "maintainability_index": maintainability,
                    "commit_count": max(1, int(np.random.poisson(3))),
                    "modification_count": max(1, int(np.random.poisson(5))),
                    "contributor_count": max(1, int(np.random.poisson(2)))
                })
                
                languages_count[lang] = languages_count.get(lang, 0) + loc
                
        # Clean up temp file
        if os.path.exists(zip_path):
            os.remove(zip_path)
            
        df_quality = pd.DataFrame(file_records)
        
        if df_quality.empty:
            return pd.DataFrame(), {}
            
        dominant_lang = max(languages_count, key=languages_count.get) if languages_count else "python"
        
        summary = {
            "repository_name": "Uploaded Codebase",
            "language": dominant_lang,
            "loc": int(df_quality["loc"].sum()),
            "commits_count": int(df_quality["commit_count"].sum()),
            "contributors_count": int(df_quality["contributor_count"].max()),
            "repository_age_days": 180 # Assumed default age
        }
        
        # Add engineered features expected by production preprocessor
        df_quality["repository_age_days"] = summary["repository_age_days"]
        df_quality["commit_frequency"] = df_quality["commit_count"] / df_quality["repository_age_days"]
        
        return df_quality, summary
