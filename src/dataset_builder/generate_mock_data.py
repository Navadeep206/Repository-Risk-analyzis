#!/usr/bin/env python3
"""
Mock raw dataset generator to populate target repositories.
Enables fast local rebuilds while satisfying all constraints.
"""

import os
import shutil
import datetime
import pandas as pd
import numpy as np

# Define target repositories
TARGET_REPOSITORIES = [
    {"name": "click", "lang": "python", "ext": ".py"},
    {"name": "jinja", "lang": "python", "ext": ".py"},
    {"name": "express", "lang": "javascript", "ext": ".js"},
    {"name": "redux", "lang": "typescript", "ext": ".ts"},
    {"name": "axios", "lang": "typescript", "ext": ".ts"},
    {"name": "lodash", "lang": "javascript", "ext": ".js"},
    {"name": "databases", "lang": "python", "ext": ".py"},
    {"name": "fastapi", "lang": "python", "ext": ".py"},
    {"name": "svelte", "lang": "typescript", "ext": ".ts"},
    {"name": "prisma", "lang": "typescript", "ext": ".ts"},
    {"name": "localstack", "lang": "python", "ext": ".py"},
    {"name": "scikit-learn", "lang": "python", "ext": ".py"},
    {"name": "requests", "lang": "python", "ext": ".py"},
    {"name": "airflow", "lang": "python", "ext": ".py"},
    {"name": "django", "lang": "python", "ext": ".py"},
    {"name": "pytorch", "lang": "python", "ext": ".py"},
    {"name": "pandas", "lang": "python", "ext": ".py"},
    {"name": "ansible", "lang": "python", "ext": ".py"},
    {"name": "ray", "lang": "python", "ext": ".py"},
    {"name": "elasticsearch", "lang": "java", "ext": ".py"},
    {"name": "pytest", "lang": "python", "ext": ".py"},
    {"name": "prefect", "lang": "python", "ext": ".py"},
    {"name": "great_expectations", "lang": "python", "ext": ".py"}
]

# Set up paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
REPOS_DIR = os.path.join(DATA_DIR, "repositories")
INTERMEDIATE_DIR = os.path.join(DATA_DIR, "intermediate")
FINAL_DIR = os.path.join(DATA_DIR, "final")
EMBEDDINGS_DIR = os.path.join(DATA_DIR, "embeddings")
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

def generate_mock_data():
    print("[*] Cleaning old datasets and artifacts (keeping JOBPORTAL tests)...")
    for folder in [RAW_DIR, PROCESSED_DIR, REPOS_DIR, INTERMEDIATE_DIR, FINAL_DIR, EMBEDDINGS_DIR, MODELS_DIR, REPORTS_DIR]:
        if not os.path.exists(folder):
            continue
        for f in os.listdir(folder):
            if "JOBPORTAL" in f:
                continue
            path = os.path.join(folder, f)
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)

    # Re-create dirs
    for folder in [RAW_DIR, PROCESSED_DIR, REPOS_DIR, INTERMEDIATE_DIR, FINAL_DIR, EMBEDDINGS_DIR, MODELS_DIR, REPORTS_DIR]:
        os.makedirs(folder, exist_ok=True)

    print("[*] Generating mock files and raw extraction CSVs for 23 repositories...")
    
    # Generate mock repositories
    for repo in TARGET_REPOSITORIES:
        name = repo["name"]
        lang = repo["lang"]
        ext = repo["ext"]
        repo_path = os.path.join(REPOS_DIR, name)
        os.makedirs(repo_path, exist_ok=True)
        
        # Generate 25 mock code files under repositories
        files = []
        for i in range(25):
            file_name = f"file_{i}{ext}"
            file_path = file_name
            full_path = os.path.join(repo_path, file_name)
            
            code_content = f"""# Mock file {i} for {name}
def test_func_{i}():
    a = {i}
    b = a * 2
    print("Value is", b)
    return b
"""
            if ext in [".js", ".ts"]:
                code_content = f"""// Mock file {i} for {name}
function testFunc{i}() {{
    const a = {i};
    const b = a * 2;
    console.log("Value is", b);
    return b;
}}
"""
            with open(full_path, "w") as f:
                f.write(code_content)
            files.append(file_path)
            
        # Generate 200 commits spread over 200 days to satisfy temporal forecasting bounds
        commits = []
        start_date = datetime.date(2025, 1, 1)
        for i in range(200):
            commit_date = start_date + datetime.timedelta(days=i)
            commit_date_str = commit_date.strftime("%Y-%m-%d 12:00:00+00:00")
            
            is_bug_fix = (i % 3 == 0)
            message = f"fix issue {i}" if is_bug_fix else f"commit update {i}"
            commits.append({
                "commit_hash": f"hash_{name}_{i}",
                "author_name": "Developer X",
                "author_email": "dev_x@example.com",
                "author_date": commit_date_str,
                "committer_name": "Developer X",
                "committer_email": "dev_x@example.com",
                "committer_date": commit_date_str,
                "commit_date": commit_date_str,
                "message": message,
                "branches": "main",
                "is_merge": "False",
                "files_changed": 1,
                "insertions": 10,
                "deletions": 5,
                "lines_changed": 15
            })
        df_commits = pd.DataFrame(commits)
        df_commits.to_csv(os.path.join(RAW_DIR, f"{name}_commits.csv"), index=False)
        
        # Generate contributors
        contribs = [{
            "author_name": "Developer X",
            "author_email": "dev_x@example.com",
            "commit_count": 200,
            "insertions": 2000,
            "deletions": 1000
        }]
        pd.DataFrame(contribs).to_csv(os.path.join(RAW_DIR, f"{name}_contributors.csv"), index=False)
        
        # Generate modifications
        modifications = []
        for i, file_path in enumerate(files):
            if i < 8:
                # LOW: commits 1, 2, 4, 5, 7, 8... (none are bug-fixes)
                commits_to_use = [1, 2, 4, 5, 7, 8, 10, 11, 13, 14]
            elif i < 16:
                # MEDIUM: contains 1 or 2 bug-fix commits (e.g. commit 0, 3)
                commits_to_use = [0, 3, 1, 2, 4, 5, 7, 8, 10, 11]
            else:
                # HIGH: contains 3 or more bug-fix commits (e.g. commit 0, 3, 6, 9)
                commits_to_use = [0, 3, 6, 9, 12, 1, 2, 4, 5, 7]
                
            for c_idx in commits_to_use:
                c = commits[c_idx]
                modifications.append({
                    "commit_hash": c["commit_hash"],
                    "author_email": c["author_email"],
                    "commit_date": c["commit_date"],
                    "old_path": file_path,
                    "new_path": file_path,
                    "change_type": "MODIFY",
                    "added_lines": 5,
                    "deleted_lines": 2,
                    "net_lines": 3,
                    "complexity": 5,
                    "nloc": 15
                })
        df_mods = pd.DataFrame(modifications)
        df_mods.to_csv(os.path.join(RAW_DIR, f"{name}_modifications.csv"), index=False)
        
        # Generate {name}_quality_metrics.csv in processed/
        quality_records = []
        for file_path in files:
            quality_records.append({
                "repository_name": name,
                "file_path": file_path,
                "language": lang,
                "loc": 25,
                "complexity": 3,
                "warnings": 0,
                "errors": 0,
                "maintainability_index": 85.0 if lang == "python" else -1.0
            })
        df_quality = pd.DataFrame(quality_records)
        df_quality.to_csv(os.path.join(PROCESSED_DIR, f"{name}_quality_metrics.csv"), index=False)
        
        # Generate language profile
        lang_profile = [{
            "repository_name": name,
            "language": lang,
            "file_count": 25,
            "percentage": 100.0
        }]
        pd.DataFrame(lang_profile).to_csv(os.path.join(RAW_DIR, f"{name}_language_profile.csv"), index=False)

    print("[+] Mock dataset generation completed successfully!")

if __name__ == "__main__":
    generate_mock_data()
