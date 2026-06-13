#!/usr/bin/env python3
"""
Source code file extraction module for Phase 5.
Scans repository directories for Python, JavaScript, and TypeScript files.
"""

import os
import sys
import pandas as pd
from typing import List, Dict

# Ensure parent directory is in path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import BASE_DIR, DATA_DIR

REPOS_DIR = os.path.join(DATA_DIR, "repositories")
INTERMEDIATE_DIR = os.path.join(DATA_DIR, "intermediate")

def extract_source_files() -> pd.DataFrame:
    """
    Scans data/repositories/ recursively and extracts source code contents.
    
    Returns:
        A pandas DataFrame with extracted source files.
    """
    os.makedirs(INTERMEDIATE_DIR, exist_ok=True)
    
    if not os.path.exists(REPOS_DIR):
        raise FileNotFoundError(f"Repositories directory does not exist: {REPOS_DIR}")
        
    records = []
    
    # Exclude typical build, test-runner, dependency, and configuration folders
    exclude_dirs = {
        "node_modules", "bower_components", "dist", "build", "out",
        "__pycache__", ".git", ".github", ".vscode", "coverage", "venv", "env"
    }
    
    # Supported file extensions
    ext_mapping = {
        ".py": "python",
        ".js": "javascript",
        ".ts": "typescript",
        ".tsx": "typescript"
    }
    
    # Iterate through each repository in data/repositories/
    for repo_name in os.listdir(REPOS_DIR):
        repo_path = os.path.join(REPOS_DIR, repo_name)
        if not os.path.isdir(repo_path):
            continue
            
        print(f"[*] Extracting files from repository: {repo_name}")
        
        for root, dirs, files in os.walk(repo_path):
            # Prune directories in place to prevent scanning excluded paths
            dirs[:] = [d for d in dirs if d not in exclude_dirs]
            
            for file in files:
                name, ext = os.path.splitext(file)
                if ext not in ext_mapping:
                    continue
                    
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, repo_path)
                
                # Attempt to read the source code content
                try:
                    with open(full_path, "r", encoding="utf-8", errors="replace") as f:
                        source_code = f.read()
                except Exception as e:
                    print(f"[!] Error reading {full_path}: {e}")
                    continue
                    
                records.append({
                    "repository_name": repo_name,
                    "file_path": rel_path,
                    "language": ext_mapping[ext],
                    "source_code": source_code
                })
                
    df = pd.DataFrame(records)
    output_path = os.path.join(INTERMEDIATE_DIR, "source_code_dataset.csv")
    df.to_csv(output_path, index=False)
    print(f"[+] Extracted {len(df)} files. Saved dataset to {output_path}")
    return df

if __name__ == "__main__":
    extract_source_files()
