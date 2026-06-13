#!/usr/bin/env python3
"""
Language detector for scanned repositories. Scans codebase and generates the language profile.
"""

import os
import argparse
import pandas as pd
import sys
from typing import Optional, Dict

# Ensure parent directory is in path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import RAW_DIR, ensure_dirs_exist

def detect_languages(target_dir: str, output_file: Optional[str] = None) -> pd.DataFrame:
    """
    Scans the target directory recursively, counts Python, JavaScript, and TypeScript files,
    and calculates their percentages.
    
    Args:
        target_dir: The root codebase directory.
        output_file: Optional path for CSV output.
        
    Returns:
        A pandas DataFrame representing the language profile.
    """
    if not os.path.exists(target_dir):
        raise ValueError(f"Target directory does not exist: {target_dir}")
        
    repo_name = os.path.basename(target_dir.rstrip("/"))
    
    # Track file counts
    counts: Dict[str, int] = {
        "python": 0,
        "javascript": 0,
        "typescript": 0
    }
    
    # Scan files, ignoring build directories and venvs
    ignored_dirs = {".venv", "venv", "node_modules", ".git", "__pycache__", "dist", "build"}
    
    for root, dirs, files in os.walk(target_dir):
        dirs[:] = [d for d in dirs if d not in ignored_dirs]
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext == ".py":
                counts["python"] += 1
            elif ext in (".js", ".jsx"):
                counts["javascript"] += 1
            elif ext in (".ts", ".tsx"):
                counts["typescript"] += 1
                
    total_files = sum(counts.values())
    
    records = []
    for lang, count in counts.items():
        if count > 0 or total_files == 0:
            percentage = (count / total_files * 100) if total_files > 0 else 0.0
            records.append({
                "repository_name": repo_name,
                "language": lang,
                "file_count": count,
                "percentage": round(percentage, 2)
            })
            
    df = pd.DataFrame(records)
    # Sort by percentage descending
    if not df.empty:
        df = df.sort_values(by="percentage", ascending=False).reset_index(drop=True)
    else:
        df = pd.DataFrame(columns=["repository_name", "language", "file_count", "percentage"])
        
    if not output_file:
        output_file = os.path.join(RAW_DIR, "repository_language_profile.csv")
        
    ensure_dirs_exist()
    df.to_csv(output_file, index=False)
    print(f"[+] Language profile saved to {output_file}.")
    return df

def main() -> None:
    parser = argparse.ArgumentParser(description="Scan target directory and generate repository language profile.")
    parser.add_argument("target_dir", help="Path to codebase directory to scan")
    parser.add_argument("--output", "-o", help="Path to output CSV file (optional)", default=None)
    args = parser.parse_args()
    try:
        detect_languages(args.target_dir, args.output)
    except Exception as e:
        print(f"[-] Error: {e}")

if __name__ == "__main__":
    main()
