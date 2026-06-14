#!/usr/bin/env python3
"""
Universal Software Quality Metrics Pipeline. Detects languages,
routes files to respective registered analyzers, and merges the datasets.
"""

import os
import argparse
import pandas as pd
import sys
from typing import Optional, Dict, List

# Ensure parent directory is in path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import RAW_DIR, PROCESSED_DIR, ensure_dirs_exist

from quality_metrics.language_detector import detect_languages
from quality_metrics.analyzers.base_analyzer import BaseAnalyzer
from quality_metrics.analyzers.python_analyzer import PythonAnalyzer
from quality_metrics.analyzers.javascript_analyzer import JavaScriptAnalyzer
from quality_metrics.analyzers.typescript_analyzer import TypeScriptAnalyzer
from quality_metrics.metric_merger import merge_metrics

# Extensible registry pattern mapping languages to their BaseAnalyzer instances
ANALYZER_REGISTRY: Dict[str, BaseAnalyzer] = {
    "python": PythonAnalyzer(),
    "javascript": JavaScriptAnalyzer(),
    "typescript": TypeScriptAnalyzer()
}

def get_files_by_language(target_dir: str) -> Dict[str, List[str]]:
    """
    Finds all Python, JS, and TS files under target_dir, grouped by language.
    
    Args:
        target_dir: Codebase directory path to scan.
        
    Returns:
        A dictionary mapping language names to lists of absolute file paths.
    """
    files_by_lang: Dict[str, List[str]] = {
        "python": [],
        "javascript": [],
        "typescript": []
    }
    ignored_dirs = {
        ".venv", "venv", "node_modules", ".git", "__pycache__", "dist", "build",
        "vendor", "third_party", "extern", "fixtures", "test_fixtures", "tests/fixtures"
    }
    
    for root, dirs, files in os.walk(target_dir):
        dirs[:] = [d for d in dirs if d not in ignored_dirs]
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            abs_path = os.path.abspath(os.path.join(root, file))
            if ext == ".py":
                files_by_lang["python"].append(abs_path)
            elif ext in (".js", ".jsx"):
                files_by_lang["javascript"].append(abs_path)
            elif ext in (".ts", ".tsx"):
                files_by_lang["typescript"].append(abs_path)
                
    return files_by_lang

def run_quality_pipeline(target_dir: str, output_processed_file: Optional[str] = None) -> pd.DataFrame:
    """
    Runs the entire multi-language quality metrics pipeline.
    
    Args:
        target_dir: Codebase directory path to analyze.
        output_processed_file: Optional path for the final merged CSV.
        
    Returns:
        A pandas DataFrame of unified quality metrics.
    """
    if not os.path.exists(target_dir):
        raise ValueError(f"Target directory does not exist: {target_dir}")
        
    print(f"[*] Executing Universal Software Quality Pipeline on: {target_dir}")
    
    # 1. Detect Languages and save profile
    print("[*] Running Language Detection...")
    detect_languages(target_dir)
    
    # Group files by language
    files_by_lang = get_files_by_language(target_dir)
    
    # 2. Run correct analyzers based on registry
    for lang, analyzer in ANALYZER_REGISTRY.items():
        files = files_by_lang.get(lang, [])
        # Delete old CSV to avoid merging stale data if files lists are empty
        raw_csv_path = os.path.join(RAW_DIR, f"{lang}_metrics.csv")
        if os.path.exists(raw_csv_path):
            os.remove(raw_csv_path)
            
        if files:
            print(f"[*] Running {lang.upper()} Analyzer on {len(files)} files...")
            analyzer.analyze(target_dir, files)
        else:
            print(f"[*] No {lang.upper()} files detected. Generating empty output with headers.")
            # Run with empty list to ensure file existence and standard headers
            analyzer.analyze(target_dir, [])
            
    # 3. Merge all results
    print("[*] Merging all metrics...")
    merged_df = merge_metrics(output_processed_file)
    
    print("[+] Universal Quality Pipeline completed successfully.")
    return merged_df

def main() -> None:
    parser = argparse.ArgumentParser(description="Universal Software Quality Metrics Pipeline.")
    parser.add_argument("target_dir", help="Path to repository directory to analyze")
    parser.add_argument("--output", "-o", help="Path to final merged processed CSV output", default=None)
    args = parser.parse_args()
    try:
        run_quality_pipeline(args.target_dir, args.output)
    except Exception as e:
        print(f"[-] Pipeline execution failed: {e}")

if __name__ == "__main__":
    main()
