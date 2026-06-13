#!/usr/bin/env python3
"""
JavaScript analyzer implementing Lizard and ESLint parsing under the BaseAnalyzer interface.
"""

import os
import json
import subprocess
import pandas as pd
import sys
from typing import Optional, List, Dict
import lizard

# Ensure parent directory is in path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from config import RAW_DIR
from quality_metrics.analyzers.base_analyzer import BaseAnalyzer

def get_eslint_metrics(target_dir: str) -> Dict[str, Dict[str, int]]:
    """
    Helper function to run ESLint recursively in directories and gather warnings/errors.
    
    Args:
        target_dir: The target codebase directory.
        
    Returns:
        A dictionary mapping target-relative file paths to warning and error counts.
    """
    eslint_data: Dict[str, Dict[str, int]] = {}
    config_dirs = set()
    
    # Recursively find directories containing eslint configs or package.json
    ignored_dirs = {".venv", "venv", "node_modules", ".git", "__pycache__", "dist", "build"}
    for root, dirs, files in os.walk(target_dir):
        dirs[:] = [d for d in dirs if d not in ignored_dirs]
        for file in files:
            if file in ("package.json", "eslint.config.js", "eslint.config.mjs", "eslint.config.cjs") or file.startswith(".eslintrc"):
                config_dirs.add(root)
                break
                
    if not config_dirs:
        config_dirs.add(target_dir)
        
    for config_dir in config_dirs:
        try:
            # We run npx eslint --format json . to lint all files in that subdirectory
            cmd = ["npx", "eslint", "--format", "json", "."]
            result = subprocess.run(
                cmd,
                cwd=config_dir,
                capture_output=True,
                text=True
            )
            stdout = result.stdout.strip()
            if stdout:
                try:
                    data = json.loads(stdout)
                    for item in data:
                        file_path = item.get("filePath")
                        if file_path:
                            rel_path = os.path.relpath(file_path, target_dir)
                            eslint_data[rel_path] = {
                                "warnings": item.get("warningCount", 0),
                                "errors": item.get("errorCount", 0)
                            }
                except json.JSONDecodeError:
                    pass
        except Exception:
            # Silently pass if ESLint fails or is not configured
            pass
            
    return eslint_data

class JavaScriptAnalyzer(BaseAnalyzer):
    def analyze(self, target_dir: str, files: List[str], output_file: Optional[str] = None) -> pd.DataFrame:
        """
        Extracts software quality metrics from JavaScript files using Lizard and ESLint.
        
        Args:
            target_dir: Absolute path of target codebase directory.
            files: List of absolute file paths to analyze.
            output_file: Optional path to save the output CSV.
            
        Returns:
            A pandas DataFrame matching the unified schema.
        """
        repo_name = os.path.basename(target_dir.rstrip("/"))
        records = []
        
        # Run ESLint to gather metrics
        print(f"[*] Running ESLint for JavaScript files in {target_dir}...")
        eslint_data = get_eslint_metrics(target_dir)
        
        for file_path in files:
            try:
                # Use Lizard to parse LOC and Complexity
                analysis = lizard.analyze_file(file_path)
                rel_path = os.path.relpath(file_path, target_dir)
                
                # Check for eslint counts
                eslint_info = eslint_data.get(rel_path, {"warnings": 0, "errors": 0})
                
                records.append({
                    "repository_name": repo_name,
                    "file_path": rel_path,
                    "language": "javascript",
                    "loc": analysis.nloc,
                    "complexity": analysis.CCN if analysis.CCN > 0 else 1,
                    "warnings": eslint_info["warnings"],
                    "errors": eslint_info["errors"],
                    "maintainability_index": -1.0  # Placeholder for JS
                })
            except Exception as e:
                print(f"[-] Warning: JS analyzer failed to process {file_path}: {e}")
                
        df = pd.DataFrame(records)
        if df.empty:
            df = pd.DataFrame(columns=[
                "repository_name", "file_path", "language", "loc", "complexity",
                "warnings", "errors", "maintainability_index"
            ])
            
        if not output_file:
            output_file = os.path.join(RAW_DIR, "javascript_metrics.csv")
            
        df.to_csv(output_file, index=False)
        print(f"[+] JavaScript metrics saved to {output_file}. Processed {len(df)} files.")
        return df
