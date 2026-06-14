#!/usr/bin/env python3
"""
Python analyzer implementing Radon metrics extraction under the BaseAnalyzer interface.
"""

import os
import pandas as pd
import sys
from typing import Optional, List
import radon.raw
import radon.complexity
import radon.metrics

# Ensure parent directory is in path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from config import RAW_DIR
from quality_metrics.analyzers.base_analyzer import BaseAnalyzer

class PythonAnalyzer(BaseAnalyzer):
    def analyze(self, target_dir: str, files: List[str], output_file: Optional[str] = None) -> pd.DataFrame:
        """
        Extracts software quality metrics from Python files using Radon.
        
        Args:
            target_dir: Absolute path of target codebase directory.
            files: List of absolute file paths to analyze.
            output_file: Optional path to save the output CSV.
            
        Returns:
            A pandas DataFrame matching the unified schema.
        """
        import signal
        
        class TimeoutException(Exception):
            pass
            
        def timeout_handler(signum, frame):
            raise TimeoutException("File analysis timed out after 2 seconds")
            
        repo_name = os.path.basename(target_dir.rstrip("/"))
        records = []
        
        # Register the signal handler for timeout
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        
        try:
            for file_path in files:
                try:
                    # 0. Skip files larger than 1MB to prevent performance degradation
                    if os.path.getsize(file_path) > 1024 * 1024:
                        print(f"[-] Warning: Skipping {file_path} because it is larger than 1MB")
                        continue
                        
                    # Set alarm for 2 seconds
                    signal.alarm(2)
                    
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                        
                    # 1. LOC
                    raw_metrics = radon.raw.analyze(content)
                    loc = raw_metrics.loc
                    
                    # 2. Complexity
                    blocks = radon.complexity.cc_visit(content)
                    top_level_cc = 0
                    for block in blocks:
                        block_type = type(block).__name__
                        if block_type == "Class":
                            top_level_cc += block.complexity
                        elif block_type == "Function" and getattr(block, "classname", None) is None:
                            top_level_cc += block.complexity
                    if top_level_cc == 0:
                        top_level_cc = 1
                        
                    # 3. Maintainability Index
                    mi_value = radon.metrics.mi_visit(content, multi=True)
                    
                    # Disable alarm
                    signal.alarm(0)
                    
                    rel_path = os.path.relpath(file_path, target_dir)
                    records.append({
                        "repository_name": repo_name,
                        "file_path": rel_path,
                        "language": "python",
                        "loc": loc,
                        "complexity": top_level_cc,
                        "warnings": 0,
                        "errors": 0,
                        "maintainability_index": mi_value
                    })
                except Exception as e:
                    signal.alarm(0)  # Make sure alarm is disabled
                    print(f"[-] Warning: Python analyzer failed to process {file_path}: {e}")
        finally:
            # Restore original signal handler and cancel any active alarm
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
                
        df = pd.DataFrame(records)
        if df.empty:
            df = pd.DataFrame(columns=[
                "repository_name", "file_path", "language", "loc", "complexity",
                "warnings", "errors", "maintainability_index"
            ])
            
        if not output_file:
            output_file = os.path.join(RAW_DIR, "python_metrics.csv")
            
        df.to_csv(output_file, index=False)
        print(f"[+] Python metrics saved to {output_file}. Processed {len(df)} files.")
        return df

