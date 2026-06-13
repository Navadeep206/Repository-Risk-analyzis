#!/usr/bin/env python3
"""
Explainability Service for Phase 10.
Manages retrieval of Random Forest feature importances, permutation scores,
and markdown analysis reports from the explainability system.
"""

import os
import sys
import pickle
import pandas as pd
from typing import Dict, List, Optional

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from config import BASE_DIR

class ExplainabilityService:
    """
    Retrieves static and dynamic explainability findings from production models and reports.
    """
    def __init__(self):
        self.exp_dir = os.path.join(BASE_DIR, "reports", "explainability")
        self.model_path = os.path.join(BASE_DIR, "models", "random_forest.pkl")
        self.preprocessor_path = os.path.join(BASE_DIR, "models", "preprocessor.pkl")
        
    def get_feature_importances(self) -> pd.DataFrame:
        """
        Retrieves feature importances. Attempts to load from reports first,
        falling back to extracting from the fitted Random Forest classifier.
        """
        feat_imp_file = os.path.join(self.exp_dir, "feature_importance.csv")
        if os.path.exists(feat_imp_file):
            return pd.read_csv(feat_imp_file)
            
        # Fallback extraction from pickle
        if os.path.exists(self.model_path) and os.path.exists(self.preprocessor_path):
            try:
                with open(self.model_path, "rb") as f:
                    rf_model = pickle.load(f)
                with open(self.preprocessor_path, "rb") as f:
                    preproc = pickle.load(f)
                    
                importances = rf_model.feature_importances_
                features = preproc.feature_names
                
                # Check alignment
                if len(importances) == len(features):
                    df = pd.DataFrame({
                        "feature_name": features,
                        "rf_intrinsic_importance": importances
                    }).sort_values("rf_intrinsic_importance", ascending=False)
                    return df
            except Exception as e:
                print(f"[-] Error extracting feature importances: {e}")
                
        # Base fallback
        return pd.DataFrame(columns=["feature_name", "rf_intrinsic_importance"])
        
    def get_permutation_importances(self) -> pd.DataFrame:
        """Loads permutation importance CSV from reports."""
        perm_file = os.path.join(self.exp_dir, "permutation_importance.csv")
        if os.path.exists(perm_file):
            return pd.read_csv(perm_file)
        return pd.DataFrame(columns=["feature_name", "rf_permutation_mean", "rf_permutation_std"])
        
    def get_global_rankings(self) -> pd.DataFrame:
        """Loads global feature ranking CSV from reports."""
        rank_file = os.path.join(self.exp_dir, "global_feature_ranking.csv")
        if os.path.exists(rank_file):
            return pd.read_csv(rank_file)
        return pd.DataFrame()
        
    def get_explainability_report(self, report_name: str) -> str:
        """
        Loads explainability markdown report content.
        
        Args:
            report_name: Name of report e.g. 'error_analysis', 'domain_shift_analysis', 'model_trustworthiness'.
            
        Returns:
            The markdown string content.
        """
        if not report_name.endswith(".md"):
            report_name += ".md"
            
        report_path = os.path.join(self.exp_dir, report_name)
        if os.path.exists(report_path):
            with open(report_path, "r") as f:
                return f.read()
        return f"### Report '{report_name}' not found."
