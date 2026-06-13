#!/usr/bin/env python3
"""
Prediction Service for Phase 10.
Loads Random Forest production models and calculates risk labels, confidence, and trust gates.
"""

import os
import sys
import pickle
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from config import BASE_DIR
from ml.preprocessing import CodeRiskPreprocessor

INV_LABEL_MAP = {
    0: "LOW",
    1: "MEDIUM",
    2: "HIGH"
}

class PredictionService:
    """
    Production Risk Prediction Service incorporating Random Forest classifier
    and Trust Gate evaluation.
    """
    def __init__(self):
        models_dir = os.path.join(BASE_DIR, "models")
        self.preprocessor_path = os.path.join(models_dir, "preprocessor.pkl")
        self.model_path = os.path.join(models_dir, "random_forest.pkl")
        
        self.preprocessor: Optional[CodeRiskPreprocessor] = None
        self.model = None
        self._load_artifacts()

    def _load_artifacts(self):
        """Loads fitted preprocessor and classifier."""
        if os.path.exists(self.preprocessor_path):
            self.preprocessor = CodeRiskPreprocessor.load(self.preprocessor_path)
        else:
            print(f"[-] Preprocessor not found at {self.preprocessor_path}")
            
        if os.path.exists(self.model_path):
            with open(self.model_path, "rb") as f:
                self.model = pickle.load(f)
        else:
            print(f"[-] Random Forest model not found at {self.model_path}")

    def is_ready(self) -> bool:
        """Checks if both artifacts are successfully loaded."""
        return self.preprocessor is not None and self.model is not None

    def evaluate_trust_gate(self, confidence: float) -> Tuple[str, str]:
        """
        Determines the Trust Gate status and rating based on prediction confidence.
        
        Returns:
            A tuple of (rating_label, color_hex).
        """
        if confidence >= 90.0:
            return "High Confidence", "#10B981"  # Emerald Green
        elif confidence >= 70.0:
            return "Moderate Confidence", "#F59E0B"  # Amber Orange
        else:
            return "Manual Review Recommended", "#EF4444"  # Red

    def predict(self, df_metrics: pd.DataFrame) -> pd.DataFrame:
        """
        Runs preprocessing and risk predictions for a DataFrame of files.
        
        Args:
            df_metrics: Dataframe containing required code and process metrics.
            
        Returns:
            The input DataFrame augmented with predictions, confidence, and trust gating columns.
        """
        if not self.is_ready():
            raise RuntimeError("Prediction Service is not fully initialized. Models missing.")
            
        df_copy = df_metrics.copy()
        
        # Ensure all columns required by preprocessor are present
        required_cols = [
            "loc", "complexity", "maintainability_index", "commit_count",
            "modification_count", "contributor_count", "commit_frequency", "repository_age_days"
        ]
        
        # Fallback values for missing columns
        for col in required_cols:
            if col not in df_copy.columns:
                df_copy[col] = 0
                
        if "language" not in df_copy.columns:
            df_copy["language"] = "python"
            
        # Run preprocessing
        X_proc = self.preprocessor.transform(df_copy)
        
        # Run model inference
        preds = self.model.predict(X_proc)
        probs = self.model.predict_proba(X_proc)
        
        # Calculate confidences and labels
        risk_labels = [INV_LABEL_MAP.get(int(p), "LOW") for p in preds]
        confidences = np.max(probs, axis=1) * 100.0
        
        df_copy["predicted_risk"] = risk_labels
        df_copy["confidence"] = confidences
        
        # Add Trust Gate classifications
        trust_labels = []
        trust_colors = []
        for conf in confidences:
            label, color = self.evaluate_trust_gate(conf)
            trust_labels.append(label)
            trust_colors.append(color)
            
        df_copy["trust_rating"] = trust_labels
        df_copy["trust_color"] = trust_colors
        
        return df_copy
