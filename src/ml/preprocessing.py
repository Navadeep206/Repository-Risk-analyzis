#!/usr/bin/env python3
"""
Preprocessing module for baseline ML pipeline.
Handles feature scaling, categorical encoding, and feature extraction.
Saves preprocessor artifact to models/preprocessor.pkl.
"""

import os
import pickle
import pandas as pd
import numpy as np
import sys
from typing import Tuple, List
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Ensure parent directory is in path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import BASE_DIR

MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

class CodeRiskPreprocessor:
    """
    Handles scaling of numeric features (optionally relative to repository) and one-hot encoding of categorical language tags.
    """
    def __init__(self, relative_scaling: bool = True) -> None:
        self.numeric_features = [
            "loc", "complexity", "commit_count",
            "modification_count", "contributor_count", "commit_frequency", "repository_age_days",
            "ownership_concentration", "contributor_entropy", "bus_factor",
            "recent_churn", "time_decayed_churn", "historical_bug_density", "time_since_last_bug_fix"
        ]
        self.categorical_features = ["language"]
        self.preprocessor = None
        self.feature_names: List[str] = []
        self.relative_scaling = relative_scaling

    def _relative_scale(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        if "repository_name" not in df.columns:
            return df
            
        for col in self.numeric_features:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0).astype(float)
            else:
                df[col] = 0.0
            
        group_cols = ["repository_name"]
        if "language" in df.columns:
            group_cols.append("language")
            
        for keys, group_idx in df.groupby(group_cols).groups.items():
            if len(group_idx) > 0:
                sub_df = df.loc[group_idx, self.numeric_features]
                scaler = StandardScaler()
                scaled_vals = scaler.fit_transform(sub_df)
                scaled_vals = np.nan_to_num(scaled_vals, nan=0.0)
                df.loc[group_idx, self.numeric_features] = scaled_vals
        return df

    def fit(self, X: pd.DataFrame) -> "CodeRiskPreprocessor":
        """
        Fits the scaler and encoder on the training data.
        """
        if self.relative_scaling:
            X_proc = self._relative_scale(X)
            num_transformer = "passthrough"
        else:
            X_proc = X.copy()
            num_transformer = StandardScaler()
            
        transformers = []
        if self.numeric_features:
            transformers.append(("num", num_transformer, self.numeric_features))
        if self.categorical_features:
            transformers.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), self.categorical_features))
            
        self.preprocessor = ColumnTransformer(transformers=transformers)
        
        # Ensure only relevant columns are present for fitting
        all_features = self.numeric_features + self.categorical_features
        X_subset = X_proc[all_features].copy()
        self.preprocessor.fit(X_subset)
        
        # Extract feature names after encoding
        self.feature_names = list(self.numeric_features)
        if self.categorical_features:
            cat_encoder = self.preprocessor.named_transformers_["cat"]
            cat_features = cat_encoder.get_feature_names_out(self.categorical_features).tolist()
            self.feature_names.extend(cat_features)
            
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Applies scaling and encoding to the input DataFrame.
        """
        if self.preprocessor is None:
            raise ValueError("Preprocessor has not been fitted yet. Call fit() first.")
            
        if self.relative_scaling:
            X_proc = self._relative_scale(X)
        else:
            X_proc = X.copy()
            
        all_features = self.numeric_features + self.categorical_features
        X_subset = X_proc[all_features].copy()
        return self.preprocessor.transform(X_subset)

    def save(self, file_path: str) -> None:
        """
        Saves the fitted preprocessor to disk.
        """
        with open(file_path, "wb") as f:
            pickle.dump(self, f)
        print(f"[+] Fitted preprocessor saved to {file_path}")

    @staticmethod
    def load(file_path: str) -> "CodeRiskPreprocessor":
        """
        Loads a fitted preprocessor from disk.
        """
        with open(file_path, "rb") as f:
            preprocessor = pickle.load(f)
        return preprocessor

if __name__ == "__main__":
    from ml.data_loader import load_all_splits
    X_train, y_train, X_val, y_val, X_test, y_test = load_all_splits()
    
    preproc = CodeRiskPreprocessor()
    preproc.fit(X_train)
    X_train_proc = preproc.transform(X_train)
    
    # Save test
    preproc_path = os.path.join(MODELS_DIR, "preprocessor.pkl")
    preproc.save(preproc_path)
    
    print(f"Preprocessed shape: {X_train_proc.shape}")
    print(f"Feature Names: {preproc.feature_names}")
