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
    Handles scaling of numeric features and one-hot encoding of categorical language tags.
    """
    def __init__(self) -> None:
        self.numeric_features = [
            "loc", "complexity", "maintainability_index", "commit_count",
            "modification_count", "contributor_count", "commit_frequency", "repository_age_days"
        ]
        self.categorical_features = ["language"]
        self.preprocessor = None
        self.feature_names: List[str] = []

    def fit(self, X: pd.DataFrame) -> "CodeRiskPreprocessor":
        """
        Fits the scaler and encoder on the training data.
        """
        # Ensure only relevant columns are present
        X_subset = X[self.numeric_features + self.categorical_features].copy()
        
        # Define transformers
        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), self.numeric_features),
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), self.categorical_features)
            ]
        )
        
        self.preprocessor.fit(X_subset)
        
        # Extract feature names after encoding
        # Cat feature names from OneHotEncoder
        cat_encoder = self.preprocessor.named_transformers_["cat"]
        cat_features = cat_encoder.get_feature_names_out(self.categorical_features).tolist()
        
        self.feature_names = self.numeric_features + cat_features
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Applies scaling and encoding to the input DataFrame.
        """
        if self.preprocessor is None:
            raise ValueError("Preprocessor has not been fitted yet. Call fit() first.")
            
        X_subset = X[self.numeric_features + self.categorical_features].copy()
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
