#!/usr/bin/env python3
"""
Inference API for the hybrid pipeline.
Loads models/hybrid_risk_predictor.pt and runs inference on joint tabular + embedding inputs.
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, Union

# Ensure parent directory is in path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import BASE_DIR
from hybrid.fusion_model import HybridRiskPredictor
from hybrid.hybrid_dataset import INV_LABEL_MAP
from ml.preprocessing import CodeRiskPreprocessor

class HybridRiskInferenceEngine:
    """
    Risk prediction engine that uses the trained hybrid deep learning checkpoint 
    and preprocessor to evaluate raw or preprocessed tabular metrics and code embeddings.
    """
    def __init__(self, model_path: str = None, preproc_path: str = None) -> None:
        if model_path is None:
            model_path = os.path.join(BASE_DIR, "models", "hybrid_risk_predictor.pt")
        if preproc_path is None:
            preproc_path = os.path.join(BASE_DIR, "models", "preprocessor.pkl")
            
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model weights file not found: {model_path}")
        if not os.path.exists(preproc_path):
            raise FileNotFoundError(f"Preprocessor file not found: {preproc_path}")
            
        # Device check
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
            
        # Initialize model and load weights
        self.model = HybridRiskPredictor()
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        # Load fitted preprocessor
        self.preprocessor = CodeRiskPreprocessor.load(preproc_path)

    def _preprocess_tabular(self, tabular_data: Union[Dict[str, Any], pd.Series, pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Standardizes tabular input to a processed 11-D numpy array.
        """
        if isinstance(tabular_data, np.ndarray):
            if tabular_data.shape == (11,):
                return tabular_data.reshape(1, 11)
            elif tabular_data.shape == (1, 11):
                return tabular_data
            else:
                raise ValueError(f"Numpy array shape must be (11,) or (1, 11), got {tabular_data.shape}")
                
        # If dict or Series, convert to DataFrame
        if isinstance(tabular_data, dict):
            df = pd.DataFrame([tabular_data])
        elif isinstance(tabular_data, pd.Series):
            df = pd.DataFrame([tabular_data])
        elif isinstance(tabular_data, pd.DataFrame):
            df = tabular_data.copy()
        else:
            raise TypeError("Unsupported tabular data type.")
            
        # Preprocess using loaded preprocessor
        return self.preprocessor.transform(df)

    def predict(
        self, 
        tabular_data: Union[Dict[str, Any], pd.Series, pd.DataFrame, np.ndarray],
        embedding_vector: np.ndarray
    ) -> Tuple[str, Dict[str, float]]:
        """
        Runs joint prediction using preprocessed tabular features and embedding vector.
        
        Args:
            tabular_data: Raw dictionary/DataFrame of metrics, or pre-scaled 11-D array.
            embedding_vector: 768-D code embedding vector.
            
        Returns:
            predicted_label: "LOW", "MEDIUM", or "HIGH"
            probabilities: Dictionary mapping classes to probability scores.
        """
        # Preprocess tabular features
        x_tab_proc = self._preprocess_tabular(tabular_data)
        
        # Format embedding vector
        if embedding_vector.ndim == 1:
            if embedding_vector.shape != (768,):
                raise ValueError(f"Invalid embedding dimensions: {embedding_vector.shape}. Expected (768,).")
            x_emb_proc = embedding_vector.reshape(1, 768)
        elif embedding_vector.ndim == 2:
            if embedding_vector.shape != (1, 768):
                raise ValueError(f"Invalid embedding dimensions: {embedding_vector.shape}. Expected (1, 768).")
            x_emb_proc = embedding_vector
        else:
            raise ValueError(f"Invalid embedding dimension: {embedding_vector.ndim}. Expected 1D or 2D array.")
            
        # Convert to tensors
        tensor_tab = torch.tensor(x_tab_proc, dtype=torch.float32).to(self.device)
        tensor_emb = torch.tensor(x_emb_proc, dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            logits = self.model(tensor_tab, tensor_emb)
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
            
        pred_idx = int(np.argmax(probs))
        pred_label = INV_LABEL_MAP[pred_idx]
        
        scores = {
            "LOW": float(probs[0]),
            "MEDIUM": float(probs[1]),
            "HIGH": float(probs[2])
        }
        
        return pred_label, scores

def run_inference_demo() -> None:
    print("[*] Testing HybridRiskInferenceEngine with random dummy inputs...")
    try:
        engine = HybridRiskInferenceEngine()
        
        # 11-D preprocessed array mock
        dummy_tab = np.random.randn(11).astype(np.float32)
        dummy_emb = np.random.randn(768).astype(np.float32)
        
        label, scores = engine.predict(dummy_tab, dummy_emb)
        print("\n" + "="*40)
        print("Hybrid Verification Inference Results:")
        print("="*40)
        for cls, prob in scores.items():
            print(f"{cls}: {prob:.2%}")
        print(f"Prediction: {label}")
        print("="*40 + "\n")
    except FileNotFoundError:
        print("[!] Model weights not found. Run hybrid_pipeline.py first.")

if __name__ == "__main__":
    run_inference_demo()
