#!/usr/bin/env python3
"""
Inference API for the deep learning pipeline.
Loads models/repository_risk_predictor.pt and runs inference on a 768-D code embedding.
"""

import os
import sys

# MUST be set before importing torch or any OpenMP-linked library
# Prevents EXC_BAD_ACCESS (SIGSEGV) from duplicate libomp on macOS ARM64
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import torch
import numpy as np
from typing import Tuple, Dict

# Ensure parent directory is in path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import BASE_DIR
from deep_learning.model import RepositoryRiskPredictor
from deep_learning.dataset_loader import INV_LABEL_MAP

class RiskInferenceEngine:
    """
    Risk prediction engine that uses the trained deep learning checkpoint to evaluate code embeddings.
    """
    def __init__(self, model_path: str = None) -> None:
        if model_path is None:
            model_path = os.path.join(BASE_DIR, "models", "repository_risk_predictor.pt")
            
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model weights file not found: {model_path}")
            
        # Device check
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
            
        self.model = RepositoryRiskPredictor()
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    def predict(self, embedding_vector: np.ndarray) -> Tuple[str, Dict[str, float]]:
        """
        Accepts a 768-dimensional numpy array, executes forward pass,
        applies Softmax, and returns the predicted label and class probability scores.
        """
        if embedding_vector.shape != (768,):
            raise ValueError(f"Invalid embedding dimensions: {embedding_vector.shape}. Expected (768,).")
            
        # Convert to tensor and add batch dimension (1, 768)
        tensor_in = torch.tensor(embedding_vector, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            logits = self.model(tensor_in)
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
    # Quick verification using a dummy embedding
    print("[*] Testing RiskInferenceEngine with random dummy vector...")
    try:
        engine = RiskInferenceEngine()
        dummy_vector = np.random.randn(768).astype(np.float32)
        label, scores = engine.predict(dummy_vector)
        print("\n" + "="*40)
        print("Verification Inference Results:")
        print("="*40)
        for cls, prob in scores.items():
            print(f"{cls}: {prob:.2%}")
        print(f"Prediction: {label}")
        print("="*40 + "\n")
    except FileNotFoundError:
        print("[!] Model weights not found. Run deep_learning_pipeline.py first.")

if __name__ == "__main__":
    run_inference_demo()
