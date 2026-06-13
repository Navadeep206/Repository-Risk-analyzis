import os
import unittest
import torch
import numpy as np
import sys

# Ensure src is in search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from config import BASE_DIR
from hybrid.hybrid_dataset import get_hybrid_dataloaders
from hybrid.fusion_model import TabularOnlyPredictor, EmbeddingOnlyPredictor, HybridRiskPredictor
from hybrid.inference import HybridRiskInferenceEngine

class TestHybridPipeline(unittest.TestCase):
    def setUp(self) -> None:
        self.models_dir = os.path.join(BASE_DIR, "models")
        self.reports_dir = os.path.join(BASE_DIR, "reports", "hybrid")
        
        self.model_file = os.path.join(self.models_dir, "hybrid_risk_predictor.pt")
        self.metrics_json = os.path.join(self.reports_dir, "best_model_metrics.json")
        self.classification_report = os.path.join(self.reports_dir, "classification_report.txt")
        self.model_comparison = os.path.join(self.reports_dir, "model_comparison.csv")
        self.ablation_study = os.path.join(self.reports_dir, "ablation_study.csv")
        self.loss_curve = os.path.join(self.reports_dir, "loss_curve.png")
        self.f1_curve = os.path.join(self.reports_dir, "f1_curve.png")
        self.cm_plot = os.path.join(self.reports_dir, "confusion_matrix.png")

    def test_dataloaders(self) -> None:
        """Verify that hybrid loaders initialize and yield correct shapes."""
        train_loader, val_loader, test_loader, weights = get_hybrid_dataloaders(batch_size=4)
        
        # Check shapes on first training batch
        for X_tab, X_emb, y_batch in train_loader:
            self.assertEqual(X_tab.shape[1], 11, "Tabular features must be 11-dimensional.")
            self.assertEqual(X_emb.shape[1], 768, "Embedding features must be 768-dimensional.")
            self.assertEqual(len(y_batch.shape), 1, "Labels must be 1-dimensional.")
            break
            
        self.assertEqual(len(weights.shape), 1, "Weights array must be 1D.")
        self.assertEqual(weights.shape[0], 3, "Weights must cover 3 classes.")

    def test_model_forward_passes(self) -> None:
        """Verify forward pass output shapes for Model A, B, and C."""
        dummy_tab = torch.randn(4, 11)
        dummy_emb = torch.randn(4, 768)
        
        m_a = TabularOnlyPredictor()
        m_b = EmbeddingOnlyPredictor()
        m_c = HybridRiskPredictor()
        
        logits_a = m_a(dummy_tab, dummy_emb)
        logits_b = m_b(dummy_tab, dummy_emb)
        logits_c = m_c(dummy_tab, dummy_emb)
        
        self.assertEqual(logits_a.shape, (4, 3), "Model A output logits shape mismatch.")
        self.assertEqual(logits_b.shape, (4, 3), "Model B output logits shape mismatch.")
        self.assertEqual(logits_c.shape, (4, 3), "Model C output logits shape mismatch.")

    def test_inference_engine(self) -> None:
        """Verify HybridRiskInferenceEngine prediction properties."""
        if os.path.exists(self.model_file):
            engine = HybridRiskInferenceEngine(model_path=self.model_file)
            dummy_tab = np.random.randn(11).astype(np.float32)
            dummy_emb = np.random.randn(768).astype(np.float32)
            
            label, scores = engine.predict(dummy_tab, dummy_emb)
            
            self.assertIn(label, ["LOW", "MEDIUM", "HIGH"], "Predicted label is invalid.")
            
            # Verify sum of probabilities equals 1.0 (approx)
            prob_sum = sum(scores.values())
            self.assertAlmostEqual(prob_sum, 1.0, places=5, msg="Probabilities must sum to 1.0.")

    def test_pipeline_outputs_exist(self) -> None:
        """Verify that training outputs, metrics, and figures are created."""
        if os.path.exists(self.model_file):
            self.assertTrue(os.path.exists(self.metrics_json), "best_model_metrics.json missing.")
            self.assertTrue(os.path.exists(self.classification_report), "classification_report.txt missing.")
            self.assertTrue(os.path.exists(self.model_comparison), "model_comparison.csv missing.")
            self.assertTrue(os.path.exists(self.ablation_study), "ablation_study.csv missing.")
            self.assertTrue(os.path.exists(self.loss_curve), "loss_curve.png missing.")
            self.assertTrue(os.path.exists(self.f1_curve), "f1_curve.png missing.")
            self.assertTrue(os.path.exists(self.cm_plot), "confusion_matrix.png missing.")

if __name__ == "__main__":
    unittest.main()
