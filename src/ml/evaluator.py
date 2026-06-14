#!/usr/bin/env python3
"""
Production-grade evaluation suite for Repository Risk prediction.
Supports classification, risk-ranking, and probability calibration.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.metrics import (
    precision_recall_fscore_support,
    roc_auc_score,
    precision_recall_curve,
    auc,
    brier_score_loss
)
from sklearn.calibration import calibration_curve

class RepositoryRiskEvaluator:
    """
    Evaluates Repository Risk predictions.
    Computes classification, ranking, and calibration metrics.
    """
    def __init__(self, y_true: np.ndarray, y_pred_class: np.ndarray, y_prob: np.ndarray):
        """
        Args:
            y_true: 1D array of actual labels (integer format: 0=LOW, 1=MEDIUM, 2=HIGH).
            y_pred_class: 1D array of predicted class labels (same format).
            y_prob: 2D array of class probabilities (shape: [n_samples, 3]).
        """
        self.y_true = np.asarray(y_true, dtype=int)
        self.y_pred_class = np.asarray(y_pred_class, dtype=int)
        self.y_prob = np.asarray(y_prob, dtype=float)
        
        # Binary target for HIGH risk (class 2) evaluations
        self.y_true_high = (self.y_true == 2).astype(int)
        self.y_prob_high = self.y_prob[:, 2]

    def evaluate_classification(self) -> Dict[str, Any]:
        """
        Computes standard multi-class classification metrics.
        """
        p_cls, r_cls, f1_cls, support = precision_recall_fscore_support(
            self.y_true, self.y_pred_class, average=None, labels=[0, 1, 2], zero_division=0
        )
        
        p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
            self.y_true, self.y_pred_class, average='macro', zero_division=0
        )
        
        try:
            roc_auc_ovr = roc_auc_score(self.y_true, self.y_prob, multi_class='ovr', average='macro')
        except ValueError:
            roc_auc_ovr = np.nan
            
        precision_curve, recall_curve, _ = precision_recall_curve(self.y_true_high, self.y_prob_high)
        pr_auc_high = auc(recall_curve, precision_curve)

        return {
            "class_metrics": {
                "LOW":    {"precision": float(p_cls[0]), "recall": float(r_cls[0]), "f1": float(f1_cls[0]), "support": int(support[0])},
                "MEDIUM": {"precision": float(p_cls[1]), "recall": float(r_cls[1]), "f1": float(f1_cls[1]), "support": int(support[1])},
                "HIGH":   {"precision": float(p_cls[2]), "recall": float(r_cls[2]), "f1": float(f1_cls[2]), "support": int(support[2])}
            },
            "macro_avg": {
                "precision": float(p_macro),
                "recall": float(r_macro),
                "f1_score": float(f1_macro)
            },
            "roc_auc_macro_ovr": float(roc_auc_ovr) if not np.isnan(roc_auc_ovr) else None,
            "pr_auc_high": float(pr_auc_high)
        }

    def evaluate_ranking(self, top_k_values: Tuple[int, ...] = (5, 10)) -> Dict[str, float]:
        """
        Computes ranking metrics (Precision@K) based on predicted HIGH-risk probabilities.
        Measures the ratio of actual HIGH-risk files in the top-K predicted files.
        """
        results = {}
        sorted_indices = np.argsort(self.y_prob_high)[::-1]
        
        for k in top_k_values:
            if len(self.y_true) < k:
                results[f"precision_at_{k}"] = 0.0
                continue
                
            top_k_indices = sorted_indices[:k]
            actual_high_count = np.sum(self.y_true[top_k_indices] == 2)
            results[f"precision_at_{k}"] = float(actual_high_count / k)
            
        return results

    def evaluate_calibration(self, n_bins: int = 5) -> Dict[str, Any]:
        """
        Computes probability calibration metrics for HIGH-risk predictions.
        """
        try:
            brier_score = brier_score_loss(self.y_true_high, self.y_prob_high)
        except ValueError:
            brier_score = 1.0
            
        try:
            prob_true, prob_pred = calibration_curve(self.y_true_high, self.y_prob_high, n_bins=n_bins, strategy='uniform')
            true_probs = prob_true.tolist()
            pred_probs = prob_pred.tolist()
        except Exception:
            true_probs = []
            pred_probs = []
            
        ece = 0.0
        n_samples = len(self.y_prob_high)
        for i in range(n_bins):
            bin_lower = i / n_bins
            bin_upper = (i + 1) / n_bins
            
            in_bin = (self.y_prob_high > bin_lower) & (self.y_prob_high <= bin_upper)
            prop_in_bin = np.mean(in_bin)
            
            if prop_in_bin > 0:
                accuracy = np.mean(self.y_true_high[in_bin])
                confidence = np.mean(self.y_prob_high[in_bin])
                ece += prop_in_bin * np.abs(accuracy - confidence)

        return {
            "brier_score_high": float(brier_score),
            "expected_calibration_error": float(ece),
            "calibration_curve_points": {
                "true_probabilities": true_probs,
                "predicted_probabilities": pred_probs
            }
        }

    def compile_all_metrics(self) -> Dict[str, Any]:
        """
        Runs the complete evaluation suite.
        """
        return {
            "classification": self.evaluate_classification(),
            "ranking":        self.evaluate_ranking(),
            "calibration":     self.evaluate_calibration()
        }
