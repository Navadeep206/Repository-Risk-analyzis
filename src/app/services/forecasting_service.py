#!/usr/bin/env python3
"""
Forecasting Service for Phase 10.
Runs multi-horizon forecasts (30d, 60d, 90d) on repositories
using Random Forest and XGBoost forecasting models.
"""

import os
import sys
import pickle
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "forecasting")))
from config import BASE_DIR
from forecasting.xgboost_forecaster import XGBoostForecaster
from forecasting.baseline_forecaster import PersistenceForecaster

class ForecastingService:
    """
    Service for loading and executing repository risk forecasters.
    """
    def __init__(self):
        self.models_dir = os.path.join(BASE_DIR, "models")
        self.dataset_file = os.path.join(BASE_DIR, "data", "final", "forecasting_dataset.csv")
        self.reports_dir = os.path.join(BASE_DIR, "reports", "forecasting")
        
        self.rf_models_dict = {}
        self.xgb_models_dict = {}
        self.features_list = []
        self._load_models()

    def _load_models(self):
        """Loads forecasting model structures."""
        # Load RF pickle
        rf_path = os.path.join(self.models_dir, "risk_forecaster.pkl")
        if os.path.exists(rf_path):
            try:
                with open(rf_path, "rb") as f:
                    data = pickle.load(f)
                    self.features_list = data.get("features", [])
                    self.rf_models_dict = data.get("models", {})
            except Exception as e:
                print(f"[-] Error loading RF forecaster pickle: {e}")
                
        # Pre-instantiate and load XGBoost models for each horizon
        for target in ["future_risk", "future_defect_count", "future_modification_intensity"]:
            for horizon in [30, 60, 90]:
                key = f"{target}_{horizon}d"
                xgb_path = os.path.join(self.models_dir, f"xgb_{target}_{horizon}d.json")
                if os.path.exists(xgb_path):
                    try:
                        xgb_model = XGBoostForecaster()
                        xgb_model.load_model(xgb_path)
                        self.xgb_models_dict[key] = xgb_model
                    except Exception as e:
                        print(f"[-] Error loading XGBoost model {key}: {e}")

    def get_forecast_dataset(self) -> pd.DataFrame:
        """Loads and returns the overall forecasting snapshot dataset."""
        if os.path.exists(self.dataset_file):
            return pd.read_csv(self.dataset_file)
        return pd.DataFrame()

    def get_forecasts(self, repo_name: str, target: str = "future_risk", horizon: int = 30) -> pd.DataFrame:
        """
        Retrieves actual future risk and computes predictions for a given repo, target, and horizon.
        
        Returns:
            A DataFrame containing date, actual, persistence, RF, and XGBoost predictions.
        """
        df_all = self.get_forecast_dataset()
        if df_all.empty:
            return pd.DataFrame()
            
        repo_df = df_all[df_all["repository_name"] == repo_name].copy()
        if repo_df.empty:
            return pd.DataFrame()
            
        repo_df = repo_df.sort_values("snapshot_date")
        
        target_col = f"{target}_{horizon}d"
        if target_col not in repo_df.columns:
            return pd.DataFrame()
            
        # A. Persistence baseline
        persistence = PersistenceForecaster()
        preds_persist = persistence.predict(repo_df, target_col, horizon)
        
        # B. Random Forest
        preds_rf = np.zeros(len(repo_df))
        if target == "future_risk":
            rf_key = f"future_risk_{horizon}d"
            if rf_key in self.rf_models_dict:
                try:
                    preds_rf = self.rf_models_dict[rf_key].predict(repo_df[self.features_list])
                except Exception:
                    pass
                    
        # C. XGBoost
        preds_xgb = np.zeros(len(repo_df))
        xgb_key = f"{target}_{horizon}d"
        if xgb_key in self.xgb_models_dict:
            try:
                preds_xgb = self.xgb_models_dict[xgb_key].predict(repo_df[self.features_list])
            except Exception:
                pass
                
        # Construct output trajectory DataFrame
        df_out = pd.DataFrame({
            "snapshot_date": repo_df["snapshot_date"],
            "actual": repo_df[target_col],
            "persistence": preds_persist,
            "random_forest": preds_rf,
            "xgboost": preds_xgb
        })
        return df_out

    def get_best_model_metrics(self) -> Dict[str, any]:
        """Loads the best model metrics JSON file from Phase 9."""
        metrics_file = os.path.join(self.reports_dir, "best_model_metrics.json")
        if os.path.exists(metrics_file):
            with open(metrics_file, "r") as f:
                return json.load(f)
        return {}
        
    def get_model_comparisons(self) -> pd.DataFrame:
        """Loads the model comparisons CSV file from Phase 9."""
        comp_file = os.path.join(self.reports_dir, "model_comparison.csv")
        if os.path.exists(comp_file):
            return pd.read_csv(comp_file)
        return pd.DataFrame()
