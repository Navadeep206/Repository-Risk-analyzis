#!/usr/bin/env python3
"""
Streamlit Page 4: Forecasting.
Renders risk forecasting curves, multi-horizon comparison charts, and displays error reports.
"""

import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from config import BASE_DIR

# Set page config
st.set_page_config(page_title="Future Risk Forecasting - Risk Intelligence", page_icon="🔮", layout="wide")

st.markdown("""
<style>
    .card {
        background: rgba(255, 255, 255, 0.8);
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        border: 1px solid rgba(229, 231, 235, 1);
        margin-bottom: 1rem;
        text-align: center;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #8B5CF6;
    }
    .metric-label {
        font-size: 0.8rem;
        font-weight: 600;
        color: #6B7280;
        text-transform: uppercase;
    }
</style>
""", unsafe_allow_html=True)

st.title("🔮 Time-Series Repository Risk Forecasting")

fore_service = st.session_state.fore_service
selected_repo = st.session_state.selected_repo

# If uploaded codebase is selected, warn that historical time-series forecasting is not available unless logs are uploaded
if selected_repo == "Uploaded Codebase":
    st.warning("⚠️ Time-Series Forecasting requires historical commit and modification timelines. Pre-mined repositories contain complete histories, while ZIP uploads only have static code metrics.")
    st.info("Please select a standard repository in the sidebar (e.g. click, axios, redux) to inspect risk forecasts.")
else:
    # Option selections
    c_opt1, c_opt2 = st.columns(2)
    with c_opt1:
        target = st.selectbox(
            "Forecast Target Metric", 
            ["future_risk", "future_defect_count", "future_modification_intensity"],
            format_func=lambda x: x.replace("_", " ").title()
        )
    with c_opt2:
        horizon = st.selectbox("Forecast Horizon (Days)", [30, 60, 90], index=0)
        
    with st.spinner("Retrieving forecasting records..."):
        df_forecast = fore_service.get_forecasts(selected_repo, target=target, horizon=horizon)
        
    if df_forecast.empty:
        st.error(f"❌ Could not load forecasting snapshots for repository '{selected_repo}'. Verify final/forecasting_dataset.csv exists.")
    else:
        # 1. Multi-horizon summary cards (using the latest snapshot date)
        st.subheader("Latest Risk Forecast Projections")
        latest_row = df_forecast.iloc[-1]
        
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"""
            <div class="card">
                <div class="metric-label">XGBoost Forecast (Latest)</div>
                <div class="metric-value">{latest_row['xgboost']:.2f}</div>
                <div style="font-size: 0.75rem; color:#6B7280; margin-top:0.25rem;">Date: {latest_row['snapshot_date']}</div>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="card">
                <div class="metric-label">Random Forest Forecast (Latest)</div>
                <div class="metric-value">{latest_row['random_forest']:.2f}</div>
                <div style="font-size: 0.75rem; color:#6B7280; margin-top:0.25rem;">Date: {latest_row['snapshot_date']}</div>
            </div>
            """, unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
            <div class="card">
                <div class="metric-label">Actual Future Value</div>
                <div class="metric-value">{latest_row['actual']:.2f}</div>
                <div style="font-size: 0.75rem; color:#6B7280; margin-top:0.25rem;">Date: {latest_row['snapshot_date']}</div>
            </div>
            """, unsafe_allow_html=True)
            
        # 2. Interactive Trajectory Curve
        st.subheader("Forecasting Trajectory Curve Over Time")
        fig = go.Figure()
        
        # Convert date to datetime
        dates = pd.to_datetime(df_forecast["snapshot_date"])
        
        fig.add_trace(go.Scatter(x=dates, y=df_forecast["actual"], name="Actual Target", line=dict(color="#3B82F6", width=2.5)))
        fig.add_trace(go.Scatter(x=dates, y=df_forecast["xgboost"], name="XGBoost Prediction", line=dict(color="#10B981", width=2, dash="dash")))
        fig.add_trace(go.Scatter(x=dates, y=df_forecast["random_forest"], name="Random Forest Prediction", line=dict(color="#8B5CF6", width=2, dash="dashdot")))
        fig.add_trace(go.Scatter(x=dates, y=df_forecast["persistence"], name="Persistence Baseline", line=dict(color="#EF4444", width=1.5, dash="dot")))
        
        fig.update_layout(
            xaxis_title="Snapshot Date",
            yaxis_title=f"{target.replace('_', ' ').title()} value ({horizon}-day horizon)",
            hovermode="x unified",
            margin=dict(t=10, b=10, l=10, r=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Tabs for details & reports
        t_details, t_comp, t_report = st.tabs(["Snapshot Data Table", "Model Comparisons", "Forecasting Error Analysis Report"])
        
        with t_details:
            st.dataframe(df_forecast, use_container_width=True)
            
        with t_comp:
            df_comp = fore_service.get_model_comparisons()
            if not df_comp.empty:
                st.dataframe(df_comp, use_container_width=True)
            else:
                st.warning("No model comparisons data resolved.")
                
        with t_report:
            report_path = os.path.join(BASE_DIR, "reports", "forecasting", "forecast_error_analysis.md")
            if os.path.exists(report_path):
                with open(report_path, "r") as f:
                    st.markdown(f.read())
            else:
                st.warning("Forecasting error analysis report markdown file not found on disk.")
                
        # Export Reports
        st.subheader("Export Forecast Report")
        csv_data = df_forecast.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Forecast Trajectory (CSV)",
            data=csv_data,
            file_name=f"{selected_repo}_{target}_{horizon}d_forecast.csv",
            mime="text/csv"
        )
st.sidebar.markdown(f"**Selected Repo**: `{selected_repo}`")
