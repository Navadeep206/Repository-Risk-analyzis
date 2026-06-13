#!/usr/bin/env python3
"""
Streamlit Page 2: Risk Prediction.
Performs classification using the Random Forest risk engine and applies the Trust Gate.
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

# Set page config
st.set_page_config(page_title="Risk Prediction - Risk Intelligence", page_icon="🛡️", layout="wide")

st.markdown("""
<style>
    .card {
        background: rgba(255, 255, 255, 0.8);
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        border: 1px solid rgba(229, 231, 235, 1);
        margin-bottom: 1.5rem;
    }
    .trust-header {
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .trust-text {
        font-size: 1rem;
        color: #374151;
    }
    .badge {
        padding: 0.35rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .badge-low { background-color: #D1FAE5; color: #065F46; }
    .badge-med { background-color: #FEF3C7; color: #92400E; }
    .badge-high { background-color: #FEE2E2; color: #991B1B; }
</style>
""", unsafe_allow_html=True)

st.title("🛡️ Production Risk Prediction Engine")

selected_repo = st.session_state.selected_repo
repo_service = st.session_state.repo_service
pred_service = st.session_state.pred_service

# Load active metrics
if selected_repo == "Uploaded Codebase" and st.session_state.custom_repo_df is not None:
    df_metrics = st.session_state.custom_repo_df
else:
    df_metrics = repo_service.load_metrics(selected_repo)
    
if df_metrics.empty:
    st.warning("⚠️ Please select or upload a valid repository dataset first.")
else:
    if not pred_service.is_ready():
        st.error("❌ Risk prediction models could not be loaded. Verify 'models/random_forest.pkl' and 'models/preprocessor.pkl' exist.")
    else:
        with st.spinner("Executing Random Forest inference & Trust Gate analysis..."):
            df_preds = pred_service.predict(df_metrics)
            
        st.success(f"Predictions calculated successfully for {len(df_preds)} files.")
        
        # 1. Trust Gate Banner (Aggregated)
        mean_conf = df_preds["confidence"].mean()
        gate_rating, gate_color = pred_service.evaluate_trust_gate(mean_conf)
        
        st.markdown(f"""
        <div class="card" style="border-left: 8px solid {gate_color}; background: {gate_color}10;">
            <div class="trust-header" style="color: {gate_color};">Trust Gate Status: {gate_rating}</div>
            <div class="trust-text">
                Average Model Prediction Confidence: <strong>{mean_conf:.2f}%</strong><br>
                Gating Rating: <em>{gate_rating}</em>. {'This codebase is highly predictable and safe for production deploy.' if mean_conf >= 90 else 'Some files require review due to model classification boundary shifts.' if mean_conf >= 70 else 'High volatility or scale mismatch detected. Manual code audit is strongly recommended.'}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        col_charts1, col_charts2 = st.columns(2)
        
        # 2. Risk Distribution Pie Chart
        with col_charts1:
            st.subheader("Predicted Risk Distribution")
            risk_counts = df_preds["predicted_risk"].value_counts().reset_index()
            risk_counts.columns = ["risk_level", "count"]
            
            # Map colors: LOW=green, MEDIUM=orange, HIGH=red
            color_map = {"LOW": "#10B981", "MEDIUM": "#F59E0B", "HIGH": "#EF4444"}
            
            fig_pie = px.pie(
                risk_counts, 
                values="count", 
                names="risk_level",
                color="risk_level",
                color_discrete_map=color_map,
                hole=0.4
            )
            fig_pie.update_layout(margin=dict(t=10, b=10, l=10, r=10))
            st.plotly_chart(fig_pie, use_container_width=True)
            
        # 3. Confidence Histograms
        with col_charts2:
            st.subheader("Model Confidence Distribution")
            fig_hist = px.histogram(
                df_preds,
                x="confidence",
                nbins=20,
                color="predicted_risk",
                color_discrete_map=color_map,
                labels={"confidence": "Confidence Percentage (%)", "count": "File Count"},
                marginal="rug"
            )
            fig_hist.update_layout(margin=dict(t=10, b=10, l=10, r=10), showlegend=True)
            st.plotly_chart(fig_hist, use_container_width=True)
            
        # 4. Filterable Table
        st.subheader("Detailed File Predictions")
        
        # Filter checkboxes
        c_filter1, c_filter2 = st.columns(2)
        with c_filter1:
            selected_levels = st.multiselect(
                "Filter by Predicted Risk", 
                ["LOW", "MEDIUM", "HIGH"], 
                default=["LOW", "MEDIUM", "HIGH"]
            )
        with c_filter2:
            search_query = st.text_input("Search file paths", "")
            
        df_filtered = df_preds[df_preds["predicted_risk"].isin(selected_levels)]
        if search_query:
            df_filtered = df_filtered[df_filtered["file_path"].str.contains(search_query, case=False, na=False)]
            
        # Format table styling before render
        display_cols = ["file_path", "language", "loc", "complexity", "maintainability_index", "predicted_risk", "confidence", "trust_rating"]
        df_display = df_filtered[display_cols].copy()
        
        # Format columns for nice appearance
        df_display["confidence"] = df_display["confidence"].map(lambda x: f"{x:.1f}%")
        df_display["maintainability_index"] = df_display["maintainability_index"].map(lambda x: f"{x:.1f}")
        
        st.dataframe(df_display, use_container_width=True, height=400)
        
        # Download Prediction Report
        st.subheader("Export Predictions Report")
        csv_preds = df_preds.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Predictions (CSV)",
            data=csv_preds,
            file_name=f"{selected_repo}_predictions_report.csv",
            mime="text/csv"
        )
st.sidebar.markdown(f"**Selected Repo**: `{selected_repo}`")
