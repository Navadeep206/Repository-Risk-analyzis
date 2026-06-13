#!/usr/bin/env python3
"""
Streamlit Page 3: Explainability.
Visualizes feature importances, global feature rankings, and prints explainability and domain shift reports.
"""

import os
import sys
import streamlit as st
import pandas as pd
import plotly.express as px

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

# Set page config
st.set_page_config(page_title="Model Explainability - Risk Intelligence", page_icon="🔍", layout="wide")

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
</style>
""", unsafe_allow_html=True)

st.title("🔍 Model Explainability & Risk Drivers")

exp_service = st.session_state.exp_service
selected_repo = st.session_state.selected_repo

# Tabs for layouts
tabs = st.tabs(["Feature Importance", "Global Rankings", "Error Analysis Report", "Domain Shift Report", "Model Trustworthiness"])

# TAB 1: Feature Importance Charts
with tabs[0]:
    st.subheader("Global Feature Importance (Random Forest)")
    df_imp = exp_service.get_feature_importances()
    
    if df_imp.empty:
        st.warning("⚠️ No feature importances available. Ensure random_forest.pkl and preprocessor.pkl are present in models/.")
    else:
        # Sort and limit to top 20
        df_imp_top = df_imp.sort_values(df_imp.columns[1], ascending=True).tail(20)
        
        # Determine the importance column
        imp_col = df_imp_top.columns[1]
        
        fig_imp = px.bar(
            df_imp_top,
            x=imp_col,
            y="feature_name",
            orientation="h",
            labels={imp_col: "Intrinsic Gini Importance", "feature_name": "Feature Name"},
            color=imp_col,
            color_continuous_scale="Viridis",
        )
        fig_imp.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=500)
        st.plotly_chart(fig_imp, use_container_width=True)
        
        # Render Permutation Importance if exists
        df_perm = exp_service.get_permutation_importances()
        if not df_perm.empty:
            st.subheader("Permutation Feature Importance (Generalization Impact)")
            df_perm_top = df_perm.sort_values("rf_permutation_mean", ascending=True).tail(20)
            
            fig_perm = px.bar(
                df_perm_top,
                x="rf_permutation_mean",
                y="feature_name",
                error_x="rf_permutation_std",
                orientation="h",
                labels={"rf_permutation_mean": "Drop in Accuracy upon Permutation", "feature_name": "Feature Name"},
                color="rf_permutation_mean",
                color_continuous_scale="Cividis"
            )
            fig_perm.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=500)
            st.plotly_chart(fig_perm, use_container_width=True)

# TAB 2: Global Rankings Table
with tabs[1]:
    st.subheader("Combined Multi-Method Feature Rankings")
    df_rank = exp_service.get_global_rankings()
    if df_rank.empty:
        # Fallback list if CSV missing
        st.info("Feature importance rankings compiled from Scikit-Learn Intrinsic and Permutation methods:")
        df_rank = df_imp.copy()
        if not df_rank.empty:
            df_rank["average_rank"] = range(1, len(df_rank) + 1)
            
    if not df_rank.empty:
        st.dataframe(df_rank, use_container_width=True)
    else:
        st.warning("No global ranking metrics resolved.")

# TAB 3: Error Analysis Report
with tabs[2]:
    st.subheader("Error Analysis Report")
    err_report = exp_service.get_explainability_report("error_analysis.md")
    st.markdown(err_report)

# TAB 4: Domain Shift Report
with tabs[3]:
    st.subheader("Domain Shift findings")
    shift_report = exp_service.get_explainability_report("domain_shift_analysis.md")
    st.markdown(shift_report)

# TAB 5: Model Trustworthiness
with tabs[4]:
    st.subheader("Model Trustworthiness Analysis")
    trust_report = exp_service.get_explainability_report("model_trustworthiness.md")
    st.markdown(trust_report)

st.sidebar.markdown(f"**Selected Repo**: `{selected_repo}`")
