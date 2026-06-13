#!/usr/bin/env python3
"""
Streamlit Page 1: Repository Analysis.
Allows selecting or uploading codebases and visualizes metrics like LOC, complexity, maintainability.
"""

import os
import sys
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from config import BASE_DIR

# Set page config
st.set_page_config(page_title="Repository Analysis - Risk Intelligence", page_icon="📈", layout="wide")

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
        color: #2563EB;
    }
    .metric-label {
        font-size: 0.8rem;
        font-weight: 600;
        color: #6B7280;
        text-transform: uppercase;
    }
</style>
""", unsafe_allow_html=True)

st.title("📈 Repository Health & Quality Analysis")

# Setup layout tabs
tabs = st.tabs(["Select Repository", "Upload Zipped Codebase", "Upload Quality CSV"])

repo_service = st.session_state.repo_service
selected_repo = st.session_state.selected_repo
df_metrics = pd.DataFrame()
summary = {}

# TAB 1: Selection
with tabs[0]:
    st.write(f"Analyzing pre-extracted production repository: **{selected_repo}**")
    df_metrics = repo_service.load_metrics(selected_repo)
    summary = repo_service.load_summary(selected_repo)
    
# TAB 2: Upload Zipped Repository
with tabs[1]:
    uploaded_zip = st.file_uploader("Upload a zipped repository folder (.zip)", type=["zip"])
    if uploaded_zip is not None:
        with st.spinner("Extracting and parsing codebase metrics..."):
            df_zip, zip_summary = repo_service.process_uploaded_zip(uploaded_zip)
            if not df_zip.empty:
                st.session_state.custom_repo_df = df_zip
                st.session_state.custom_repo_summary = zip_summary
                st.success("Zip file parsed successfully!")
            else:
                st.error("No valid code files (Python/JS/TS) found in ZIP archive.")
                
    if st.session_state.custom_repo_summary is not None:
        if st.button("Use Uploaded ZIP Repository"):
            st.session_state.selected_repo = "Uploaded Codebase"
            st.rerun()

# TAB 3: Upload Quality CSV
with tabs[2]:
    uploaded_csv = st.file_uploader("Upload pre-computed quality metrics CSV (must contain loc, complexity, maintainability_index, language)", type=["csv"])
    if uploaded_csv is not None:
        try:
            df_csv = pd.read_csv(uploaded_csv)
            # Verify columns
            req = ["loc", "complexity", "maintainability_index"]
            if all(col in df_csv.columns for col in req):
                summary_csv = {
                    "repository_name": "Custom Uploaded CSV",
                    "language": df_csv["language"].iloc[0] if "language" in df_csv.columns else "python",
                    "loc": int(df_csv["loc"].sum()),
                    "commits_count": int(df_csv["commit_count"].sum()) if "commit_count" in df_csv.columns else 100,
                    "contributors_count": int(df_csv["contributor_count"].max()) if "contributor_count" in df_csv.columns else 5,
                    "repository_age_days": int(df_csv["repository_age_days"].max()) if "repository_age_days" in df_csv.columns else 365
                }
                
                # Check engineered fields
                if "repository_age_days" not in df_csv.columns:
                    df_csv["repository_age_days"] = summary_csv["repository_age_days"]
                if "commit_count" not in df_csv.columns:
                    df_csv["commit_count"] = 10
                if "commit_frequency" not in df_csv.columns:
                    df_csv["commit_frequency"] = df_csv["commit_count"] / df_csv["repository_age_days"]
                if "modification_count" not in df_csv.columns:
                    df_csv["modification_count"] = 15
                if "contributor_count" not in df_csv.columns:
                    df_csv["contributor_count"] = 3
                
                st.session_state.custom_repo_df = df_csv
                st.session_state.custom_repo_summary = summary_csv
                st.success("CSV file loaded successfully!")
                
                if st.button("Use Uploaded CSV Repository"):
                    st.session_state.selected_repo = "Uploaded Codebase"
                    st.rerun()
            else:
                st.error(f"CSV missing one or more required columns: {req}")
        except Exception as e:
            st.error(f"Error reading CSV: {e}")

# Resolve active dataset
if selected_repo == "Uploaded Codebase" and st.session_state.custom_repo_df is not None:
    df_metrics = st.session_state.custom_repo_df
    summary = st.session_state.custom_repo_summary
elif selected_repo == "Uploaded Codebase":
    st.warning("No uploaded codebase found. Defaulting back to clicking repository.")
    st.session_state.selected_repo = "click"
    st.rerun()

# Display summary metrics cards
if not df_metrics.empty:
    st.subheader(f"Repository Health Overview: {summary.get('repository_name', selected_repo)}")
    
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    
    with c1:
        st.markdown(f'<div class="card"><div class="metric-label">Total LOC</div><div class="metric-value">{summary.get("loc", 0):,}</div></div>', unsafe_allow_html=True)
    with c2:
        avg_comp = df_metrics["complexity"].mean() if "complexity" in df_metrics.columns else 0.0
        st.markdown(f'<div class="card"><div class="metric-label">Avg Complexity</div><div class="metric-value">{avg_comp:.2f}</div></div>', unsafe_allow_html=True)
    with c3:
        avg_maint = df_metrics["maintainability_index"].mean() if "maintainability_index" in df_metrics.columns else 100.0
        st.markdown(f'<div class="card"><div class="metric-label">Avg Maintainability</div><div class="metric-value">{avg_maint:.1f}</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="card"><div class="metric-label">Total Commits</div><div class="metric-value">{summary.get("commits_count", 0):,}</div></div>', unsafe_allow_html=True)
    with c5:
        st.markdown(f'<div class="card"><div class="metric-label">Contributors</div><div class="metric-value">{summary.get("contributors_count", 0)}</div></div>', unsafe_allow_html=True)
    with c6:
        st.markdown(f'<div class="card"><div class="metric-label">Repo Age (Days)</div><div class="metric-value">{summary.get("repository_age_days", 1)}</div></div>', unsafe_allow_html=True)

    # Visualization plots
    col_plot1, col_plot2 = st.columns(2)
    
    with col_plot1:
        st.subheader("Language Distribution")
        # Language counts
        lang_counts = df_metrics.groupby("language")["loc"].sum().reset_index()
        fig_lang = px.pie(lang_counts, values="loc", names="language", hole=0.4, 
                          color_discrete_sequence=px.colors.qualitative.Pastel)
        fig_lang.update_layout(margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(fig_lang, use_container_width=True)
        
    with col_plot2:
        st.subheader("Complexity vs. Maintainability Index")
        fig_scatter = px.scatter(
            df_metrics,
            x="complexity",
            y="maintainability_index",
            size="loc",
            color="language",
            hover_name="file_path",
            opacity=0.7,
            labels={"complexity": "Cyclomatic Complexity", "maintainability_index": "Maintainability Index"},
            color_discrete_sequence=px.colors.qualitative.Safe
        )
        fig_scatter.update_layout(margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(fig_scatter, use_container_width=True)
        
    # Exporter
    st.subheader("Export Repository Analysis Data")
    csv_data = df_metrics.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Repository Analysis (CSV)",
        data=csv_data,
        file_name=f"{selected_repo}_analysis_report.csv",
        mime="text/csv"
    )
else:
    st.warning("No data found for the selected repository. Please analyze or upload one.")
st.sidebar.markdown(f"**Selected Repo**: `{selected_repo}`")
