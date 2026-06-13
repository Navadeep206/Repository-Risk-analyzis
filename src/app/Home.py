#!/usr/bin/env python3
"""
Home Page of the Repository Risk Intelligence Platform.
Entry point for the Streamlit dashboard.
"""

import os
import sys
import streamlit as st

# Add parent directory to path to enable imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import services
from app.services.repository_service import RepositoryService
from app.services.prediction_service import PredictionService
from app.services.explainability_service import ExplainabilityService
from app.services.forecasting_service import ForecastingService

# Set page config
st.set_page_config(
    page_title="Repository Risk Intelligence Platform",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium CSS styling (curated slate/indigo theme with glassmorphism)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #6366F1 0%, #3B82F6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    .sub-header {
        font-size: 1.25rem;
        color: #6B7280;
        margin-bottom: 2rem;
    }
    
    .card {
        background: rgba(255, 255, 255, 0.8);
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
        border: 1px solid rgba(229, 231, 235, 1);
        margin-bottom: 1.5rem;
    }
    
    .metric-value {
        font-size: 2.25rem;
        font-weight: 700;
        color: #1E3A8A;
    }
    
    .metric-label {
        font-size: 0.875rem;
        font-weight: 600;
        color: #4B5563;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .badge-success {
        background-color: #D1FAE5;
        color: #065F46;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.875rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Initialize services in session state to cache across pages
if "repo_service" not in st.session_state:
    st.session_state.repo_service = RepositoryService()
if "pred_service" not in st.session_state:
    st.session_state.pred_service = PredictionService()
if "exp_service" not in st.session_state:
    st.session_state.exp_service = ExplainabilityService()
if "fore_service" not in st.session_state:
    st.session_state.fore_service = ForecastingService()
    
# Keep track of active repository in session
if "selected_repo" not in st.session_state:
    repos = st.session_state.repo_service.list_repositories()
    st.session_state.selected_repo = repos[0] if repos else "click"
if "custom_repo_df" not in st.session_state:
    st.session_state.custom_repo_df = None
if "custom_repo_summary" not in st.session_state:
    st.session_state.custom_repo_summary = None

# Title block
st.markdown('<div class="main-header">Repository Risk Intelligence Platform</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Production-grade risk evaluation, explainability, and future technical debt forecasting.</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

# 1. Project Overview & Stats
with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Platform Stats")
    
    repos_list = st.session_state.repo_service.list_repositories()
    
    # Render layout using metric layout
    m_col1, m_col2 = st.columns(2)
    with m_col1:
        st.markdown('<div class="metric-label">Mined Repositories</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{len(repos_list)}</div>', unsafe_allow_html=True)
    with m_col2:
        st.markdown('<div class="metric-label">Platform Status</div>', unsafe_allow_html=True)
        st.markdown('<div style="margin-top: 0.5rem;"><span class="badge-success">Operational</span></div>', unsafe_allow_html=True)
        
    st.markdown("""
    <br>
    **Supported Languages**:
    - Python 🐍
    - JavaScript 💛
    - TypeScript 💙
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# 2. Production Model Details (Random Forest)
with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Core Risk Engine")
    st.markdown("**Model type**: Random Forest Classifier")
    st.markdown("*Selected as the production model due to superior generalization on disjoint repository splits.*")
    
    m_col1, m_col2 = st.columns(2)
    with m_col1:
        st.markdown('<div class="metric-label">Macro F1 Score</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-value">0.6714</div>', unsafe_allow_html=True)
    with m_col2:
        st.markdown('<div class="metric-label">Weighted F1 Score</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-value">0.6845</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# 3. Forecasting Engine
with col3:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Time-Series Forecaster")
    st.markdown("**Model type**: XGBoost Regressor")
    st.markdown("*Forecasts future repository risk indices, defect frequency, and edit volatility.*")
    
    m_col1, m_col2 = st.columns(2)
    with m_col1:
        st.markdown('<div class="metric-label">Horizons</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-value">30 / 60 / 90</div>', unsafe_allow_html=True)
    with m_col2:
        st.markdown('<div class="metric-label">Best Forecaster</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-value" style="font-size:1.75rem;">XGBoost</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Side bar active repo selector
st.sidebar.title("Repository Selection")
active_repo = st.sidebar.selectbox(
    "Active Repository",
    repos_list,
    index=repos_list.index(st.session_state.selected_repo) if st.session_state.selected_repo in repos_list else 0
)
st.session_state.selected_repo = active_repo

st.sidebar.markdown("""
---
🛡️ **Platform Capabilities**:
- **Repository Analysis**: Deep metrics inspection.
- **Risk Prediction**: File-level classifications.
- **Explainability**: Code risk driver reports.
- **Risk Forecasting**: Time-series projections.
- **System Monitor**: Execution performance tracker.
""")

st.info("💡 **Getting Started**: Select a repository in the sidebar and navigate using the page selector on the left to analyze health, predict code risk, review explainability, or check forecasts!")
