"""
Scrap Reduction Intelligence
============================
Main application entry point.

A Streamlit app with Databricks Oat Medium branding for:
- Viewing executive dashboards (via Databricks iframe embed)
- Running what-if scenarios for scrap reduction
- AI assistant integration (Genie/multi-agent chatbots)

Deployable to Databricks Apps or locally.
"""

import streamlit as st
from config import app_config
from components import apply_custom_styles, render_header, render_sidebar
from views import render_landing_page, render_dashboard_page, render_scenarios_page, render_assistant_page
# from views import render_ml_insights_page  # Commented out - moved to Dashboard tabs


# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title=app_config.title,
    page_icon=app_config.icon,
    layout=app_config.layout,
    initial_sidebar_state="expanded"
)


# =============================================================================
# APPLY STYLES & HEADER
# =============================================================================

apply_custom_styles()
render_header()


# =============================================================================
# SIDEBAR & NAVIGATION
# =============================================================================

selected_page = render_sidebar()


# =============================================================================
# PAGE ROUTING
# =============================================================================

if selected_page == "🏠 Overview":
    render_landing_page()

elif selected_page == "📊 Executive Dashboard":
    render_dashboard_page()

elif selected_page == "🎯 What-If Scenarios":
    render_scenarios_page()

# elif selected_page == "🤖 ML Insights":
#     render_ml_insights_page()  # Commented out - moved to Dashboard tabs

elif selected_page == "💬 AI Assistant":
    render_assistant_page()


# =============================================================================
# FOOTER WITH DATABRICKS BRANDING
# =============================================================================

st.markdown("""
<div style="
    text-align: center; 
    padding: 40px 0 20px; 
    border-top: 1px solid #e0dfdc; 
    margin-top: 40px;
">
    <p style="
        color: #6b6b69; 
        font-size: 0.875rem; 
        margin: 0;
        font-family: 'DM Sans', sans-serif;
    ">
        Scrap Reduction Intelligence Platform | Built with Streamlit & Databricks
    </p>
</div>
""", unsafe_allow_html=True)
