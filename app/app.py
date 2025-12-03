"""
Scrap Reduction Intelligence
============================
Main application entry point.

A modern, dark-mode Streamlit app for:
- Viewing executive dashboards (via Databricks iframe embed)
- Running what-if scenarios for scrap reduction
- AI assistant integration (Genie/multi-agent chatbots)

Deployable to Databricks Apps or locally.
"""

import streamlit as st
from config import app_config
from components import apply_custom_styles, render_sidebar
from views import render_dashboard_page, render_scenarios_page, render_assistant_page


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
# APPLY STYLES
# =============================================================================

apply_custom_styles()


# =============================================================================
# SIDEBAR & NAVIGATION
# =============================================================================

selected_page = render_sidebar()


# =============================================================================
# PAGE ROUTING
# =============================================================================

if selected_page == "📊 Executive Dashboard":
    render_dashboard_page()

elif selected_page == "🎯 What-If Scenarios":
    render_scenarios_page()

elif selected_page == "🤖 AI Assistant":
    render_assistant_page()


# =============================================================================
# FOOTER
# =============================================================================

st.markdown("""
<div style="text-align: center; padding: 40px 0 20px; border-top: 1px solid var(--border-color); margin-top: 40px;">
    <p style="color: var(--text-muted); font-size: 0.875rem;">
        Scrap Reduction Intelligence Platform | Built with Streamlit & Databricks
    </p>
</div>
""", unsafe_allow_html=True)
