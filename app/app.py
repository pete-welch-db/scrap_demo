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
import base64
from pathlib import Path
from config import app_config
from components import apply_custom_styles, render_sidebar
from views import render_dashboard_page, render_scenarios_page, render_ml_insights_page, render_assistant_page


def get_databricks_logo_base64():
    """Load and encode the Databricks logo as base64."""
    logo_path = Path(__file__).parent / "assets" / "Databricks_Logo.png"
    if logo_path.exists():
        with open(logo_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return None


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

elif selected_page == "🤖 ML Insights":
    render_ml_insights_page()

elif selected_page == "💬 AI Assistant":
    render_assistant_page()


# =============================================================================
# FOOTER WITH DATABRICKS BRANDING
# =============================================================================

logo_b64 = get_databricks_logo_base64()
if logo_b64:
    st.markdown(f"""
    <div style="
        text-align: center; 
        padding: 40px 0 20px; 
        border-top: 1px solid var(--border-color); 
        margin-top: 40px;
    ">
        <div style="
            display: inline-block;
            background: white;
            padding: 12px 24px;
            border-radius: 8px;
            margin-bottom: 16px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        ">
            <img src="data:image/png;base64,{logo_b64}" 
                 style="max-height: 32px; width: auto;" 
                 alt="Powered by Databricks">
        </div>
        <p style="color: var(--text-muted); font-size: 0.875rem; margin: 0;">
            Scrap Reduction Intelligence Platform | Built with Streamlit & Databricks
        </p>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div style="text-align: center; padding: 40px 0 20px; border-top: 1px solid var(--border-color); margin-top: 40px;">
        <p style="color: var(--text-muted); font-size: 0.875rem;">
            Scrap Reduction Intelligence Platform | Built with Streamlit & Databricks
        </p>
    </div>
    """, unsafe_allow_html=True)
