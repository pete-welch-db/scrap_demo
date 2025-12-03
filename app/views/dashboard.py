"""
Executive Dashboard Page
========================
Main dashboard with embedded Databricks dashboard and KPIs.
"""

import streamlit as st
import base64
from pathlib import Path
from config import databricks_config, app_config
from components.metrics import create_metric_card
from data.mock_data import get_live_kpis, get_mock_kpis


def get_logo_base64():
    """Load Databricks logo as base64 string."""
    current_dir = Path(__file__).parent.parent
    logo_path = current_dir / "assets" / "Databricks_Logo.png"
    
    if logo_path.exists():
        with open(logo_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return None


def format_number(value: float, prefix: str = "", suffix: str = "") -> str:
    """Format large numbers with K, M, B suffixes."""
    if value >= 1_000_000_000:
        return f"{prefix}{value/1_000_000_000:.2f}B{suffix}"
    elif value >= 1_000_000:
        return f"{prefix}{value/1_000_000:.2f}M{suffix}"
    elif value >= 1_000:
        return f"{prefix}{value/1_000:.2f}K{suffix}"
    else:
        return f"{prefix}{value:.0f}{suffix}"


def render_dashboard_page():
    """Render the Executive Dashboard page."""
    
    # Header with Databricks branding
    col_title, col_logo = st.columns([3, 1])
    
    with col_title:
        st.markdown("""
        <h1>📊 Executive Dashboard</h1>
        <p style="font-size: 1.1rem; color: #64748b; margin-top: -8px;">
            Real-time scrap analytics and KPIs from your Databricks workspace
        </p>
        """, unsafe_allow_html=True)
    
    with col_logo:
        logo_b64 = get_logo_base64()
        if logo_b64:
            st.markdown(f"""
            <div style="text-align: right; padding-top: 10px;">
                <p style="font-size: 0.7rem; color: #64748b; margin-bottom: 8px;">Powered by</p>
                <img src="data:image/png;base64,{logo_b64}" 
                     alt="Databricks" 
                     style="height: 40px;">
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="text-align: right; padding-top: 10px;">
                <p style="font-size: 0.7rem; color: #64748b; margin-bottom: 4px;">Powered by</p>
                <p style="color: #FF3621; font-weight: bold; font-size: 1.2rem;">Databricks</p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Load KPIs - try live data first, fall back to mock
    try:
        kpis = get_live_kpis()
    except:
        kpis = get_mock_kpis()
    
    # KPI Cards (these show properly, unlike embedded counter widgets)
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(create_metric_card(
            format_number(kpis['ytd_scrap_cost'], prefix="$"),
            "Total Scrap Cost (YTD)", 
            kpis.get('ytd_scrap_delta')
        ), unsafe_allow_html=True)
    
    with col2:
        st.markdown(create_metric_card(
            format_number(kpis['ytd_units_scrapped']),
            "Units Scrapped (YTD)", 
            kpis.get('units_delta')
        ), unsafe_allow_html=True)
    
    with col3:
        st.markdown(create_metric_card(
            format_number(kpis.get('scrap_incidents', 0)),
            "Scrap Incidents (YTD)", 
            kpis.get('incidents_delta')
        ), unsafe_allow_html=True)
    
    with col4:
        st.markdown(create_metric_card(
            format_number(kpis.get('open_8d_reports', 0)),
            "Open 8D Reports", 
            kpis.get('eight_d_delta')
        ), unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Dashboard section
    st.markdown("### 📈 Dashboard Charts")
    
    # Info about KPI widgets
    st.info("💡 **Note:** KPI counter widgets don't render in embedded mode. The metrics above are pulled directly from your data.", icon="ℹ️")
    
    # Dashboard controls
    dash_col1, dash_col2 = st.columns([3, 1])
    
    with dash_col2:
        dashboard_height = st.slider(
            "Chart Height", 
            400, 1200, 
            app_config.default_dashboard_height, 
            50,
            key="dash_height_slider"
        )
    
    # Embed Databricks Dashboard via iframe
    iframe_html = f"""
    <div style="border-radius: 16px; overflow: hidden; border: 1px solid #2d2d44; background: #12121a;">
        <iframe
            src="{databricks_config.dashboard_embed_url}"
            width="100%"
            height="{dashboard_height}px"
            frameborder="0"
            style="border: none; background: #12121a;">
        </iframe>
    </div>
    """
    
    st.markdown(iframe_html, unsafe_allow_html=True)
    
    # Dashboard links
    st.markdown("---")
    
    col_link1, col_link2 = st.columns(2)
    
    with col_link1:
        dashboard_url = f"https://{databricks_config.host}/sql/dashboards/{databricks_config.dashboard_embed_url.split('/')[-1].split('?')[0]}"
        st.link_button(
            "🔗 Open Full Dashboard in Databricks",
            f"https://{databricks_config.host}/dashboardsv3/01f0cfcf667f1b8ba6b3f8158b94f453",
            use_container_width=True
        )
    
    with col_link2:
        st.markdown(f"""
        <div style="text-align: center; padding: 8px;">
            <span style="color: #64748b; font-size: 0.75rem;">
                Dashboard ID: <code style="color: #a855f7;">01f0cfcf667f1b8ba6b3f8158b94f453</code>
            </span>
        </div>
        """, unsafe_allow_html=True)
