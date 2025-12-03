"""
Sidebar Component
=================
Navigation and status sidebar for the application.
"""

import streamlit as st
from config import databricks_config
from data.connection import test_db_connection


def render_sidebar() -> str:
    """
    Render the sidebar with navigation and status.
    
    Returns:
        str: The selected page name
    """
    with st.sidebar:
        # Logo and title
        st.markdown("""
        <div style="text-align: center; padding: 20px 0;">
            <h2 style="margin: 0; font-size: 1.5rem;">🔧 Scrap Intelligence</h2>
            <p style="color: #64748b; font-size: 0.875rem; margin-top: 8px;">
                Automotive Quality Analytics
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        
        # Navigation
        page = st.radio(
            "Navigation",
            ["📊 Executive Dashboard", "🎯 What-If Scenarios", "🤖 ML Insights", "💬 AI Assistant"],
            label_visibility="collapsed"
        )
        
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        
        # Connection Status
        st.markdown("### 🔌 Connection")
        db_connected = test_db_connection()
        
        if db_connected:
            st.markdown(
                '<span class="status-badge connected">✅ Databricks Connected</span>', 
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<span class="status-badge disconnected">❌ Using Mock Data</span>', 
                unsafe_allow_html=True
            )
        
        host_display = databricks_config.host[:30] + "..." if len(databricks_config.host) > 30 else databricks_config.host
        st.markdown(f"""
        <p style="font-size: 0.7rem; color: #64748b; margin-top: 8px;">
            Host: <code>{host_display}</code>
        </p>
        """, unsafe_allow_html=True)
        
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        
        # Quick Stats
        st.markdown("### Quick Stats")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("YTD Scrap", "$4.2M", "-8.3%")
        with col2:
            st.metric("Open 8Ds", "47", "-12")
        
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        
        # Settings
        with st.expander("⚙️ Settings"):
            if st.button("🔄 Refresh Data"):
                st.cache_data.clear()
                st.rerun()
            
            st.markdown("**Data Source**")
            data_source = st.radio(
                "Source",
                ["Auto (DB if available)", "Mock Data Only"],
                label_visibility="collapsed",
                key="data_source_setting"
            )
            st.session_state.data_source = data_source
        
        return page

