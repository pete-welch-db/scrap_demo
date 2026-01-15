"""
Sidebar Component
=================
Navigation and status sidebar for the application.
Styled with Databricks Oat Medium theme.
"""

import streamlit as st
from config import databricks_config, genie_config
from data.connection import test_db_connection


def render_sidebar() -> str:
    """
    Render the sidebar with navigation and status.
    
    Returns:
        str: The selected page name
    """
    with st.sidebar:
        # App title - compact
        st.markdown("""
        <div style="text-align: center; padding: 4px 0;">
            <h2 style="
                margin: 0; 
                font-size: 1.35rem; 
                color: #1B3139;
                font-family: 'DM Sans', sans-serif;
                font-weight: 600;
            ">🔧 Scrap Intelligence</h2>
            <p style="
                color: #6b6b69; 
                font-size: 0.8rem; 
                margin: 4px 0 0 0;
                font-family: 'DM Sans', sans-serif;
            ">Automotive Quality Analytics</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('<div style="height: 1px; background: #e0dfdc; margin: 12px 0;"></div>', unsafe_allow_html=True)
        
        # Navigation
        page = st.radio(
            "Navigation",
            [
                "🏠 Overview",
                "📊 Scrap Insights", 
                "🎯 What-If Scenarios",
                # "🤖 ML Insights",  # Removed - content moved to Scrap Insights tabs
                "💬 AI Assistant"
            ],
            label_visibility="collapsed"
        )
        
        # Settings at bottom (positioned via CSS flexbox)
        with st.expander("⚙️ Settings"):
            # Database Connection Status
            st.markdown("**🔌 Database**")
            db_connected = test_db_connection()
            
            if db_connected:
                st.markdown(
                    '<span class="status-badge connected">✅ Connected</span>', 
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    '<span class="status-badge disconnected">❌ Mock Data</span>', 
                    unsafe_allow_html=True
                )
            
            # Genie API Status
            st.markdown("**🧞 Genie API**")
            if genie_config.is_configured():
                st.markdown(
                    f'<span class="status-badge connected">✅ {genie_config.space_name}</span>', 
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    '<span class="status-badge disconnected">⚠️ Not Configured</span>', 
                    unsafe_allow_html=True
                )
            
            st.markdown("---")
            
            # Data Source
            st.markdown("**📊 Data Source**")
            data_source = st.radio(
                "Source",
                ["Live Data", "Mock Data"],
                label_visibility="collapsed",
                key="data_source_setting"
            )
            st.session_state.use_mock_data = (data_source == "Mock Data")
            
            st.markdown("---")
            
            # Refresh button
            if st.button("🔄 Refresh", width='stretch'):
                st.cache_data.clear()
                st.rerun()
        
        return page
