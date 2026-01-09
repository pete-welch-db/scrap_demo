"""
Sidebar Component
=================
Navigation and status sidebar for the application.
Styled with Databricks Oat Medium theme.
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
        # App title with Databricks styling
        st.markdown("""
        <div style="text-align: center; padding: 16px 0 8px;">
            <h2 style="
                margin: 0; 
                font-size: 1.5rem; 
                color: #1B3139;
                font-family: 'DM Sans', sans-serif;
                font-weight: 600;
            ">🔧 Scrap Intelligence</h2>
            <p style="
                color: #6b6b69; 
                font-size: 0.875rem; 
                margin-top: 8px;
                font-family: 'DM Sans', sans-serif;
            ">
                Automotive Quality Analytics
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        
        # Quick Stats (moved above navigation)
        st.markdown("### 📈 Quick Stats")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("YTD Scrap", "$4.2M", "-8.3%")
        with col2:
            st.metric("Open 8Ds", "47", "-12")
        
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        
        # Navigation
        page = st.radio(
            "Navigation",
            ["📊 Executive Dashboard", "🎯 What-If Scenarios", "🤖 ML Insights", "💬 AI Assistant"],
            label_visibility="collapsed"
        )
        
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        
        # Settings (now includes connection status and debug mode)
        with st.expander("⚙️ Settings"):
            # Connection Status
            st.markdown("**🔌 Connection**")
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
            <p style="
                font-size: 0.7rem; 
                color: #6b6b69; 
                margin-top: 4px; 
                margin-bottom: 12px;
                font-family: 'DM Sans', sans-serif;
            ">
                Host: <code style="
                    background: #e0dfdc;
                    padding: 2px 6px;
                    border-radius: 4px;
                    font-size: 0.65rem;
                ">{host_display}</code>
            </p>
            """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Data Source
            st.markdown("**📊 Data Source**")
            data_source = st.radio(
                "Source",
                ["Live Data (Database)", "Mock Data Only"],
                label_visibility="collapsed",
                key="data_source_setting"
            )
            # Set flag for use across the app
            st.session_state.use_mock_data = (data_source == "Mock Data Only")
            
            st.markdown("---")
            
            # Debug Mode (for Genie)
            st.markdown("**🔧 Debug Options**")
            debug_mode = st.checkbox("Enable Genie Debug Mode", value=False, key="genie_debug_setting")
            if "genie_client" in st.session_state:
                st.session_state.genie_client.debug = debug_mode
            
            st.markdown("---")
            
            # Refresh button
            if st.button("🔄 Refresh Data", use_container_width=True):
                st.cache_data.clear()
                st.rerun()
        
        return page
