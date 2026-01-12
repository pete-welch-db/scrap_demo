"""
Custom Styles
=============
Databricks Oat Medium theme CSS styling for the application.
Uses official Databricks brand colors and DM Sans font.
"""

import streamlit as st


def apply_custom_styles():
    """Apply Databricks Oat Medium theme CSS styling to the app."""
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,100..1000;1,9..40,100..1000&display=swap');
        
        :root {
            /* Databricks Primary Palette */
            --lava-600: #FF3621;
            --lava-500: #FF5F46;
            --navy-800: #1B3139;
            --navy-900: #0B2026;
            --oat-light: #F9F7F4;
            --oat-medium: #EEEDE9;
            
            /* Semantic colors */
            --bg-primary: #EEEDE9;
            --bg-secondary: #F9F7F4;
            --bg-card: #FFFFFF;
            --text-primary: #151513;
            --text-secondary: #4a4a48;
            --text-muted: #6b6b69;
            --border-color: #e0dfdc;
            --accent-primary: #FF3621;
            --accent-secondary: #1B3139;
            --success: #10b981;
            --warning: #f59e0b;
            --danger: #ef4444;
        }
        
        /* Base App Styling */
        .stApp {
            background-color: var(--bg-primary);
            font-family: "DM Sans", sans-serif;
            font-variation-settings: "opsz" 14;
        }
        
        /* Hide default Streamlit header to use custom */
        header[data-testid="stHeader"] {
            display: none;
        }
        
        /* Reduce top padding in main content area */
        .stMainBlockContainer {
            padding-top: 0 !important;
        }
        
        [data-testid="stAppViewBlockContainer"] {
            padding-top: 0 !important;
        }
        
        .block-container {
            padding-top: 0.5rem !important;
        }
        
        /* Sidebar */
        [data-testid="stSidebar"] {
            background: var(--bg-secondary);
            border-right: 1px solid var(--border-color);
        }
        
        /* Make sidebar use flexbox to push settings to bottom */
        [data-testid="stSidebar"] [data-testid="stSidebarContent"] {
            display: flex !important;
            flex-direction: column !important;
            height: calc(100vh - 60px) !important;
            padding-bottom: 20px !important;
        }
        
        /* Settings expander pushed to bottom */
        [data-testid="stSidebar"] [data-testid="stExpander"] {
            margin-top: auto !important;
        }
        
        [data-testid="stSidebar"] .stMarkdown {
            color: var(--text-primary);
        }
        
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
            color: var(--text-secondary);
        }
        
        /* Typography */
        h1, h2, h3, h4, h5, h6 {
            color: var(--text-primary) !important;
            font-family: "DM Sans", sans-serif;
            font-weight: 600;
        }
        
        h1 {
            color: var(--navy-800) !important;
            font-weight: 700;
        }
        
        p, span, label, .stMarkdown {
            color: var(--text-secondary);
            font-family: "DM Sans", sans-serif;
        }
        
        /* Metric Cards */
        .metric-card {
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 24px;
            margin: 8px 0;
            transition: all 0.2s ease;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
        }
        
        .metric-card:hover {
            border-color: var(--lava-500);
            box-shadow: 0 4px 12px rgba(255, 54, 33, 0.1);
        }
        
        .metric-value {
            font-size: 2.5rem;
            font-weight: 700;
            color: var(--text-primary);
            font-family: "DM Sans", sans-serif;
        }
        
        .metric-label {
            font-size: 0.875rem;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-top: 8px;
        }
        
        .metric-delta-positive {
            color: var(--success);
            font-size: 0.875rem;
        }
        
        .metric-delta-negative {
            color: var(--danger);
            font-size: 0.875rem;
        }
        
        /* Streamlit native metric styling */
        [data-testid="stMetric"] {
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 16px;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
        }
        
        [data-testid="stMetricLabel"] {
            color: var(--text-muted) !important;
        }
        
        [data-testid="stMetricValue"] {
            color: var(--text-primary) !important;
        }
        
        /* Buttons */
        .stButton > button {
            background: var(--lava-600);
            color: white;
            border: none;
            border-radius: 8px;
            padding: 12px 24px;
            font-weight: 600;
            font-family: "DM Sans", sans-serif;
            transition: all 0.2s ease;
        }
        
        .stButton > button:hover {
            background: var(--lava-500);
            transform: translateY(-1px);
            box-shadow: 0 4px 12px rgba(255, 54, 33, 0.25);
        }
        
        /* Secondary buttons */
        .stButton > button[kind="secondary"] {
            background: var(--bg-card);
            color: var(--text-primary);
            border: 1px solid var(--border-color);
        }
        
        .stButton > button[kind="secondary"]:hover {
            background: var(--oat-light);
            border-color: var(--lava-600);
        }
        
        /* Form inputs */
        .stSelectbox > div > div,
        .stMultiSelect > div > div {
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            color: var(--text-primary);
        }
        
        .stTextInput > div > div > input,
        .stNumberInput > div > div > input {
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            color: var(--text-primary);
            font-family: "DM Sans", sans-serif;
        }
        
        .stTextInput > div > div > input:focus,
        .stNumberInput > div > div > input:focus {
            border-color: var(--lava-600);
            box-shadow: 0 0 0 2px rgba(255, 54, 33, 0.15);
        }
        
        /* Slider */
        .stSlider > div > div > div {
            background: var(--lava-600);
        }
        
        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            background: var(--bg-card);
            border-radius: 8px;
            padding: 4px;
            gap: 4px;
            border: 1px solid var(--border-color);
        }
        
        .stTabs [data-baseweb="tab"] {
            background: transparent;
            border-radius: 6px;
            color: var(--text-secondary);
            font-weight: 500;
            font-family: "DM Sans", sans-serif;
            padding: 8px 16px;
        }
        
        .stTabs [aria-selected="true"] {
            background: var(--lava-600) !important;
            color: white !important;
        }
        
        .stTabs [data-baseweb="tab"]:hover {
            background: rgba(255, 54, 33, 0.1);
            color: var(--lava-600);
        }
        
        /* DataFrames */
        .stDataFrame {
            background: var(--bg-card);
            border-radius: 8px;
            border: 1px solid var(--border-color);
        }
        
        /* Chat components */
        .stChatInput > div {
            background: var(--bg-card) !important;
            border: 1px solid var(--border-color);
            border-radius: 8px;
        }
        
        /* Position chat input above footer and style container */
        [data-testid="stChatInput"] {
            bottom: 60px !important;
            background: var(--bg-primary) !important;
            padding: 12px 0 !important;
        }
        
        [data-testid="stBottom"] {
            background: var(--bg-primary) !important;
        }
        
        [data-testid="stBottomBlockContainer"] {
            background: var(--bg-primary) !important;
        }
        
        .stChatMessage {
            background: var(--bg-card);
            border-radius: 8px;
            border: 1px solid var(--border-color);
        }
        
        /* Expander - light theme styling */
        .streamlit-expanderHeader,
        [data-testid="stExpander"] summary,
        [data-testid="stExpander"] > div:first-child {
            background: var(--bg-card) !important;
            border-radius: 8px;
            color: var(--text-primary) !important;
            font-family: "DM Sans", sans-serif;
        }
        
        [data-testid="stExpander"] {
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 8px;
        }
        
        [data-testid="stExpander"] summary {
            padding: 12px 16px;
        }
        
        [data-testid="stExpander"] summary:hover {
            background: var(--oat-light) !important;
        }
        
        [data-testid="stExpander"] svg {
            fill: var(--text-primary) !important;
        }
        
        /* Section divider */
        .section-divider {
            height: 1px;
            background: var(--border-color);
            margin: 24px 0;
        }
        
        /* Cards */
        .glass-card {
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 24px;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
        }
        
        /* Status badges */
        .status-badge {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 6px 12px;
            border-radius: 20px;
            font-size: 0.75rem;
            font-weight: 500;
            font-family: "DM Sans", sans-serif;
        }
        
        .status-badge.connected {
            background: rgba(16, 185, 129, 0.1);
            color: #059669;
            border: 1px solid rgba(16, 185, 129, 0.2);
        }
        
        .status-badge.disconnected {
            background: rgba(239, 68, 68, 0.1);
            color: #dc2626;
            border: 1px solid rgba(239, 68, 68, 0.2);
        }
        
        /* Hide default elements */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        
        /* Dashboard container */
        .dashboard-container {
            border-radius: 12px;
            overflow: hidden;
            border: 1px solid var(--border-color);
            background: var(--bg-card);
        }
        
        .dashboard-container iframe {
            border: none;
            border-radius: 12px;
        }
        
        /* Radio buttons in sidebar */
        [data-testid="stSidebar"] .stRadio > div {
            gap: 8px;
        }
        
        [data-testid="stSidebar"] .stRadio > div > label {
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 12px 16px;
            margin: 0;
            transition: all 0.2s ease;
            cursor: pointer;
        }
        
        [data-testid="stSidebar"] .stRadio > div > label:hover {
            border-color: var(--lava-500);
            background: white;
        }
        
        [data-testid="stSidebar"] .stRadio > div > label[data-checked="true"] {
            background: var(--navy-800);
            border-color: var(--navy-800);
            color: white;
        }
        
        /* Alert/info boxes */
        .stAlert {
            border-radius: 8px;
            font-family: "DM Sans", sans-serif;
        }
        
        /* Plotly charts background */
        .js-plotly-plot .plotly .modebar {
            background: var(--bg-card) !important;
        }
    </style>
    """, unsafe_allow_html=True)


def render_header():
    """Render the Databricks branded header."""
    # Databricks logo SVG (official)
    databricks_logo_svg = '''<svg width="132" height="22" viewBox="0 0 132 22" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="m18.318 9.275-8.631 4.859L.445 8.942 0 9.182v3.77l9.687 5.431 8.63-4.84v1.995l-8.63 4.86-9.242-5.192-.445.24v.646l9.687 5.432 9.668-5.432v-3.769l-.445-.24-9.223 5.173-8.65-4.84V10.42l8.65 4.84 9.668-5.43V6.114l-.482-.277-9.186 5.155L1.482 6.41l8.205-4.6 6.741 3.787.593-.332v-.462L9.687.684 0 6.115v.592l9.687 5.432 8.63-4.86z" fill="#EE3D2C"/><path d="M37.449 18.443V1.852h-2.556v6.207c0 .093-.056.167-.148.204a.23.23 0 0 1-.24-.056c-.871-1.016-2.223-1.589-3.705-1.589-3.167 0-5.65 2.66-5.65 6.06 0 1.663.575 3.197 1.63 4.324a5.44 5.44 0 0 0 4.02 1.736c1.463 0 2.815-.61 3.704-1.662.056-.074.167-.093.24-.074.093.037.15.11.15.203v1.238zm-6.093-2.014c-2.038 0-3.63-1.644-3.63-3.75 0-2.107 1.592-3.751 3.63-3.751s3.63 1.644 3.63 3.75-1.593 3.75-3.63 3.75m19.762 2.016V6.896h-2.537V8.06c0 .093-.056.166-.149.203a.2.2 0 0 1-.24-.073c-.852-1.017-2.186-1.59-3.705-1.59-3.167 0-5.649 2.661-5.649 6.06 0 3.4 2.482 6.06 5.65 6.06 1.463 0 2.815-.61 3.704-1.68.055-.075.166-.093.24-.075.093.037.149.111.149.204v1.256h2.537zm-6.056-2.014c-2.038 0-3.63-1.645-3.63-3.75 0-2.107 1.592-3.751 3.63-3.751s3.63 1.644 3.63 3.75-1.593 3.75-3.63 3.75m27.781 2.015V6.896h-2.538V8.06c0 .093-.055.166-.148.203s-.185 0-.24-.073c-.853-1.017-2.186-1.59-3.705-1.59-3.186 0-5.649 2.661-5.649 6.08 0 3.417 2.482 6.06 5.649 6.06 1.463 0 2.815-.61 3.704-1.682.056-.074.167-.093.241-.074.093.037.148.11.148.203v1.256zm-6.057-2.014c-2.037 0-3.63-1.645-3.63-3.75 0-2.107 1.593-3.751 3.63-3.751s3.63 1.644 3.63 3.75-1.593 3.75-3.63 3.75m10.706.647c.019 0 .056-.019.074-.019.056 0 .13.037.167.074.87 1.016 2.222 1.589 3.704 1.589 3.167 0 5.65-2.66 5.65-6.06 0-1.663-.575-3.196-1.63-4.323a5.44 5.44 0 0 0-4.02-1.737c-1.463 0-2.815.61-3.704 1.663-.056.074-.148.092-.24.074-.093-.037-.149-.111-.149-.204V1.852h-2.556v16.59h2.556V17.28c0-.093.056-.166.148-.203m-.26-4.398c0-2.106 1.594-3.75 3.631-3.75s3.63 1.644 3.63 3.75-1.593 3.75-3.63 3.75-3.63-1.662-3.63-3.75m17.244-3.416c.24 0 .463.019.61.056V6.695a2.4 2.4 0 0 0-.425-.037c-1.334 0-2.556.684-3.204 1.774-.056.092-.149.13-.241.092a.22.22 0 0 1-.167-.203V6.898h-2.537v11.566h2.556v-5.1c0-2.53 1.296-4.1 3.408-4.1m4.815-2.367h-2.593v11.566h2.593zM97.958 1.87a1.571 1.571 0 1 0 0 3.141 1.571 1.571 0 1 0 0-3.14m8.928 4.729c-3.556 0-6.131 2.55-6.131 6.08 0 1.717.612 3.25 1.704 4.36 1.112 1.108 2.667 1.718 4.408 1.718 1.445 0 2.556-.277 4.668-1.83l-1.463-1.533c-1.038.684-2.001 1.016-2.945 1.016-2.149 0-3.76-1.607-3.76-3.732s1.611-3.732 3.76-3.732c1.018 0 1.963.333 2.908 1.016l1.629-1.533c-1.907-1.626-3.63-1.83-4.778-1.83m9.149 6.762a.2.2 0 0 1 .149-.055h.018c.056 0 .111.037.167.073l4.093 5.063h3.149l-5.297-6.393c-.075-.092-.075-.222.018-.295l4.871-4.86h-3.13l-4.204 4.213c-.056.055-.148.074-.241.055a.23.23 0 0 1-.13-.203V1.87h-2.574v16.591h2.556v-4.508c0-.055.018-.13.074-.166z" fill="#000"/><path d="M127.776 18.739c2.093 0 4.223-1.275 4.223-3.695 0-1.589-1-2.68-3.037-3.344l-1.39-.462c-.944-.314-1.389-.758-1.389-1.367 0-.702.63-1.183 1.519-1.183.852 0 1.611.555 2.093 1.515l2.056-1.108c-.759-1.552-2.334-2.513-4.149-2.513-2.297 0-3.963 1.478-3.963 3.492 0 1.607.963 2.679 2.944 3.307l1.427.462c1 .314 1.426.72 1.426 1.367 0 .98-.908 1.33-1.686 1.33-1.037 0-1.963-.665-2.407-1.755l-2.093 1.109c.685 1.755 2.37 2.845 4.426 2.845m-69.546-.111c.815 0 1.538-.074 1.945-.13v-2.216a14 14 0 0 1-1.278.073c-1.037 0-1.833-.184-1.833-2.42V9.187c0-.13.092-.222.222-.222h2.5V6.877h-2.5a.214.214 0 0 1-.222-.221V3.33h-2.556v3.344c0 .13-.093.222-.223.222h-1.778v2.088h1.778c.13 0 .223.092.223.221v5.377c0 4.046 2.704 4.046 3.722 4.046" fill="#000"/></svg>'''
    
    st.markdown(f"""
    <header style="
        display: flex;
        align-items: center;
        justify-content: space-between;
        white-space: nowrap;
        border-bottom: 1px solid #e0dfdc;
        padding: 8px 24px;
        background: white;
        margin: -1rem -1rem 0.25rem -1rem;
        width: calc(100% + 2rem);
        min-height: 48px;
    ">
        <div style="display: flex; align-items: center; gap: 16px; color: #151513;">
            <a href="https://dbricks.co/manufacturing" target="_blank" 
               style="display: flex; align-items: center; text-decoration: none;">
                {databricks_logo_svg}
            </a>
            <h2 style="
                color: #151513 !important; 
                font-size: 1.25rem; 
                font-weight: 500; 
                margin: 0;
                font-family: 'DM Sans', sans-serif;
            ">Manufacturing</h2>
        </div>
        <div style="display: flex; align-items: center; gap: 24px;">
            <span style="
                color: #151513; 
                font-size: 1rem;
                font-family: 'DM Sans', sans-serif;
            ">Scrap Intelligence Platform</span>
        </div>
    </header>
    """, unsafe_allow_html=True)
