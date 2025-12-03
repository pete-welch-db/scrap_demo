"""
Custom Styles
=============
Dark mode CSS styling for the application.
"""

import streamlit as st


def apply_custom_styles():
    """Apply custom dark mode CSS styling to the app."""
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
        
        :root {
            --bg-primary: #0a0a0f;
            --bg-secondary: #12121a;
            --bg-tertiary: #1a1a2e;
            --bg-card: #16162a;
            --accent-primary: #6366f1;
            --accent-secondary: #8b5cf6;
            --accent-tertiary: #a855f7;
            --text-primary: #f1f5f9;
            --text-secondary: #94a3b8;
            --text-muted: #64748b;
            --border-color: #2d2d44;
            --success: #10b981;
            --warning: #f59e0b;
            --danger: #ef4444;
            --gradient-start: #6366f1;
            --gradient-end: #a855f7;
        }
        
        .stApp {
            background: linear-gradient(180deg, var(--bg-primary) 0%, var(--bg-secondary) 100%);
            font-family: 'Inter', sans-serif;
        }
        
        [data-testid="stSidebar"] {
            background: var(--bg-secondary);
            border-right: 1px solid var(--border-color);
        }
        
        [data-testid="stSidebar"] .stMarkdown {
            color: var(--text-primary);
        }
        
        h1, h2, h3, h4, h5, h6 {
            color: var(--text-primary) !important;
            font-family: 'Inter', sans-serif;
            font-weight: 600;
        }
        
        h1 {
            background: linear-gradient(90deg, var(--gradient-start), var(--gradient-end));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        
        p, span, label, .stMarkdown {
            color: var(--text-secondary);
        }
        
        .metric-card {
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            padding: 24px;
            margin: 8px 0;
            transition: all 0.3s ease;
        }
        
        .metric-card:hover {
            border-color: var(--accent-primary);
            box-shadow: 0 0 30px rgba(99, 102, 241, 0.15);
        }
        
        .metric-value {
            font-size: 2.5rem;
            font-weight: 700;
            color: var(--text-primary);
            font-family: 'JetBrains Mono', monospace;
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
        
        .stButton > button {
            background: linear-gradient(135deg, var(--accent-primary), var(--accent-secondary));
            color: white;
            border: none;
            border-radius: 12px;
            padding: 12px 24px;
            font-weight: 600;
            font-family: 'Inter', sans-serif;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(99, 102, 241, 0.3);
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(99, 102, 241, 0.4);
        }
        
        .stSelectbox > div > div {
            background: var(--bg-tertiary);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            color: var(--text-primary);
        }
        
        .stSlider > div > div > div {
            background: var(--accent-primary);
        }
        
        .stTabs [data-baseweb="tab-list"] {
            background: var(--bg-tertiary);
            border-radius: 12px;
            padding: 4px;
            gap: 4px;
        }
        
        .stTabs [data-baseweb="tab"] {
            background: transparent;
            border-radius: 8px;
            color: var(--text-secondary);
            font-weight: 500;
        }
        
        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, var(--accent-primary), var(--accent-secondary));
            color: white;
        }
        
        .stDataFrame {
            background: var(--bg-card);
            border-radius: 12px;
            border: 1px solid var(--border-color);
        }
        
        .stChatInput > div {
            background: var(--bg-tertiary);
            border: 1px solid var(--border-color);
            border-radius: 12px;
        }
        
        .stChatMessage {
            background: var(--bg-card);
            border-radius: 12px;
            border: 1px solid var(--border-color);
        }
        
        .streamlit-expanderHeader {
            background: var(--bg-tertiary);
            border-radius: 12px;
            color: var(--text-primary);
        }
        
        .stNumberInput > div > div > input {
            background: var(--bg-tertiary);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            color: var(--text-primary);
        }
        
        .section-divider {
            height: 1px;
            background: linear-gradient(90deg, transparent, var(--border-color), transparent);
            margin: 32px 0;
        }
        
        .glass-card {
            background: rgba(22, 22, 42, 0.7);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 20px;
            padding: 24px;
        }
        
        .status-badge {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 6px 12px;
            border-radius: 20px;
            font-size: 0.75rem;
            font-weight: 500;
        }
        .status-badge.connected {
            background: rgba(16, 185, 129, 0.15);
            color: #10b981;
            border: 1px solid rgba(16, 185, 129, 0.3);
        }
        .status-badge.disconnected {
            background: rgba(239, 68, 68, 0.15);
            color: #ef4444;
            border: 1px solid rgba(239, 68, 68, 0.3);
        }
        
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        
        .dashboard-container {
            border-radius: 16px;
            overflow: hidden;
            border: 1px solid var(--border-color);
            background: var(--bg-card);
        }
        
        .dashboard-container iframe {
            border: none;
            border-radius: 16px;
        }
        
        /* Logo styling */
        .logo-container {
            display: flex;
            align-items: center;
            gap: 12px;
            margin-bottom: 8px;
        }
        
        .logo-container img {
            height: 32px;
        }
        
        .powered-by {
            font-size: 0.7rem;
            color: var(--text-muted);
            margin-top: 4px;
        }
    </style>
    """, unsafe_allow_html=True)

