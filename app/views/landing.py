"""
Landing Page
============
A compelling pitch-style landing page for the Scrap Reduction Intelligence Platform.
Tells the story: WHAT, WHY, WHEN, HOW.
"""

import streamlit as st


def render_landing_page():
    """Render the landing page with key pitch content."""
    
    # Hero Section
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #1B3139 0%, #0B2026 100%);
        border-radius: 16px;
        padding: 48px 40px;
        margin-bottom: 32px;
        position: relative;
        overflow: hidden;
    ">
        <div style="
            position: absolute;
            top: -50px;
            right: -50px;
            width: 300px;
            height: 300px;
            background: radial-gradient(circle, rgba(255,54,33,0.3) 0%, transparent 70%);
            border-radius: 50%;
        "></div>
        <div style="position: relative; z-index: 1;">
            <p style="
                color: #FF3621;
                font-size: 0.875rem;
                font-weight: 600;
                letter-spacing: 2px;
                text-transform: uppercase;
                margin-bottom: 12px;
                font-family: 'DM Sans', sans-serif;
            ">DATABRICKS FOR MANUFACTURING</p>
            <p style="
                color: #FFFFFF !important;
                font-size: 2.75rem;
                font-weight: 700;
                line-height: 1.15;
                margin: 0 0 16px 0;
                font-family: 'DM Sans', sans-serif;
                text-shadow: 0 2px 4px rgba(0,0,0,0.2);
            ">Smart Scrap Reduction</p>
            <p style="
                color: rgba(255,255,255,0.85);
                font-size: 1.25rem;
                font-weight: 400;
                margin: 0;
                max-width: 600px;
                font-family: 'DM Sans', sans-serif;
            ">Machine Learning and AI for Scenario Planning</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Quote Section
    st.markdown("""
    <div style="
        background: #FF3621;
        border-radius: 12px;
        padding: 32px 40px;
        margin-bottom: 32px;
        text-align: center;
    ">
        <p style="
            color: white;
            font-size: 1.5rem;
            font-weight: 500;
            font-style: italic;
            margin: 0 0 12px 0;
            font-family: 'DM Sans', sans-serif;
            line-height: 1.4;
        ">"The most dangerous kind of waste is the waste we do not recognize."</p>
        <p style="
            color: rgba(255,255,255,0.9);
            font-size: 1rem;
            margin: 0;
            font-family: 'DM Sans', sans-serif;
        ">— Shigeo Shingo, Toyota Production System Pioneer</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Problem Section
    st.markdown("""
    <h2 style="
        color: #1B3139;
        font-size: 1.5rem;
        font-weight: 600;
        margin: 0 0 20px 0;
        font-family: 'DM Sans', sans-serif;
    ">📊 The Problem: Manufacturing Scrap is a Massive Cost Driver</h2>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style="
            background: white;
            border: 1px solid #e0dfdc;
            border-radius: 12px;
            padding: 24px;
            text-align: center;
            height: 100%;
        ">
            <p style="
                color: #FF3621;
                font-size: 2.5rem;
                font-weight: 700;
                margin: 0;
                font-family: 'DM Sans', sans-serif;
            ">$32-58B</p>
            <p style="
                color: #4a4a48;
                font-size: 0.875rem;
                margin: 8px 0 0 0;
                font-family: 'DM Sans', sans-serif;
            ">Annual U.S. defect-related costs<br>(scrap, rework, warranty)</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="
            background: white;
            border: 1px solid #e0dfdc;
            border-radius: 12px;
            padding: 24px;
            text-align: center;
            height: 100%;
        ">
            <p style="
                color: #FF3621;
                font-size: 2.5rem;
                font-weight: 700;
                margin: 0;
                font-family: 'DM Sans', sans-serif;
            ">15-25%</p>
            <p style="
                color: #4a4a48;
                font-size: 0.875rem;
                margin: 8px 0 0 0;
                font-family: 'DM Sans', sans-serif;
            ">Of manufacturing revenue consumed<br>by quality-related costs (ASQ)</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="
            background: white;
            border: 1px solid #e0dfdc;
            border-radius: 12px;
            padding: 24px;
            text-align: center;
            height: 100%;
        ">
            <p style="
                color: #FF3621;
                font-size: 2.5rem;
                font-weight: 700;
                margin: 0;
                font-family: 'DM Sans', sans-serif;
            ">3-10%</p>
            <p style="
                color: #4a4a48;
                font-size: 0.875rem;
                margin: 8px 0 0 0;
                font-family: 'DM Sans', sans-serif;
            ">Of total material costs lost<br>depending on industry maturity</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<div style='height: 32px;'></div>", unsafe_allow_html=True)
    
    # Opportunity Gap Section
    st.markdown("""
    <h2 style="
        color: #1B3139;
        font-size: 1.5rem;
        font-weight: 600;
        margin: 0 0 20px 0;
        font-family: 'DM Sans', sans-serif;
    ">🎯 The Opportunity Gap: High Intent, Low Visibility</h2>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style="
            background: #1B3139;
            border-radius: 12px;
            padding: 24px;
            text-align: center;
        ">
            <p style="
                color: #FF3621;
                font-size: 2.5rem;
                font-weight: 700;
                margin: 0;
                font-family: 'DM Sans', sans-serif;
            ">66%</p>
            <p style="
                color: white;
                font-size: 0.875rem;
                margin: 8px 0 0 0;
                font-family: 'DM Sans', sans-serif;
            ">Identify waste reduction<br>as a primary priority</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="
            background: #1B3139;
            border-radius: 12px;
            padding: 24px;
            text-align: center;
        ">
            <p style="
                color: #FF3621;
                font-size: 2.5rem;
                font-weight: 700;
                margin: 0;
                font-family: 'DM Sans', sans-serif;
            ">99%</p>
            <p style="
                color: white;
                font-size: 0.875rem;
                margin: 8px 0 0 0;
                font-family: 'DM Sans', sans-serif;
            ">Say they need more support<br>to execute on scrap reduction</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="
            background: #1B3139;
            border-radius: 12px;
            padding: 24px;
            text-align: center;
        ">
            <p style="
                color: #FF3621;
                font-size: 2.5rem;
                font-weight: 700;
                margin: 0;
                font-family: 'DM Sans', sans-serif;
            ">48%</p>
            <p style="
                color: white;
                font-size: 0.875rem;
                margin: 8px 0 0 0;
                font-family: 'DM Sans', sans-serif;
            ">Identify robust data as<br>mission critical for success</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="
        background: #1B3139;
        border-radius: 12px;
        padding: 20px 24px;
        margin: 24px 0 32px 0;
        display: flex;
        align-items: flex-start;
        gap: 12px;
    ">
        <span style="font-size: 1.5rem;">💡</span>
        <div>
            <p style="
                color: white;
                font-size: 1.1rem;
                font-weight: 600;
                margin: 0 0 4px 0;
                font-family: 'DM Sans', sans-serif;
            ">This is not a motivation problem — it's an execution problem.</p>
            <p style="
                color: rgba(255,255,255,0.75);
                font-size: 0.95rem;
                font-weight: 400;
                margin: 0;
                font-family: 'DM Sans', sans-serif;
            ">Manufacturers lack the data foundation and predictive tools to execute.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Why Scenario Planning Section
    st.markdown("""
    <h2 style="
        color: #1B3139;
        font-size: 1.5rem;
        font-weight: 600;
        margin: 0 0 20px 0;
        font-family: 'DM Sans', sans-serif;
    ">⚡ The Shift: From Visibility to Impact</h2>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div style="
            background: white;
            border: 1px solid #e0dfdc;
            border-radius: 12px;
            padding: 24px;
        ">
            <p style="
                color: #FF3621;
                font-size: 0.875rem;
                font-weight: 600;
                text-transform: uppercase;
                letter-spacing: 1px;
                margin: 0 0 12px 0;
                font-family: 'DM Sans', sans-serif;
            ">❌ Current State</p>
            <p style="
                color: #4a4a48;
                font-size: 0.95rem;
                margin: 0;
                font-family: 'DM Sans', sans-serif;
                line-height: 1.6;
            ">Scrap data is fragmented across systems, so teams <strong>debate numbers</strong> instead of fixing problems. Without a trusted view of true scrap cost, reduction planning breaks down.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="
            background: white;
            border: 2px solid #10b981;
            border-radius: 12px;
            padding: 24px;
        ">
            <p style="
                color: #10b981;
                font-size: 0.875rem;
                font-weight: 600;
                text-transform: uppercase;
                letter-spacing: 1px;
                margin: 0 0 12px 0;
                font-family: 'DM Sans', sans-serif;
            ">✓ With Scenario Planning</p>
            <p style="
                color: #4a4a48;
                font-size: 0.95rem;
                margin: 0;
                font-family: 'DM Sans', sans-serif;
                line-height: 1.6;
            ">Move from reactive reporting to <strong>proactive decision-making</strong> — identifying which actions deliver the biggest return <em>before</em> money is spent.</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)
    
    # Impact metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style="
            border-left: 3px solid #FF3621;
            padding-left: 16px;
        ">
            <p style="
                color: #1B3139;
                font-size: 1rem;
                font-weight: 600;
                margin: 0 0 4px 0;
                font-family: 'DM Sans', sans-serif;
            ">💰 Margin Protection</p>
            <p style="
                color: #6b6b69;
                font-size: 0.875rem;
                margin: 0;
                font-family: 'DM Sans', sans-serif;
            ">Scrap reduction goes straight to profitability</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="
            border-left: 3px solid #FF3621;
            padding-left: 16px;
        ">
            <p style="
                color: #1B3139;
                font-size: 1rem;
                font-weight: 600;
                margin: 0 0 4px 0;
                font-family: 'DM Sans', sans-serif;
            ">⚙️ Production Velocity</p>
            <p style="
                color: #6b6b69;
                font-size: 0.875rem;
                margin: 0;
                font-family: 'DM Sans', sans-serif;
            ">Higher OEE, fewer late-stage disruptions</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="
            border-left: 3px solid #FF3621;
            padding-left: 16px;
        ">
            <p style="
                color: #1B3139;
                font-size: 1rem;
                font-weight: 600;
                margin: 0 0 4px 0;
                font-family: 'DM Sans', sans-serif;
            ">🎯 Operational Confidence</p>
            <p style="
                color: #6b6b69;
                font-size: 0.875rem;
                margin: 0;
                font-family: 'DM Sans', sans-serif;
            ">One set of numbers, clear priorities</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<div style='height: 32px;'></div>", unsafe_allow_html=True)
    
    # How Databricks Does It
    st.markdown("""
    <h2 style="
        color: #1B3139;
        font-size: 1.5rem;
        font-weight: 600;
        margin: 0 0 20px 0;
        font-family: 'DM Sans', sans-serif;
    ">🚀 How: Scrap Reduction on Databricks</h2>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #1B3139 0%, #0B2026 100%);
            border-radius: 12px;
            padding: 24px;
            margin-bottom: 16px;
        ">
            <p style="
                color: #FF3621;
                font-size: 1.25rem;
                font-weight: 600;
                margin: 0 0 8px 0;
                font-family: 'DM Sans', sans-serif;
            ">🗄️ Single Source of Truth</p>
            <p style="
                color: rgba(255,255,255,0.85);
                font-size: 0.9rem;
                margin: 0;
                font-family: 'DM Sans', sans-serif;
                line-height: 1.5;
            ">Unify ERP, MES, and quality data — including unstructured reports — into one trusted view of scrap.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #1B3139 0%, #0B2026 100%);
            border-radius: 12px;
            padding: 24px;
        ">
            <p style="
                color: #FF3621;
                font-size: 1.25rem;
                font-weight: 600;
                margin: 0 0 8px 0;
                font-family: 'DM Sans', sans-serif;
            ">🤖 Best-in-Breed ML Ops</p>
            <p style="
                color: rgba(255,255,255,0.85);
                font-size: 0.9rem;
                margin: 0;
                font-family: 'DM Sans', sans-serif;
                line-height: 1.5;
            ">Train, deploy, and govern predictive and scenario models in one integrated environment.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #1B3139 0%, #0B2026 100%);
            border-radius: 12px;
            padding: 24px;
            margin-bottom: 16px;
        ">
            <p style="
                color: #FF3621;
                font-size: 1.25rem;
                font-weight: 600;
                margin: 0 0 8px 0;
                font-family: 'DM Sans', sans-serif;
            ">📈 Unlimited Scalability</p>
            <p style="
                color: rgba(255,255,255,0.85);
                font-size: 0.9rem;
                margin: 0;
                font-family: 'DM Sans', sans-serif;
                line-height: 1.5;
            ">Scale scrap analytics from one plant to the entire enterprise without replatforming.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #1B3139 0%, #0B2026 100%);
            border-radius: 12px;
            padding: 24px;
        ">
            <p style="
                color: #FF3621;
                font-size: 1.25rem;
                font-weight: 600;
                margin: 0 0 8px 0;
                font-family: 'DM Sans', sans-serif;
            ">📱 Business-Friendly Apps</p>
            <p style="
                color: rgba(255,255,255,0.85);
                font-size: 0.9rem;
                margin: 0;
                font-family: 'DM Sans', sans-serif;
                line-height: 1.5;
            ">Deliver real-time what-if scenarios to business users — not just data scientists.</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<div style='height: 32px;'></div>", unsafe_allow_html=True)
    
    # CTA Section
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #FF3621 0%, #FF5F46 100%);
        border-radius: 12px;
        padding: 32px 40px;
        text-align: center;
    ">
        <p style="
            color: white;
            font-size: 1.5rem;
            font-weight: 600;
            margin: 0 0 8px 0;
            font-family: 'DM Sans', sans-serif;
        ">Ready to see it in action?</p>
        <p style="
            color: rgba(255,255,255,0.9);
            font-size: 1rem;
            margin: 0;
            font-family: 'DM Sans', sans-serif;
        ">Use the sidebar to explore the Executive Dashboard, What-If Scenarios, and AI Assistant.</p>
    </div>
    """, unsafe_allow_html=True)
