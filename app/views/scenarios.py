"""
What-If Scenarios Page
======================
Interactive scenario planning for scrap reduction strategies.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from components.metrics import create_metric_card, create_plotly_theme


# Cache mock data at module level to avoid reload issues
@st.cache_data
def get_scenario_data():
    """Generate mock data for scenarios."""
    np.random.seed(42)
    
    plants = [
        "Detroit Assembly", 
        "Louisville Stamping", 
        "San Antonio Body", 
        "Oakville Paint", 
        "Saltillo Powertrain", 
        "Silao Components"
    ]
    
    parts = [f"PRT-{str(i).zfill(5)}" for i in range(1, 51)]
    part_names = [f"Component {chr(65 + i % 26)}-{i}" for i in range(50)]
    
    work_centers = [
        "Stamping Press Line 1", 
        "Welding Cell A", 
        "CNC Machining", 
        "Injection Molding", 
        "Assembly Line", 
        "Paint Booth"
    ]
    
    # Monthly data
    months = pd.date_range('2022-01-01', '2024-11-01', freq='MS')
    monthly_data = []
    for month in months:
        for plant in plants:
            base_cost = np.random.uniform(50000, 150000)
            seasonal = 1.3 if month.month in [7, 8] else 1.0
            monthly_data.append({
                'month': month,
                'plant': plant,
                'scrap_cost': base_cost * seasonal * np.random.uniform(0.8, 1.2),
            })
    monthly_df = pd.DataFrame(monthly_data)
    
    # Parts data
    parts_data = []
    for part, name in zip(parts, part_names):
        parts_data.append({
            'part_id': part,
            'part_name': name,
            'monthly_scrap_cost': np.random.uniform(5000, 50000),
            'defect_rate': np.random.uniform(0.01, 0.08),
            'unit_cost': np.random.uniform(10, 200)
        })
    parts_df = pd.DataFrame(parts_data)
    
    # Work center data
    wc_data = []
    for wc in work_centers:
        for plant in plants:
            wc_data.append({
                'work_center': wc,
                'plant': plant,
                'monthly_scrap_cost': np.random.uniform(10000, 80000),
                'oee': np.random.uniform(0.65, 0.95),
                'equipment_age': np.random.randint(2, 18)
            })
    wc_df = pd.DataFrame(wc_data)
    
    return monthly_df, parts_df, wc_df, plants, work_centers


def render_scenarios_page():
    """Render the What-If Scenarios page."""
    
    st.markdown("""
    <h1>🎯 What-If Scenarios</h1>
    <p style="font-size: 1.1rem; color: #64748b; margin-bottom: 24px;">
        Simulate scrap reduction strategies and estimate cost savings
    </p>
    """, unsafe_allow_html=True)
    
    # Load data once
    monthly_df, parts_df, wc_df, plants, work_centers = get_scenario_data()
    
    # Scenario tabs
    tab1, tab2, tab3 = st.tabs(["🏭 By Plant", "🔧 By Work Center", "📦 By Part"])
    
    with tab1:
        render_plant_scenario(monthly_df, plants)
    
    with tab2:
        render_work_center_scenario(wc_df, work_centers)
    
    with tab3:
        render_part_scenario(parts_df)


def get_plant_recommendations(plant: str, reduction_target: int, roi_months: float, annual_savings: float) -> list:
    """Generate context-aware recommendations for plant scenario."""
    recommendations = []
    
    # Based on reduction target
    if reduction_target >= 30:
        recommendations.append({
            "icon": "🎯",
            "title": "Aggressive Target",
            "desc": "Consider phased implementation to manage risk",
            "priority": "high"
        })
        recommendations.append({
            "icon": "👥",
            "title": "Cross-Functional Team",
            "desc": "Form dedicated task force with quality, ops, and engineering",
            "priority": "high"
        })
    
    if reduction_target >= 20:
        recommendations.append({
            "icon": "📊",
            "title": "Root Cause Analysis",
            "desc": "Conduct Pareto analysis on top defect categories",
            "priority": "high"
        })
    
    # Based on ROI
    if roi_months <= 12:
        recommendations.append({
            "icon": "🚀",
            "title": "Quick Win Opportunity",
            "desc": "Fast ROI - prioritize this initiative",
            "priority": "high"
        })
    elif roi_months > 24:
        recommendations.append({
            "icon": "⚠️",
            "title": "Long Payback Period",
            "desc": "Consider reducing investment or increasing scope",
            "priority": "medium"
        })
    
    # Plant-specific recommendations
    if "Stamping" in plant:
        recommendations.append({
            "icon": "🔧",
            "title": "Die Maintenance",
            "desc": "Focus on die wear monitoring and predictive maintenance",
            "priority": "medium"
        })
    elif "Assembly" in plant:
        recommendations.append({
            "icon": "🤖",
            "title": "Assembly Automation",
            "desc": "Review torque specs and automated assembly verification",
            "priority": "medium"
        })
    elif "Paint" in plant:
        recommendations.append({
            "icon": "🌡️",
            "title": "Environment Control",
            "desc": "Monitor booth temperature/humidity and paint viscosity",
            "priority": "medium"
        })
    
    # General recommendations
    recommendations.append({
        "icon": "📈",
        "title": "Track Leading Indicators",
        "desc": "Monitor weekly scrap trends, not just monthly",
        "priority": "low"
    })
    
    recommendations.append({
        "icon": "🏆",
        "title": "Operator Incentives",
        "desc": "Implement quality bonuses tied to scrap reduction",
        "priority": "low"
    })
    
    return recommendations[:6]  # Return top 6


def get_work_center_recommendations(work_center: str, avg_oee: float, avg_age: float, target_oee: int) -> list:
    """Generate context-aware recommendations for work center scenario."""
    recommendations = []
    
    # Based on OEE
    if avg_oee < 0.75:
        recommendations.append({
            "icon": "🔴",
            "title": "Critical OEE",
            "desc": "OEE below 75% - major improvement potential",
            "priority": "high"
        })
        recommendations.append({
            "icon": "📉",
            "title": "Identify Six Big Losses",
            "desc": "Analyze breakdown, setup, idle, speed, defects, startup",
            "priority": "high"
        })
    
    # Based on equipment age
    if avg_age > 12:
        recommendations.append({
            "icon": "⚙️",
            "title": "Aging Equipment",
            "desc": "Consider capital refresh or rebuild program",
            "priority": "high"
        })
        recommendations.append({
            "icon": "🔮",
            "title": "Predictive Maintenance",
            "desc": "Install sensors for vibration, temperature monitoring",
            "priority": "medium"
        })
    elif avg_age > 8:
        recommendations.append({
            "icon": "🛠️",
            "title": "Enhanced PM Program",
            "desc": "Increase preventive maintenance frequency",
            "priority": "medium"
        })
    
    # Work center specific
    if "Stamping" in work_center:
        recommendations.append({
            "icon": "📐",
            "title": "Tool & Die Program",
            "desc": "Implement die tracking and wear prediction",
            "priority": "medium"
        })
    elif "Welding" in work_center:
        recommendations.append({
            "icon": "⚡",
            "title": "Weld Parameter Control",
            "desc": "Real-time monitoring of current, voltage, wire feed",
            "priority": "medium"
        })
    elif "CNC" in work_center:
        recommendations.append({
            "icon": "🎯",
            "title": "Tool Life Management",
            "desc": "Track tool wear and implement automatic offsets",
            "priority": "medium"
        })
    elif "Injection" in work_center:
        recommendations.append({
            "icon": "🌡️",
            "title": "Process Monitoring",
            "desc": "Monitor barrel temp, injection pressure, cycle time",
            "priority": "medium"
        })
    
    # General
    recommendations.append({
        "icon": "👨‍🏫",
        "title": "Operator Training",
        "desc": "Certify operators on TPM and quality standards",
        "priority": "low"
    })
    
    recommendations.append({
        "icon": "📋",
        "title": "Standard Work",
        "desc": "Document and enforce standard operating procedures",
        "priority": "low"
    })
    
    return recommendations[:6]


def get_part_recommendations(num_parts: int, total_cost: float, avg_defect_rate: float, reduction_target: int) -> list:
    """Generate context-aware recommendations for part scenario."""
    recommendations = []
    
    # Based on defect rate
    if avg_defect_rate > 0.05:
        recommendations.append({
            "icon": "🔴",
            "title": "High Defect Rate",
            "desc": f"Avg {avg_defect_rate*100:.1f}% - root cause investigation needed",
            "priority": "high"
        })
        recommendations.append({
            "icon": "🔍",
            "title": "PFMEA Review",
            "desc": "Update failure modes analysis for high-scrap parts",
            "priority": "high"
        })
    
    # Based on number of parts selected
    if num_parts >= 5:
        recommendations.append({
            "icon": "🎯",
            "title": "Focus Strategy",
            "desc": "Consider tackling top 3 parts first for faster wins",
            "priority": "medium"
        })
    
    # Based on total cost
    if total_cost > 100000:
        recommendations.append({
            "icon": "💰",
            "title": "High-Value Target",
            "desc": f"${total_cost/1000:.0f}K/month - executive sponsorship recommended",
            "priority": "high"
        })
    
    # Based on reduction target
    if reduction_target >= 50:
        recommendations.append({
            "icon": "🔬",
            "title": "Design Review",
            "desc": "Consider part redesign or material change",
            "priority": "medium"
        })
    
    # General recommendations
    recommendations.append({
        "icon": "🏭",
        "title": "Supplier Quality",
        "desc": "Review incoming inspection and supplier PPM",
        "priority": "medium"
    })
    
    recommendations.append({
        "icon": "📊",
        "title": "SPC Implementation",
        "desc": "Add statistical process control on critical dimensions",
        "priority": "medium"
    })
    
    recommendations.append({
        "icon": "✅",
        "title": "Error Proofing",
        "desc": "Implement poka-yoke at key process steps",
        "priority": "low"
    })
    
    recommendations.append({
        "icon": "🔄",
        "title": "8D Follow-Up",
        "desc": "Review historical 8D reports for these parts",
        "priority": "low"
    })
    
    return recommendations[:6]


def render_recommendations(recommendations: list, title: str = "💡 Recommended Actions"):
    """Render the recommendations panel."""
    
    priority_colors = {
        "high": "#ef4444",
        "medium": "#f59e0b",
        "low": "#10b981"
    }
    
    st.markdown(f"#### {title}")
    
    for rec in recommendations:
        color = priority_colors.get(rec.get("priority", "low"), "#6366f1")
        st.markdown(f"""
        <div style="background: #1a1a2e; border-left: 3px solid {color}; border-radius: 8px; padding: 12px; margin-bottom: 8px;">
            <div style="display: flex; align-items: flex-start; gap: 10px;">
                <span style="font-size: 1.2rem;">{rec['icon']}</span>
                <div>
                    <p style="color: #f1f5f9; font-weight: 600; margin: 0; font-size: 0.9rem;">{rec['title']}</p>
                    <p style="color: #94a3b8; margin: 4px 0 0 0; font-size: 0.8rem;">{rec['desc']}</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Priority legend
    st.markdown("""
    <div style="display: flex; gap: 16px; margin-top: 12px; padding: 8px 12px; background: #12121a; border-radius: 8px;">
        <div style="display: flex; align-items: center; gap: 6px;">
            <div style="width: 12px; height: 12px; background: #ef4444; border-radius: 2px;"></div>
            <span style="color: #94a3b8; font-size: 0.75rem;">High</span>
        </div>
        <div style="display: flex; align-items: center; gap: 6px;">
            <div style="width: 12px; height: 12px; background: #f59e0b; border-radius: 2px;"></div>
            <span style="color: #94a3b8; font-size: 0.75rem;">Medium</span>
        </div>
        <div style="display: flex; align-items: center; gap: 6px;">
            <div style="width: 12px; height: 12px; background: #10b981; border-radius: 2px;"></div>
            <span style="color: #94a3b8; font-size: 0.75rem;">Low</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_plant_scenario(monthly_df: pd.DataFrame, plants: list):
    """Render the plant-level scenario tab."""
    
    st.markdown("### Plant-Level Scrap Reduction")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("#### Scenario Parameters")
        
        plant_index = st.selectbox(
            "Select Plant",
            range(len(plants)),
            format_func=lambda x: plants[x],
            key="scenario_plant_select"
        )
        selected_plant = plants[plant_index]
        
        reduction_target = st.slider(
            "Scrap Reduction Target (%)",
            min_value=5,
            max_value=50,
            value=20,
            step=5,
            key="scenario_plant_reduction"
        )
        
        implementation_months = st.slider(
            "Implementation Timeline (months)",
            min_value=3,
            max_value=24,
            value=12,
            key="scenario_plant_timeline"
        )
        
        investment = st.number_input(
            "Estimated Investment ($)",
            min_value=0,
            max_value=5000000,
            value=250000,
            step=50000,
            key="scenario_plant_investment"
        )
        
        # Calculate metrics for recommendations
        plant_data = monthly_df[monthly_df['plant'] == selected_plant]
        current_annual_cost = plant_data['scrap_cost'].sum() / 3
        annual_savings = current_annual_cost * (reduction_target / 100)
        roi_months = investment / (annual_savings / 12) if annual_savings > 0 else 999
        
        st.markdown("---")
        
        # Recommendations in left column
        recommendations = get_plant_recommendations(
            selected_plant, reduction_target, roi_months, annual_savings
        )
        render_recommendations(recommendations)
    
    with col2:
        five_year_savings = annual_savings * 5 - investment
        
        st.markdown("#### Projected Impact")
        
        res_col1, res_col2, res_col3 = st.columns(3)
        
        with res_col1:
            st.markdown(create_metric_card(
                f"${current_annual_cost/1e6:.2f}M",
                "Current Annual Scrap"
            ), unsafe_allow_html=True)
        
        with res_col2:
            st.markdown(create_metric_card(
                f"${annual_savings/1e6:.2f}M",
                "Annual Savings"
            ), unsafe_allow_html=True)
        
        with res_col3:
            st.markdown(create_metric_card(
                f"{min(roi_months, 999):.1f}",
                "Months to ROI"
            ), unsafe_allow_html=True)
        
        st.markdown("#### Cost Trajectory")
        
        months_proj = list(range(1, 37))
        baseline = [current_annual_cost / 12] * 36
        
        improved = []
        for m in months_proj:
            if m <= implementation_months:
                progress = m / implementation_months
                reduction = progress * (reduction_target / 100)
            else:
                reduction = reduction_target / 100
            improved.append((current_annual_cost / 12) * (1 - reduction))
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=months_proj,
            y=baseline,
            mode='lines',
            name='Baseline',
            line=dict(color='#ef4444', width=2, dash='dash')
        ))
        
        fig.add_trace(go.Scatter(
            x=months_proj,
            y=improved,
            mode='lines',
            name='With Improvement',
            line=dict(color='#10b981', width=3),
            fill='tonexty',
            fillcolor='rgba(16, 185, 129, 0.1)'
        ))
        
        fig.update_layout(
            **create_plotly_theme(),
            height=300,
            margin=dict(l=0, r=0, t=40, b=0),
            legend=dict(orientation='h', y=1.1),
            hovermode='x unified',
            xaxis_title="Month",
            yaxis_title="Monthly Scrap Cost ($)"
        )
        
        st.plotly_chart(fig, use_container_width=True, key="plant_trajectory_chart")
        
        # 5-year summary
        roi_pct = ((annual_savings * 5) / investment - 1) * 100 if investment > 0 else 0
        benefit_color = "#10b981" if five_year_savings > 0 else "#ef4444"
        
        st.markdown(f"""
        <div class="glass-card">
            <h4 style="color: #f1f5f9; margin-bottom: 16px;">📊 5-Year Summary</h4>
            <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 16px;">
                <div>
                    <p style="color: #64748b; margin: 0;">Total Investment</p>
                    <p style="color: #f1f5f9; font-size: 1.25rem; font-weight: 600; margin: 4px 0;">${investment:,.0f}</p>
                </div>
                <div>
                    <p style="color: #64748b; margin: 0;">Cumulative Savings</p>
                    <p style="color: #10b981; font-size: 1.25rem; font-weight: 600; margin: 4px 0;">${annual_savings * 5:,.0f}</p>
                </div>
                <div>
                    <p style="color: #64748b; margin: 0;">Net Benefit</p>
                    <p style="color: {benefit_color}; font-size: 1.25rem; font-weight: 600; margin: 4px 0;">${five_year_savings:,.0f}</p>
                </div>
                <div>
                    <p style="color: #64748b; margin: 0;">ROI</p>
                    <p style="color: #6366f1; font-size: 1.25rem; font-weight: 600; margin: 4px 0;">{roi_pct:.0f}%</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)


def render_work_center_scenario(wc_df: pd.DataFrame, work_centers: list):
    """Render the work center scenario tab."""
    
    st.markdown("### Work Center / Production Line Optimization")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("#### Scenario Parameters")
        
        wc_index = st.selectbox(
            "Select Work Center Type",
            range(len(work_centers)),
            format_func=lambda x: work_centers[x],
            key="scenario_wc_select"
        )
        selected_wc = work_centers[wc_index]
        
        target_oee = st.slider(
            "Target OEE Improvement (%)",
            min_value=5,
            max_value=30,
            value=10,
            step=5,
            key="scenario_wc_oee"
        )
        
        maintenance_invest = st.number_input(
            "Maintenance Investment ($)",
            min_value=0,
            max_value=2000000,
            value=150000,
            step=25000,
            key="scenario_wc_investment"
        )
        
        # Get data for recommendations
        wc_data = wc_df[wc_df['work_center'] == selected_wc].copy()
        avg_oee = wc_data['oee'].mean()
        avg_age = wc_data['equipment_age'].mean()
        
        st.markdown("---")
        
        # Recommendations
        recommendations = get_work_center_recommendations(
            selected_wc, avg_oee, avg_age, target_oee
        )
        render_recommendations(recommendations)
    
    with col2:
        current_wc_cost = wc_data['monthly_scrap_cost'].sum()
        
        scrap_reduction_factor = target_oee * 1.5 / 100
        estimated_savings = current_wc_cost * 12 * scrap_reduction_factor
        
        st.markdown("#### Current State Analysis")
        
        res_col1, res_col2, res_col3 = st.columns(3)
        
        with res_col1:
            st.markdown(create_metric_card(
                f"{avg_oee*100:.1f}%",
                "Current Avg OEE"
            ), unsafe_allow_html=True)
        
        with res_col2:
            st.markdown(create_metric_card(
                f"{avg_age:.0f} yrs",
                "Avg Equipment Age"
            ), unsafe_allow_html=True)
        
        with res_col3:
            st.markdown(create_metric_card(
                f"${estimated_savings/1e3:.0f}K",
                "Est. Annual Savings"
            ), unsafe_allow_html=True)
        
        st.markdown("#### OEE Impact on Scrap (by Plant)")
        
        fig = px.scatter(
            wc_data,
            x='oee',
            y='monthly_scrap_cost',
            color='plant',
            size='equipment_age',
            hover_data=['plant', 'equipment_age'],
            labels={
                'oee': 'OEE',
                'monthly_scrap_cost': 'Monthly Scrap Cost ($)',
                'equipment_age': 'Equipment Age'
            }
        )
        
        fig.update_layout(
            **create_plotly_theme(),
            height=350,
            margin=dict(l=0, r=0, t=20, b=0),
            showlegend=True,
            legend=dict(orientation='h', y=-0.2)
        )
        
        fig.update_traces(marker=dict(line=dict(width=1, color='white')))
        
        st.plotly_chart(fig, use_container_width=True, key="wc_scatter_chart")


def render_part_scenario(parts_df: pd.DataFrame):
    """Render the part-level scenario tab."""
    
    st.markdown("### Part-Level Scrap Reduction")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("#### Select Parts to Analyze")
        
        parts_sorted = parts_df.sort_values('monthly_scrap_cost', ascending=False).reset_index(drop=True)
        
        part_options = parts_sorted['part_id'].tolist()
        part_display = {
            pid: f"{pid} - ${parts_sorted[parts_sorted['part_id']==pid]['monthly_scrap_cost'].values[0]:,.0f}/mo"
            for pid in part_options
        }
        
        default_parts = part_options[:5]
        
        selected_parts = st.multiselect(
            "Select Parts (Top Scrap Contributors)",
            options=part_options,
            default=default_parts,
            format_func=lambda x: part_display.get(x, x),
            key="scenario_part_select"
        )
        
        defect_reduction = st.slider(
            "Defect Rate Reduction Target (%)",
            min_value=10,
            max_value=75,
            value=30,
            step=5,
            key="scenario_part_defect"
        )
        
        # Get data for recommendations
        if selected_parts:
            selected_data = parts_df[parts_df['part_id'].isin(selected_parts)].copy()
            total_cost = selected_data['monthly_scrap_cost'].sum()
            avg_defect_rate = selected_data['defect_rate'].mean()
            
            st.markdown("---")
            
            recommendations = get_part_recommendations(
                len(selected_parts), total_cost, avg_defect_rate, defect_reduction
            )
            render_recommendations(recommendations)
    
    with col2:
        if selected_parts:
            selected_data = parts_df[parts_df['part_id'].isin(selected_parts)].copy()
            
            current_cost = selected_data['monthly_scrap_cost'].sum()
            projected_savings = current_cost * (defect_reduction / 100) * 12
            
            st.markdown("#### Selected Parts Analysis")
            
            res_col1, res_col2 = st.columns(2)
            
            with res_col1:
                st.markdown(create_metric_card(
                    f"${current_cost/1e3:.0f}K",
                    "Monthly Scrap Cost"
                ), unsafe_allow_html=True)
            
            with res_col2:
                st.markdown(create_metric_card(
                    f"${projected_savings/1e3:.0f}K",
                    "Projected Annual Savings"
                ), unsafe_allow_html=True)
            
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=selected_data['part_name'],
                y=selected_data['monthly_scrap_cost'],
                name='Current',
                marker_color='#ef4444'
            ))
            
            fig.add_trace(go.Bar(
                x=selected_data['part_name'],
                y=selected_data['monthly_scrap_cost'] * (1 - defect_reduction/100),
                name='After Improvement',
                marker_color='#10b981'
            ))
            
            fig.update_layout(
                **create_plotly_theme(),
                height=300,
                margin=dict(l=0, r=0, t=40, b=0),
                barmode='group',
                legend=dict(orientation='h', y=1.1),
                xaxis_tickangle=-45
            )
            
            st.plotly_chart(fig, use_container_width=True, key="part_bar_chart")
            
            st.markdown("#### Part Details")
            display_df = selected_data[['part_id', 'part_name', 'monthly_scrap_cost', 
                                        'defect_rate', 'unit_cost']].copy()
            display_df['defect_rate'] = display_df['defect_rate'].apply(lambda x: f"{x*100:.2f}%")
            display_df['monthly_scrap_cost'] = display_df['monthly_scrap_cost'].apply(lambda x: f"${x:,.0f}")
            display_df['unit_cost'] = display_df['unit_cost'].apply(lambda x: f"${x:.2f}")
            display_df.columns = ['Part ID', 'Part Name', 'Monthly Scrap', 'Defect Rate', 'Unit Cost']
            
            st.dataframe(display_df, use_container_width=True, hide_index=True)
        else:
            st.info("👆 Select one or more parts to analyze")
