"""
What-If Scenarios Page
======================
Interactive scenario planning for scrap reduction strategies.

Data Sources:
- Live data from Databricks Gold tables when available
- Mock data fallback for development/demo when DB unavailable
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from components.metrics import create_metric_card, create_plotly_theme
from data.queries import (
    load_plant_monthly_data,
    load_work_center_data,
    load_supplier_scenario_data,
    load_part_scrap_data,
    get_plants_list,
    get_work_center_types,
    get_supplier_alternatives
)


# =============================================================================
# Data Loading Functions - Live Data with Mock Fallback
# =============================================================================

def get_data_source_indicator(using_live: bool) -> str:
    """Return a small indicator showing data source."""
    if using_live:
        return "🟢 Live Data"
    return "🟡 Demo Data"


def should_use_mock_data() -> bool:
    """Check if user explicitly requested mock data."""
    return st.session_state.get('use_mock_data', False)


@st.cache_data(ttl=300)
def _load_scenario_data_from_db():
    """Load scenario data from database (cached)."""
    monthly_df = load_plant_monthly_data()
    parts_df = load_part_scrap_data()
    wc_df = load_work_center_data()
    plants_list = get_plants_list()
    wc_types = get_work_center_types()
    return monthly_df, parts_df, wc_df, plants_list, wc_types


def get_scenario_data():
    """
    Load scenario data from Databricks or generate mock data.
    
    Returns:
        Tuple of (monthly_df, parts_df, wc_df, plants, work_centers, is_live)
    """
    # Check if user explicitly wants mock data
    if should_use_mock_data():
        return _generate_mock_scenario_data()
    
    # Load from database
    monthly_df, parts_df, wc_df, plants_list, wc_types = _load_scenario_data_from_db()
    
    # Check if we got live data
    if monthly_df is not None and len(monthly_df) > 0:
        # Transform to expected format
        if 'month_date' in monthly_df.columns:
            monthly_df['month'] = pd.to_datetime(monthly_df['month_date'])
        elif 'month' not in monthly_df.columns:
            monthly_df['month'] = pd.Timestamp.now()
        
        if plants_list:
            plants = plants_list
        else:
            plants = monthly_df["plant"].dropna().unique().tolist()
        work_centers = wc_types if wc_types else []
        if not work_centers and wc_df is not None and 'work_center_type' in wc_df.columns:
            work_centers = wc_df['work_center_type'].unique().tolist()
        
        # Transform parts data - ensure required columns exist
        if parts_df is not None and len(parts_df) > 0:
            # Add missing columns if not present
            if 'supplier_name' not in parts_df.columns:
                parts_df['supplier_name'] = 'Unknown Supplier'
            if 'part_category' not in parts_df.columns:
                parts_df['part_category'] = 'General'
        else:
            # Show error - no parts data from DB
            st.warning("⚠️ Parts data not available from database")
            parts_df = pd.DataFrame({'part_id': [], 'part_name': [], 'monthly_scrap_cost': [], 'defect_rate': [], 'unit_cost': [], 'supplier_name': [], 'part_category': []})
        
        # Transform work center data  
        if wc_df is not None and len(wc_df) > 0:
            # Use work_center_type for grouping (different work center types)
            group_cols = []
            if 'work_center_type' in wc_df.columns:
                group_cols.append('work_center_type')
            if 'plant' in wc_df.columns:
                group_cols.append('plant')
            
            if group_cols:
                agg_dict = {}
                if 'monthly_scrap_cost' in wc_df.columns:
                    agg_dict['monthly_scrap_cost'] = 'mean'
                if 'oee' in wc_df.columns:
                    agg_dict['oee'] = 'mean'
                if 'avg_equipment_age' in wc_df.columns:
                    agg_dict['avg_equipment_age'] = 'mean'
                
                if agg_dict:
                    wc_df = wc_df.groupby(group_cols).agg(agg_dict).reset_index()
                    
            # Rename columns to expected names
            rename_map = {}
            if 'work_center_type' in wc_df.columns:
                rename_map['work_center_type'] = 'work_center'
            if 'avg_equipment_age' in wc_df.columns:
                rename_map['avg_equipment_age'] = 'equipment_age'
            if rename_map:
                wc_df = wc_df.rename(columns=rename_map)
        else:
            # Show error - no work center data from DB
            st.warning("⚠️ Work center data not available from database")
            wc_df = pd.DataFrame({'work_center': [], 'plant': [], 'monthly_scrap_cost': [], 'oee': [], 'equipment_age': []})
        
        return monthly_df, parts_df, wc_df, plants, work_centers, True
    
    # No data from database - show error (don't fall back to mock)
    st.error("❌ Could not load data from database. Check connection in Settings or enable Mock Data.")
    # Return empty dataframes to prevent crashes
    return (
        pd.DataFrame({'month': [], 'plant': [], 'scrap_cost': []}),
        pd.DataFrame({'part_id': [], 'part_name': [], 'monthly_scrap_cost': [], 'defect_rate': [], 'unit_cost': [], 'supplier_name': [], 'part_category': []}),
        pd.DataFrame({'work_center': [], 'plant': [], 'monthly_scrap_cost': [], 'oee': [], 'equipment_age': []}),
        [],
        [],
        False
    )


def _generate_mock_parts_data():
    """Generate mock parts data only."""
    np.random.seed(42)
    parts = [f"PRT-{str(i).zfill(5)}" for i in range(1, 51)]
    part_names = [f"Component {chr(65 + i % 26)}-{i}" for i in range(50)]
    supplier_names = [
        "Precision Metal Works", "Global Plastics Inc", "AutoParts Direct",
        "Quality Components Ltd", "Premium Electronics", "CNC Masters",
        "Midwest Stamping", "Southern Castings", "Atlantic Assembly"
    ]
    parts_data = []
    for i, (part, name) in enumerate(zip(parts, part_names)):
        supplier = supplier_names[i % len(supplier_names)]
        parts_data.append({
            'part_id': part,
            'part_name': name,
            'monthly_scrap_cost': np.random.uniform(5000, 50000),
            'defect_rate': np.random.uniform(0.01, 0.08),
            'unit_cost': np.random.uniform(10, 200),
            'supplier_name': supplier,
            'part_category': ['Stamping', 'Injection Molding', 'Machining', 'Electronics', 'Casting'][i % 5]
        })
    return pd.DataFrame(parts_data)


def _generate_mock_wc_data():
    """Generate mock work center data only."""
    np.random.seed(42)
    plants = ["Detroit Assembly", "Louisville Stamping", "San Antonio Body", 
              "Oakville Paint", "Saltillo Powertrain", "Silao Components"]
    work_centers = ["Stamping Press", "Welding Cell", "CNC Machining", 
                    "Injection Molding", "Assembly Line", "Paint Booth"]
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
    return pd.DataFrame(wc_data)


def _generate_mock_scenario_data():
    """Generate mock data for scenarios when live data unavailable."""
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
        "Stamping Press", 
        "Welding Cell", 
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
    
    # Parts data with supplier info
    supplier_names = [
        "Precision Metal Works", "Global Plastics Inc", "AutoParts Direct",
        "Quality Components Ltd", "Premium Electronics", "CNC Masters",
        "Midwest Stamping", "Southern Castings", "Atlantic Assembly"
    ]
    parts_data = []
    for i, (part, name) in enumerate(zip(parts, part_names)):
        supplier = supplier_names[i % len(supplier_names)]
        parts_data.append({
            'part_id': part,
            'part_name': name,
            'monthly_scrap_cost': np.random.uniform(5000, 50000),
            'defect_rate': np.random.uniform(0.01, 0.08),
            'unit_cost': np.random.uniform(10, 200),
            'supplier_name': supplier,
            'part_category': ['Stamping', 'Injection Molding', 'Machining', 'Electronics', 'Casting'][i % 5]
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
    
    return monthly_df, parts_df, wc_df, plants, work_centers, False


def get_supplier_data():
    """
    Load supplier data from Databricks or generate mock data.
    
    Returns:
        Tuple of (suppliers_df, is_live)
    """
    # Check if user explicitly wants mock data
    if should_use_mock_data():
        return _generate_mock_supplier_data()
    
    # Try loading live data
    suppliers_df = load_supplier_scenario_data()
    
    if suppliers_df is not None and len(suppliers_df) > 0:
        # Rename columns to match expected format
        suppliers_df = suppliers_df.rename(columns={
            'supplier_id': 'id',
            'name': 'name',
            'tier': 'tier',
            'category': 'category',
            'region': 'region',
            'quality_score': 'quality_score',
            'on_time_delivery': 'on_time_delivery',
            'parts_supplied': 'parts_supplied',
            'annual_spend': 'annual_spend',
            'scrap_cost': 'scrap_cost',
            'scrap_rate': 'scrap_rate',
            'capacity_util': 'capacity_util',
            'available_capacity': 'available_capacity',
            'risk_score': 'risk_score',
            'unit_cost_index': 'unit_cost_index'
        })
        return suppliers_df, True
    
    # No data from database - return empty DataFrame with required columns
    st.error("❌ Could not load supplier data from database.")
    empty_df = pd.DataFrame({
        'id': [], 'name': [], 'tier': [], 'category': [], 'region': [],
        'quality_score': [], 'on_time_delivery': [], 'parts_supplied': [],
        'annual_spend': [], 'scrap_cost': [], 'scrap_rate': [],
        'capacity_util': [], 'available_capacity': [], 'risk_score': [],
        'unit_cost_index': []
    })
    return empty_df, False


def _generate_mock_supplier_data():
    """Generate mock supplier scenario data."""
    np.random.seed(42)
    
    # Supplier master data
    suppliers = [
        {"id": "SUP-001", "name": "Precision Metal Works", "tier": "Tier 1", "category": "Stamping", "region": "Midwest"},
        {"id": "SUP-002", "name": "Allied Steel Components", "tier": "Tier 1", "category": "Stamping", "region": "Southeast"},
        {"id": "SUP-003", "name": "TechForm Industries", "tier": "Tier 1", "category": "Forming", "region": "Midwest"},
        {"id": "SUP-004", "name": "Premier Plastics Inc", "tier": "Tier 1", "category": "Injection Molding", "region": "Southwest"},
        {"id": "SUP-005", "name": "Global Plastics LLC", "tier": "Tier 2", "category": "Injection Molding", "region": "Mexico"},
        {"id": "SUP-006", "name": "Elite Machining Corp", "tier": "Tier 1", "category": "Machining", "region": "Midwest"},
        {"id": "SUP-007", "name": "Rapid CNC Solutions", "tier": "Tier 2", "category": "Machining", "region": "Canada"},
        {"id": "SUP-008", "name": "Superior Weld Systems", "tier": "Tier 1", "category": "Welding", "region": "Midwest"},
        {"id": "SUP-009", "name": "AutoWeld Technologies", "tier": "Tier 2", "category": "Welding", "region": "Southeast"},
        {"id": "SUP-010", "name": "Coating Solutions Inc", "tier": "Tier 1", "category": "Coating", "region": "Southwest"},
        {"id": "SUP-011", "name": "PaintTech Industries", "tier": "Tier 2", "category": "Coating", "region": "Mexico"},
        {"id": "SUP-012", "name": "Assembly Masters", "tier": "Tier 1", "category": "Assembly", "region": "Midwest"},
    ]
    
    # Add performance metrics
    for sup in suppliers:
        # Quality score (higher = better, 0.7-0.99)
        base_quality = 0.92 if sup["tier"] == "Tier 1" else 0.85
        sup["quality_score"] = min(0.99, max(0.70, base_quality + np.random.uniform(-0.08, 0.07)))
        
        # On-time delivery (0.8-0.99)
        sup["on_time_delivery"] = np.random.uniform(0.82, 0.98)
        
        # Annual spend
        if sup["tier"] == "Tier 1":
            sup["annual_spend"] = np.random.uniform(2_000_000, 8_000_000)
        else:
            sup["annual_spend"] = np.random.uniform(500_000, 2_500_000)
        
        # Scrap rate (lower = better)
        sup["scrap_rate"] = (1 - sup["quality_score"]) * np.random.uniform(0.8, 1.5)
        
        # Scrap cost contribution
        sup["scrap_cost"] = sup["annual_spend"] * sup["scrap_rate"]
        
        # Parts supplied count
        sup["parts_supplied"] = np.random.randint(5, 25)
        
        # Capacity utilization
        sup["capacity_util"] = np.random.uniform(0.60, 0.95)
        
        # Available capacity (inverse of utilization)
        sup["available_capacity"] = 1 - sup["capacity_util"]
        
        # Cost per unit (relative)
        sup["unit_cost_index"] = 1.0 if sup["tier"] == "Tier 1" else np.random.uniform(0.85, 1.05)
        
        # Risk score (0-100, lower = better)
        risk_base = 25 if sup["tier"] == "Tier 1" else 40
        sup["risk_score"] = max(5, min(95, risk_base + np.random.uniform(-15, 25)))
    
    suppliers_df = pd.DataFrame(suppliers)
    
    return suppliers_df, False  # Return False for is_live flag


def render_scenarios_page():
    """Render the What-If Scenarios page."""
    
    # Load data once
    monthly_df, parts_df, wc_df, plants, work_centers, data_is_live = get_scenario_data()
    suppliers_df, supplier_data_is_live = get_supplier_data()
    
    # Show data source indicator
    is_live = data_is_live or supplier_data_is_live
    data_source = get_data_source_indicator(is_live)
    
    st.markdown(f"""
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 24px;">
        <div>
            <h1 style="margin: 0;">🎯 What-If Scenarios</h1>
            <p style="font-size: 1.1rem; color: #64748b; margin: 8px 0 0 0;">
        Simulate scrap reduction strategies and estimate cost savings
    </p>
        </div>
        <div style="background: #1a1a2e; padding: 8px 16px; border-radius: 8px; font-size: 0.85rem;">
            {data_source}
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Scenario tabs - now with 4 tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🏭 By Plant", "🔧 By Work Center", "📦 By Part", "🏢 By Supplier"])
    
    with tab1:
        render_plant_scenario(monthly_df, plants)
    
    with tab2:
        render_work_center_scenario(wc_df, work_centers)
    
    with tab3:
        render_part_scenario(parts_df)
    
    with tab4:
        render_supplier_scenario(suppliers_df)


def get_supplier_recommendations(supplier: dict, suppliers_df: pd.DataFrame, spend_change: int, action: str) -> list:
    """Generate context-aware recommendations for supplier scenario."""
    recommendations = []
    
    # Get alternative suppliers in same category
    alternatives = suppliers_df[
        (suppliers_df['category'] == supplier['category']) & 
        (suppliers_df['id'] != supplier['id'])
    ].sort_values('quality_score', ascending=False)
    
    if action == "reduce":
        if supplier['quality_score'] < 0.85:
            recommendations.append({
                "icon": "⚠️",
                "title": "Quality Concern",
                "desc": f"Quality score {supplier['quality_score']:.1%} is below target. Consider accelerating reduction.",
                "priority": "high"
            })
        
        if spend_change >= 30:
            recommendations.append({
                "icon": "📋",
                "title": "Supplier Development Plan",
                "desc": "Major reduction requires formal communication and timeline",
                "priority": "high"
            })
        
        if len(alternatives) > 0:
            best_alt = alternatives.iloc[0]
            if best_alt['available_capacity'] > 0.15:
                recommendations.append({
                    "icon": "✅",
                    "title": f"Shift to {best_alt['name']}",
                    "desc": f"Quality: {best_alt['quality_score']:.1%}, {best_alt['available_capacity']:.0%} capacity available",
                    "priority": "high"
                })
        
        if supplier['parts_supplied'] > 15:
            recommendations.append({
                "icon": "📦",
                "title": "Multi-Source Parts",
                "desc": f"Supplier has {supplier['parts_supplied']} parts. Prioritize highest-scrap items first.",
                "priority": "medium"
            })
    
    else:  # increase
        if supplier['quality_score'] >= 0.95:
            recommendations.append({
                "icon": "⭐",
                "title": "Top Performer",
                "desc": "Excellent quality score supports increased allocation",
                "priority": "high"
            })
        
        if supplier['capacity_util'] > 0.85:
            recommendations.append({
                "icon": "⚠️",
                "title": "Capacity Constraint",
                "desc": f"Supplier at {supplier['capacity_util']:.0%} capacity. Verify ability to scale.",
                "priority": "high"
            })
        
        if supplier['tier'] == "Tier 2":
            recommendations.append({
                "icon": "📈",
                "title": "Tier Upgrade Candidate",
                "desc": "Consider upgrading to Tier 1 with increased volume commitment",
                "priority": "medium"
            })
    
    # General recommendations
    recommendations.append({
        "icon": "📊",
        "title": "Track Quality Metrics",
        "desc": "Monitor PPM, delivery, and scrap weekly during transition",
        "priority": "low"
    })
    
    recommendations.append({
        "icon": "💰",
        "title": "Negotiate Terms",
        "desc": "Volume changes may warrant pricing renegotiation",
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


def render_supplier_scenario(suppliers_df: pd.DataFrame):
    """Render the supplier what-if scenario tab."""
    
    st.markdown("### Supplier Spend & Allocation Analysis")
    
    # Handle empty DataFrame
    if suppliers_df is None or len(suppliers_df) == 0:
        st.warning("⚠️ No supplier data available. Enable Mock Data in Settings or check database connection.")
        return
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("#### Scenario Parameters")
        
        # Supplier selection
        supplier_options = suppliers_df["name"].tolist()
        selected_name = st.selectbox(
            "Select Supplier",
            supplier_options,
            key="scenario_supplier_select",
        )
        selected_supplier = (
            suppliers_df[suppliers_df["name"] == selected_name].iloc[0].to_dict()
        )
        
        # Spend adjustment direction
        spend_action = st.radio(
            "Spend Adjustment",
            ["Reduce Spend", "Increase Spend"],
            key="scenario_spend_action",
            horizontal=True
        )
        
        # Spend change percentage
        spend_change = st.slider(
            f"{'Reduction' if spend_action == 'Reduce Spend' else 'Increase'} Target (%)",
            min_value=5,
            max_value=50,
            value=20,
            step=5,
            key="scenario_spend_change"
        )
        
        # Show current supplier info
        st.markdown("---")
        st.markdown("#### Current Supplier Profile")
        
        tier_color = "#10b981" if selected_supplier['tier'] == "Tier 1" else "#f59e0b"
        st.markdown(f"""
        <div style="background: #1a1a2e; border-radius: 12px; padding: 16px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                <span style="color: #f1f5f9; font-weight: 600;">{selected_supplier['name']}</span>
                <span style="background: {tier_color}22; color: {tier_color}; padding: 4px 8px; border-radius: 4px; font-size: 0.75rem;">{selected_supplier['tier']}</span>
            </div>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px; font-size: 0.85rem;">
                <div><span style="color: #64748b;">Category:</span> <span style="color: #f1f5f9;">{selected_supplier['category']}</span></div>
                <div><span style="color: #64748b;">Region:</span> <span style="color: #f1f5f9;">{selected_supplier['region']}</span></div>
                <div><span style="color: #64748b;">Quality:</span> <span style="color: {'#10b981' if selected_supplier['quality_score'] >= 0.90 else '#f59e0b'};">{selected_supplier['quality_score']:.1%}</span></div>
                <div><span style="color: #64748b;">On-Time:</span> <span style="color: #f1f5f9;">{selected_supplier['on_time_delivery']:.1%}</span></div>
                <div><span style="color: #64748b;">Capacity:</span> <span style="color: #f1f5f9;">{selected_supplier['capacity_util']:.0%} used</span></div>
                <div><span style="color: #64748b;">Risk Score:</span> <span style="color: {'#ef4444' if selected_supplier['risk_score'] > 50 else '#10b981'};">{selected_supplier['risk_score']:.0f}/100</span></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Recommendations
        action = "reduce" if spend_action == "Reduce Spend" else "increase"
        recommendations = get_supplier_recommendations(selected_supplier, suppliers_df, spend_change, action)
        render_recommendations(recommendations)
    
    with col2:
        # Calculate impact
        current_spend = selected_supplier['annual_spend']
        current_scrap = selected_supplier['scrap_cost']
        
        if spend_action == "Reduce Spend":
            new_spend = current_spend * (1 - spend_change / 100)
            spend_delta = current_spend - new_spend
            new_scrap = current_scrap * (1 - spend_change / 100)
            scrap_delta = current_scrap - new_scrap
        else:
            new_spend = current_spend * (1 + spend_change / 100)
            spend_delta = new_spend - current_spend
            # Increasing spend with good supplier may reduce overall scrap
            quality_factor = selected_supplier['quality_score']
            new_scrap = current_scrap * (1 + spend_change / 100 * (1 - quality_factor))
            scrap_delta = new_scrap - current_scrap
        
        st.markdown("#### Projected Impact")
        
        # KPI cards
        kpi_col1, kpi_col2, kpi_col3 = st.columns(3)
        
        with kpi_col1:
            st.markdown(create_metric_card(
                f"${current_spend/1e6:.2f}M",
                "Current Annual Spend"
            ), unsafe_allow_html=True)
        
        with kpi_col2:
            delta_sign = "-" if spend_action == "Reduce Spend" else "+"
            st.markdown(create_metric_card(
                f"${new_spend/1e6:.2f}M",
                "Projected Spend",
                -spend_change if spend_action == "Reduce Spend" else spend_change
            ), unsafe_allow_html=True)
        
        with kpi_col3:
            scrap_change_pct = (scrap_delta / current_scrap * 100) if current_scrap > 0 else 0
            st.markdown(create_metric_card(
                f"${abs(scrap_delta)/1e3:.0f}K",
                "Scrap Impact",
                -abs(scrap_change_pct) if spend_action == "Reduce Spend" else scrap_change_pct
            ), unsafe_allow_html=True)
        
        # Alternative suppliers section
        st.markdown("#### 🔄 Alternative Suppliers")
        
        alternatives = suppliers_df[
            (suppliers_df['category'] == selected_supplier['category']) & 
            (suppliers_df['id'] != selected_supplier['id'])
        ].sort_values('quality_score', ascending=False)
        
        if len(alternatives) > 0:
            st.markdown(f"*Suppliers in **{selected_supplier['category']}** category that could absorb demand:*")
            
            # Create comparison chart
            fig = go.Figure()
            
            # Add current supplier
            fig.add_trace(go.Bar(
                name='Current',
                x=[selected_supplier['name'][:20]],
                y=[selected_supplier['quality_score'] * 100],
                marker_color='#6366f1',
                text=[f"{selected_supplier['quality_score']:.1%}"],
                textposition='outside'
            ))
            
            # Add alternatives
            for _, alt in alternatives.head(4).iterrows():
                color = '#10b981' if alt['quality_score'] > selected_supplier['quality_score'] else '#f59e0b'
                fig.add_trace(go.Bar(
                    name=alt['name'][:15],
                    x=[alt['name'][:20]],
                    y=[alt['quality_score'] * 100],
                    marker_color=color,
                    text=[f"{alt['quality_score']:.1%}"],
                    textposition='outside'
                ))
            
            theme = create_plotly_theme()
            fig.update_layout(
                paper_bgcolor=theme.get('paper_bgcolor'),
                plot_bgcolor=theme.get('plot_bgcolor'),
                font=theme.get('font'),
                height=300,
                margin=dict(l=60, r=20, t=30, b=80),
                showlegend=False,
                xaxis_title="Supplier",
                yaxis_title="Quality Score (%)",
                yaxis_range=[70, 105],
                xaxis=dict(
                    tickangle=-30,
                    tickfont=dict(size=10, color='#4a4a48'),
                    title_font=dict(size=12, color='#1B3139')
                ),
                yaxis=dict(
                    ticksuffix='%',
                    tickfont=dict(size=11, color='#4a4a48'),
                    title_font=dict(size=12, color='#1B3139')
                )
            )
            
            st.plotly_chart(fig, use_container_width=True, key="supplier_quality_chart")
            
            # Detailed alternatives table
            st.markdown("##### Available Capacity & Details")
            
            alt_display = alternatives.head(4)[['name', 'tier', 'quality_score', 'available_capacity', 'risk_score', 'unit_cost_index']].copy()
            alt_display['quality_score'] = alt_display['quality_score'].apply(lambda x: f"{x:.1%}")
            alt_display['available_capacity'] = alt_display['available_capacity'].apply(lambda x: f"{x:.0%}")
            alt_display['risk_score'] = alt_display['risk_score'].apply(lambda x: f"{x:.0f}")
            alt_display['unit_cost_index'] = alt_display['unit_cost_index'].apply(lambda x: f"{x:.2f}x")
            alt_display.columns = ['Supplier', 'Tier', 'Quality', 'Available Cap.', 'Risk', 'Cost Index']
            
            st.dataframe(alt_display, use_container_width=True, hide_index=True)
            
            # Reallocation recommendation
            if spend_action == "Reduce Spend" and len(alternatives) > 0:
                best_alt = alternatives.iloc[0]
                reallocate_amount = spend_delta * 0.8  # Assume 80% can be reallocated
                
                st.markdown(f"""
                <div style="background: #1B3139; border-radius: 12px; padding: 20px; margin: 12px 0;">
                    <h4 style="color: #FFFFFF; margin-bottom: 12px;">📊 Recommended Reallocation</h4>
                    <p style="color: #FF3621; margin-bottom: 12px;">
                        Shift <strong style="color: #10b981;">${reallocate_amount/1e6:.2f}M</strong> 
                        ({spend_change * 0.8:.0f}% of reduction) to <strong style="color: #FFFFFF;">{best_alt['name']}</strong>
                    </p>
                    <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; font-size: 0.85rem;">
                        <div style="background: #10b981; padding: 12px; border-radius: 8px; text-align: center;">
                            <p style="color: #FFFFFF; margin: 0;">Quality Gain</p>
                            <p style="color: #FFFFFF; font-size: 1.25rem; font-weight: 600; margin: 4px 0;">
                                +{(best_alt['quality_score'] - selected_supplier['quality_score']) * 100:.1f}%
                            </p>
                        </div>
                        <div style="background: {'#10b981' if best_alt['available_capacity'] > 0.2 else '#f59e0b'}; padding: 12px; border-radius: 8px; text-align: center;">
                            <p style="color: #FFFFFF; margin: 0;">Capacity Fit</p>
                            <p style="color: #FFFFFF; font-size: 1.25rem; font-weight: 600; margin: 4px 0;">
                                {best_alt['available_capacity']:.0%}
                            </p>
                        </div>
                        <div style="background: #1B3139; border: 2px solid #10b981; padding: 12px; border-radius: 8px; text-align: center;">
                            <p style="color: #FFFFFF; margin: 0;">Est. Scrap Savings</p>
                            <p style="color: #10b981; font-size: 1.25rem; font-weight: 600; margin: 4px 0;">
                                ${(scrap_delta * (best_alt['quality_score'] / selected_supplier['quality_score']))/1e3:.0f}K
                            </p>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.warning(f"No alternative suppliers found in the {selected_supplier['category']} category.")
        
        # Supplier comparison scatter
        st.markdown("#### 📈 All Suppliers: Quality vs Scrap Cost")
        
        fig2 = px.scatter(
            suppliers_df,
            x='quality_score',
            y='scrap_cost',
            size='annual_spend',
            color='tier',
            hover_name='name',
            hover_data=['category', 'capacity_util', 'risk_score'],
            color_discrete_map={'Tier 1': '#6366f1', 'Tier 2': '#f59e0b'},
            labels={
                'quality_score': 'Quality Score',
                'scrap_cost': 'Annual Scrap Cost ($)',
                'tier': 'Tier'
            }
        )
        
        # Highlight selected supplier
        fig2.add_trace(go.Scatter(
            x=[selected_supplier['quality_score']],
            y=[selected_supplier['scrap_cost']],
            mode='markers',
            marker=dict(size=25, color='#ef4444', symbol='star', line=dict(width=2, color='white')),
            name='Selected',
            hoverinfo='skip'
        ))
        
        theme = create_plotly_theme()
        fig2.update_layout(
            paper_bgcolor=theme.get('paper_bgcolor'),
            plot_bgcolor=theme.get('plot_bgcolor'),
            font=theme.get('font'),
            height=350,
            margin=dict(l=80, r=20, t=30, b=80),
            legend=dict(orientation='h', y=-0.25, font=dict(color='#4a4a48')),
            xaxis_title="Quality Score",
            yaxis_title="Annual Scrap Cost ($)",
            xaxis=dict(
                tickformat='.0%',
                showticklabels=True,
                tickfont=dict(size=11, color='#4a4a48'),
                title_font=dict(size=12, color='#1B3139')
            ),
            yaxis=dict(
                tickprefix='$',
                tickformat=',.0f',
                showticklabels=True,
                tickfont=dict(size=11, color='#4a4a48'),
                title_font=dict(size=12, color='#1B3139')
            )
        )
        
        st.plotly_chart(fig2, use_container_width=True, key="supplier_scatter_chart")


# =============================================================================
# Other scenario functions (Plant, Work Center, Part) - kept from original
# =============================================================================

def get_plant_recommendations(plant: str, reduction_target: int, roi_months: float, annual_savings: float) -> list:
    """Generate context-aware recommendations for plant scenario."""
    recommendations = []
    
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
    
    return recommendations[:6]


def get_work_center_recommendations(work_center: str, avg_oee: float, avg_age: float, target_oee: int) -> list:
    """Generate context-aware recommendations for work center scenario."""
    recommendations = []
    
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


def get_part_recommendations(num_parts: int, total_cost: float, avg_defect_rate: float, 
                              reduction_target: int, supplier_info: dict = None, 
                              alt_suppliers: list = None) -> list:
    """Generate context-aware recommendations for part scenario with re-sourcing options."""
    recommendations = []
    
    # Re-sourcing recommendation if supplier has quality issues
    if supplier_info and supplier_info.get('quality_score', 1.0) < 0.85:
        recommendations.append({
            "icon": "🔄",
            "title": "Re-Source Part",
            "desc": f"Supplier quality {supplier_info.get('quality_score', 0)*100:.0f}% - evaluate alternate sources",
            "priority": "high"
        })
        if alt_suppliers and len(alt_suppliers) > 0:
            best_alt = alt_suppliers[0]
            recommendations.append({
                "icon": "🏆",
                "title": f"Alternate: {best_alt.get('name', 'N/A')[:25]}",
                "desc": f"Quality: {best_alt.get('quality_score', 0)*100:.0f}% | Capacity: {best_alt.get('available_capacity', 0)*100:.0f}% avail",
                "priority": "high"
            })
    
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
    
    if num_parts >= 5:
        recommendations.append({
            "icon": "🎯",
            "title": "Focus Strategy",
            "desc": "Consider tackling top 3 parts first for faster wins",
            "priority": "medium"
        })
    
    if total_cost > 100000:
        recommendations.append({
            "icon": "💰",
            "title": "High-Value Target",
            "desc": f"${total_cost/1000:.0f}K/month - executive sponsorship recommended",
            "priority": "high"
        })
    
    # Supplier-specific recommendations
    if supplier_info:
        if supplier_info.get('on_time_delivery', 1.0) < 0.90:
            recommendations.append({
                "icon": "⏰",
                "title": "Delivery Issues",
                "desc": f"On-time {supplier_info.get('on_time_delivery', 0)*100:.0f}% - consider dual-sourcing",
                "priority": "medium"
            })
        if supplier_info.get('capacity_util', 0) > 0.90:
            recommendations.append({
                "icon": "⚠️",
                "title": "Capacity Constrained",
                "desc": f"{supplier_info.get('capacity_util', 0)*100:.0f}% utilized - qualify backup supplier",
                "priority": "medium"
            })
    
    if reduction_target >= 50:
        recommendations.append({
            "icon": "🔬",
            "title": "Design Review",
            "desc": "Consider part redesign or material change",
            "priority": "medium"
        })
    
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


def get_mock_part_supplier_info(supplier_name: str) -> dict:
    """Get mock supplier info for a part."""
    import random
    random.seed(hash(supplier_name) % 1000)
    
    tiers = ["Tier 1", "Tier 2", "Tier 3"]
    regions = ["North America", "Mexico", "Canada", "EMEA", "APAC"]
    categories = ["Stamping", "Plastic Injection", "Electronics", "Machining", "Casting"]
    
    return {
        'name': supplier_name,
        'tier': random.choice(tiers),
        'category': random.choice(categories),
        'region': random.choice(regions),
        'quality_score': random.uniform(0.70, 0.98),
        'on_time_delivery': random.uniform(0.75, 0.98),
        'capacity_util': random.uniform(0.60, 0.95),
        'risk_score': random.randint(20, 80)
    }


def get_mock_alternative_suppliers(category: str, current_supplier: str) -> list:
    """Get mock alternative suppliers for the same category."""
    import random
    random.seed(42)
    
    all_suppliers = [
        {'name': 'Precision Metal Works', 'quality_score': 0.96, 'available_capacity': 0.25, 'tier': 'Tier 1'},
        {'name': 'Global Plastics Inc', 'quality_score': 0.94, 'available_capacity': 0.30, 'tier': 'Tier 1'},
        {'name': 'AutoParts Direct', 'quality_score': 0.91, 'available_capacity': 0.40, 'tier': 'Tier 2'},
        {'name': 'Quality Components Ltd', 'quality_score': 0.89, 'available_capacity': 0.35, 'tier': 'Tier 2'},
        {'name': 'Premium Electronics', 'quality_score': 0.95, 'available_capacity': 0.20, 'tier': 'Tier 1'},
        {'name': 'CNC Masters', 'quality_score': 0.92, 'available_capacity': 0.45, 'tier': 'Tier 1'},
    ]
    
    # Filter out current supplier and return top alternatives
    return [s for s in all_suppliers if s['name'] != current_supplier][:3]


def render_plant_scenario(monthly_df: pd.DataFrame, plants: list):
    """Render the plant-level scenario tab."""
    
    st.markdown("### Plant-Level Scrap Reduction")
    
    if not plants:
        st.warning(
            "⚠️ No plants available for scenarios. Check database data or enable Mock Data in Settings."
        )
        return
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("#### Scenario Parameters")
        
        # Use plant names as options (not indices) so widget state stays valid when the list size changes
        selected_plant = st.selectbox(
            "Select Plant",
            plants,
            key="scenario_plant_select",
        )
        
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
    
        plant_data = monthly_df[monthly_df['plant'] == selected_plant]
        current_annual_cost = plant_data['scrap_cost'].sum() / 3
        annual_savings = current_annual_cost * (reduction_target / 100)
        roi_months = investment / (annual_savings / 12) if annual_savings > 0 else 999
        
        st.markdown("---")
        
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
        
        theme = create_plotly_theme()
        fig.update_layout(
            paper_bgcolor=theme.get('paper_bgcolor'),
            plot_bgcolor=theme.get('plot_bgcolor'),
            font=theme.get('font'),
            height=300,
            margin=dict(l=80, r=20, t=40, b=60),
            legend=dict(orientation='h', y=1.1, font=dict(color='#4a4a48')),
            hovermode='x unified',
            xaxis_title="Month",
            yaxis_title="Monthly Scrap Cost ($)",
            xaxis=dict(
                tickfont=dict(size=11, color='#4a4a48'),
                title_font=dict(size=12, color='#1B3139')
            ),
            yaxis=dict(
                tickformat='$,.0f',
                tickfont=dict(size=11, color='#4a4a48'),
                title_font=dict(size=12, color='#1B3139')
            )
        )
        
        st.plotly_chart(fig, use_container_width=True, key="plant_trajectory_chart")
        
        roi_pct = ((annual_savings * 5) / investment - 1) * 100 if investment > 0 else 0
        benefit_color = "#10b981" if five_year_savings > 0 else "#ef4444"
        
        st.markdown(f"""
        <div class="glass-card">
            <h4 style="color: #1B3139; margin-bottom: 16px;">📊 5-Year Summary</h4>
            <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 16px;">
                <div>
                    <p style="color: #6b6b69; margin: 0;">Total Investment</p>
                    <p style="color: #1B3139; font-size: 1.25rem; font-weight: 600; margin: 4px 0;">${investment:,.0f}</p>
                </div>
                <div>
                    <p style="color: #6b6b69; margin: 0;">Cumulative Savings</p>
                    <p style="color: #10b981; font-size: 1.25rem; font-weight: 600; margin: 4px 0;">${annual_savings * 5:,.0f}</p>
                </div>
                <div>
                    <p style="color: #6b6b69; margin: 0;">Net Benefit</p>
                    <p style="color: {benefit_color}; font-size: 1.25rem; font-weight: 600; margin: 4px 0;">${five_year_savings:,.0f}</p>
                </div>
                <div>
                    <p style="color: #6b6b69; margin: 0;">ROI</p>
                    <p style="color: #FF3621; font-size: 1.25rem; font-weight: 600; margin: 4px 0;">{roi_pct:.0f}%</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)


def render_work_center_scenario(wc_df: pd.DataFrame, work_centers: list):
    """Render the work center scenario tab."""
    
    st.markdown("### Work Center / Production Line Optimization")
    
    if not work_centers:
        st.warning(
            "⚠️ No work center types available. Check database data or enable Mock Data in Settings."
        )
        return
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("#### Scenario Parameters")
        
        selected_wc = st.selectbox(
            "Select Work Center Type",
            work_centers,
            key="scenario_wc_select",
        )
        
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
    
        wc_data = wc_df[wc_df['work_center'] == selected_wc].copy()
        avg_oee = wc_data['oee'].mean()
        avg_age = wc_data['equipment_age'].mean()
        
        st.markdown("---")
        
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
        
        theme = create_plotly_theme()
        fig.update_layout(
            paper_bgcolor=theme.get('paper_bgcolor'),
            plot_bgcolor=theme.get('plot_bgcolor'),
            font=theme.get('font'),
            height=350,
            margin=dict(l=80, r=20, t=20, b=80),
            showlegend=True,
            legend=dict(orientation='h', y=-0.3, font=dict(color='#4a4a48')),
            xaxis_title="OEE (%)",
            yaxis_title="Monthly Scrap Cost ($)",
            xaxis=dict(
                tickformat='.0%',
                tickfont=dict(size=11, color='#4a4a48'),
                title_font=dict(size=12, color='#1B3139')
            ),
            yaxis=dict(
                tickformat='$,.0f',
                tickfont=dict(size=11, color='#4a4a48'),
                title_font=dict(size=12, color='#1B3139')
            )
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
        
        if selected_parts:
            selected_data = parts_df[parts_df['part_id'].isin(selected_parts)].copy()
            total_cost = selected_data['monthly_scrap_cost'].sum()
            avg_defect_rate = selected_data['defect_rate'].mean()
            
            # Get supplier info for recommendations
            supplier_info = None
            alt_suppliers = []
            
            # Show supplier profile if single part selected
            if len(selected_parts) == 1:
                part_row = selected_data.iloc[0]
                supplier_name = part_row.get('supplier_name', 'Unknown Supplier')
                part_category = part_row.get('part_category', 'General')
                supplier_info = get_mock_part_supplier_info(supplier_name)
                alt_suppliers = get_mock_alternative_suppliers(part_category, supplier_name)
                
                st.markdown("---")
                st.markdown("#### Supplier Profile")
                
                tier_color = "#10b981" if supplier_info['tier'] == "Tier 1" else "#f59e0b" if supplier_info['tier'] == "Tier 2" else "#ef4444"
                quality_color = "#10b981" if supplier_info['quality_score'] >= 0.90 else "#f59e0b" if supplier_info['quality_score'] >= 0.80 else "#ef4444"
                
                st.markdown(f"""
                <div style="background: #1a1a2e; border-radius: 12px; padding: 16px;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                        <span style="color: #f1f5f9; font-weight: 600;">{supplier_info['name']}</span>
                        <span style="background: {tier_color}22; color: {tier_color}; padding: 4px 8px; border-radius: 4px; font-size: 0.75rem;">{supplier_info['tier']}</span>
                    </div>
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px; font-size: 0.85rem;">
                        <div><span style="color: #64748b;">Category:</span> <span style="color: #f1f5f9;">{supplier_info['category']}</span></div>
                        <div><span style="color: #64748b;">Region:</span> <span style="color: #f1f5f9;">{supplier_info['region']}</span></div>
                        <div><span style="color: #64748b;">Quality:</span> <span style="color: {quality_color};">{supplier_info['quality_score']:.1%}</span></div>
                        <div><span style="color: #64748b;">On-Time:</span> <span style="color: #f1f5f9;">{supplier_info['on_time_delivery']:.1%}</span></div>
                        <div><span style="color: #64748b;">Capacity:</span> <span style="color: #f1f5f9;">{supplier_info['capacity_util']:.0%} used</span></div>
                        <div><span style="color: #64748b;">Risk Score:</span> <span style="color: {'#ef4444' if supplier_info['risk_score'] > 50 else '#10b981'};">{supplier_info['risk_score']:.0f}/100</span></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Show alternative suppliers if current has quality issues
                if supplier_info['quality_score'] < 0.90 and alt_suppliers:
                    st.markdown("##### 🔄 Alternative Suppliers")
                    for alt in alt_suppliers[:2]:
                        alt_tier_color = "#10b981" if alt['tier'] == "Tier 1" else "#f59e0b"
                        st.markdown(f"""
                        <div style="background: #0f172a; border-radius: 8px; padding: 10px; margin: 4px 0; border-left: 3px solid #10b981;">
                            <div style="display: flex; justify-content: space-between;">
                                <span style="color: #f1f5f9; font-weight: 500;">{alt['name']}</span>
                                <span style="color: {alt_tier_color}; font-size: 0.75rem;">{alt['tier']}</span>
                            </div>
                            <div style="font-size: 0.8rem; color: #64748b;">
                                Quality: <span style="color: #10b981;">{alt['quality_score']:.0%}</span> | 
                                Capacity Available: <span style="color: #f1f5f9;">{alt['available_capacity']:.0%}</span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            recommendations = get_part_recommendations(
                len(selected_parts), total_cost, avg_defect_rate, defect_reduction,
                supplier_info, alt_suppliers
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
            
            theme = create_plotly_theme()
            fig.update_layout(
                paper_bgcolor=theme.get('paper_bgcolor'),
                plot_bgcolor=theme.get('plot_bgcolor'),
                font=theme.get('font'),
                height=350,
                margin=dict(l=80, r=20, t=40, b=100),
                barmode='group',
                legend=dict(orientation='h', y=1.1, font=dict(color='#4a4a48')),
                xaxis_tickangle=-45,
                xaxis_title="Part Name",
                yaxis_title="Monthly Scrap Cost ($)",
                xaxis=dict(
                    tickfont=dict(size=10, color='#4a4a48'),
                    title_font=dict(size=12, color='#1B3139')
                ),
                yaxis=dict(
                    tickformat='$,.0f',
                    tickfont=dict(size=11, color='#4a4a48'),
                    title_font=dict(size=12, color='#1B3139')
                )
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
