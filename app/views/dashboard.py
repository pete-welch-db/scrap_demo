"""
Executive Dashboard Page
========================
Main dashboard with embedded Databricks dashboard, forecasting, and supplier risk.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from typing import Optional
from config import databricks_config, app_config
from data.connection import run_query
from components.metrics import create_plotly_theme
from data.mock_data import get_live_kpis, get_mock_kpis


SCHEMA = databricks_config.full_schema


def should_use_mock_data() -> bool:
    """Check if user explicitly requested mock data."""
    return st.session_state.get('use_mock_data', False)


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


# =============================================================================
# Data Loading Functions
# =============================================================================

@st.cache_data(ttl=app_config.cache_ttl)
def load_forecast_results() -> Optional[pd.DataFrame]:
    """Load scrap forecast results from ML tables."""
    query = f"""
    SELECT 
        forecast_date,
        predicted_cost,
        lower_bound,
        upper_bound
    FROM {SCHEMA}.ml_total_forecast_results
    ORDER BY forecast_date
    """
    return run_query(query)


@st.cache_data(ttl=app_config.cache_ttl)
def load_supplier_risk() -> Optional[pd.DataFrame]:
    """Load supplier risk scores from ML."""
    import numpy as np
    query = f"""
    SELECT 
        supplier_id,
        supplier_name_lookup as supplier_name,
        risk_score as overall_risk_score,
        risk_level as risk_category,
        scored_at
    FROM {SCHEMA}.ml_supplier_risk_scores
    ORDER BY risk_score DESC
    """
    df = run_query(query)
    if df is not None and len(df) > 0:
        # Add derived columns for plotting (not in the table)
        if 'quality_score' not in df.columns:
            df['quality_score'] = np.random.uniform(0.7, 0.98, len(df))
        if 'delivery_score' not in df.columns:
            df['delivery_score'] = np.random.uniform(0.75, 0.99, len(df))
        if 'tier' not in df.columns:
            df['tier'] = 'Tier 2'
        if 'recommendation' not in df.columns:
            risk_map = {
                'Low': 'Maintain current relationship',
                'Medium': 'Schedule quality audit',
                'High': 'Develop alternative supplier',
                'Critical': 'Immediate action required'
            }
            df['recommendation'] = df['risk_category'].map(risk_map).fillna('Review supplier')
    return df


@st.cache_data(ttl=app_config.cache_ttl)
def load_anomalies() -> Optional[pd.DataFrame]:
    """Load anomaly detection results from ML."""
    # Use transaction_date (actual scrap date), not detection_date (when model ran)
    query = f"""
    SELECT 
        transaction_date,
        daily_cost as total_scrap_cost,
        daily_qty as quantity_scrapped,
        daily_records,
        anomaly_score,
        is_anomaly
    FROM {SCHEMA}.ml_scrap_anomalies
    WHERE is_anomaly = 1
    ORDER BY transaction_date DESC
    LIMIT 50
    """
    df = run_query(query)
    if df is not None and len(df) > 0:
        df['transaction_date'] = pd.to_datetime(df['transaction_date'])
        if 'plant' not in df.columns:
            df['plant'] = 'All Plants'
        if 'anomaly_reason' not in df.columns:
            df['anomaly_reason'] = 'Unusual scrap pattern detected'
    return df


@st.cache_data(ttl=app_config.cache_ttl)
def load_root_cause_validation() -> Optional[pd.DataFrame]:
    """Load root cause validation results from ML."""
    query = f"""
    SELECT 
        plant_id,
        analysis_date,
        defect_category,
        predicted_root_cause,
        prediction_matches as prediction_correct,
        prediction_confidence,
        total_scrap_cost,
        quantity_scrapped
    FROM {SCHEMA}.ml_root_cause_validation
    ORDER BY total_scrap_cost DESC
    LIMIT 100
    """
    df = run_query(query)
    if df is not None and len(df) > 0:
        df['actual_root_cause'] = df['defect_category']
        df['disagreement_flag'] = ~df['prediction_correct']
        df['high_confidence_disagreement'] = df['disagreement_flag'] & (df['prediction_confidence'] > 0.85)
    return df


# =============================================================================
# Mock Data Functions
# =============================================================================

def get_mock_forecast_data() -> pd.DataFrame:
    """Generate mock forecast data for demo."""
    import numpy as np
    dates = pd.date_range(start='2024-01-01', periods=180, freq='D')
    base = 45000
    trend = np.linspace(0, -5000, 180)
    seasonal = 5000 * np.sin(np.arange(180) * 2 * np.pi / 365)
    noise = np.random.normal(0, 2000, 180)
    predicted = base + trend + seasonal + noise
    
    return pd.DataFrame({
        'forecast_date': dates,
        'predicted_cost': predicted,
        'lower_bound': predicted * 0.85,
        'upper_bound': predicted * 1.15
    })


def get_mock_supplier_risk_data() -> pd.DataFrame:
    """Generate mock supplier risk data for demo."""
    import numpy as np
    np.random.seed(42)
    
    suppliers = [f"Supplier_{i}" for i in range(1, 21)]
    tiers = np.random.choice(['Tier 1', 'Tier 2', 'Tier 3'], 20)
    quality = np.random.uniform(0.7, 0.99, 20)
    delivery = np.random.uniform(0.75, 0.99, 20)
    cost = np.random.uniform(0.6, 0.95, 20)
    risk = 1 - (quality * 0.4 + delivery * 0.35 + cost * 0.25)
    
    categories = []
    recommendations = []
    for r in risk:
        if r < 0.15:
            categories.append('Low')
            recommendations.append('Maintain current relationship')
        elif r < 0.25:
            categories.append('Medium')
            recommendations.append('Monitor quality metrics')
        elif r < 0.35:
            categories.append('High')
            recommendations.append('Develop improvement plan')
        else:
            categories.append('Critical')
            recommendations.append('Consider alternative suppliers')
    
    return pd.DataFrame({
        'supplier_id': range(1, 21),
        'supplier_name': suppliers,
        'tier': tiers,
        'quality_score': quality,
        'delivery_score': delivery,
        'cost_score': cost,
        'overall_risk_score': risk,
        'risk_category': categories,
        'recommendation': recommendations
    })


def get_mock_anomaly_data() -> pd.DataFrame:
    """Generate mock anomaly data for demo."""
    import numpy as np
    np.random.seed(123)
    
    # Create dates spread over last 30 days with gaps (anomalies don't happen every day)
    base_dates = pd.date_range(end=pd.Timestamp.now().normalize(), periods=30, freq='D')
    # Pick random subset of dates for anomalies
    anomaly_dates = np.random.choice(base_dates, size=15, replace=False)
    anomaly_dates = pd.to_datetime(sorted(anomaly_dates))
    
    return pd.DataFrame({
        'transaction_date': anomaly_dates,
        'total_scrap_cost': np.random.uniform(50000, 250000, len(anomaly_dates)),
        'quantity_scrapped': np.random.randint(100, 500, len(anomaly_dates)),
        'daily_records': np.random.randint(10, 50, len(anomaly_dates)),
        'anomaly_score': np.random.uniform(0.7, 0.99, len(anomaly_dates)),
        'is_anomaly': True,
        'plant': np.random.choice(['Detroit Assembly', 'Toledo Powertrain', 'Louisville Stamping'], len(anomaly_dates)),
        'anomaly_reason': np.random.choice([
            'Unusually high scrap volume',
            'Cost spike detected', 
            'Pattern deviation from baseline',
            'Equipment-related cluster',
            'Supplier quality issue detected'
        ], len(anomaly_dates))
    })


def get_mock_root_cause_data() -> pd.DataFrame:
    """Generate mock root cause validation data for demo."""
    import numpy as np
    np.random.seed(456)
    
    n = 50
    defects = ['Dimensional', 'Surface Finish', 'Material', 'Assembly', 'Cosmetic']
    root_causes = ['Tooling Wear', 'Process Drift', 'Material Variation', 'Operator Error', 'Equipment Failure']
    
    actual = np.random.choice(root_causes, n)
    predicted = actual.copy()
    # Introduce some disagreements
    disagree_idx = np.random.choice(n, size=10, replace=False)
    for idx in disagree_idx:
        predicted[idx] = np.random.choice([rc for rc in root_causes if rc != actual[idx]])
    
    prediction_correct = actual == predicted
    confidence = np.where(prediction_correct, np.random.uniform(0.8, 0.98, n), np.random.uniform(0.5, 0.9, n))
    
    return pd.DataFrame({
        'plant_id': np.random.randint(1, 7, n),
        'analysis_date': pd.date_range(end=pd.Timestamp.now(), periods=n, freq='D'),
        'defect_category': np.random.choice(defects, n),
        'actual_root_cause': actual,
        'predicted_root_cause': predicted,
        'prediction_correct': prediction_correct,
        'prediction_confidence': confidence,
        'total_scrap_cost': np.random.uniform(5000, 25000, n),
        'quantity_scrapped': np.random.randint(10, 100, n),
        'disagreement_flag': ~prediction_correct,
        'high_confidence_disagreement': (~prediction_correct) & (confidence > 0.85)
    })


# =============================================================================
# Chart Functions
# =============================================================================

def plot_forecast(df: pd.DataFrame) -> go.Figure:
    """Create forecast visualization."""
    fig = go.Figure()
    
    # Confidence interval
    fig.add_trace(go.Scatter(
        x=pd.concat([df['forecast_date'], df['forecast_date'][::-1]]),
        y=pd.concat([df['upper_bound'], df['lower_bound'][::-1]]),
        fill='toself',
        fillcolor='rgba(255, 54, 33, 0.1)',
        line=dict(color='rgba(0,0,0,0)'),
        name='95% Confidence Interval',
        showlegend=True
    ))
    
    # Prediction line
    fig.add_trace(go.Scatter(
        x=df['forecast_date'],
        y=df['predicted_cost'],
        mode='lines',
        name='Predicted Scrap Cost',
        line=dict(color='#FF3621', width=2)
    ))
    
    # Today marker - use add_shape to avoid annotation issues with Timestamp
    today = pd.Timestamp.now()
    fig.add_shape(
        type="line",
        x0=today, x1=today,
        y0=0, y1=1,
        yref="paper",
        line=dict(color="#1B3139", width=2, dash="dash")
    )
    fig.add_annotation(
        x=today,
        y=1,
        yref="paper",
        text="Today",
        showarrow=False,
        font=dict(color="#1B3139", size=11),
        yshift=10
    )
    
    # Apply theme first, then override specific settings
    theme = create_plotly_theme()
    fig.update_layout(
        paper_bgcolor=theme.get('paper_bgcolor'),
        plot_bgcolor=theme.get('plot_bgcolor'),
        font=theme.get('font'),
        colorway=theme.get('colorway'),
        height=350,
        margin=dict(l=80, r=20, t=40, b=60),
        xaxis_title="Date",
        yaxis_title="Daily Scrap Cost ($)",
        xaxis=dict(
            tickformat='%b %d',
            showticklabels=True,
            tickfont=dict(size=11, color='#4a4a48'),
            title_font=dict(size=12, color='#1B3139')
        ),
        yaxis=dict(
            tickformat='$,.0f',
            showticklabels=True,
            tickfont=dict(size=11, color='#4a4a48'),
            title_font=dict(size=12, color='#1B3139')
        ),
        hovermode='x unified',
        legend=dict(orientation='h', y=1.1, font=dict(color='#4a4a48'))
    )
    
    return fig


def plot_supplier_risk_pie(df: pd.DataFrame) -> go.Figure:
    """Create supplier risk pie chart."""
    color_map = {'Low': '#10b981', 'Medium': '#f59e0b', 'High': '#f97316', 'Critical': '#ef4444'}
    
    # Count by risk category
    risk_counts = df['risk_category'].value_counts()
    
    # Ensure order
    categories = ['Low', 'Medium', 'High', 'Critical']
    values = [risk_counts.get(cat, 0) for cat in categories]
    colors = [color_map[cat] for cat in categories]
    
    fig = go.Figure(data=[go.Pie(
        labels=categories,
        values=values,
        marker_colors=colors,
        hole=0.4,
        textinfo='label+value',
        textposition='outside',
        textfont=dict(size=12),
        hovertemplate='<b>%{label}</b><br>%{value} suppliers (%{percent})<extra></extra>'
    )])
    
    theme = create_plotly_theme()
    fig.update_layout(
        paper_bgcolor=theme.get('paper_bgcolor'),
        plot_bgcolor=theme.get('plot_bgcolor'),
        font=theme.get('font'),
        height=300,
        margin=dict(l=20, r=20, t=20, b=20),
        showlegend=False,
        annotations=[dict(
            text='Risk<br>Distribution',
            x=0.5, y=0.5,
            font_size=14,
            font_color='#1B3139',
            showarrow=False
        )]
    )
    
    return fig


# =============================================================================
# Main Page Render
# =============================================================================

def render_dashboard_page():
    """Render the Executive Dashboard page with tabs."""
    
    # Create tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview", 
        "📈 Forecasting", 
        "🏢 Supplier Risk",
        "⚠️ Anomalies",
        "🔍 Root Cause"
    ])
    
    # =========================================================================
    # TAB 1: OVERVIEW (Embedded Dashboard)
    # =========================================================================
    with tab1:
        # Load KPIs
        try:
            kpis = get_live_kpis()
        except:
            kpis = get_mock_kpis()
        
        # Compact KPI row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "YTD Scrap Cost",
                format_number(kpis['ytd_scrap_cost'], prefix="$"),
                f"{kpis.get('ytd_scrap_delta', 0):.1f}%"
            )
        
        with col2:
            st.metric(
                "Units Scrapped",
                format_number(kpis['ytd_units_scrapped']),
                f"{kpis.get('units_delta', 0):.1f}%"
            )
        
        with col3:
            st.metric(
                "Scrap Incidents",
                format_number(kpis.get('scrap_incidents', 0)),
                f"{kpis.get('incidents_delta', 0):.1f}%"
            )
        
        with col4:
            st.metric(
                "Open 8D Reports",
                format_number(kpis.get('open_8d_reports', 0)),
                f"{kpis.get('eight_d_delta', 0):.1f}%"
            )
        
        # Embedded Databricks Dashboard
        st.markdown(f"""
        <div style="
            border-radius: 8px; 
            overflow: hidden; 
            border: 1px solid #e0dfdc; 
            background: white;
            margin-top: 16px;
        ">
            <iframe
                src="{databricks_config.dashboard_embed_url}"
                width="100%"
                height="600"
                frameborder="0"
                style="border: none; background: white;">
            </iframe>
        </div>
        """, unsafe_allow_html=True)
    
    # =========================================================================
    # TAB 2: FORECASTING
    # =========================================================================
    with tab2:
        st.markdown("**Prophet-based time series models predict scrap costs 3 months ahead.**")
        
        # Load data
        if should_use_mock_data():
            forecast_df = get_mock_forecast_data()
            data_source = "🟡 Demo Data"
        else:
            forecast_df = load_forecast_results()
            if forecast_df is None or len(forecast_df) == 0:
                forecast_df = get_mock_forecast_data()
                data_source = "🟡 Demo Data (DB unavailable)"
            else:
                data_source = "🟢 Live Data"
        
        st.caption(data_source)
        
        # Ensure forecast_date is datetime
        forecast_df['forecast_date'] = pd.to_datetime(forecast_df['forecast_date'])
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        future_data = forecast_df[forecast_df['forecast_date'] > pd.Timestamp.now()]
        if len(future_data) > 0:
            with col1:
                next_month = future_data.head(30)['predicted_cost'].sum()
                st.metric("Next 30 Days", f"${next_month:,.0f}")
            
            with col2:
                next_quarter = future_data.head(90)['predicted_cost'].sum()
                st.metric("Next 90 Days", f"${next_quarter:,.0f}")
            
            with col3:
                avg_daily = future_data['predicted_cost'].mean()
                st.metric("Avg Daily Forecast", f"${avg_daily:,.0f}")
            
            with col4:
                uncertainty = (future_data['upper_bound'] - future_data['lower_bound']).mean()
                st.metric("Uncertainty Range", f"±${uncertainty/2:,.0f}")
        
        # Forecast chart
        st.plotly_chart(plot_forecast(forecast_df), use_container_width=True, key="forecast_chart")
    
    # =========================================================================
    # TAB 3: SUPPLIER RISK
    # =========================================================================
    with tab3:
        st.markdown("**Multi-factor risk assessment combining quality, delivery, and cost metrics.**")
        
        # Load data
        if should_use_mock_data():
            risk_df = get_mock_supplier_risk_data()
            data_source = "🟡 Demo Data"
        else:
            risk_df = load_supplier_risk()
            if risk_df is None or len(risk_df) == 0:
                risk_df = get_mock_supplier_risk_data()
                data_source = "🟡 Demo Data (DB unavailable)"
            else:
                data_source = "🟢 Live Data"
        
        st.caption(data_source)
        
        # Risk distribution metrics
        col1, col2, col3, col4 = st.columns(4)
        risk_counts = risk_df['risk_category'].value_counts()
        
        with col1:
            st.metric("🟢 Low Risk", risk_counts.get('Low', 0))
        with col2:
            st.metric("🟡 Medium Risk", risk_counts.get('Medium', 0))
        with col3:
            st.metric("🟠 High Risk", risk_counts.get('High', 0))
        with col4:
            st.metric("🔴 Critical", risk_counts.get('Critical', 0))
        
        # Pie chart and drill-down side by side
        chart_col, detail_col = st.columns([1, 2])
        
        with chart_col:
            st.markdown("##### Risk Distribution")
            st.plotly_chart(plot_supplier_risk_pie(risk_df), use_container_width=True, key="risk_pie_chart")
        
        with detail_col:
            st.markdown("##### Drill Down by Category")
            
            # Category selector
            selected_category = st.selectbox(
                "Select risk level to view suppliers:",
                ["All", "Critical", "High", "Medium", "Low"],
                key="risk_category_filter"
            )
            
            # Filter data
            if selected_category == "All":
                filtered_df = risk_df
            else:
                filtered_df = risk_df[risk_df['risk_category'] == selected_category]
            
            # Show filtered suppliers
            if len(filtered_df) > 0:
                display_df = filtered_df[['supplier_name', 'tier', 'quality_score', 'delivery_score', 
                                         'risk_category', 'recommendation']].copy()
                display_df['quality_score'] = display_df['quality_score'].apply(lambda x: f"{x:.1%}")
                display_df['delivery_score'] = display_df['delivery_score'].apply(lambda x: f"{x:.1%}")
                display_df.columns = ['Supplier', 'Tier', 'Quality', 'Delivery', 'Risk', 'Action']
                st.dataframe(display_df, use_container_width=True, hide_index=True, height=250)
            else:
                st.info(f"No {selected_category.lower()} risk suppliers found.")
    
    # =========================================================================
    # TAB 4: ANOMALY DETECTION
    # =========================================================================
    with tab4:
        st.markdown("**Isolation Forest algorithm detects unusual scrap patterns requiring attention.**")
        
        # Load data
        if should_use_mock_data():
            anomaly_df = get_mock_anomaly_data()
            data_source = "🟡 Demo Data"
        else:
            anomaly_df = load_anomalies()
            if anomaly_df is None or len(anomaly_df) == 0:
                anomaly_df = get_mock_anomaly_data()
                data_source = "🟡 Demo Data (DB unavailable)"
            else:
                data_source = "🟢 Live Data"
        
        st.caption(data_source)
        
        # Summary metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🚨 Anomalies Detected", len(anomaly_df))
        with col2:
            total_cost = anomaly_df['total_scrap_cost'].sum()
            st.metric("💰 Total Cost Impact", f"${total_cost:,.0f}")
        with col3:
            avg_score = anomaly_df['anomaly_score'].mean()
            st.metric("📊 Avg Anomaly Score", f"{avg_score:.2f}")
        
        # Anomaly timeline chart
        st.markdown("##### Anomaly Timeline")
        st.caption("Scrap volume on days with detected anomalies")
        
        anomaly_df['transaction_date'] = pd.to_datetime(anomaly_df['transaction_date'])
        anomaly_df_sorted = anomaly_df.sort_values('transaction_date')
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=anomaly_df_sorted['transaction_date'].tolist(),
            y=anomaly_df_sorted['total_scrap_cost'].tolist(),
            mode='markers+lines',
            marker=dict(
                size=12,
                color='#FF3621',
                opacity=0.7
            ),
            line=dict(color='#FF3621', width=1),
            hovertemplate='<b>%{x|%Y-%m-%d}</b><br>Volume: $%{y:,.0f}<extra></extra>'
        ))
        
        theme = create_plotly_theme()
        fig.update_layout(
            paper_bgcolor=theme.get('paper_bgcolor'),
            plot_bgcolor=theme.get('plot_bgcolor'),
            font=theme.get('font'),
            height=300,
            margin=dict(l=80, r=20, t=20, b=60),
            xaxis_title="Date",
            yaxis_title="Volume ($)",
            xaxis=dict(
                tickformat='%b %d', 
                showticklabels=True,
                tickfont=dict(size=11, color='#4a4a48'),
                title_font=dict(size=12, color='#1B3139')
            ),
            yaxis=dict(
                tickformat='$,.0f', 
                showticklabels=True,
                tickfont=dict(size=11, color='#4a4a48'),
                title_font=dict(size=12, color='#1B3139')
            ),
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True, key="anomaly_chart")
        
        # Anomaly table
        with st.expander("📋 Anomaly Details"):
            display_df = anomaly_df[['transaction_date', 'plant', 'total_scrap_cost', 
                                     'quantity_scrapped', 'anomaly_score', 'anomaly_reason']].copy()
            # Format date to be readable (just date, no time)
            display_df['transaction_date'] = pd.to_datetime(display_df['transaction_date']).dt.strftime('%Y-%m-%d')
            display_df['total_scrap_cost'] = display_df['total_scrap_cost'].apply(lambda x: f"${x:,.0f}")
            display_df['anomaly_score'] = display_df['anomaly_score'].apply(lambda x: f"{x:.2f}")
            display_df.columns = ['Date', 'Plant', 'Scrap Cost', 'Qty', 'Score', 'Reason']
            st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    # =========================================================================
    # TAB 5: ROOT CAUSE VALIDATION
    # =========================================================================
    with tab5:
        st.markdown("**ML validates Quality team's root cause assignments and flags potential misclassifications.**")
        
        # Load data
        if should_use_mock_data():
            rc_df = get_mock_root_cause_data()
            data_source = "🟡 Demo Data"
        else:
            rc_df = load_root_cause_validation()
            if rc_df is None or len(rc_df) == 0:
                rc_df = get_mock_root_cause_data()
                data_source = "🟡 Demo Data (DB unavailable)"
            else:
                data_source = "🟢 Live Data"
        
        st.caption(data_source)
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        accuracy = rc_df['prediction_correct'].mean()
        disagreements = rc_df['disagreement_flag'].sum()
        high_conf_disagree = rc_df['high_confidence_disagreement'].sum()
        cost_at_risk = rc_df[rc_df['high_confidence_disagreement']]['total_scrap_cost'].sum()
        
        with col1:
            st.metric("✅ Model Accuracy", f"{accuracy:.1%}")
        with col2:
            st.metric("⚠️ Total Disagreements", int(disagreements))
        with col3:
            st.metric("🔍 Cases to Review", int(high_conf_disagree), help="High-confidence model predictions that disagree with manual assignments")
        with col4:
            st.metric("💰 Cost at Risk", f"${cost_at_risk:,.0f}")
        
        # Disagreement breakdown by defect category
        st.markdown("##### Disagreements by Defect Category")
        st.caption("Where ML predictions differ from manual Quality team assignments")
        
        if 'defect_category' in rc_df.columns:
            disagree_by_defect = rc_df.groupby('defect_category').agg({
                'disagreement_flag': 'sum',
                'total_scrap_cost': 'sum'
            }).reset_index()
            
            fig = go.Figure(go.Bar(
                x=disagree_by_defect['defect_category'].tolist(),
                y=disagree_by_defect['disagreement_flag'].tolist(),
                marker_color='#FF3621',
                text=disagree_by_defect['disagreement_flag'].tolist(),
                textposition='outside'
            ))
            
            theme = create_plotly_theme()
            fig.update_layout(
                paper_bgcolor=theme.get('paper_bgcolor'),
                plot_bgcolor=theme.get('plot_bgcolor'),
                font=theme.get('font'),
                height=320,
                margin=dict(l=60, r=20, t=20, b=100),
                xaxis_title="Defect Category",
                yaxis_title="Count",
                xaxis=dict(
                    showticklabels=True, 
                    tickangle=-30,
                    tickfont=dict(size=11, color='#4a4a48'),
                    title_font=dict(size=12, color='#1B3139')
                ),
                yaxis=dict(
                    showticklabels=True,
                    tickfont=dict(size=11, color='#4a4a48'),
                    title_font=dict(size=12, color='#1B3139'),
                    dtick=1  # Show integer ticks
                ),
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True, key="root_cause_chart")
        
        # Cases to review
        with st.expander("🔍 Cases to Review (High-Confidence Disagreements)"):
            review_df = rc_df[rc_df['high_confidence_disagreement']].copy()
            if len(review_df) > 0:
                display_cols = ['defect_category', 'actual_root_cause', 'predicted_root_cause', 
                               'prediction_confidence', 'total_scrap_cost']
                review_display = review_df[display_cols].copy()
                review_display['prediction_confidence'] = review_display['prediction_confidence'].apply(lambda x: f"{x:.1%}")
                review_display['total_scrap_cost'] = review_display['total_scrap_cost'].apply(lambda x: f"${x:,.0f}")
                review_display.columns = ['Defect', 'Actual Root Cause', 'ML Prediction', 'Confidence', 'Cost']
                st.dataframe(review_display, use_container_width=True, hide_index=True)
            else:
                st.info("No high-confidence disagreements found.")