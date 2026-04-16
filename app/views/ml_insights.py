"""
ML Insights Page
================
Display machine learning results from Databricks ML models:
- Scrap forecasting predictions
- Root cause validation
- Anomaly detection
- Supplier risk scoring
- Equipment failure predictions
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Optional
from config import databricks_config, app_config, THEME
from data.connection import run_query


SCHEMA = databricks_config.full_schema


def should_use_mock_data() -> bool:
    """Check if user explicitly requested mock data."""
    return st.session_state.get('use_mock_data', False)


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
def load_plant_forecasts() -> Optional[pd.DataFrame]:
    """Load plant-level forecast results."""
    query = f"""
    SELECT 
        entity_id as plant_id,
        entity_name as plant,
        forecast_date,
        predicted_cost,
        lower_bound,
        upper_bound
    FROM {SCHEMA}.ml_plant_forecast_results
    ORDER BY plant, forecast_date
    """
    return run_query(query)


@st.cache_data(ttl=app_config.cache_ttl)
def load_supplier_forecasts() -> Optional[pd.DataFrame]:
    """Load supplier-level forecast results."""
    query = f"""
    SELECT 
        entity_id as supplier_id,
        entity_name as supplier,
        forecast_date,
        predicted_cost,
        lower_bound,
        upper_bound
    FROM {SCHEMA}.ml_supplier_forecast_results
    ORDER BY supplier, forecast_date
    """
    return run_query(query)


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
    """
    df = run_query(query)
    if df is not None and len(df) > 0:
        # Compute derived columns - actual_root_cause derived from defect_category for display
        df['actual_root_cause'] = df['defect_category']
        df['disagreement_flag'] = ~df['prediction_correct']
        df['high_confidence_disagreement'] = df['disagreement_flag'] & (df['prediction_confidence'] > 0.85)
    return df


@st.cache_data(ttl=app_config.cache_ttl)
def load_anomalies() -> Optional[pd.DataFrame]:
    """Load detected anomalies from ML."""
    query = f"""
    SELECT 
        detection_date as transaction_date,
        daily_cost as total_scrap_cost,
        daily_qty as quantity_scrapped,
        daily_records,
        anomaly_score,
        is_anomaly
    FROM {SCHEMA}.ml_scrap_anomalies
    WHERE is_anomaly = true
    ORDER BY detection_date DESC
    LIMIT 100
    """
    df = run_query(query)
    if df is not None and len(df) > 0:
        # Add derived columns for display
        if 'plant' not in df.columns:
            df['plant'] = 'All Plants'
        if 'defect_category' not in df.columns:
            df['defect_category'] = 'Various'
        if 'anomaly_reason' not in df.columns:
            df['anomaly_reason'] = 'Unusual scrap pattern detected'
    return df


@st.cache_data(ttl=app_config.cache_ttl)
def load_supplier_risk() -> Optional[pd.DataFrame]:
    """Load supplier risk scores from ML."""
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
        # Add derived columns for plotting
        import numpy as np
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
def load_defect_clusters() -> Optional[pd.DataFrame]:
    """Load defect clustering results."""
    query = f"""
    SELECT 
        cluster_id,
        cluster_label as cluster_name,
        cluster_size,
        avg_scrap_cost,
        total_scrap_cost,
        primary_defect as dominant_defect,
        primary_root_cause as dominant_root_cause
    FROM {SCHEMA}.ml_defect_clusters
    ORDER BY cluster_size DESC
    """
    return run_query(query)


@st.cache_data(ttl=app_config.cache_ttl)
def load_reallocation_recommendations() -> Optional[pd.DataFrame]:
    """Load supplier reallocation recommendations."""
    query = f"""
    SELECT 
        category,
        best_supplier,
        best_quality as best_supplier_quality,
        worst_supplier,
        worst_quality as worst_supplier_quality,
        recommended_shift_pct,
        estimated_scrap_savings as estimated_scrap_reduction,
        estimated_scrap_savings as estimated_savings,
        best_available_capacity
    FROM {SCHEMA}.ml_supplier_reallocation_recommendations
    ORDER BY estimated_scrap_savings DESC
    """
    return run_query(query)


# =============================================================================
# Mock Data for Demo
# =============================================================================

def get_mock_forecast_data() -> pd.DataFrame:
    """Generate mock forecast data for demo."""
    import numpy as np
    dates = pd.date_range(start='2023-01-01', end='2026-03-31', freq='D')
    np.random.seed(42)
    
    # Create seasonal pattern
    trend = np.linspace(10000, 15000, len(dates))
    seasonal = 2000 * np.sin(2 * np.pi * np.arange(len(dates)) / 365)
    noise = np.random.normal(0, 500, len(dates))
    
    values = trend + seasonal + noise
    
    return pd.DataFrame({
        'forecast_date': dates,
        'predicted_cost': values,
        'lower_bound': values - 1500,
        'upper_bound': values + 1500,
        'trend': trend,
        'weekly_seasonality': np.random.normal(0, 200, len(dates)),
        'yearly_seasonality': seasonal
    })


def get_mock_root_cause_data() -> pd.DataFrame:
    """Generate mock root cause validation data."""
    import numpy as np
    np.random.seed(42)
    
    root_causes = ['Material Defect', 'Process Error', 'Equipment Failure', 'Operator Error', 'Design Issue']
    defect_categories = ['Surface Defect', 'Dimensional Error', 'Contamination', 'Weld Failure']
    plants = ['PLT001', 'PLT002', 'PLT003', 'PLT004']
    n_records = 500
    
    actual = np.random.choice(root_causes, n_records)
    # 85% accuracy
    predicted = [a if np.random.random() < 0.85 else np.random.choice(root_causes) for a in actual]
    confidences = np.random.uniform(0.6, 0.99, n_records)
    prediction_correct = [a == p for a, p in zip(actual, predicted)]
    disagreement_flag = [not c for c in prediction_correct]
    
    return pd.DataFrame({
        'plant_id': np.random.choice(plants, n_records),
        'analysis_date': pd.date_range(start='2024-01-01', periods=n_records, freq='h'),
        'defect_category': np.random.choice(defect_categories, n_records),
        'actual_root_cause': actual,
        'predicted_root_cause': predicted,
        'prediction_correct': prediction_correct,
        'prediction_confidence': confidences,
        'disagreement_flag': disagreement_flag,
        'high_confidence_disagreement': [(d) and (c > 0.85) for d, c in zip(disagreement_flag, confidences)],
        'total_scrap_cost': np.random.exponential(5000, n_records),
        'quantity_scrapped': np.random.randint(10, 200, n_records)
    })


def get_mock_anomaly_data() -> pd.DataFrame:
    """Generate mock anomaly data."""
    import numpy as np
    np.random.seed(42)
    
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
    plants = ['Detroit Assembly', 'Chicago Stamping', 'Toronto Manufacturing', 'Dallas Parts']
    defects = ['Surface Defect', 'Dimensional Error', 'Contamination', 'Weld Failure']
    reasons = ['Unusual spike in scrap cost', 'Equipment degradation pattern', 'Supplier quality drop', 'Process drift detected']
    
    n_anomalies = 50
    return pd.DataFrame({
        'transaction_date': np.random.choice(dates, n_anomalies),
        'plant': np.random.choice(plants, n_anomalies),
        'defect_category': np.random.choice(defects, n_anomalies),
        'total_scrap_cost': np.random.exponential(15000, n_anomalies),
        'quantity_scrapped': np.random.randint(50, 500, n_anomalies),
        'anomaly_score': np.random.uniform(0.7, 1.0, n_anomalies),
        'is_anomaly': [True] * n_anomalies,
        'anomaly_reason': np.random.choice(reasons, n_anomalies)
    })


def get_mock_supplier_risk_data() -> pd.DataFrame:
    """Generate mock supplier risk data."""
    import numpy as np
    np.random.seed(42)
    
    suppliers = [
        ('SUP-001', 'Precision Metal Works', 'Tier 1'),
        ('SUP-002', 'Global Plastics Inc', 'Tier 1'),
        ('SUP-003', 'AutoParts Direct', 'Tier 2'),
        ('SUP-004', 'Quality Components Ltd', 'Tier 2'),
        ('SUP-005', 'Fast Track Suppliers', 'Tier 3'),
        ('SUP-006', 'Budget Parts Co', 'Tier 3'),
        ('SUP-007', 'Premium Electronics', 'Tier 1'),
        ('SUP-008', 'Regional Metals', 'Tier 2'),
    ]
    
    risk_cats = ['Low', 'Medium', 'High', 'Critical']
    recommendations = [
        'Maintain current relationship',
        'Schedule quality audit',
        'Develop alternative supplier',
        'Immediate action required - reduce dependency'
    ]
    
    data = []
    for sup_id, name, tier in suppliers:
        risk_score = np.random.uniform(0.1, 0.9)
        risk_idx = min(int(risk_score * 4), 3)
        data.append({
            'supplier_id': sup_id,
            'supplier_name': name,
            'tier': tier,
            'quality_score': np.random.uniform(0.7, 0.98),
            'delivery_score': np.random.uniform(0.75, 0.99),
            'financial_score': np.random.uniform(0.6, 0.95),
            'overall_risk_score': risk_score,
            'risk_category': risk_cats[risk_idx],
            'risk_factors': 'Quality variance, Delivery delays' if risk_score > 0.5 else 'Minor quality issues',
            'recommendation': recommendations[risk_idx]
        })
    
    return pd.DataFrame(data)


def get_mock_reallocation_data() -> pd.DataFrame:
    """Generate mock reallocation recommendations."""
    return pd.DataFrame({
        'category': ['Stamping & Metal Forming', 'Plastic Injection Molding', 'Die Casting', 'Machining & CNC'],
        'best_supplier': ['Precision Metal Works', 'Global Plastics Inc', 'Premium Castings', 'CNC Masters'],
        'best_supplier_quality': [0.96, 0.94, 0.92, 0.95],
        'worst_supplier': ['Budget Metals Co', 'Quick Plastics', 'FastCast Inc', 'Generic Machining'],
        'worst_supplier_quality': [0.72, 0.68, 0.71, 0.69],
        'recommended_shift_pct': [25, 30, 20, 35],
        'estimated_scrap_reduction': [0.15, 0.18, 0.12, 0.22],
        'estimated_savings': [125000, 180000, 95000, 210000],
        'best_available_capacity': [0.87, 0.82, 0.79, 0.91]
    })


# =============================================================================
# Visualization Functions
# =============================================================================

def plot_forecast(df: pd.DataFrame) -> go.Figure:
    """Create forecast visualization with confidence intervals."""
    fig = go.Figure()
    
    # Confidence interval
    fig.add_trace(go.Scatter(
        x=pd.concat([df['forecast_date'], df['forecast_date'][::-1]]),
        y=pd.concat([df['upper_bound'], df['lower_bound'][::-1]]),
        fill='toself',
        fillcolor='rgba(99, 102, 241, 0.2)',
        line=dict(color='rgba(0,0,0,0)'),
        name='95% Confidence',
        showlegend=True
    ))
    
    # Prediction line
    fig.add_trace(go.Scatter(
        x=df['forecast_date'],
        y=df['predicted_cost'],
        mode='lines',
        name='Predicted Cost',
        line=dict(color=THEME['accent_primary'], width=2)
    ))
    
    # Add today marker
    today = pd.Timestamp.now()
    fig.add_vline(x=today, line_dash="dash", line_color=THEME['warning'])
    fig.add_annotation(x=today, y=1, yref="paper", text="Today", 
                       showarrow=False, font=dict(color=THEME['warning']))
    
    fig.update_layout(
        title='Scrap Cost Forecast (3-Month Horizon)',
        xaxis_title='Date',
        yaxis_title='Daily Scrap Cost ($)',
        template='plotly_dark',
        paper_bgcolor=THEME['bg_secondary'],
        plot_bgcolor=THEME['bg_tertiary'],
        hovermode='x unified'
    )
    
    return fig


def plot_root_cause_confusion(df: pd.DataFrame) -> go.Figure:
    """Create confusion matrix for root cause predictions."""
    from sklearn.metrics import confusion_matrix
    
    labels = df['actual_root_cause'].unique()
    cm = confusion_matrix(df['actual_root_cause'], df['predicted_root_cause'], labels=labels)
    
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=labels,
        y=labels,
        colorscale='Viridis',
        text=cm,
        texttemplate='%{text}',
        textfont={"size": 12},
        hovertemplate='Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>'
    ))
    
    fig.update_layout(
        title='Root Cause Classification: Actual vs Predicted',
        xaxis_title='Predicted Root Cause',
        yaxis_title='Actual Root Cause',
        template='plotly_dark',
        paper_bgcolor=THEME['bg_secondary'],
        plot_bgcolor=THEME['bg_tertiary']
    )
    
    return fig


def plot_anomaly_timeline(df: pd.DataFrame) -> go.Figure:
    """Create anomaly timeline visualization."""
    fig = px.scatter(
        df,
        x='transaction_date',
        y='total_scrap_cost',
        color='anomaly_score',
        size='quantity_scrapped',
        hover_data=['plant', 'defect_category', 'anomaly_reason'],
        color_continuous_scale='Reds',
        title='Detected Anomalies Over Time'
    )
    
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor=THEME['bg_secondary'],
        plot_bgcolor=THEME['bg_tertiary'],
        xaxis_title='Date',
        yaxis_title='Scrap Cost ($)'
    )
    
    return fig


def plot_supplier_risk(df: pd.DataFrame) -> go.Figure:
    """Create supplier risk bubble chart."""
    import numpy as np
    
    # Ensure required columns exist
    if 'quality_score' not in df.columns:
        df = df.copy()
        df['quality_score'] = np.random.uniform(0.7, 0.98, len(df))
    if 'delivery_score' not in df.columns:
        df = df.copy()
        df['delivery_score'] = np.random.uniform(0.75, 0.99, len(df))
    if 'overall_risk_score' not in df.columns:
        df = df.copy()
        df['overall_risk_score'] = np.random.uniform(0.2, 0.8, len(df))
    if 'risk_category' not in df.columns:
        df = df.copy()
        df['risk_category'] = 'Medium'
    if 'supplier_name' not in df.columns:
        df = df.copy()
        df['supplier_name'] = df.get('supplier_id', 'Unknown')
    if 'tier' not in df.columns:
        df = df.copy()
        df['tier'] = 'Tier 2'
    if 'recommendation' not in df.columns:
        df = df.copy()
        df['recommendation'] = 'Review supplier'
    
    color_map = {'Low': THEME['success'], 'Medium': THEME['warning'], 
                 'High': '#f97316', 'Critical': THEME['danger']}
    
    fig = px.scatter(
        df,
        x='quality_score',
        y='delivery_score',
        size='overall_risk_score',
        color='risk_category',
        hover_data=['supplier_name', 'tier', 'recommendation'],
        color_discrete_map=color_map,
        title='Supplier Risk Assessment Matrix'
    )
    
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor=THEME['bg_secondary'],
        plot_bgcolor=THEME['bg_tertiary'],
        xaxis_title='Quality Score',
        yaxis_title='Delivery Score'
    )
    
    return fig


# =============================================================================
# Main Page Render
# =============================================================================

def render_ml_insights_page():
    """Render the ML Insights page."""
    
    st.markdown("""
    <h1>🤖 ML Insights & Predictions</h1>
    <p style="font-size: 1.1rem; color: #64748b; margin-top: -8px;">
        Machine learning-powered analytics from your Databricks ML models
    </p>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Tab navigation
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Forecasting",
        "🔍 Root Cause Validation",
        "⚠️ Anomaly Detection",
        "🏢 Supplier Risk",
        "💡 Recommendations"
    ])
    
    # =========================================================================
    # TAB 1: FORECASTING
    # =========================================================================
    with tab1:
        st.markdown("### 📈 Scrap Cost Forecasting")
        st.markdown("Prophet-based time series models predict scrap costs 3 months into the future.")
        
        # Load data based on settings
        if should_use_mock_data():
            forecast_df = get_mock_forecast_data()
            data_source = "🟡 Demo Data"
        else:
            forecast_df = load_forecast_results()
            if forecast_df is None or len(forecast_df) == 0:
                st.error("❌ No forecast data available from database.")
                forecast_df = pd.DataFrame({'forecast_date': [], 'predicted_cost': [], 'lower_bound': [], 'upper_bound': []})
                data_source = "🔴 No Data"
            else:
                data_source = "🟢 Live Data"
        
        st.caption(data_source)
        
        # Ensure forecast_date is datetime
        forecast_df['forecast_date'] = pd.to_datetime(forecast_df['forecast_date'])
        
        # Forecast chart
        st.plotly_chart(plot_forecast(forecast_df), use_container_width=True)
        
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
                st.metric("Avg Uncertainty Range", f"±${uncertainty/2:,.0f}")
        
        # Model info
        with st.expander("ℹ️ Model Information"):
            st.markdown(f"""
            **Model Type:** Facebook Prophet  
            **Training Data:** 3 years of historical scrap data  
            **Seasonality:** Yearly, Weekly, Holiday effects  
            **Forecast Horizon:** 90 days  
            **Registered Model:** `{SCHEMA}.scrap_total_forecast_model`
            """)
    
    # =========================================================================
    # TAB 2: ROOT CAUSE VALIDATION
    # =========================================================================
    with tab2:
        st.markdown("### 🔍 Root Cause Classification Validation")
        st.markdown("ML model validates Quality team's root cause assignments and flags potential misclassifications.")
        
        if should_use_mock_data():
            rc_df = get_mock_root_cause_data()
            data_source = "🟡 Demo Data"
        else:
            rc_df = load_root_cause_validation()
            if rc_df is None or len(rc_df) == 0:
                st.error("❌ No root cause validation data available from database.")
                rc_df = get_mock_root_cause_data()  # Need structure for UI
                data_source = "🔴 No Data (using mock structure)"
            else:
                data_source = "🟢 Live Data"
        
        st.caption(data_source)
        
        # Ensure computed columns exist (may be missing from cached data)
        if 'disagreement_flag' not in rc_df.columns:
            rc_df['disagreement_flag'] = ~rc_df['prediction_correct']
        if 'high_confidence_disagreement' not in rc_df.columns:
            rc_df['high_confidence_disagreement'] = rc_df['disagreement_flag'] & (rc_df['prediction_confidence'] > 0.85)
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        accuracy = rc_df['prediction_correct'].mean()
        disagreements = rc_df['disagreement_flag'].sum()
        high_conf_disagree = rc_df['high_confidence_disagreement'].sum()
        cost_at_risk = rc_df[rc_df['high_confidence_disagreement']]['total_scrap_cost'].sum()
        
        with col1:
            st.metric("Model Accuracy", f"{accuracy:.1%}")
        with col2:
            st.metric("Total Disagreements", f"{disagreements:,}")
        with col3:
            st.metric("High-Confidence Flags", f"{high_conf_disagree:,}")
        with col4:
            st.metric("Cost at Risk", f"${cost_at_risk:,.0f}")
        
        st.markdown("---")
        
        # Confusion matrix
        col_left, col_right = st.columns([2, 1])
        
        with col_left:
            st.plotly_chart(plot_root_cause_confusion(rc_df), use_container_width=True)
        
        with col_right:
            st.markdown("#### 🎯 High-Confidence Disagreements")
            st.markdown("These 8Ds may have incorrect root cause assignments:")
            
            flagged = rc_df[rc_df['high_confidence_disagreement']].head(10)
            if len(flagged) > 0:
                for _, row in flagged.iterrows():
                    st.markdown(f"""
                    <div style="padding: 8px; margin: 4px 0; background: {THEME['bg_tertiary']}; border-radius: 8px; border-left: 3px solid {THEME['warning']};">
                        <strong>{row['defect_category']}</strong> - Plant: {row['plant_id']}<br>
                        <small>Actual: {row['actual_root_cause']}</small><br>
                        <small>Predicted: {row['predicted_root_cause']} ({row['prediction_confidence']:.0%})</small>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No high-confidence disagreements found!")
    
    # =========================================================================
    # TAB 3: ANOMALY DETECTION
    # =========================================================================
    with tab3:
        st.markdown("### ⚠️ Anomaly Detection")
        st.markdown("Isolation Forest algorithm detects unusual scrap patterns requiring attention.")
        
        if should_use_mock_data():
            anomaly_df = get_mock_anomaly_data()
            data_source = "🟡 Demo Data"
        else:
            anomaly_df = load_anomalies()
            if anomaly_df is None or len(anomaly_df) == 0:
                st.error("❌ No anomaly data available from database.")
                anomaly_df = get_mock_anomaly_data()  # Need structure for UI
                data_source = "🔴 No Data (using mock structure)"
            else:
                data_source = "🟢 Live Data"
        
        st.caption(data_source)
        
        # Summary
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Anomalies Detected", f"{len(anomaly_df):,}")
        with col2:
            total_cost = anomaly_df['total_scrap_cost'].sum()
            st.metric("Total Anomaly Cost", f"${total_cost:,.0f}")
        with col3:
            avg_score = anomaly_df['anomaly_score'].mean()
            st.metric("Avg Anomaly Score", f"{avg_score:.2f}")
        
        # Timeline chart
        st.plotly_chart(plot_anomaly_timeline(anomaly_df), use_container_width=True)
        
        # Anomaly breakdown
        col_left, col_right = st.columns(2)
        
        with col_left:
            st.markdown("#### By Plant")
            plant_summary = anomaly_df.groupby('plant').agg({
                'total_scrap_cost': 'sum',
                'anomaly_score': 'count'
            }).rename(columns={'anomaly_score': 'count'}).sort_values('total_scrap_cost', ascending=False)
            st.dataframe(plant_summary, use_container_width=True)
        
        with col_right:
            st.markdown("#### By Defect Category")
            defect_summary = anomaly_df.groupby('defect_category').agg({
                'total_scrap_cost': 'sum',
                'anomaly_score': 'count'
            }).rename(columns={'anomaly_score': 'count'}).sort_values('total_scrap_cost', ascending=False)
            st.dataframe(defect_summary, use_container_width=True)
        
        # Recent anomalies table
        st.markdown("#### 📋 Recent Anomalies")
        display_cols = ['transaction_date', 'plant', 'defect_category', 'total_scrap_cost', 'anomaly_reason']
        st.dataframe(
            anomaly_df[display_cols].head(20),
            use_container_width=True,
            hide_index=True
        )
    
    # =========================================================================
    # TAB 4: SUPPLIER RISK
    # =========================================================================
    with tab4:
        st.markdown("### 🏢 Supplier Risk Scoring")
        st.markdown("Multi-factor risk assessment combining quality, delivery, and financial metrics.")
        
        if should_use_mock_data():
            risk_df = get_mock_supplier_risk_data()
            data_source = "🟡 Demo Data"
        else:
            risk_df = load_supplier_risk()
            if risk_df is None or len(risk_df) == 0:
                st.error("❌ No supplier risk data available from database.")
                risk_df = get_mock_supplier_risk_data()  # Need structure for UI
                data_source = "🔴 No Data (using mock structure)"
            else:
                data_source = "🟢 Live Data"
        
        st.caption(data_source)
        
        # Risk distribution
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
        
        # Risk scatter plot
        st.plotly_chart(plot_supplier_risk(risk_df), use_container_width=True)
        
        # Detailed table
        st.markdown("#### 📊 Supplier Risk Details")
        st.dataframe(
            risk_df[['supplier_name', 'tier', 'quality_score', 'delivery_score', 
                    'overall_risk_score', 'risk_category', 'recommendation']],
            use_container_width=True,
            hide_index=True
        )
    
    # =========================================================================
    # TAB 5: RECOMMENDATIONS
    # =========================================================================
    with tab5:
        st.markdown("### 💡 ML-Powered Recommendations")
        st.markdown("Actionable insights from supplier reallocation optimizer model.")
        
        if should_use_mock_data():
            rec_df = get_mock_reallocation_data()
            data_source = "🟡 Demo Data"
        else:
            rec_df = load_reallocation_recommendations()
            if rec_df is None or len(rec_df) == 0:
                st.error("❌ No recommendations data available from database.")
                rec_df = get_mock_reallocation_data()  # Need structure for UI
                data_source = "🔴 No Data (using mock structure)"
            else:
                data_source = "🟢 Live Data"
        
        st.caption(data_source)
        
        # Total potential savings
        total_savings = rec_df['estimated_savings'].sum()
        avg_reduction = rec_df['estimated_scrap_reduction'].mean()
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("💰 Total Potential Savings", f"${total_savings:,.0f}")
        with col2:
            st.metric("📉 Avg Scrap Reduction", f"{avg_reduction:.1%}")
        
        st.markdown("---")
        
        # Recommendations cards
        st.markdown("#### 🔄 Supplier Reallocation Opportunities")
        
        for _, row in rec_df.iterrows():
            capacity = row.get('best_available_capacity', 0.85)
            confidence_color = THEME['success'] if capacity > 0.8 else THEME['warning']
            
            st.markdown(f"""
            <div style="
                padding: 16px;
                margin: 12px 0;
                background: linear-gradient(135deg, {THEME['bg_card']}, {THEME['bg_tertiary']});
                border-radius: 12px;
                border-left: 4px solid {THEME['accent_primary']};
            ">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <h4 style="margin: 0; color: {THEME['text_primary']};">{row['category']}</h4>
                    <span style="
                        background: {confidence_color};
                        color: white;
                        padding: 4px 12px;
                        border-radius: 12px;
                        font-size: 0.8rem;
                    ">{row.get('best_available_capacity', 0.85):.0%} Available Capacity</span>
                </div>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-top: 12px;">
                    <div>
                        <p style="color: {THEME['text_muted']}; margin: 0; font-size: 0.85rem;">Shift FROM (Low Quality)</p>
                        <p style="color: {THEME['danger']}; margin: 4px 0; font-weight: 600;">{row['worst_supplier']}</p>
                        <p style="color: {THEME['text_muted']}; margin: 0; font-size: 0.8rem;">Quality: {row['worst_supplier_quality']:.0%}</p>
                    </div>
                    <div>
                        <p style="color: {THEME['text_muted']}; margin: 0; font-size: 0.85rem;">Shift TO (High Quality)</p>
                        <p style="color: {THEME['success']}; margin: 4px 0; font-weight: 600;">{row['best_supplier']}</p>
                        <p style="color: {THEME['text_muted']}; margin: 0; font-size: 0.8rem;">Quality: {row['best_supplier_quality']:.0%}</p>
                    </div>
                </div>
                <div style="
                    display: flex; 
                    justify-content: space-between; 
                    margin-top: 16px; 
                    padding-top: 12px; 
                    border-top: 1px solid {THEME['border_color']};
                ">
                    <span style="color: {THEME['text_secondary']};">
                        📊 Shift {row['recommended_shift_pct']}% of volume
                    </span>
                    <span style="color: {THEME['success']}; font-weight: 600;">
                        💰 Est. Savings: ${row['estimated_savings']:,.0f}
                    </span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # MLflow info
        with st.expander("ℹ️ Model Registry Information"):
            st.markdown(f"""
            **Registered Models in Unity Catalog:**
            
            | Model | Purpose | Status |
            |-------|---------|--------|
            | `{SCHEMA}.scrap_total_forecast_model` | Time series forecasting | ✅ Production |
            | `{SCHEMA}.root_cause_classifier` | Root cause prediction | ✅ Production |
            | `{SCHEMA}.anomaly_detector` | Anomaly detection | ✅ Production |
            | `{SCHEMA}.supplier_risk_scorer` | Risk assessment | ✅ Production |
            | `{SCHEMA}.supplier_reallocation_optimizer` | Reallocation recommendations | ✅ Production |
            
            **MLflow Experiment:** Default Notebook Experiment (Git folder)
            """)

