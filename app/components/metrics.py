"""
Metric Components
=================
Reusable metric cards and chart theming.
Styled with Databricks Oat Medium theme.
"""

from typing import Optional


def create_metric_card(
    value: str, 
    label: str, 
    delta: Optional[float] = None,
    delta_color: str = "normal"
) -> str:
    """
    Create a styled metric card HTML.
    
    Args:
        value: The main metric value to display
        label: Label describing the metric
        delta: Optional percentage change (positive or negative)
        delta_color: Color scheme for delta ("normal" uses auto-coloring)
    
    Returns:
        HTML string for the metric card
    """
    delta_html = ""
    if delta is not None:
        delta_class = "metric-delta-positive" if delta >= 0 else "metric-delta-negative"
        delta_symbol = "↑" if delta >= 0 else "↓"
        delta_html = f'<div class="{delta_class}">{delta_symbol} {abs(delta):.1f}%</div>'
    
    return f"""
    <div class="metric-card">
        <div class="metric-value">{value}</div>
        <div class="metric-label">{label}</div>
        {delta_html}
    </div>
    """


def create_plotly_theme() -> dict:
    """
    Return consistent Plotly theme configuration for Databricks Oat Medium theme.
    
    Returns:
        Dictionary of Plotly layout settings
    """
    return {
        'paper_bgcolor': 'rgba(255,255,255,0)',
        'plot_bgcolor': 'rgba(255,255,255,0)',
        'font': {
            'color': '#4a4a48', 
            'family': 'DM Sans, sans-serif'
        },
        'title': {
            'font': {
                'color': '#151513',
                'family': 'DM Sans, sans-serif',
                'size': 16
            }
        },
        'xaxis': {
            'gridcolor': 'rgba(224, 223, 220, 0.7)',
            'linecolor': '#e0dfdc',
            'tickfont': {'color': '#4a4a48'}
        },
        'yaxis': {
            'gridcolor': 'rgba(224, 223, 220, 0.7)',
            'linecolor': '#e0dfdc',
            'tickfont': {'color': '#4a4a48'}
        },
        'colorway': [
            '#FF3621',  # Lava 600 (primary)
            '#1B3139',  # Navy 800
            '#FF5F46',  # Lava 500
            '#0B2026',  # Navy 900
            '#10b981',  # Success green
            '#f59e0b',  # Warning orange
        ],
        'legend': {
            'font': {'color': '#4a4a48'},
            'bgcolor': 'rgba(255,255,255,0.8)'
        }
    }
