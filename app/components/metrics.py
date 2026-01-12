"""
Metric Components
=================
Reusable metric cards and chart theming.
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
    Return consistent Plotly theme configuration for dark mode.
    
    Returns:
        Dictionary of Plotly layout settings
    """
    return {
        'paper_bgcolor': 'rgba(0,0,0,0)',
        'plot_bgcolor': 'rgba(0,0,0,0)',
        'font': {'color': '#94a3b8', 'family': 'Inter'},
        'xaxis': {
            'gridcolor': 'rgba(45, 45, 68, 0.5)',
            'linecolor': 'rgba(45, 45, 68, 0.5)'
        },
        'yaxis': {
            'gridcolor': 'rgba(45, 45, 68, 0.5)',
            'linecolor': 'rgba(45, 45, 68, 0.5)'
        }
    }

