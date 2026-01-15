"""
Application Views
=================
Individual view modules for each section of the app.

Views:
- dashboard: Executive KPIs and embedded Databricks dashboard
- scenarios: What-if scenario analysis for scrap reduction
- ml_insights: ML model results (forecasting, anomaly detection, etc.)
- assistant: AI-powered Genie assistant integration
"""

from .landing import render_landing_page
from .dashboard import render_dashboard_page
from .scenarios import render_scenarios_page
# from .ml_insights import render_ml_insights_page  # Commented out - moved to Scrap Insights tabs
from .assistant import render_assistant_page

__all__ = [
    "render_landing_page",
    "render_dashboard_page",
    "render_scenarios_page",
    # "render_ml_insights_page",  # Commented out - moved to Scrap Insights tabs
    "render_assistant_page"
]

