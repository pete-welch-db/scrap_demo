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

from .dashboard import render_dashboard_page
from .scenarios import render_scenarios_page
from .assistant import render_assistant_page
# from .ml_insights import render_ml_insights_page  # Commented out - moved to Dashboard tabs

__all__ = [
    "render_dashboard_page",
    "render_scenarios_page",
    # "render_ml_insights_page",  # Commented out - moved to Dashboard tabs
    "render_assistant_page"
]

