"""
Views Package
=============
Page rendering functions for the Scrap Reduction Intelligence app.

INTEGRATION NOTE:
- Base: scrap_demo-main (includes landing page, no separate ML insights)
- ML Branch: Scrap (no landing, has separate ML insights)
- Using main's version: Landing page included, ML insights kept for backward compatibility
"""

from .landing import render_landing_page
from .dashboard import render_dashboard_page
from .scenarios import render_scenarios_page
from .assistant import render_assistant_page
from .ml_insights import render_ml_insights_page  # Kept for compatibility, but integrated into dashboard

__all__ = [
    "render_landing_page",
    "render_dashboard_page",
    "render_scenarios_page",
    "render_assistant_page",
    "render_ml_insights_page"  # Available if needed separately
]
