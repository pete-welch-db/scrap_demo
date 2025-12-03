"""
Application Views
=================
Individual view modules for each section of the app.
"""

from .dashboard import render_dashboard_page
from .scenarios import render_scenarios_page
from .assistant import render_assistant_page

__all__ = [
    "render_dashboard_page",
    "render_scenarios_page",
    "render_assistant_page"
]

