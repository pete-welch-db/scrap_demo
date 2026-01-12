"""
Components Package
==================
Reusable UI components for the Scrap Reduction Intelligence app.

INTEGRATION NOTE:
- Base: scrap_demo-main (includes render_header)
- ML Branch: Scrap (no render_header)
- Using main's version with header component
"""

from .styles import apply_custom_styles, render_header
from .sidebar import render_sidebar

__all__ = [
    "apply_custom_styles",
    "render_header",
    "render_sidebar"
]
