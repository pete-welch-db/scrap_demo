"""
Data Layer
==========
Database connections, queries, and data management.
"""

from .connection import get_databricks_connection, test_db_connection, run_query
from .mock_data import load_mock_data
from .genie_client import GenieClient, get_genie_client

__all__ = [
    "get_databricks_connection",
    "test_db_connection", 
    "run_query",
    "load_mock_data",
    "GenieClient",
    "get_genie_client"
]

