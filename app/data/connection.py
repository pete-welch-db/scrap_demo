"""
Database Connection
===================
Databricks SQL connection management.
"""

import streamlit as st
import pandas as pd
from typing import Optional
from config import databricks_config, app_config


def get_databricks_connection():
    """
    Create a connection to Databricks SQL Warehouse.

    Uses different authentication methods:
    - LOCAL: Personal Access Token
    - DBX: Service Principal (OAuth M2M with client_id/secret)

    Returns:
        Connection object or None if connection fails
    """
    print(f"[DEBUG] Connecting to Databricks ({app_config.environment} mode)")

    try:
        from databricks import sql
        from databricks.sdk.core import Config
        import os

        if app_config.is_local():
            # LOCAL: Use personal access token
            connection = sql.connect(
                server_hostname=databricks_config.host,
                http_path=databricks_config.http_path,
                access_token=databricks_config.token
            )
        else:
            # DBX: Use SDK Config credentials provider with OAuth M2M
            client_id = os.getenv("DATABRICKS_CLIENT_ID")
            client_secret = os.getenv("DATABRICKS_CLIENT_SECRET")

            if not client_id or not client_secret:
                raise ValueError("Service principal credentials not found. DATABRICKS_CLIENT_ID and DATABRICKS_CLIENT_SECRET must be set.")

            cfg = Config(host=databricks_config.host)
            connection = sql.connect(
                server_hostname=databricks_config.host,
                http_path=databricks_config.http_path,
                credentials_provider=lambda: cfg.authenticate
            )

        return connection

    except ImportError as e:
        import traceback
        st.warning(f"Required package not installed: {str(e)}. Using mock data.")
        print(f"[ERROR] ImportError: {str(e)}")
        traceback.print_exc()
        return None
    except Exception as e:
        import traceback
        st.error(f"Connection error: {str(e)}")
        print(f"[ERROR] Connection error: {str(e)}")
        traceback.print_exc()
        return None


def test_db_connection() -> bool:
    """
    Test database connection.
    
    Returns:
        True if connection successful, False otherwise
    """
    if not databricks_config.is_configured():
        return False
        
    try:
        conn = get_databricks_connection()
        if conn:
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            cursor.close()
            conn.close()
            return True
    except:
        pass
    return False


@st.cache_data(ttl=app_config.cache_ttl)
def run_query(query: str) -> Optional[pd.DataFrame]:
    """
    Execute a SQL query and return results as DataFrame.

    Args:
        query: SQL query string to execute

    Returns:
        DataFrame with query results or None if query fails
    """
    # print(f"[DEBUG] Executing query: {query[:100]}...")

    conn = get_databricks_connection()
    if conn is None:
        return None

    try:
        cursor = conn.cursor()
        cursor.execute(query)
        columns = [desc[0] for desc in cursor.description]
        data = cursor.fetchall()
        cursor.close()
        conn.close()

        return pd.DataFrame(data, columns=columns)

    except Exception as e:
        import traceback
        st.error(f"Query error: {str(e)}")
        print(f"[ERROR] Query failed: {str(e)}")
        traceback.print_exc()
        return None

