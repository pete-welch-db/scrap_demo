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
    
    Returns:
        Connection object or None if connection fails
    """
    try:
        from databricks import sql
        
        connection = sql.connect(
            server_hostname=databricks_config.host,
            http_path=databricks_config.http_path,
            access_token=databricks_config.token
        )
        return connection
    except ImportError:
        st.warning("databricks-sql-connector not installed. Using mock data.")
        return None
    except Exception as e:
        st.error(f"Connection error: {str(e)}")
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
        st.error(f"Query error: {str(e)}")
        return None

