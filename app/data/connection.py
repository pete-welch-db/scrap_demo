"""
Database Connection
===================
Databricks SQL connection management.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

import pandas as pd
import streamlit as st

from config import databricks_config, app_config

logger = logging.getLogger(__name__)


def _bare_hostname_for_sql(host: str) -> str:
    """Databricks SQL connector expects hostname only (no scheme, no path)."""
    h = (host or "").strip()
    if h.startswith("https://"):
        h = h[8:]
    elif h.startswith("http://"):
        h = h[7:]
    return h.split("/")[0].rstrip("/")


def _sdk_workspace_url() -> str:
    """
    Full workspace URL for databricks.sdk Config (OIDC + OAuth).
    Prefer live DATABRICKS_HOST from Databricks Apps runtime.
    """
    raw = (os.getenv("DATABRICKS_HOST") or "").strip()
    if not raw:
        return databricks_config.sdk_host_url()
    if raw.startswith("https://"):
        return raw.rstrip("/")
    if raw.startswith("http://"):
        return "https://" + raw[7:].split("/")[0].rstrip("/")
    return f"https://{raw.split('/')[0].rstrip('/')}"


def _warehouse_http_path() -> str:
    """Resolve /sql/1.0/warehouses/{id} from bundle env or config."""
    explicit = (os.getenv("DATABRICKS_HTTP_PATH") or "").strip()
    if explicit:
        return explicit if explicit.startswith("/") else f"/{explicit}"
    wh = (os.getenv("DATABRICKS_WAREHOUSE_ID") or "").strip()
    if wh:
        return f"/sql/1.0/warehouses/{wh}"
    return databricks_config.http_path


def _connect_with_token(sql_mod: Any, server_hostname: str, http_path: str, access_token: str) -> Any:
    """Open SQL session with a bearer token (most reliable for Thrift in Apps)."""
    # Queries use fully qualified names ({catalog}.{schema}.table); omit session catalog/schema
    # to avoid open_session failures on some warehouse / UC combinations.
    return sql_mod.connect(
        server_hostname=server_hostname,
        http_path=http_path,
        access_token=access_token,
        _socket_timeout=120,
    )


def get_databricks_connection():
    """
    Create a connection to Databricks SQL Warehouse.

    Uses different authentication methods:
    - LOCAL: Personal Access Token
    - DBX: OAuth M2M — prefer bearer token (oauth_token) + access_token= for SQL connector

    Returns:
        Connection object or None if connection fails
    """
    logger.info("Connecting to Databricks (%s mode)", app_config.environment)

    try:
        from databricks import sql
        from databricks.sdk.core import Config, oauth_service_principal

        if app_config.is_local():
            catalog = (os.getenv("DATABRICKS_CATALOG") or databricks_config.catalog or "").strip()
            schema = (os.getenv("DATABRICKS_SCHEMA") or databricks_config.schema or "").strip()
            return sql.connect(
                server_hostname=databricks_config.host,
                http_path=databricks_config.http_path,
                access_token=databricks_config.token,
                catalog=catalog or None,
                schema=schema or None,
            )

        client_id = os.getenv("DATABRICKS_CLIENT_ID")
        client_secret = os.getenv("DATABRICKS_CLIENT_SECRET")
        if not client_id or not client_secret:
            raise ValueError(
                "DATABRICKS_CLIENT_ID and DATABRICKS_CLIENT_SECRET must be set in the app runtime."
            )

        workspace_url = _sdk_workspace_url()
        server_hostname = _bare_hostname_for_sql(workspace_url)
        http_path = _warehouse_http_path()

        logger.info(
            "SQL connect: host=%s path=%s",
            server_hostname,
            http_path[:40] + "..." if len(http_path) > 40 else http_path,
        )

        def build_cfg() -> Config:
            return Config(
                host=workspace_url,
                client_id=client_id,
                client_secret=client_secret,
            )

        cfg = build_cfg()

        # --- 1) Bearer token from OAuth M2M provider (best for SQL Thrift; avoids credentials_provider quirks) ---
        m2m = oauth_service_principal(cfg)
        if m2m is not None:
            try:
                tok = m2m.oauth_token()
                at = (tok.access_token or "").strip() if tok else ""
                if at:
                    return _connect_with_token(sql, server_hostname, http_path, at)
            except Exception as e:
                logger.warning("M2M oauth_token for SQL failed: %s", e)

        # --- 2) Config.oauth_token() (when inner factory is OAuthCredentialsProvider) ---
        try:
            tok = cfg.oauth_token()
            at = (tok.access_token or "").strip() if tok else ""
            if at:
                return _connect_with_token(sql, server_hostname, http_path, at)
        except Exception as e:
            logger.warning("cfg.oauth_token() failed: %s", e)

        # --- 3) Fresh Config() from env (Apps injects DATABRICKS_HOST + SP creds) ---
        cfg_env = Config()
        m2m_env = oauth_service_principal(cfg_env)
        if m2m_env is not None:
            try:
                tok = m2m_env.oauth_token()
                at = (tok.access_token or "").strip() if tok else ""
                if at:
                    host_env = _bare_hostname_for_sql(
                        (cfg_env.host or workspace_url or "").strip() or workspace_url
                    )
                    return _connect_with_token(sql, host_env, http_path, at)
            except Exception as e:
                logger.warning("env Config M2M oauth_token failed: %s", e)

        # --- 4) credentials_provider fallbacks ---
        if m2m is not None:
            try:
                return sql.connect(
                    server_hostname=server_hostname,
                    http_path=http_path,
                    credentials_provider=lambda: m2m,
                )
            except Exception as e:
                logger.warning("credentials_provider M2M failed: %s", e)

        try:
            return sql.connect(
                server_hostname=server_hostname,
                http_path=http_path,
                credentials_provider=lambda: cfg.authenticate,
            )
        except Exception as e:
            logger.warning("credentials_provider cfg.authenticate failed: %s", e)

        host_env = _bare_hostname_for_sql(
            (cfg_env.host or workspace_url or "").strip() or workspace_url
        )
        if m2m_env is not None:
            return sql.connect(
                server_hostname=host_env,
                http_path=http_path,
                credentials_provider=lambda: m2m_env,
            )
        return sql.connect(
            server_hostname=host_env,
            http_path=http_path,
            credentials_provider=lambda: cfg_env.authenticate,
        )

    except ImportError as e:
        import traceback

        st.warning(f"Required package not installed: {str(e)}. Using mock data.")
        logger.warning("ImportError: %s", e, exc_info=True)
        traceback.print_exc()
        return None
    except Exception as e:
        import traceback

        logger.exception("Databricks SQL connection failed: %s", e)
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
    except Exception:
        logger.exception("test_db_connection failed")
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
        import traceback

        logger.exception("Query failed: %s", e)
        traceback.print_exc()
        return None
