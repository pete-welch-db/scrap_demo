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


def _workspace_url_for_sql() -> str:
    """
    Workspace URL used for OAuth + SQL Thrift. Must be the **workspace** host, not the App URL.
    """
    url = _sdk_workspace_url()
    if "databricksapps.com" in url.lower():
        logger.error(
            "DATABRICKS_HOST must be the workspace URL (e.g. https://xxx.cloud.databricks.com), "
            "not the Databricks Apps URL (%s). Using config fallback.",
            url,
        )
        return databricks_config.sdk_host_url()
    return url


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


def _sql_connect_kwargs() -> dict:
    """Shared connector kwargs (Databricks SQL Thrift)."""
    return {"_socket_timeout": 120}


def get_databricks_connection():
    """
    Create a connection to Databricks SQL Warehouse.

    Uses different authentication methods:
    - LOCAL: Personal Access Token (+ optional session catalog/schema)
    - DBX: Databricks Apps runtime — follow official cookbook: default ``Config()`` + ``cfg.authenticate``
      (see https://apps-cookbook.dev/docs/streamlit/tables/tables_read ), then fall back to explicit
      OAuth M2M / bearer token if needed.

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

        http_path = _warehouse_http_path()
        kw = _sql_connect_kwargs()
        workspace_url = _workspace_url_for_sql()
        server_hostname = _bare_hostname_for_sql(workspace_url)
        client_id = (os.getenv("DATABRICKS_CLIENT_ID") or "").strip()
        client_secret = (os.getenv("DATABRICKS_CLIENT_SECRET") or "").strip()

        logger.info(
            "DBX SQL target: workspace_host=%s warehouse_path=%s",
            server_hostname,
            http_path[:48] + "..." if len(http_path) > 48 else http_path,
        )

        # --- 1) OAuth M2M first when Apps injects client id/secret (explicit auth_type avoids ambiguous Config defaults) ---
        # See: https://docs.databricks.com/aws/en/dev-tools/auth/oauth-m2m.html
        if client_id and client_secret:
            try:
                cfg_m2m = Config(
                    host=workspace_url,
                    client_id=client_id,
                    client_secret=client_secret,
                    auth_type="oauth-m2m",
                )
            except TypeError:
                cfg_m2m = Config(
                    host=workspace_url,
                    client_id=client_id,
                    client_secret=client_secret,
                )
            m2m = oauth_service_principal(cfg_m2m)
            if m2m is not None:
                try:
                    tok = m2m.oauth_token()
                    at = (tok.access_token or "").strip() if tok else ""
                    if at:
                        return _connect_with_token(sql, server_hostname, http_path, at)
                except Exception as e:
                    logger.warning("M2M oauth_token (oauth-m2m Config) failed: %s", e)

            try:
                tok = cfg_m2m.oauth_token()
                at = (tok.access_token or "").strip() if tok else ""
                if at:
                    return _connect_with_token(sql, server_hostname, http_path, at)
            except Exception as e:
                logger.warning("cfg_m2m.oauth_token() failed: %s", e)

            if m2m is not None:
                try:
                    return sql.connect(
                        server_hostname=server_hostname,
                        http_path=http_path,
                        credentials_provider=lambda: m2m,
                        **kw,
                    )
                except Exception as e:
                    logger.warning("credentials_provider M2M failed: %s", e)

            try:
                return sql.connect(
                    server_hostname=server_hostname,
                    http_path=http_path,
                    credentials_provider=lambda: cfg_m2m.authenticate,
                    **kw,
                )
            except Exception as e:
                logger.warning("SQL connect (oauth-m2m cfg.authenticate) failed: %s", e)

        # --- 2) Cookbook path: default Config() + authenticate (runtime-injected profile) ---
        # https://apps-cookbook.dev/docs/streamlit/tables/tables_read
        cfg_apps = Config()
        raw_host = (cfg_apps.host or os.getenv("DATABRICKS_HOST") or "").strip()
        if not raw_host:
            logger.warning("DBX: DATABRICKS_HOST / Config.host missing; cannot open SQL session.")
            return None
        server_apps = _bare_hostname_for_sql(raw_host)
        logger.info(
            "SQL connect (cookbook default Config): host=%s path=%s",
            server_apps,
            http_path[:48] + "..." if len(http_path) > 48 else http_path,
        )
        try:
            return sql.connect(
                server_hostname=server_apps,
                http_path=http_path,
                credentials_provider=lambda: cfg_apps.authenticate,
                **kw,
            )
        except Exception as e:
            logger.warning("Cookbook SQL connect (Config.authenticate) failed: %s", e)

        # --- 3) Legacy explicit Config without auth_type (older SDK / runtimes) ---
        if client_id and client_secret:
            cfg = Config(
                host=workspace_url,
                client_id=client_id,
                client_secret=client_secret,
            )
            m2m = oauth_service_principal(cfg)
            if m2m is not None:
                try:
                    tok = m2m.oauth_token()
                    at = (tok.access_token or "").strip() if tok else ""
                    if at:
                        return _connect_with_token(sql, server_hostname, http_path, at)
                except Exception as e:
                    logger.warning("M2M oauth_token (legacy Config) failed: %s", e)

            try:
                tok = cfg.oauth_token()
                at = (tok.access_token or "").strip() if tok else ""
                if at:
                    return _connect_with_token(sql, server_hostname, http_path, at)
            except Exception as e:
                logger.warning("cfg.oauth_token() failed: %s", e)

            if m2m is not None:
                try:
                    return sql.connect(
                        server_hostname=server_hostname,
                        http_path=http_path,
                        credentials_provider=lambda: m2m,
                        **kw,
                    )
                except Exception as e:
                    logger.warning("credentials_provider M2M (legacy) failed: %s", e)

            try:
                return sql.connect(
                    server_hostname=server_hostname,
                    http_path=http_path,
                    credentials_provider=lambda: cfg.authenticate,
                    **kw,
                )
            except Exception as e:
                logger.warning("credentials_provider explicit cfg.authenticate failed: %s", e)

        logger.error(
            "DBX: All SQL connection strategies failed (Thrift open_session / OAuth). "
            "This is independent of DLT/jobs: fix workspace host, OAuth for the app identity, "
            "warehouse CAN USE, and UC grants. Pipeline running does not block SQL."
        )
        return None

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
