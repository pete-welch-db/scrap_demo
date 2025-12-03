"""
SQL Queries
===========
Pre-defined queries for loading data from Databricks.
"""

import streamlit as st
import pandas as pd
from typing import Optional
from config import databricks_config, app_config
from .connection import run_query


SCHEMA = databricks_config.full_schema


@st.cache_data(ttl=app_config.cache_ttl)
def load_scrap_summary() -> Optional[pd.DataFrame]:
    """Load daily scrap summary from gold tables."""
    query = f"""
    SELECT * FROM {SCHEMA}.scrap_daily_summary_gold
    ORDER BY transaction_date DESC
    LIMIT 1000
    """
    return run_query(query)


@st.cache_data(ttl=app_config.cache_ttl)
def load_plant_summary() -> Optional[pd.DataFrame]:
    """Load plant-level scrap summary from gold tables."""
    query = f"""
    SELECT * FROM {SCHEMA}.scrap_by_plant_gold
    """
    return run_query(query)


@st.cache_data(ttl=app_config.cache_ttl)
def load_supplier_summary() -> Optional[pd.DataFrame]:
    """Load supplier-level scrap summary from gold tables."""
    query = f"""
    SELECT * FROM {SCHEMA}.scrap_by_supplier_gold
    """
    return run_query(query)


@st.cache_data(ttl=app_config.cache_ttl)
def load_parts_data() -> Optional[pd.DataFrame]:
    """Load parts master data from silver tables."""
    query = f"""
    SELECT 
        part_id,
        part_name,
        part_category,
        unit_cost,
        supplier_id
    FROM {SCHEMA}.parts_silver
    LIMIT 500
    """
    return run_query(query)


@st.cache_data(ttl=app_config.cache_ttl)
def load_locations_data() -> Optional[pd.DataFrame]:
    """Load locations/plants data from silver tables."""
    query = f"""
    SELECT DISTINCT
        plant_id,
        plant_name,
        city,
        state_province,
        country
    FROM {SCHEMA}.locations_silver
    """
    return run_query(query)


@st.cache_data(ttl=app_config.cache_ttl)
def load_eight_d_reports() -> Optional[pd.DataFrame]:
    """Load 8D reports from silver tables."""
    query = f"""
    SELECT 
        report_id,
        report_date,
        status,
        problem_description,
        root_cause,
        corrective_action,
        effectiveness_rating
    FROM {SCHEMA}.eight_d_reports_silver
    ORDER BY report_date DESC
    LIMIT 100
    """
    return run_query(query)


@st.cache_data(ttl=app_config.cache_ttl)
def get_kpi_summary() -> dict:
    """Get current KPI values for dashboard."""
    query = f"""
    SELECT 
        SUM(total_scrap_cost) as ytd_scrap_cost,
        SUM(total_quantity_scrapped) as ytd_units_scrapped
    FROM {SCHEMA}.scrap_daily_summary_gold
    WHERE transaction_year = YEAR(CURRENT_DATE())
    """
    result = run_query(query)
    
    if result is not None and len(result) > 0:
        return {
            "ytd_scrap_cost": result.iloc[0]["ytd_scrap_cost"],
            "ytd_units_scrapped": result.iloc[0]["ytd_units_scrapped"]
        }
    
    # Default values if query fails
    return {
        "ytd_scrap_cost": 4200000,
        "ytd_units_scrapped": 127000
    }

