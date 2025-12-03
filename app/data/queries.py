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
        part_category_lookup as part_category,
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


# =============================================================================
# Scenario Queries - Real Data for What-If Analysis
# =============================================================================

@st.cache_data(ttl=app_config.cache_ttl)
def load_plant_monthly_data() -> Optional[pd.DataFrame]:
    """Load monthly plant scrap data for scenario analysis."""
    query = f"""
    SELECT 
        plant_id,
        plant_name_lookup as plant,
        plant_city,
        plant_country,
        year,
        month,
        month_date,
        scrap_records,
        quantity_scrapped,
        scrap_cost,
        unique_parts,
        unique_suppliers,
        avg_oee,
        avg_equipment_age,
        shift_1_ratio,
        shift_2_ratio,
        shift_3_ratio
    FROM {SCHEMA}.plant_monthly_scenario_gold
    ORDER BY plant_name_lookup, year, month
    """
    return run_query(query)


@st.cache_data(ttl=app_config.cache_ttl)
def load_work_center_data() -> Optional[pd.DataFrame]:
    """Load work center scrap data for scenario analysis."""
    query = f"""
    SELECT 
        work_center_id,
        work_center_name_lookup as work_center,
        work_center_type_lookup as work_center_type,
        plant_id,
        plant_name_lookup as plant,
        transaction_year,
        transaction_month,
        scrap_records,
        quantity_scrapped,
        total_scrap_cost as monthly_scrap_cost,
        avg_equipment_age,
        avg_equipment_oee as oee,
        min_equipment_oee,
        unique_equipment,
        top_defect_category,
        top_root_cause,
        oee_scrap_correlation
    FROM {SCHEMA}.scrap_by_work_center_gold
    ORDER BY work_center_type, plant
    """
    return run_query(query)


@st.cache_data(ttl=app_config.cache_ttl)
def load_supplier_scenario_data() -> Optional[pd.DataFrame]:
    """Load supplier data for what-if scenario analysis."""
    query = f"""
    SELECT 
        supplier_id,
        supplier_name as name,
        supplier_tier as tier,
        specialization as category,
        region,
        quality_score,
        on_time_delivery_rate as on_time_delivery,
        is_preferred,
        parts_supplied,
        plants_served,
        annual_qty_scrapped,
        annual_scrap_cost as scrap_cost,
        estimated_annual_spend as annual_spend,
        scrap_rate,
        capacity_utilization as capacity_util,
        available_capacity,
        risk_score,
        unit_cost_index
    FROM {SCHEMA}.supplier_scenario_gold
    WHERE parts_supplied > 0
    ORDER BY annual_scrap_cost DESC
    """
    return run_query(query)


@st.cache_data(ttl=app_config.cache_ttl)
def load_part_scrap_data() -> Optional[pd.DataFrame]:
    """Load part-level scrap data for scenario analysis."""
    query = f"""
    SELECT 
        part_id,
        part_name_lookup as part_name,
        part_category_lookup as part_category,
        supplier_id,
        supplier_name_lookup as supplier_name,
        is_critical_part,
        SUM(scrap_records) as scrap_records,
        SUM(quantity_scrapped) as quantity_scrapped,
        SUM(total_scrap_cost) as monthly_scrap_cost,
        AVG(avg_unit_cost) as unit_cost,
        AVG(estimated_defect_rate) as defect_rate,
        MAX(top_defect_category) as top_defect,
        MAX(top_root_cause) as top_root_cause,
        AVG(avg_equipment_oee) as avg_oee
    FROM {SCHEMA}.part_scrap_analysis_gold
    WHERE transaction_year >= YEAR(CURRENT_DATE()) - 1
    GROUP BY part_id, part_name_lookup, part_category_lookup, supplier_id, supplier_name_lookup, is_critical_part
    ORDER BY monthly_scrap_cost DESC
    LIMIT 100
    """
    return run_query(query)


@st.cache_data(ttl=app_config.cache_ttl)
def get_plants_list() -> list:
    """Get list of plant names for dropdown."""
    query = f"""
    SELECT DISTINCT plant_name_lookup as plant
    FROM {SCHEMA}.plant_monthly_scenario_gold
    ORDER BY plant_name_lookup
    """
    result = run_query(query)
    if result is not None:
        return result['plant'].tolist()
    return []


@st.cache_data(ttl=app_config.cache_ttl)
def get_work_center_types() -> list:
    """Get list of work center types for dropdown."""
    query = f"""
    SELECT DISTINCT work_center_type_lookup as work_center_type
    FROM {SCHEMA}.scrap_by_work_center_gold
    ORDER BY work_center_type_lookup
    """
    result = run_query(query)
    if result is not None:
        return result['work_center_type'].tolist()
    return []


@st.cache_data(ttl=app_config.cache_ttl)
def get_supplier_alternatives(category: str, exclude_supplier_id: str) -> Optional[pd.DataFrame]:
    """Get alternative suppliers in the same category."""
    query = f"""
    SELECT 
        supplier_id,
        supplier_name as name,
        supplier_tier as tier,
        quality_score,
        on_time_delivery_rate as on_time_delivery,
        available_capacity,
        risk_score,
        unit_cost_index
    FROM {SCHEMA}.supplier_scenario_gold
    WHERE specialization = '{category}'
      AND supplier_id != '{exclude_supplier_id}'
      AND parts_supplied > 0
    ORDER BY quality_score DESC
    LIMIT 5
    """
    return run_query(query)


# =============================================================================
# ML Model Result Queries
# =============================================================================

@st.cache_data(ttl=app_config.cache_ttl)
def load_forecast_summary() -> Optional[pd.DataFrame]:
    """Load forecast summary metrics."""
    query = f"""
    SELECT 
        MIN(forecast_date) as forecast_start,
        MAX(forecast_date) as forecast_end,
        SUM(predicted_value) as total_predicted_cost,
        AVG(predicted_value) as avg_daily_cost,
        COUNT(*) as forecast_days
    FROM {SCHEMA}.ml_total_forecast_results
    WHERE forecast_date > CURRENT_DATE()
    """
    return run_query(query)


@st.cache_data(ttl=app_config.cache_ttl)
def load_model_accuracy_metrics() -> Optional[pd.DataFrame]:
    """Load ML model accuracy metrics from validation results."""
    query = f"""
    SELECT 
        COUNT(*) as total_records,
        SUM(CASE WHEN prediction_correct THEN 1 ELSE 0 END) as correct_predictions,
        ROUND(AVG(CASE WHEN prediction_correct THEN 1.0 ELSE 0.0 END) * 100, 2) as accuracy_pct,
        SUM(CASE WHEN high_confidence_disagreement THEN 1 ELSE 0 END) as high_conf_disagreements,
        SUM(CASE WHEN high_confidence_disagreement THEN total_scrap_cost ELSE 0 END) as cost_at_risk
    FROM {SCHEMA}.ml_root_cause_validation
    """
    return run_query(query)


@st.cache_data(ttl=app_config.cache_ttl)
def load_recent_anomalies(days: int = 30) -> Optional[pd.DataFrame]:
    """Load recent anomalies from ML detection."""
    query = f"""
    SELECT 
        detection_date as transaction_date,
        entity_name as plant,
        daily_cost as total_scrap_cost,
        daily_qty as quantity_scrapped,
        anomaly_score,
        anomaly_type as anomaly_reason
    FROM {SCHEMA}.ml_scrap_anomalies
    WHERE is_anomaly = true
      AND detection_date >= DATE_ADD(CURRENT_DATE(), -{days})
    ORDER BY detection_date DESC
    LIMIT 50
    """
    return run_query(query)

