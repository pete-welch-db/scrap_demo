"""
Mock Data
=========
Synthetic data generation for demo/fallback purposes.
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Tuple, List


@st.cache_data
def load_mock_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Generate mock data for the demo when database is not available.
    
    Returns:
        Tuple of (monthly_df, parts_df, work_center_df, plants_list)
    """
    np.random.seed(42)
    
    # Plants
    plants = [
        "Detroit Assembly", 
        "Louisville Stamping", 
        "San Antonio Body", 
        "Oakville Paint", 
        "Saltillo Powertrain", 
        "Silao Components"
    ]
    
    # Parts
    parts = [f"PRT-{str(i).zfill(5)}" for i in range(1, 51)]
    part_names = [f"Component {chr(65 + i % 26)}-{i}" for i in range(50)]
    
    # Work centers
    work_centers = [
        "Stamping Press Line 1", 
        "Welding Cell A", 
        "CNC Machining", 
        "Injection Molding", 
        "Assembly Line", 
        "Paint Booth"
    ]
    
    # Generate monthly scrap data
    months = pd.date_range('2022-01-01', '2024-11-01', freq='MS')
    
    monthly_data = []
    for month in months:
        for plant in plants:
            base_cost = np.random.uniform(50000, 150000)
            # Summer spike pattern
            seasonal = 1.3 if month.month in [7, 8] else 1.0
            
            monthly_data.append({
                'month': month,
                'plant': plant,
                'scrap_cost': base_cost * seasonal * np.random.uniform(0.8, 1.2),
                'quantity_scrapped': int(base_cost / 10 * np.random.uniform(0.8, 1.2)),
                'defect_count': int(np.random.uniform(50, 200))
            })
    
    monthly_df = pd.DataFrame(monthly_data)
    
    # Generate parts data
    parts_data = []
    for part, name in zip(parts, part_names):
        parts_data.append({
            'part_id': part,
            'part_name': name,
            'monthly_scrap_cost': np.random.uniform(5000, 50000),
            'monthly_quantity': int(np.random.uniform(100, 1000)),
            'defect_rate': np.random.uniform(0.01, 0.08),
            'unit_cost': np.random.uniform(10, 200)
        })
    
    parts_df = pd.DataFrame(parts_data)
    
    # Generate work center data
    wc_data = []
    for wc in work_centers:
        for plant in plants:
            wc_data.append({
                'work_center': wc,
                'plant': plant,
                'monthly_scrap_cost': np.random.uniform(10000, 80000),
                'oee': np.random.uniform(0.65, 0.95),
                'equipment_age': np.random.randint(2, 18)
            })
    
    wc_df = pd.DataFrame(wc_data)
    
    return monthly_df, parts_df, wc_df, plants


def get_mock_kpis() -> dict:
    """Return mock KPI values."""
    return {
        "ytd_scrap_cost": 36010000,  # $36.01M from dashboard
        "ytd_units_scrapped": 252790,  # 252.79K from dashboard
        "scrap_incidents": 18400,  # 18.4K from dashboard
        "open_8d_reports": 1820,  # 1.82K from dashboard
        "ytd_scrap_delta": -8.3,
        "units_delta": -12.1,
        "incidents_delta": -5.2,
        "eight_d_delta": -3.1
    }


@st.cache_data(ttl=300)
def get_live_kpis() -> dict:
    """
    Fetch live KPI values from Databricks.
    Falls back to mock values if query fails.
    """
    from .connection import run_query
    from config import databricks_config
    
    schema = f"{databricks_config.catalog}.{databricks_config.schema}"
    
    try:
        # Query for YTD scrap metrics
        scrap_query = f"""
        SELECT 
            COALESCE(SUM(total_scrap_cost), 0) as ytd_scrap_cost,
            COALESCE(SUM(total_quantity_scrapped), 0) as ytd_units_scrapped,
            COALESCE(COUNT(*), 0) as scrap_incidents
        FROM {schema}.scrap_daily_summary_gold
        WHERE transaction_year = YEAR(CURRENT_DATE())
        """
        
        scrap_result = run_query(scrap_query)
        
        # Query for open 8D reports
        eight_d_query = f"""
        SELECT COUNT(*) as open_8d_reports
        FROM {schema}.eight_d_reports_silver
        WHERE status IN ('Open', 'In Progress', 'Pending Verification')
        """
        
        eight_d_result = run_query(eight_d_query)
        
        if scrap_result is not None and len(scrap_result) > 0:
            kpis = {
                "ytd_scrap_cost": float(scrap_result.iloc[0]["ytd_scrap_cost"]),
                "ytd_units_scrapped": float(scrap_result.iloc[0]["ytd_units_scrapped"]),
                "scrap_incidents": int(scrap_result.iloc[0]["scrap_incidents"]),
                "open_8d_reports": int(eight_d_result.iloc[0]["open_8d_reports"]) if eight_d_result is not None else 0,
                "ytd_scrap_delta": -8.3,  # Would need prior year comparison
                "units_delta": -12.1,
                "incidents_delta": -5.2,
                "eight_d_delta": -3.1
            }
            return kpis
    except Exception as e:
        pass
    
    # Fallback to mock
    return get_mock_kpis()

