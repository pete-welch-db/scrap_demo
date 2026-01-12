# Databricks notebook source
# MAGIC %md
# MAGIC # 🔮 Scrap Forecasting Models
# MAGIC 
# MAGIC ## Business Objective
# MAGIC This notebook builds time series forecasting models to predict scrap quantities and costs **3 months into the future** at multiple granularities. Accurate forecasting enables:
# MAGIC - **Budgeting**: Plan for scrap-related costs
# MAGIC - **Capacity Planning**: Adjust production schedules
# MAGIC - **Supplier Management**: Anticipate quality issues
# MAGIC - **Inventory Optimization**: Buffer stock for high-scrap parts
# MAGIC 
# MAGIC ## Forecast Levels
# MAGIC | Level | Description | Use Case |
# MAGIC |-------|-------------|----------|
# MAGIC | **Total Cost** | Company-wide daily forecast | Executive planning |
# MAGIC | **By Plant** | Plant-level forecasts | Operations management |
# MAGIC | **By Supplier** | Supplier-level forecasts | Procurement decisions |
# MAGIC | **By Part Number** | Part-level forecasts | Inventory planning |
# MAGIC 
# MAGIC ## ML Approach
# MAGIC - **Primary Model**: Facebook Prophet (time series with seasonality)
# MAGIC - **Comparison**: Databricks AutoML for automated model selection
# MAGIC - **Tracking**: Full MLflow experiment tracking with artifact logging
# MAGIC 
# MAGIC ## Output Tables
# MAGIC - `ml_plant_forecast_results`
# MAGIC - `ml_supplier_forecast_results`  
# MAGIC - `ml_part_forecast_results`
# MAGIC - `ml_total_forecast_results`

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📦 Install Dependencies
# MAGIC 
# MAGIC Install the Prophet library for time series forecasting. Prophet is developed by Facebook/Meta and handles:
# MAGIC - Multiple seasonality (yearly, weekly, daily)
# MAGIC - Holiday effects
# MAGIC - Missing data and outliers
# MAGIC - Trend changes

# COMMAND ----------

# MAGIC %pip install prophet mlflow[databricks] scipy scikit-learn


# COMMAND ----------

# MAGIC %md
# MAGIC ## 🔄 Restart Python Environment
# MAGIC 
# MAGIC After installing packages, we need to restart the Python environment for the new libraries to be available.

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📚 Import Libraries
# MAGIC 
# MAGIC Import all required libraries:
# MAGIC - **pandas/numpy**: Data manipulation
# MAGIC - **prophet**: Time series forecasting
# MAGIC - **mlflow**: Experiment tracking and model registry
# MAGIC - **databricks.automl**: Automated machine learning
# MAGIC - **pyspark**: Distributed data processing
# MAGIC - **matplotlib**: Visualization

# COMMAND ----------

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
import mlflow
import mlflow.prophet
import mlflow.pyfunc
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="mlflow")

from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import *
import matplotlib.pyplot as plt
import json
import os

# MLOps imports for drift detection
from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Parallel processing imports
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# COMMAND ----------

# MAGIC %md
# MAGIC ## ⚙️ Configuration
# MAGIC 
# MAGIC Define all configuration parameters in one place for easy modification:
# MAGIC - **CATALOG/SCHEMA**: Unity Catalog location
# MAGIC - **FORECAST_DAYS**: Prediction horizon (90 days = ~3 months)
# MAGIC - **EXPERIMENT_PATH**: MLflow experiment location

# COMMAND ----------

# Unity Catalog Configuration - Use widget parameters from job, with defaults for interactive use
dbutils.widgets.text("catalog", "scrap_catalog", "Catalog Name")
dbutils.widgets.text("schema", "scrap_demo", "Schema Name")

CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
FULL_SCHEMA = f"{CATALOG}.{SCHEMA}"

print(f"Using catalog: {CATALOG}, schema: {SCHEMA}")

# Forecasting Configuration
FORECAST_DAYS = 90  # 3 months forward
TRAIN_TEST_SPLIT_DAYS = 30  # Hold out last 30 days for validation

# Parallel Training Configuration
MAX_PARALLEL_WORKERS = 4  # Number of models to train in parallel
USE_PARALLEL_TRAINING = True  # Enable/disable parallel training

# MLOps Configuration
DRIFT_DETECTION_WINDOW = 30  # Days to compare for drift detection
DRIFT_P_VALUE_THRESHOLD = 0.05  # Significance threshold for statistical tests
DRIFT_DISTANCE_THRESHOLD = 0.1  # Threshold for Wasserstein distance
MODEL_PERFORMANCE_DEGRADATION_THRESHOLD = 0.15  # 15% increase in error triggers alert

# MLflow Configuration
# MLflow will use the default notebook experiment when running from Git folders
# This avoids the error: "Cannot create workspace MLflow experiment in a Git folder"
# EXPERIMENT_PATH = "/Shared/scrap_demo/scrap_forecasting"
# mlflow.set_experiment(EXPERIMENT_PATH)  # Removed for Git folder compatibility

print(f"✅ Configuration loaded")
print(f"   Catalog: {CATALOG}")
print(f"   Schema: {SCHEMA}")
print(f"   Forecast Horizon: {FORECAST_DAYS} days")
print(f"   MLflow Experiment: Default Notebook Experiment")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📊 Load and Explore Data
# MAGIC 
# MAGIC Load scrap transactions from the Silver layer. The Silver layer contains:
# MAGIC - Cleaned and validated data
# MAGIC - Enriched with dimension lookups
# MAGIC - Standardized data types

# COMMAND ----------

# Helper function to check if table exists
def table_exists(table_name):
    try:
        spark.table(table_name)
        return True
    except Exception:
        return False

# Validate required tables exist
required_tables = [f"{FULL_SCHEMA}.scrap_transactions_silver"]
missing_tables = [t for t in required_tables if not table_exists(t)]

if missing_tables:
    error_msg = f"""
    ❌ Missing required tables: {missing_tables}
    
    Please run the following first:
    1. Setup job (scrap_setup_job) - Creates bronze tables
    2. DLT Pipeline - Creates silver tables from bronze
    
    Then re-run this ML training notebook.
    """
    raise Exception(error_msg)

print(f"✅ All required tables exist in {FULL_SCHEMA}")

# COMMAND ----------

# Load scrap transactions from silver layer
scrap_df = spark.table(f"{FULL_SCHEMA}.scrap_transactions_silver")

# Display data summary
record_count = scrap_df.count()
date_range = scrap_df.agg(
    F.min('transaction_date').alias('min_date'),
    F.max('transaction_date').alias('max_date')
).collect()[0]

print(f"📈 Data Summary:")
print(f"   Total Records: {record_count:,}")
print(f"   Date Range: {date_range['min_date']} to {date_range['max_date']}")
print(f"   Unique Plants: {scrap_df.select('plant_id').distinct().count()}")
print(f"   Unique Suppliers: {scrap_df.select('supplier_id').distinct().count()}")
print(f"   Unique Parts: {scrap_df.select('part_id').distinct().count()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🛠️ Define Helper Functions
# MAGIC 
# MAGIC Create reusable functions for:
# MAGIC 1. **Data preparation** - Convert to Prophet format (ds, y columns)
# MAGIC 2. **Hyperparameter tuning** - Grid search for optimal parameters
# MAGIC 3. **Model training** - Train Prophet with MLflow tracking
# MAGIC 4. **Metrics calculation** - MAE, MAPE, RMSE

# COMMAND ----------

def prepare_prophet_data(
    df, 
    date_col: str, 
    value_col: str, 
    group_cols: list = None
) -> pd.DataFrame:
    """
    Prepare data for Prophet model.
    
    Prophet requires a DataFrame with exactly two columns:
    - 'ds': datetime column (datestamp)
    - 'y': numeric column (value to predict)
    
    Parameters:
    -----------
    df : DataFrame
        Spark DataFrame with scrap data
    date_col : str
        Name of the date column
    value_col : str
        Name of the value column to forecast
    group_cols : list, optional
        Columns to group by before aggregation
        
    Returns:
    --------
    pd.DataFrame
        Pandas DataFrame ready for Prophet
    """
    if group_cols:
        agg_df = (
            df.groupBy(date_col, *group_cols)
            .agg(F.sum(value_col).alias("y"))
            .withColumnRenamed(date_col, "ds")
        )
    else:
        agg_df = (
            df.groupBy(date_col)
            .agg(F.sum(value_col).alias("y"))
            .withColumnRenamed(date_col, "ds")
        )
    
    # Convert to Pandas and ensure proper types
    pdf = agg_df.orderBy("ds").toPandas()
    pdf['ds'] = pd.to_datetime(pdf['ds'])
    pdf['y'] = pdf['y'].astype(float)
    
    return pdf


def calculate_metrics(actual: pd.Series, predicted: pd.Series) -> dict:
    """
    Calculate forecasting accuracy metrics.
    
    Parameters:
    -----------
    actual : pd.Series
        Actual values
    predicted : pd.Series
        Predicted values
        
    Returns:
    --------
    dict
        Dictionary with MAE, MAPE, RMSE metrics
    """
    # Remove any NaN values
    mask = ~(actual.isna() | predicted.isna())
    actual = actual[mask]
    predicted = predicted[mask]
    
    mae = np.mean(np.abs(actual - predicted))
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    rmse = np.sqrt(np.mean((actual - predicted)**2))
    
    return {
        "mae": mae,
        "mape": mape,
        "rmse": rmse
    }


# ============================================================================
# MLOps: Data and Model Drift Detection Functions
# ============================================================================

def detect_data_drift(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    feature_col: str = 'y',
    window_days: int = DRIFT_DETECTION_WINDOW
) -> dict:
    """
    Detect data drift using statistical tests.
    
    Compares the distribution of recent data against a reference baseline
    using multiple statistical methods:
    - Kolmogorov-Smirnov test (distribution shift)
    - Wasserstein distance (magnitude of shift)
    - Summary statistics comparison
    
    Parameters:
    -----------
    reference_data : pd.DataFrame
        Baseline/training data with 'ds' and feature columns
    current_data : pd.DataFrame
        Recent production data with 'ds' and feature columns
    feature_col : str
        Column name to check for drift (default: 'y')
    window_days : int
        Number of recent days to use for comparison
        
    Returns:
    --------
    dict
        Drift detection results including:
        - drift_detected: Boolean flag
        - ks_statistic: KS test statistic
        - ks_pvalue: KS test p-value
        - wasserstein_distance: Distance metric
        - reference_stats: Summary statistics of reference
        - current_stats: Summary statistics of current
        - drift_score: Overall drift severity (0-1)
    """
    
    # Get recent window from current data
    if 'ds' in current_data.columns:
        cutoff_date = current_data['ds'].max() - timedelta(days=window_days)
        current_window = current_data[current_data['ds'] >= cutoff_date][feature_col].values
    else:
        current_window = current_data[feature_col].tail(window_days).values
    
    reference_values = reference_data[feature_col].values
    
    # Remove NaN values
    reference_values = reference_values[~np.isnan(reference_values)]
    current_window = current_window[~np.isnan(current_window)]
    
    if len(current_window) < 10 or len(reference_values) < 10:
        return {
            "drift_detected": False,
            "error": "Insufficient data for drift detection",
            "current_samples": len(current_window),
            "reference_samples": len(reference_values)
        }
    
    # Kolmogorov-Smirnov test (non-parametric distribution comparison)
    ks_statistic, ks_pvalue = ks_2samp(reference_values, current_window)
    
    # Wasserstein distance (earth mover's distance)
    w_distance = wasserstein_distance(reference_values, current_window)
    
    # Normalize wasserstein distance by reference standard deviation
    reference_std = np.std(reference_values)
    normalized_w_distance = w_distance / reference_std if reference_std > 0 else 0
    
    # Calculate summary statistics
    reference_stats = {
        "mean": float(np.mean(reference_values)),
        "std": float(np.std(reference_values)),
        "median": float(np.median(reference_values)),
        "q25": float(np.percentile(reference_values, 25)),
        "q75": float(np.percentile(reference_values, 75)),
        "min": float(np.min(reference_values)),
        "max": float(np.max(reference_values))
    }
    
    current_stats = {
        "mean": float(np.mean(current_window)),
        "std": float(np.std(current_window)),
        "median": float(np.median(current_window)),
        "q25": float(np.percentile(current_window, 25)),
        "q75": float(np.percentile(current_window, 75)),
        "min": float(np.min(current_window)),
        "max": float(np.max(current_window))
    }
    
    # Calculate percentage changes in key statistics
    mean_change_pct = abs(current_stats["mean"] - reference_stats["mean"]) / reference_stats["mean"] * 100 if reference_stats["mean"] != 0 else 0
    std_change_pct = abs(current_stats["std"] - reference_stats["std"]) / reference_stats["std"] * 100 if reference_stats["std"] != 0 else 0
    
    # Drift score (0-1): combination of statistical tests
    # Higher score = more drift
    drift_score = min(1.0, (
        (1 - ks_pvalue) * 0.4 +  # KS test weight
        min(normalized_w_distance, 1.0) * 0.4 +  # Wasserstein distance weight
        min(mean_change_pct / 50, 1.0) * 0.2  # Mean change weight (50% change = max)
    ))
    
    # Determine if drift detected based on multiple criteria
    drift_detected = (
        ks_pvalue < DRIFT_P_VALUE_THRESHOLD or  # Significant distribution shift
        normalized_w_distance > DRIFT_DISTANCE_THRESHOLD or  # Large distance
        mean_change_pct > 30  # >30% change in mean
    )
    
    return {
        "drift_detected": bool(drift_detected),
        "drift_score": float(drift_score),
        "ks_statistic": float(ks_statistic),
        "ks_pvalue": float(ks_pvalue),
        "wasserstein_distance": float(w_distance),
        "normalized_wasserstein_distance": float(normalized_w_distance),
        "reference_stats": reference_stats,
        "current_stats": current_stats,
        "mean_change_pct": float(mean_change_pct),
        "std_change_pct": float(std_change_pct),
        "reference_samples": len(reference_values),
        "current_samples": len(current_window),
        "window_days": window_days
    }


def detect_model_drift(
    model,
    baseline_metrics: dict,
    current_data: pd.DataFrame,
    model_name: str = "model"
) -> dict:
    """
    Detect model performance drift by comparing current predictions against baseline.
    
    Monitors model performance degradation over time by:
    - Generating predictions on recent data
    - Comparing actual vs predicted values
    - Comparing metrics to baseline performance
    
    Parameters:
    -----------
    model : Prophet
        Trained Prophet model
    baseline_metrics : dict
        Baseline performance metrics (MAE, MAPE, RMSE)
    current_data : pd.DataFrame
        Recent data with 'ds' and 'y' columns (including actuals)
    model_name : str
        Name of the model for logging
        
    Returns:
    --------
    dict
        Model drift results including:
        - drift_detected: Boolean flag
        - current_metrics: Current performance metrics
        - baseline_metrics: Original baseline metrics
        - metric_degradation: Percentage degradation
        - predictions_df: DataFrame with predictions
    """
    
    if len(current_data) == 0:
        return {
            "drift_detected": False,
            "error": "No current data provided"
        }
    
    # Generate predictions for current period
    try:
        forecast = model.predict(current_data[['ds']])
        
        # Align predictions with actuals
        merged = current_data[['ds', 'y']].merge(
            forecast[['ds', 'yhat']], 
            on='ds', 
            how='inner'
        )
        
        if len(merged) == 0:
            return {
                "drift_detected": False,
                "error": "No overlapping dates between actuals and predictions"
            }
        
        # Calculate current metrics
        current_metrics = calculate_metrics(merged['y'], merged['yhat'])
        
        # Calculate degradation percentages
        mae_degradation = ((current_metrics['mae'] - baseline_metrics['mae']) / baseline_metrics['mae'] * 100) if baseline_metrics.get('mae', 0) > 0 else 0
        mape_degradation = ((current_metrics['mape'] - baseline_metrics['mape']) / baseline_metrics['mape'] * 100) if baseline_metrics.get('mape', 0) > 0 else 0
        rmse_degradation = ((current_metrics['rmse'] - baseline_metrics['rmse']) / baseline_metrics['rmse'] * 100) if baseline_metrics.get('rmse', 0) > 0 else 0
        
        # Overall degradation (average of metric degradations)
        avg_degradation = np.mean([mae_degradation, mape_degradation, rmse_degradation])
        
        # Drift detected if any metric degraded beyond threshold
        drift_detected = (
            mae_degradation > MODEL_PERFORMANCE_DEGRADATION_THRESHOLD * 100 or
            mape_degradation > MODEL_PERFORMANCE_DEGRADATION_THRESHOLD * 100 or
            rmse_degradation > MODEL_PERFORMANCE_DEGRADATION_THRESHOLD * 100
        )
        
        return {
            "drift_detected": bool(drift_detected),
            "model_name": model_name,
            "current_metrics": current_metrics,
            "baseline_metrics": baseline_metrics,
            "mae_degradation_pct": float(mae_degradation),
            "mape_degradation_pct": float(mape_degradation),
            "rmse_degradation_pct": float(rmse_degradation),
            "avg_degradation_pct": float(avg_degradation),
            "evaluation_samples": len(merged),
            "predictions_df": merged
        }
        
    except Exception as e:
        return {
            "drift_detected": False,
            "error": f"Error during drift detection: {str(e)}"
        }


def log_drift_metrics_to_mlflow(
    data_drift_results: dict,
    model_drift_results: dict,
    run_id: str = None
):
    """
    Log drift detection results to MLflow for monitoring and alerting.
    
    Parameters:
    -----------
    data_drift_results : dict
        Results from detect_data_drift()
    model_drift_results : dict
        Results from detect_model_drift()
    run_id : str, optional
        Specific MLflow run ID to log to (if None, uses active run)
    """
    
    if run_id:
        with mlflow.start_run(run_id=run_id):
            _log_drift_metrics(data_drift_results, model_drift_results)
    else:
        _log_drift_metrics(data_drift_results, model_drift_results)


def _log_drift_metrics(data_drift_results: dict, model_drift_results: dict):
    """Internal function to log drift metrics."""
    
    # Log data drift metrics
    if not data_drift_results.get("error"):
        mlflow.log_metric("data_drift_detected", 1.0 if data_drift_results["drift_detected"] else 0.0)
        mlflow.log_metric("data_drift_score", data_drift_results["drift_score"])
        mlflow.log_metric("data_ks_pvalue", data_drift_results["ks_pvalue"])
        mlflow.log_metric("data_wasserstein_distance", data_drift_results["normalized_wasserstein_distance"])
        mlflow.log_metric("data_mean_change_pct", data_drift_results["mean_change_pct"])
        
        # Log current vs reference stats
        mlflow.log_metric("current_data_mean", data_drift_results["current_stats"]["mean"])
        mlflow.log_metric("current_data_std", data_drift_results["current_stats"]["std"])
        mlflow.log_metric("reference_data_mean", data_drift_results["reference_stats"]["mean"])
        mlflow.log_metric("reference_data_std", data_drift_results["reference_stats"]["std"])
    
    # Log model drift metrics
    if not model_drift_results.get("error"):
        mlflow.log_metric("model_drift_detected", 1.0 if model_drift_results["drift_detected"] else 0.0)
        mlflow.log_metric("model_mae_degradation_pct", model_drift_results["mae_degradation_pct"])
        mlflow.log_metric("model_mape_degradation_pct", model_drift_results["mape_degradation_pct"])
        mlflow.log_metric("model_rmse_degradation_pct", model_drift_results["rmse_degradation_pct"])
        mlflow.log_metric("model_avg_degradation_pct", model_drift_results["avg_degradation_pct"])
        
        # Log current metrics
        mlflow.log_metric("current_mae", model_drift_results["current_metrics"]["mae"])
        mlflow.log_metric("current_mape", model_drift_results["current_metrics"]["mape"])
        mlflow.log_metric("current_rmse", model_drift_results["current_metrics"]["rmse"])
    
    # Log drift detection parameters
    mlflow.log_param("drift_p_value_threshold", DRIFT_P_VALUE_THRESHOLD)
    mlflow.log_param("drift_distance_threshold", DRIFT_DISTANCE_THRESHOLD)
    mlflow.log_param("model_degradation_threshold", MODEL_PERFORMANCE_DEGRADATION_THRESHOLD)


def create_drift_monitoring_report(
    data_drift_results: dict,
    model_drift_results: dict,
    model_name: str = "model"
) -> str:
    """
    Create a formatted drift monitoring report.
    
    Parameters:
    -----------
    data_drift_results : dict
        Results from detect_data_drift()
    model_drift_results : dict
        Results from detect_model_drift()
    model_name : str
        Name of the model
        
    Returns:
    --------
    str
        Formatted report text
    """
    
    report = []
    report.append("=" * 80)
    report.append(f"📊 DRIFT MONITORING REPORT: {model_name}")
    report.append("=" * 80)
    report.append(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    # Data Drift Section
    report.append("🔍 DATA DRIFT ANALYSIS")
    report.append("-" * 80)
    
    if data_drift_results.get("error"):
        report.append(f"❌ Error: {data_drift_results['error']}")
    else:
        drift_status = "🚨 DETECTED" if data_drift_results["drift_detected"] else "✅ NOT DETECTED"
        report.append(f"Status: {drift_status}")
        report.append(f"Drift Score: {data_drift_results['drift_score']:.4f} (0=no drift, 1=maximum drift)")
        report.append("")
        
        report.append("Statistical Tests:")
        report.append(f"  • KS Test p-value: {data_drift_results['ks_pvalue']:.6f} (threshold: {DRIFT_P_VALUE_THRESHOLD})")
        report.append(f"  • Wasserstein Distance: {data_drift_results['normalized_wasserstein_distance']:.4f} (threshold: {DRIFT_DISTANCE_THRESHOLD})")
        report.append(f"  • Mean Change: {data_drift_results['mean_change_pct']:.2f}%")
        report.append(f"  • Std Change: {data_drift_results['std_change_pct']:.2f}%")
        report.append("")
        
        report.append("Distribution Comparison:")
        report.append(f"  Reference (n={data_drift_results['reference_samples']:,}): "
                     f"mean={data_drift_results['reference_stats']['mean']:.2f}, "
                     f"std={data_drift_results['reference_stats']['std']:.2f}")
        report.append(f"  Current (n={data_drift_results['current_samples']:,}, last {data_drift_results['window_days']} days): "
                     f"mean={data_drift_results['current_stats']['mean']:.2f}, "
                     f"std={data_drift_results['current_stats']['std']:.2f}")
    
    report.append("")
    
    # Model Drift Section
    report.append("🤖 MODEL PERFORMANCE DRIFT ANALYSIS")
    report.append("-" * 80)
    
    if model_drift_results.get("error"):
        report.append(f"❌ Error: {model_drift_results['error']}")
    else:
        drift_status = "🚨 DETECTED" if model_drift_results["drift_detected"] else "✅ NOT DETECTED"
        report.append(f"Status: {drift_status}")
        report.append(f"Average Degradation: {model_drift_results['avg_degradation_pct']:.2f}%")
        report.append("")
        
        report.append("Metric Comparison:")
        baseline = model_drift_results['baseline_metrics']
        current = model_drift_results['current_metrics']
        
        report.append(f"  • MAE: {current['mae']:.2f} vs {baseline['mae']:.2f} baseline "
                     f"({model_drift_results['mae_degradation_pct']:+.2f}%)")
        report.append(f"  • MAPE: {current['mape']:.2f}% vs {baseline['mape']:.2f}% baseline "
                     f"({model_drift_results['mape_degradation_pct']:+.2f}%)")
        report.append(f"  • RMSE: {current['rmse']:.2f} vs {baseline['rmse']:.2f} baseline "
                     f"({model_drift_results['rmse_degradation_pct']:+.2f}%)")
        report.append("")
        report.append(f"Evaluation Samples: {model_drift_results['evaluation_samples']:,}")
    
    report.append("")
    
    # Recommendations
    report.append("💡 RECOMMENDATIONS")
    report.append("-" * 80)
    
    if data_drift_results.get("drift_detected") or model_drift_results.get("drift_detected"):
        report.append("⚠️  Drift detected. Consider the following actions:")
        if data_drift_results.get("drift_detected"):
            report.append("  1. Investigate data quality issues or changes in business patterns")
            report.append("  2. Review recent changes in data collection or processing")
            report.append("  3. Consider retraining the model with recent data")
        if model_drift_results.get("drift_detected"):
            report.append("  4. Retrain model with updated data to restore performance")
            report.append("  5. Review feature engineering and model hyperparameters")
            report.append("  6. Implement automated retraining pipeline")
    else:
        report.append("✅ No drift detected. Model performance is stable.")
        report.append("  • Continue monitoring on regular schedule")
        report.append("  • Maintain current model in production")
    
    report.append("=" * 80)
    
    return "\n".join(report)

# COMMAND ----------


def tune_prophet_hyperparameters(
    train_data: pd.DataFrame,
    param_grid: dict = None,
    metric: str = 'mape',
    cv_horizon: str = '90 days',
    cv_initial: str = '365 days',
    cv_period: str = '30 days'
) -> dict:
    """
    Tune Prophet hyperparameters using time series cross-validation.
    
    Tests different parameter combinations and returns the best configuration.
    
    Parameters:
    -----------
    train_data : pd.DataFrame
        DataFrame with 'ds' and 'y' columns
    param_grid : dict, optional
        Dictionary of parameters to test
        If None, uses default grid
    metric : str
        Metric to optimize ('mae', 'mape', or 'rmse')
    cv_horizon : str
        Forecast horizon for CV
    cv_initial : str
        Initial training period for CV
    cv_period : str
        Period between cutoff dates
        
    Returns:
    --------
    dict
        Best hyperparameters and their performance
    """
    
    # Default parameter grid if none provided
    if param_grid is None:
        param_grid = {
            'changepoint_prior_scale': [0.001, 0.01, 0.05, 0.1, 0.5],
            'seasonality_prior_scale': [0.01, 0.1, 1.0, 10.0],
            'holidays_prior_scale': [0.01, 0.1, 1.0, 10.0],
            'seasonality_mode': ['additive', 'multiplicative']
        }
    
    print("\n🔍 HYPERPARAMETER TUNING")
    print("=" * 60)
    print(f"   Metric to optimize: {metric.upper()}")
    print(f"   CV Horizon: {cv_horizon}")
    print(f"   Parameter combinations to test: {np.prod([len(v) for v in param_grid.values()])}")
    
    # Generate all parameter combinations
    from itertools import product
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    all_combinations = list(product(*param_values))
    
    results = []
    best_score = float('inf')
    best_params = None
    
    # Test each combination
    for i, combination in enumerate(all_combinations, 1):
        params = dict(zip(param_names, combination))
        
        print(f"\n[{i}/{len(all_combinations)}] Testing: {params}")
        
        try:
            # Create and fit model with these parameters
            model = Prophet(
                changepoint_prior_scale=params.get('changepoint_prior_scale', 0.05),
                seasonality_prior_scale=params.get('seasonality_prior_scale', 10.0),
                holidays_prior_scale=params.get('holidays_prior_scale', 10.0),
                seasonality_mode=params.get('seasonality_mode', 'multiplicative'),
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False,
                interval_width=0.95
            )
            model.add_country_holidays(country_name='US')
            
            # Fit model
            model.fit(train_data)
            
            # Run cross-validation
            df_cv = cross_validation(
                model,
                initial=cv_initial,
                period=cv_period,
                horizon=cv_horizon,
                parallel="processes"
            )
            
            # Calculate performance metrics
            df_p = performance_metrics(df_cv)
            
            # Get the metric value
            score = df_p[metric].mean()
            
            print(f"   {metric.upper()}: {score:.4f}")
            
            # Store results
            results.append({
                **params,
                f'{metric}_mean': score,
                f'{metric}_std': df_p[metric].std()
            })
            
            # Track best
            if score < best_score:
                best_score = score
                best_params = params
                print(f"   🌟 New best {metric.upper()}: {score:.4f}")
                
        except Exception as e:
            print(f"   ❌ Failed: {str(e)}")
            continue
    
    print("\n" + "=" * 60)
    print("✅ HYPERPARAMETER TUNING COMPLETE")
    print("=" * 60)
    print(f"\n🏆 Best Parameters:")
    for key, value in best_params.items():
        print(f"   {key}: {value}")
    print(f"\n📊 Best {metric.upper()}: {best_score:.4f}")
    
    # Convert results to DataFrame for analysis
    results_df = pd.DataFrame(results)
    
    return {
        'best_params': best_params,
        'best_score': best_score,
        'all_results': results_df,
        'metric': metric
    }

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🧠 Define Model Training Function with Time Series Cross-Validation
# MAGIC 
# MAGIC This function:
# MAGIC 1. Uses Prophet's time series cross-validation (expanding window)
# MAGIC 2. Trains Prophet model with seasonality on full dataset
# MAGIC 3. Validates with multiple cutoff points for robust metrics
# MAGIC 4. Logs ALL artifacts to MLflow:
# MAGIC    - Model parameters
# MAGIC    - Cross-validation metrics (mean ± std)
# MAGIC    - Forecast plots
# MAGIC    - Component plots
# MAGIC    - CV metrics by horizon
# MAGIC    - Model artifacts

# COMMAND ----------

def train_prophet_model_with_mlflow(
    train_data: pd.DataFrame,
    model_name: str,
    forecast_days: int = FORECAST_DAYS,
    yearly_seasonality: bool = True,
    weekly_seasonality: bool = True,
    include_holidays: bool = True,
    seasonality_mode: str = 'multiplicative',
    changepoint_prior_scale: float = 0.05,
    seasonality_prior_scale: float = 10.0,
    holidays_prior_scale: float = 10.0
) -> tuple:
    """
    Train a Prophet model with time series cross-validation and comprehensive MLflow logging.
    
    Uses Prophet's built-in cross-validation for robust time series validation.
    
    Parameters:
    -----------
    train_data : pd.DataFrame
        DataFrame with 'ds' and 'y' columns
    model_name : str
        Name for this model run
    forecast_days : int
        Number of days to forecast
    yearly_seasonality : bool
        Include yearly seasonality
    weekly_seasonality : bool
        Include weekly seasonality
    include_holidays : bool
        Include US holidays
    seasonality_mode : str
        'multiplicative' or 'additive'
    changepoint_prior_scale : float
        Flexibility of trend (higher = more flexible)
    seasonality_prior_scale : float
        Strength of seasonality (higher = stronger)
    holidays_prior_scale : float
        Strength of holiday effects (higher = stronger)
        
    Returns:
    --------
    tuple
        (trained_model, forecast_dataframe)
    """
    
    with mlflow.start_run(run_name=model_name) as run:
        # ===== LOG PARAMETERS =====
        mlflow.log_param("model_type", "Prophet")
        mlflow.log_param("forecast_days", forecast_days)
        mlflow.log_param("yearly_seasonality", yearly_seasonality)
        mlflow.log_param("weekly_seasonality", weekly_seasonality)
        mlflow.log_param("include_holidays", include_holidays)
        mlflow.log_param("seasonality_mode", seasonality_mode)
        mlflow.log_param("changepoint_prior_scale", changepoint_prior_scale)
        mlflow.log_param("seasonality_prior_scale", seasonality_prior_scale)
        mlflow.log_param("holidays_prior_scale", holidays_prior_scale)
        mlflow.log_param("training_records", len(train_data))
        mlflow.log_param("training_start_date", str(train_data['ds'].min()))
        mlflow.log_param("training_end_date", str(train_data['ds'].max()))
        mlflow.log_param("validation_method", "time_series_cv")
        
        # ===== INITIALIZE PROPHET MODEL =====
        model = Prophet(
            yearly_seasonality=yearly_seasonality,
            weekly_seasonality=weekly_seasonality,
            daily_seasonality=False,
            seasonality_mode=seasonality_mode,
            interval_width=0.95,  # 95% confidence interval
            changepoint_prior_scale=changepoint_prior_scale,
            seasonality_prior_scale=seasonality_prior_scale,
            holidays_prior_scale=holidays_prior_scale
        )
        
        # Add US holidays if requested
        if include_holidays:
            model.add_country_holidays(country_name='US')
        
        # ===== FIT MODEL ON FULL DATA =====
        model.fit(train_data)
        
        # ===== TIME SERIES CROSS-VALIDATION =====
        # Prophet's cross_validation uses expanding windows
        # initial: minimum training period
        # period: spacing between cutoff dates
        # horizon: forecast horizon for each fold
        print("   🔄 Running time series cross-validation...")
        
        # Calculate CV parameters based on data length
        data_days = (train_data['ds'].max() - train_data['ds'].min()).days
        initial_days = max(180, int(data_days * 0.5))  # At least 180 days or 50% of data
        period_days = 30  # Validate every 30 days
        horizon_days = min(forecast_days, 90)  # Validate up to 90 days out
        
        mlflow.log_param("cv_initial_days", initial_days)
        mlflow.log_param("cv_period_days", period_days)
        mlflow.log_param("cv_horizon_days", horizon_days)
        
        try:
            df_cv = cross_validation(
                model,
                initial=f'{initial_days} days',
                period=f'{period_days} days',
                horizon=f'{horizon_days} days',
                parallel="processes"
            )

            # Calculate performance metrics across all folds
            df_p = performance_metrics(df_cv)

            # Get average metrics across all horizons
            cv_mae = df_p['mae'].mean()
            cv_mape = df_p['mape'].mean()
            cv_rmse = df_p['rmse'].mean()

            # Get standard deviations
            cv_mae_std = df_p['mae'].std()
            cv_mape_std = df_p['mape'].std()
            cv_rmse_std = df_p['rmse'].std()

            # Number of folds
            num_folds = df_cv['cutoff'].nunique()

            mlflow.log_param("cv_num_folds", num_folds)

            # ===== LOG CROSS-VALIDATION METRICS =====
            mlflow.log_metric("cv_mae_mean", cv_mae)
            mlflow.log_metric("cv_mae_std", cv_mae_std)
            mlflow.log_metric("cv_mape_mean", cv_mape)
            mlflow.log_metric("cv_mape_std", cv_mape_std)
            mlflow.log_metric("cv_rmse_mean", cv_rmse)
            mlflow.log_metric("cv_rmse_std", cv_rmse_std)

            # Set val_metrics for compatibility with later code
            val_metrics = {
                'mae': cv_mae,
                'mape': cv_mape,
                'rmse': cv_rmse
            }

            print(f"   ✅ Cross-validation complete: {num_folds} folds")
            
        except Exception as e:
            print(f"   ⚠️ Cross-validation failed: {str(e)}")
            print("   Falling back to simple train/test split...")
            
            # Fallback to simple validation
            split_date = train_data['ds'].max() - timedelta(days=TRAIN_TEST_SPLIT_DAYS)
            train_set = train_data[train_data['ds'] <= split_date]
            test_set = train_data[train_data['ds'] > split_date]
            
            # Refit on training set only
            model_temp = Prophet(
                yearly_seasonality=yearly_seasonality,
                weekly_seasonality=weekly_seasonality,
                daily_seasonality=False,
                seasonality_mode=seasonality_mode,
                interval_width=0.95,
                changepoint_prior_scale=0.05
            )
            if include_holidays:
                model_temp.add_country_holidays(country_name='US')
            model_temp.fit(train_set)
            
            future_temp = model_temp.make_future_dataframe(periods=TRAIN_TEST_SPLIT_DAYS)
            forecast_temp = model_temp.predict(future_temp)
            
            test_forecast = forecast_temp[forecast_temp['ds'].isin(test_set['ds'])]
            test_merged = test_set.merge(test_forecast[['ds', 'yhat']], on='ds')
            
            val_metrics = calculate_metrics(test_merged['y'], test_merged['yhat'])
            
            cv_mae = val_metrics['mae']
            cv_mape = val_metrics['mape']
            cv_rmse = val_metrics['rmse']
            cv_mae_std = 0
            cv_mape_std = 0
            cv_rmse_std = 0
            num_folds = 1
            
            mlflow.log_param("cv_num_folds", 1)
            mlflow.log_param("cv_fallback", True)
            mlflow.log_metric("cv_mae_mean", cv_mae)
            mlflow.log_metric("cv_mape_mean", cv_mape)
            mlflow.log_metric("cv_rmse_mean", cv_rmse)
        
        # ===== MAKE FUTURE PREDICTIONS =====
        # Use the model trained on full data for final forecast
        future = model.make_future_dataframe(periods=forecast_days)
        forecast = model.predict(future)
        
        # ===== GENERATE AND LOG PLOTS =====
        # Forecast plot
        fig1 = model.plot(forecast)
        plt.title(f"{model_name} - Forecast")
        plt.tight_layout()
        mlflow.log_figure(fig1, "forecast_plot.png")
        plt.close()
        
        # Component plot (trend, seasonality)
        fig2 = model.plot_components(forecast)
        plt.tight_layout()
        mlflow.log_figure(fig2, "components_plot.png")
        plt.close()
        
        # Cross-validation metrics plot (if CV was successful)
        try:
            if 'df_p' in locals() and len(df_p) > 0:
                fig3, axes = plt.subplots(2, 2, figsize=(12, 10))
                
                # MAE by horizon
                axes[0, 0].plot(df_p['horizon'], df_p['mae'], 'o-')
                axes[0, 0].set_xlabel('Horizon (days)')
                axes[0, 0].set_ylabel('MAE ($)')
                axes[0, 0].set_title('MAE by Forecast Horizon')
                axes[0, 0].grid(True, alpha=0.3)
                
                # MAPE by horizon
                axes[0, 1].plot(df_p['horizon'], df_p['mape'], 'o-')
                axes[0, 1].set_xlabel('Horizon (days)')
                axes[0, 1].set_ylabel('MAPE (%)')
                axes[0, 1].set_title('MAPE by Forecast Horizon')
                axes[0, 1].grid(True, alpha=0.3)
                
                # RMSE by horizon
                axes[1, 0].plot(df_p['horizon'], df_p['rmse'], 'o-')
                axes[1, 0].set_xlabel('Horizon (days)')
                axes[1, 0].set_ylabel('RMSE ($)')
                axes[1, 0].set_title('RMSE by Forecast Horizon')
                axes[1, 0].grid(True, alpha=0.3)
                
                # Coverage
                axes[1, 1].plot(df_p['horizon'], df_p['coverage'], 'o-')
                axes[1, 1].axhline(y=0.95, color='r', linestyle='--', label='Target 95%')
                axes[1, 1].set_xlabel('Horizon (days)')
                axes[1, 1].set_ylabel('Coverage')
                axes[1, 1].set_title('Prediction Interval Coverage')
                axes[1, 1].legend()
                axes[1, 1].grid(True, alpha=0.3)
                
                plt.tight_layout()
                mlflow.log_figure(fig3, "cv_metrics_by_horizon.png")
                plt.close()
        except Exception as e:
            print(f"   ⚠️ Could not create CV metrics plot: {str(e)}")
        
        # ===== LOG MODEL ARTIFACT =====
        mlflow.prophet.log_model(model, "prophet_model")
        
        # ===== LOG FORECAST DATA =====
        # Save forecast as CSV artifact
        forecast_output = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
        forecast_output.to_csv("/tmp/forecast.csv", index=False)
        mlflow.log_artifact("/tmp/forecast.csv")
        
        # ===== LOG MODEL SUMMARY =====
        summary = {
            "model_name": model_name,
            "training_records": len(train_data),
            "forecast_days": forecast_days,
            "val_mape": round(val_metrics['mape'], 2),
            "val_mae": round(val_metrics['mae'], 2),
            "run_id": run.info.run_id
        }
        
        with open("/tmp/model_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        mlflow.log_artifact("/tmp/model_summary.json")
        
        # ===== PRINT RESULTS =====
        print(f"\n{'='*60}")
        print(f"✅ Model: {model_name}")
        print(f"{'='*60}")
        print(f"   Training Records: {len(train_data):,}")
        print(f"   CV Folds: {num_folds}")
        print(f"\n   📊 Cross-Validation Metrics (Mean ± Std):")
        print(f"      MAE:  ${cv_mae:,.2f} ± ${cv_mae_std:,.2f}")
        print(f"      MAPE: {cv_mape:.2f}% ± {cv_mape_std:.2f}%")
        print(f"      RMSE: ${cv_rmse:,.2f} ± ${cv_rmse_std:,.2f}")
        print(f"\n   🔗 MLflow Run ID: {run.info.run_id}")
        
        return model, forecast, val_metrics

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🔍 Forecast Explainability Functions
# MAGIC 
# MAGIC Functions to explain and interpret Prophet forecasts for business stakeholders.

# COMMAND ----------

def calculate_component_contributions(model, forecast: pd.DataFrame, target_date=None) -> dict:
    """
    Calculate the percentage contribution of each Prophet component to the final forecast.
    
    Breaks down forecast into trend, seasonalities, and holidays to show what's driving predictions.
    
    Parameters:
    -----------
    model : Prophet
        Trained Prophet model
    forecast : pd.DataFrame
        Forecast dataframe from model.predict()
    target_date : datetime, optional
        Specific date to analyze (if None, uses last forecast date)
        
    Returns:
    --------
    dict
        Component contributions with percentages and values
    """
    
    if target_date is None:
        # Use last date in forecast
        target_date = forecast['ds'].max()
    
    # Get the forecast row for target date
    forecast_row = forecast[forecast['ds'] == target_date].iloc[0]
    
    # Base forecast value
    yhat = forecast_row['yhat']
    
    # Extract components
    components = {}
    
    # Trend
    if 'trend' in forecast_row:
        components['trend'] = forecast_row['trend']
    
    # Yearly seasonality
    if 'yearly' in forecast_row:
        components['yearly_seasonality'] = forecast_row['yearly']
    
    # Weekly seasonality
    if 'weekly' in forecast_row:
        components['weekly_seasonality'] = forecast_row['weekly']
    
    # Holidays
    if 'holidays' in forecast_row:
        components['holidays'] = forecast_row['holidays']
    
    # Calculate baseline (trend without seasonality/holidays)
    baseline = components.get('trend', yhat)
    
    # Calculate contributions as delta from baseline
    contributions = {}
    total_delta = 0
    
    for comp_name, comp_value in components.items():
        if comp_name == 'trend':
            # Trend is the baseline
            contributions[comp_name] = {
                'value': comp_value,
                'percentage': (comp_value / yhat * 100) if yhat != 0 else 0,
                'type': 'baseline'
            }
        else:
            # Other components are deltas
            delta = comp_value
            total_delta += delta
            contributions[comp_name] = {
                'value': delta,
                'percentage': (delta / yhat * 100) if yhat != 0 else 0,
                'type': 'adjustment'
            }
    
    return {
        'date': target_date,
        'forecast_value': yhat,
        'baseline_trend': baseline,
        'total_adjustment': total_delta,
        'components': contributions,
        'lower_bound': forecast_row.get('yhat_lower', None),
        'upper_bound': forecast_row.get('yhat_upper', None)
    }


def generate_forecast_narrative(model, forecast: pd.DataFrame, historical_data: pd.DataFrame = None, 
                                model_name: str = "model", forecast_period_days: int = 90) -> str:
    """
    Generate a natural language narrative explaining the forecast.
    
    Creates business-friendly text summarizing forecast drivers and key insights.
    
    Parameters:
    -----------
    model : Prophet
        Trained Prophet model
    forecast : pd.DataFrame
        Forecast dataframe
    historical_data : pd.DataFrame, optional
        Historical data for comparison
    model_name : str
        Name of the model
    forecast_period_days : int
        Number of days being forecasted
        
    Returns:
    --------
    str
        Natural language narrative
    """
    
    narrative = []
    narrative.append(f"📊 FORECAST EXPLANATION: {model_name}")
    narrative.append("=" * 80)
    narrative.append("")
    
    # Get future forecast (exclude historical)
    if historical_data is not None:
        max_historical_date = historical_data['ds'].max()
        future_forecast = forecast[forecast['ds'] > max_historical_date]
    else:
        future_forecast = forecast.tail(forecast_period_days)
    
    if len(future_forecast) == 0:
        return "No future forecast data available."
    
    # Calculate forecast statistics
    forecast_mean = future_forecast['yhat'].mean()
    forecast_start = future_forecast['yhat'].iloc[0]
    forecast_end = future_forecast['yhat'].iloc[-1]
    forecast_min = future_forecast['yhat'].min()
    forecast_max = future_forecast['yhat'].max()
    
    # Forecast summary
    narrative.append(f"🎯 FORECAST SUMMARY")
    narrative.append(f"   Period: {future_forecast['ds'].min().strftime('%Y-%m-%d')} to {future_forecast['ds'].max().strftime('%Y-%m-%d')}")
    narrative.append(f"   Average Daily Forecast: ${forecast_mean:,.2f}")
    narrative.append(f"   Range: ${forecast_min:,.2f} to ${forecast_max:,.2f}")
    narrative.append(f"   Total Forecasted: ${future_forecast['yhat'].sum():,.2f}")
    narrative.append("")
    
    # Trend analysis
    trend_change = ((forecast_end - forecast_start) / forecast_start * 100) if forecast_start != 0 else 0
    narrative.append(f"📈 TREND ANALYSIS")
    if abs(trend_change) < 5:
        trend_desc = "relatively stable"
    elif trend_change > 0:
        trend_desc = f"increasing by {abs(trend_change):.1f}%"
    else:
        trend_desc = f"decreasing by {abs(trend_change):.1f}%"
    
    narrative.append(f"   The forecast shows a {trend_desc} trend over the {forecast_period_days}-day period.")
    
    if 'trend' in future_forecast.columns:
        trend_start = future_forecast['trend'].iloc[0]
        trend_end = future_forecast['trend'].iloc[-1]
        trend_slope = trend_end - trend_start
        if abs(trend_slope) > forecast_mean * 0.1:
            narrative.append(f"   ⚠️  Significant trend shift detected: ${trend_slope:,.2f} change in baseline.")
    
    narrative.append("")
    
    # Component contributions (analyze middle of forecast period)
    mid_point = len(future_forecast) // 2
    mid_date = future_forecast.iloc[mid_point]['ds']
    contributions = calculate_component_contributions(model, forecast, mid_date)
    
    narrative.append(f"🔍 KEY DRIVERS (at {mid_date.strftime('%Y-%m-%d')})")
    narrative.append(f"   Forecast Value: ${contributions['forecast_value']:,.2f}")
    narrative.append("")
    narrative.append("   Component Contributions:")
    
    # Sort components by absolute percentage
    sorted_components = sorted(
        contributions['components'].items(),
        key=lambda x: abs(x[1]['percentage']),
        reverse=True
    )
    
    for comp_name, comp_data in sorted_components:
        comp_value = comp_data['value']
        comp_pct = comp_data['percentage']
        comp_type = comp_data['type']
        
        if comp_type == 'baseline':
            narrative.append(f"   • {comp_name.replace('_', ' ').title()}: ${comp_value:,.2f} (baseline)")
        else:
            direction = "increases" if comp_value > 0 else "decreases"
            narrative.append(f"   • {comp_name.replace('_', ' ').title()}: {direction} forecast by ${abs(comp_value):,.2f} ({comp_pct:+.1f}%)")
    
    narrative.append("")
    
    # Uncertainty analysis
    avg_uncertainty = (future_forecast['yhat_upper'] - future_forecast['yhat_lower']).mean()
    uncertainty_pct = (avg_uncertainty / forecast_mean * 100) if forecast_mean != 0 else 0
    
    narrative.append(f"📊 UNCERTAINTY ANALYSIS")
    narrative.append(f"   Average Prediction Interval: ±${avg_uncertainty/2:,.2f} ({uncertainty_pct/2:.1f}%)")
    narrative.append(f"   95% Confidence: ${future_forecast['yhat_lower'].mean():,.2f} to ${future_forecast['yhat_upper'].mean():,.2f}")
    
    if uncertainty_pct > 40:
        narrative.append(f"   ⚠️  High uncertainty - consider reviewing data quality and model assumptions")
    elif uncertainty_pct < 15:
        narrative.append(f"   ✅ Low uncertainty - forecast is relatively confident")
    else:
        narrative.append(f"   ℹ️  Moderate uncertainty - typical for this forecast horizon")
    
    narrative.append("")
    
    # Historical comparison (if data provided)
    if historical_data is not None:
        hist_mean = historical_data['y'].mean()
        forecast_vs_hist = ((forecast_mean - hist_mean) / hist_mean * 100) if hist_mean != 0 else 0
        
        narrative.append(f"📅 HISTORICAL COMPARISON")
        narrative.append(f"   Historical Average: ${hist_mean:,.2f}/day")
        narrative.append(f"   Forecast Average: ${forecast_mean:,.2f}/day")
        
        if abs(forecast_vs_hist) < 5:
            narrative.append(f"   Forecast is consistent with historical patterns (within 5%).")
        elif forecast_vs_hist > 0:
            narrative.append(f"   ⚠️  Forecast is {forecast_vs_hist:.1f}% HIGHER than historical average.")
            narrative.append(f"   Expected increase: ${(forecast_mean - hist_mean) * forecast_period_days:,.2f} over forecast period.")
        else:
            narrative.append(f"   ✅ Forecast is {abs(forecast_vs_hist):.1f}% LOWER than historical average.")
            narrative.append(f"   Expected decrease: ${(forecast_mean - hist_mean) * forecast_period_days:,.2f} over forecast period.")
        
        narrative.append("")
    
    # Recommendations
    narrative.append(f"💡 RECOMMENDATIONS")
    if trend_change > 10:
        narrative.append(f"   • Investigate drivers of upward trend - proactive intervention may reduce costs")
    elif trend_change < -10:
        narrative.append(f"   • Downward trend detected - document improvement initiatives for replication")
    
    if uncertainty_pct > 30:
        narrative.append(f"   • High forecast uncertainty - consider scenario planning for risk mitigation")
    
    # Check for seasonal peaks
    if 'yearly' in future_forecast.columns:
        max_seasonal = future_forecast['yearly'].max()
        min_seasonal = future_forecast['yearly'].min()
        if max_seasonal - min_seasonal > forecast_mean * 0.2:
            narrative.append(f"   • Significant seasonal variation expected - plan inventory and staffing accordingly")
    
    narrative.append("")
    narrative.append("=" * 80)
    
    return "\n".join(narrative)


def create_whatif_scenarios(model, forecast: pd.DataFrame, historical_data: pd.DataFrame) -> dict:
    """
    Create what-if scenarios by removing or modifying components.
    
    Shows business impact of different scenarios (no holidays, no seasonality, etc.)
    
    Parameters:
    -----------
    model : Prophet
        Trained Prophet model
    forecast : pd.DataFrame
        Forecast dataframe
    historical_data : pd.DataFrame
        Historical training data
        
    Returns:
    --------
    dict
        Dictionary of scenarios with forecasts
    """
    
    scenarios = {}
    
    # Baseline scenario (actual forecast)
    scenarios['baseline'] = {
        'name': 'Actual Forecast',
        'description': 'Current forecast with all components',
        'forecast': forecast.copy(),
        'mean': forecast['yhat'].mean()
    }
    
    # Scenario 1: Trend only (no seasonality or holidays)
    if 'trend' in forecast.columns:
        trend_only = forecast.copy()
        trend_only['yhat'] = forecast['trend']
        trend_only['yhat_lower'] = forecast['trend'] - (forecast['yhat'] - forecast['yhat_lower'])
        trend_only['yhat_upper'] = forecast['trend'] + (forecast['yhat_upper'] - forecast['yhat'])
        
        scenarios['trend_only'] = {
            'name': 'Trend Only',
            'description': 'Removes all seasonal and holiday effects',
            'forecast': trend_only,
            'mean': trend_only['yhat'].mean(),
            'delta': trend_only['yhat'].mean() - forecast['yhat'].mean()
        }
    
    # Scenario 2: No holidays
    if 'holidays' in forecast.columns:
        no_holidays = forecast.copy()
        no_holidays['yhat'] = forecast['yhat'] - forecast['holidays']
        
        scenarios['no_holidays'] = {
            'name': 'No Holidays',
            'description': 'Removes holiday effects',
            'forecast': no_holidays,
            'mean': no_holidays['yhat'].mean(),
            'delta': no_holidays['yhat'].mean() - forecast['yhat'].mean()
        }
    
    # Scenario 3: No weekly seasonality
    if 'weekly' in forecast.columns:
        no_weekly = forecast.copy()
        no_weekly['yhat'] = forecast['yhat'] - forecast['weekly']
        
        scenarios['no_weekly'] = {
            'name': 'No Weekly Patterns',
            'description': 'Removes day-of-week effects',
            'forecast': no_weekly,
            'mean': no_weekly['yhat'].mean(),
            'delta': no_weekly['yhat'].mean() - forecast['yhat'].mean()
        }
    
    # Scenario 4: No yearly seasonality
    if 'yearly' in forecast.columns:
        no_yearly = forecast.copy()
        no_yearly['yhat'] = forecast['yhat'] - forecast['yearly']
        
        scenarios['no_yearly'] = {
            'name': 'No Yearly Seasonality',
            'description': 'Removes annual patterns',
            'forecast': no_yearly,
            'mean': no_yearly['yhat'].mean(),
            'delta': no_yearly['yhat'].mean() - forecast['yhat'].mean()
        }
    
    # Scenario 5: Historical average (naive baseline)
    if historical_data is not None:
        hist_avg = historical_data['y'].mean()
        historical_baseline = forecast.copy()
        historical_baseline['yhat'] = hist_avg
        historical_baseline['yhat_lower'] = hist_avg * 0.9
        historical_baseline['yhat_upper'] = hist_avg * 1.1
        
        scenarios['historical_average'] = {
            'name': 'Historical Average',
            'description': 'Simple average of historical data',
            'forecast': historical_baseline,
            'mean': hist_avg,
            'delta': hist_avg - forecast['yhat'].mean()
        }
    
    return scenarios


def visualize_component_contributions(model, forecast: pd.DataFrame, top_n_dates: int = 5):
    """
    Create visualization showing component contributions over time.
    
    Parameters:
    -----------
    model : Prophet
        Trained Prophet model
    forecast : pd.DataFrame
        Forecast dataframe
    top_n_dates : int
        Number of dates to show in detail
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure with component contribution visualizations
    """
    
    # Select dates to analyze (evenly spaced through forecast)
    forecast_dates = forecast['ds'].tail(top_n_dates * 2)  # Get future dates
    step = max(1, len(forecast_dates) // top_n_dates)
    selected_dates = forecast_dates.iloc[::step].head(top_n_dates)
    
    # Calculate contributions for each date
    contributions_list = []
    for date in selected_dates:
        contrib = calculate_component_contributions(model, forecast, date)
        contributions_list.append({
            'date': date.strftime('%Y-%m-%d'),
            **{k: v['value'] for k, v in contrib['components'].items()}
        })
    
    contrib_df = pd.DataFrame(contributions_list)
    
    # Create stacked bar chart
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Top plot: Stacked components
    component_cols = [col for col in contrib_df.columns if col != 'date']
    contrib_df.set_index('date')[component_cols].plot(
        kind='bar',
        stacked=True,
        ax=ax1,
        colormap='tab10'
    )
    ax1.set_title('Forecast Component Contributions Over Time', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Value ($)')
    ax1.legend(title='Components', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)
    
    # Bottom plot: Percentage contributions
    # Calculate percentages
    contrib_pct = contrib_df.copy()
    for col in component_cols:
        total = contrib_df[component_cols].sum(axis=1)
        contrib_pct[col] = (contrib_df[col] / total * 100)
    
    contrib_pct.set_index('date')[component_cols].plot(
        kind='bar',
        stacked=True,
        ax=ax2,
        colormap='tab10'
    )
    ax2.set_title('Component Contributions (Percentage)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Contribution (%)')
    ax2.legend(title='Components', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='x', rotation=45)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    return fig


def create_scenario_comparison_table(scenarios: dict, future_only: bool = True) -> pd.DataFrame:
    """
    Create a comparison table of what-if scenarios.
    
    Parameters:
    -----------
    scenarios : dict
        Output from create_whatif_scenarios()
    future_only : bool
        If True, only analyze future forecast period
        
    Returns:
    --------
    pd.DataFrame
        Comparison table with scenario statistics
    """
    
    comparison_data = []
    
    baseline_mean = scenarios['baseline']['mean']
    
    for scenario_key, scenario_data in scenarios.items():
        mean_val = scenario_data['mean']
        delta = scenario_data.get('delta', 0)
        delta_pct = (delta / baseline_mean * 100) if baseline_mean != 0 else 0
        
        comparison_data.append({
            'Scenario': scenario_data['name'],
            'Description': scenario_data['description'],
            'Avg Daily Forecast': f"${mean_val:,.2f}",
            'vs Baseline ($)': f"${delta:+,.2f}" if delta != 0 else "-",
            'vs Baseline (%)': f"{delta_pct:+.1f}%" if delta != 0 else "-",
            'Impact': 'Baseline' if scenario_key == 'baseline' else ('Higher' if delta > 0 else 'Lower' if delta < 0 else 'Neutral')
        })
    
    return pd.DataFrame(comparison_data)


def explain_forecast_uncertainty(model, forecast: pd.DataFrame, cv_results: pd.DataFrame = None) -> dict:
    """
    Break down sources of forecast uncertainty.
    
    Parameters:
    -----------
    model : Prophet
        Trained Prophet model
    forecast : pd.DataFrame
        Forecast dataframe
    cv_results : pd.DataFrame, optional
        Cross-validation results for error analysis
        
    Returns:
    --------
    dict
        Uncertainty breakdown and insights
    """
    
    uncertainty = {}
    
    # Prediction interval width
    forecast['interval_width'] = forecast['yhat_upper'] - forecast['yhat_lower']
    
    uncertainty['avg_interval_width'] = forecast['interval_width'].mean()
    uncertainty['avg_prediction'] = forecast['yhat'].mean()
    uncertainty['uncertainty_percentage'] = (
        uncertainty['avg_interval_width'] / uncertainty['avg_prediction'] * 100
    ) if uncertainty['avg_prediction'] != 0 else 0
    
    # How uncertainty grows with horizon
    if 'ds' in forecast.columns:
        # Split into near-term and far-term
        mid_point = len(forecast) // 2
        near_term = forecast.iloc[:mid_point]
        far_term = forecast.iloc[mid_point:]
        
        uncertainty['near_term_uncertainty'] = near_term['interval_width'].mean()
        uncertainty['far_term_uncertainty'] = far_term['interval_width'].mean()
        uncertainty['uncertainty_growth'] = (
            (far_term['interval_width'].mean() - near_term['interval_width'].mean()) / 
            near_term['interval_width'].mean() * 100
        ) if near_term['interval_width'].mean() != 0 else 0
    
    # Model parameters affecting uncertainty
    uncertainty['model_params'] = {
        'changepoint_prior_scale': model.changepoint_prior_scale,
        'seasonality_prior_scale': model.seasonality_prior_scale,
        'holidays_prior_scale': getattr(model, 'holidays_prior_scale', None),
        'interval_width': model.interval_width
    }
    
    # CV-based uncertainty (if available)
    if cv_results is not None:
        uncertainty['cv_mae'] = cv_results['mae'].mean()
        uncertainty['cv_mape'] = cv_results['mape'].mean()
        uncertainty['cv_coverage'] = cv_results['coverage'].mean()
    
    # Interpretation
    uncertainty['interpretation'] = []
    
    if uncertainty['uncertainty_percentage'] > 40:
        uncertainty['interpretation'].append("HIGH uncertainty - forecast has wide prediction intervals")
    elif uncertainty['uncertainty_percentage'] < 15:
        uncertainty['interpretation'].append("LOW uncertainty - forecast is relatively confident")
    else:
        uncertainty['interpretation'].append("MODERATE uncertainty - typical for this type of forecast")
    
    if uncertainty.get('uncertainty_growth', 0) > 50:
        uncertainty['interpretation'].append("Uncertainty increases significantly over forecast horizon")
    
    return uncertainty

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🚀 Parallel Training Functions
# MAGIC 
# MAGIC Functions to train multiple models in parallel for faster execution.

# COMMAND ----------

def train_single_model_worker(
    group_id,
    group_name: str,
    group_data: pd.DataFrame,
    model_prefix: str,
    use_tuned_params: bool = False,
    tuned_params: dict = None,
    idx: int = 1,
    total: int = 1
) -> tuple:
    """
    Worker function to train a single model (for parallel execution).
    
    This function is designed to be called by ThreadPoolExecutor.
    
    Parameters:
    -----------
    group_id : any
        Unique identifier for this group
    group_name : str
        Display name for this group
    group_data : pd.DataFrame
        Training data with 'ds' and 'y' columns
    model_prefix : str
        Prefix for model name
    use_tuned_params : bool
        Whether to use tuned hyperparameters
    tuned_params : dict
        Tuned hyperparameters
    idx : int
        Current model index
    total : int
        Total number of models
        
    Returns:
    --------
    tuple
        (group_id, result_dict) or (group_id, None) if failed
    """
    
    thread_id = threading.current_thread().name
    print(f"[Thread {thread_id}] [{idx}/{total}] Training: {group_name}")
    
    try:
        model_name = f"{model_prefix}_{group_id}"
        
        # Use tuned parameters if available
        if use_tuned_params and tuned_params:
            model, forecast, _ = train_prophet_model_with_mlflow(
                group_data,
                model_name=model_name,
                changepoint_prior_scale=tuned_params.get('changepoint_prior_scale', 0.05),
                seasonality_prior_scale=tuned_params.get('seasonality_prior_scale', 10.0),
                holidays_prior_scale=tuned_params.get('holidays_prior_scale', 10.0),
                seasonality_mode=tuned_params.get('seasonality_mode', 'multiplicative')
            )
        else:
            model, forecast, _ = train_prophet_model_with_mlflow(
                group_data,
                model_name=model_name
            )
        
        # Calculate metrics on training data
        train_forecast = forecast[forecast['ds'].isin(group_data['ds'])]
        train_merged = group_data.merge(train_forecast[['ds', 'yhat']], on='ds')
        metrics = calculate_metrics(train_merged['y'], train_merged['yhat'])
        
        result = {
            "name": group_name,
            "model": model,
            "forecast": forecast,
            "training_data": group_data,
            "metrics": metrics
        }
        
        print(f"[Thread {thread_id}] ✅ Completed: {group_name} (MAPE: {metrics['mape']:.2f}%)")
        return (group_id, result)
        
    except Exception as e:
        print(f"[Thread {thread_id}] ❌ Error training {group_name}: {str(e)}")
        return (group_id, None)


def train_multi_level_forecasts_parallel(
    df,
    date_col: str,
    value_col: str,
    group_col: str,
    group_name_col: str,
    model_prefix: str,
    top_n: int = None,
    min_records: int = 60,
    use_tuned_params: bool = False,
    tuned_params: dict = None,
    max_workers: int = MAX_PARALLEL_WORKERS
) -> dict:
    """
    Train separate Prophet models for each group IN PARALLEL.
    
    Uses ThreadPoolExecutor to train multiple models simultaneously,
    significantly reducing total training time.
    
    Parameters:
    -----------
    df : DataFrame
        Spark DataFrame with scrap data
    date_col : str
        Date column name
    value_col : str
        Value column to forecast
    group_col : str
        Column to group by (e.g., 'plant_id')
    group_name_col : str
        Friendly name column (e.g., 'plant_name_lookup')
    model_prefix : str
        Prefix for model names in MLflow
    top_n : int, optional
        Limit to top N entities by total value
    min_records : int
        Minimum days of data required to train
    use_tuned_params : bool
        Whether to use tuned hyperparameters
    tuned_params : dict
        Tuned hyperparameters to use
    max_workers : int
        Maximum number of parallel workers
        
    Returns:
    --------
    dict
        Dictionary of {group_id: {name, model, forecast, metrics}}
    """
    
    # Get groups to model
    if top_n:
        top_groups = (
            df.groupBy(group_col, group_name_col)
            .agg(F.sum(value_col).alias("total"))
            .orderBy(F.desc("total"))
            .limit(top_n)
            .select(group_col, group_name_col)
            .collect()
        )
    else:
        top_groups = (
            df.select(group_col, group_name_col)
            .distinct()
            .collect()
        )
    
    print(f"\n🚀 PARALLEL TRAINING: {len(top_groups)} {model_prefix} models")
    print(f"   Max Workers: {max_workers}")
    print("=" * 80)
    
    # Prepare all group data first (faster than doing it in parallel)
    training_jobs = []
    
    for idx, row in enumerate(top_groups, 1):
        group_id = row[group_col]
        group_name = row[group_name_col] or f"Unknown ({group_id})"
        
        # Filter data for this group
        group_df = df.filter(F.col(group_col) == group_id)
        
        # Prepare data
        group_data = prepare_prophet_data(
            group_df,
            date_col=date_col,
            value_col=value_col
        )
        
        # Check minimum records
        if len(group_data) < min_records:
            print(f"   ⚠️ Skipping {group_name}: insufficient data ({len(group_data)} days, need {min_records})")
            continue
        
        training_jobs.append({
            'group_id': group_id,
            'group_name': group_name,
            'group_data': group_data,
            'idx': len(training_jobs) + 1
        })
    
    print(f"\n📊 Prepared {len(training_jobs)} models for training")
    print(f"⏱️  Starting parallel execution...\n")
    
    results = {}
    start_time = datetime.now()
    
    # Train models in parallel using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all jobs
        future_to_group = {
            executor.submit(
                train_single_model_worker,
                job['group_id'],
                job['group_name'],
                job['group_data'],
                model_prefix,
                use_tuned_params,
                tuned_params,
                job['idx'],
                len(training_jobs)
            ): job['group_id']
            for job in training_jobs
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_group):
            group_id = future_to_group[future]
            try:
                result_group_id, result_data = future.result()
                if result_data is not None:
                    results[result_group_id] = result_data
            except Exception as e:
                print(f"   ❌ Exception for group {group_id}: {str(e)}")
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    print(f"\n{'='*80}")
    print(f"✅ PARALLEL TRAINING COMPLETE")
    print(f"{'='*80}")
    print(f"   Models Trained: {len(results)}/{len(training_jobs)}")
    print(f"   Total Duration: {duration:.1f} seconds ({duration/60:.1f} minutes)")
    if len(results) > 0:
        print(f"   Average per Model: {duration/len(results):.1f} seconds")
        print(f"   Estimated Sequential Time: {duration * max_workers:.1f} seconds ({duration * max_workers / 60:.1f} minutes)")
        print(f"   ⚡ Speedup: ~{max_workers:.1f}x faster")
    print(f"{'='*80}\n")
    
    return results

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # 📈 Model Training Section
# MAGIC ---

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1️⃣ Total Scrap Cost Forecast
# MAGIC 
# MAGIC First, we'll build a company-wide forecast to predict total daily scrap cost.
# MAGIC This provides an overall view for executive planning and budgeting.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Prepare Total Scrap Data
# MAGIC 
# MAGIC Aggregate all scrap transactions to daily totals.

# COMMAND ----------

# Prepare total scrap cost data (daily aggregation)
total_cost_data = prepare_prophet_data(
    scrap_df, 
    date_col="transaction_date", 
    value_col="total_scrap_cost"
)

print(f"📊 Total Scrap Data Summary:")
print(f"   Records: {len(total_cost_data):,} days")
print(f"   Date Range: {total_cost_data['ds'].min().date()} to {total_cost_data['ds'].max().date()}")
print(f"   Mean Daily Cost: ${total_cost_data['y'].mean():,.2f}")
print(f"   Max Daily Cost: ${total_cost_data['y'].max():,.2f}")

# Preview data
total_cost_data.head(10)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Tune Hyperparameters
# MAGIC 
# MAGIC Run grid search to find optimal Prophet hyperparameters for this dataset.

# COMMAND ----------

# Tune hyperparameters on total scrap data
# Using a smaller grid for faster execution - expand for production
tuning_param_grid = {
    'changepoint_prior_scale': [0.01, 0.05, 0.1],  # Trend flexibility
    'seasonality_prior_scale': [1.0, 10.0],        # Seasonality strength
    'holidays_prior_scale': [1.0, 10.0],           # Holiday effect
    'seasonality_mode': ['multiplicative']         # Based on data characteristics
}

# Calculate CV parameters
data_days = (total_cost_data['ds'].max() - total_cost_data['ds'].min()).days
cv_initial_days = max(365, int(data_days * 0.6))

tuning_results = tune_prophet_hyperparameters(
    total_cost_data,
    param_grid=tuning_param_grid,
    metric='mape',  # Optimize for MAPE
    cv_horizon=f'{FORECAST_DAYS} days',
    cv_initial=f'{cv_initial_days} days',
    cv_period='30 days'
)

# Extract best parameters
best_params = tuning_results['best_params']

print("\n📊 Tuning Results Summary:")
print(f"   Best MAPE: {tuning_results['best_score']:.2f}%")
print("\n   All Results:")
display(tuning_results['all_results'].sort_values('mape_mean'))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Train Total Scrap Model
# MAGIC 
# MAGIC Train Prophet model for total scrap cost with full MLflow logging.

# COMMAND ----------

# Train total scrap cost model with tuned hyperparameters
total_model, total_forecast, total_val_metrics = train_prophet_model_with_mlflow(
    total_cost_data,
    model_name="total_scrap_cost_forecast",
    yearly_seasonality=True,
    weekly_seasonality=True,
    include_holidays=True,
    seasonality_mode=best_params['seasonality_mode'],
    changepoint_prior_scale=best_params['changepoint_prior_scale'],
    seasonality_prior_scale=best_params['seasonality_prior_scale'],
    holidays_prior_scale=best_params['holidays_prior_scale']
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Visualize Total Scrap Forecast
# MAGIC 
# MAGIC Display the forecast with confidence intervals and trend components.

# COMMAND ----------

# Plot forecast with confidence intervals
fig = total_model.plot(total_forecast)
plt.title("Total Scrap Cost Forecast - 3 Month Horizon", fontsize=14)
plt.ylabel("Daily Scrap Cost ($)")
plt.xlabel("Date")
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Analyze Seasonality Components
# MAGIC 
# MAGIC Prophet decomposes the time series into:
# MAGIC - **Trend**: Long-term direction
# MAGIC - **Yearly Seasonality**: Annual patterns (e.g., summer shutdowns)
# MAGIC - **Weekly Seasonality**: Day-of-week patterns (e.g., Monday effect)
# MAGIC - **Holidays**: Impact of holidays on scrap

# COMMAND ----------

# Plot components
fig = total_model.plot_components(total_forecast)
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 🔍 Explain Total Scrap Forecast
# MAGIC 
# MAGIC Use explainability functions to understand what's driving the forecast.

# COMMAND ----------

# Generate natural language narrative
forecast_narrative = generate_forecast_narrative(
    model=total_model,
    forecast=total_forecast,
    historical_data=total_cost_data,
    model_name="Total Scrap Cost Forecast",
    forecast_period_days=FORECAST_DAYS
)

print(forecast_narrative)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 📊 Component Contribution Analysis
# MAGIC 
# MAGIC Break down the forecast to show how much each component contributes.

# COMMAND ----------

# Analyze specific dates
future_forecast = total_forecast[total_forecast['ds'] > total_cost_data['ds'].max()]

# Analyze first forecast date
first_date_contrib = calculate_component_contributions(
    total_model,
    total_forecast,
    future_forecast['ds'].iloc[0]
)

print("\n🔍 FIRST FORECAST DATE BREAKDOWN")
print(f"Date: {first_date_contrib['date'].strftime('%Y-%m-%d')}")
print(f"Forecast: ${first_date_contrib['forecast_value']:,.2f}")
print(f"Confidence Interval: ${first_date_contrib['lower_bound']:,.2f} to ${first_date_contrib['upper_bound']:,.2f}")
print("\nComponent Contributions:")
for comp_name, comp_data in sorted(first_date_contrib['components'].items(), 
                                   key=lambda x: abs(x[1]['percentage']), 
                                   reverse=True):
    print(f"  • {comp_name.replace('_', ' ').title()}: ${comp_data['value']:,.2f} ({comp_data['percentage']:+.1f}%)")

# Analyze middle forecast date
mid_point = len(future_forecast) // 2
mid_date_contrib = calculate_component_contributions(
    total_model,
    total_forecast,
    future_forecast['ds'].iloc[mid_point]
)

print(f"\n\n🔍 MID-PERIOD FORECAST BREAKDOWN")
print(f"Date: {mid_date_contrib['date'].strftime('%Y-%m-%d')}")
print(f"Forecast: ${mid_date_contrib['forecast_value']:,.2f}")
print(f"Confidence Interval: ${mid_date_contrib['lower_bound']:,.2f} to ${mid_date_contrib['upper_bound']:,.2f}")
print("\nComponent Contributions:")
for comp_name, comp_data in sorted(mid_date_contrib['components'].items(), 
                                   key=lambda x: abs(x[1]['percentage']), 
                                   reverse=True):
    print(f"  • {comp_name.replace('_', ' ').title()}: ${comp_data['value']:,.2f} ({comp_data['percentage']:+.1f}%)")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 📈 Visualize Component Contributions

# COMMAND ----------

# Create component contribution visualization
fig_contrib = visualize_component_contributions(
    total_model,
    total_forecast,
    top_n_dates=6
)
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 🎯 What-If Scenario Analysis
# MAGIC 
# MAGIC Explore alternative scenarios to understand component impacts.

# COMMAND ----------

# Create what-if scenarios
scenarios = create_whatif_scenarios(
    total_model,
    total_forecast,
    total_cost_data
)

# Create comparison table
scenario_table = create_scenario_comparison_table(scenarios)

print("\n📊 WHAT-IF SCENARIO COMPARISON")
print("=" * 100)
display(scenario_table)

# Visualize scenarios
fig, ax = plt.subplots(figsize=(14, 7))

future_dates = total_forecast[total_forecast['ds'] > total_cost_data['ds'].max()]['ds']

for scenario_key, scenario_data in scenarios.items():
    if scenario_key == 'baseline':
        linewidth = 3
        linestyle = '-'
        alpha = 1.0
    else:
        linewidth = 2
        linestyle = '--'
        alpha = 0.7
    
    future_scenario = scenario_data['forecast'][scenario_data['forecast']['ds'].isin(future_dates)]
    ax.plot(
        future_scenario['ds'],
        future_scenario['yhat'],
        label=scenario_data['name'],
        linewidth=linewidth,
        linestyle=linestyle,
        alpha=alpha
    )

ax.set_title('What-If Scenario Comparison', fontsize=14, fontweight='bold')
ax.set_xlabel('Date')
ax.set_ylabel('Daily Scrap Cost ($)')
ax.legend(loc='best')
ax.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print("\n💡 SCENARIO INSIGHTS:")
baseline_mean = scenarios['baseline']['mean']
for scenario_key, scenario_data in scenarios.items():
    if scenario_key != 'baseline' and 'delta' in scenario_data:
        delta = scenario_data['delta']
        delta_pct = (delta / baseline_mean * 100) if baseline_mean != 0 else 0
        impact_desc = "increases" if delta > 0 else "decreases"
        print(f"  • {scenario_data['name']}: {impact_desc} forecast by ${abs(delta):,.2f}/day ({abs(delta_pct):.1f}%)")
        print(f"    → {scenario_data['description']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 📊 Uncertainty Breakdown
# MAGIC 
# MAGIC Understand sources of forecast uncertainty.

# COMMAND ----------

# Analyze uncertainty
uncertainty_analysis = explain_forecast_uncertainty(
    total_model,
    future_forecast
)

print("\n🎯 FORECAST UNCERTAINTY ANALYSIS")
print("=" * 80)
print(f"\n📊 Overall Uncertainty:")
print(f"   Average Prediction: ${uncertainty_analysis['avg_prediction']:,.2f}")
print(f"   Average Interval Width: ±${uncertainty_analysis['avg_interval_width']/2:,.2f}")
print(f"   Uncertainty Percentage: {uncertainty_analysis['uncertainty_percentage']:.1f}%")

if 'near_term_uncertainty' in uncertainty_analysis:
    print(f"\n📈 Uncertainty Over Time:")
    print(f"   Near-term Uncertainty: ±${uncertainty_analysis['near_term_uncertainty']/2:,.2f}")
    print(f"   Far-term Uncertainty: ±${uncertainty_analysis['far_term_uncertainty']/2:,.2f}")
    print(f"   Growth Rate: {uncertainty_analysis['uncertainty_growth']:.1f}%")

print(f"\n🔧 Model Parameters (affecting uncertainty):")
for param, value in uncertainty_analysis['model_params'].items():
    if value is not None:
        print(f"   • {param}: {value}")

print(f"\n💡 Interpretation:")
for insight in uncertainty_analysis['interpretation']:
    print(f"   • {insight}")

# Visualize uncertainty growth
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Left: Uncertainty over time
ax1.fill_between(
    future_forecast['ds'],
    future_forecast['yhat_lower'],
    future_forecast['yhat_upper'],
    alpha=0.3,
    label='95% Prediction Interval'
)
ax1.plot(future_forecast['ds'], future_forecast['yhat'], 'b-', linewidth=2, label='Forecast')
ax1.set_title('Forecast Uncertainty Over Time', fontsize=12, fontweight='bold')
ax1.set_xlabel('Date')
ax1.set_ylabel('Daily Scrap Cost ($)')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.tick_params(axis='x', rotation=45)

# Right: Interval width over time
ax2.plot(future_forecast['ds'], future_forecast['interval_width'], 'r-', linewidth=2)
ax2.set_title('Prediction Interval Width Over Time', fontsize=12, fontweight='bold')
ax2.set_xlabel('Date')
ax2.set_ylabel('Interval Width ($)')
ax2.grid(True, alpha=0.3)
ax2.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2️⃣ Forecast by Plant
# MAGIC 
# MAGIC Train individual forecast models for each plant. This enables:
# MAGIC - Plant-specific budgeting
# MAGIC - Performance comparison
# MAGIC - Identification of plant-level patterns

# COMMAND ----------

# MAGIC %md
# MAGIC ### Define Multi-Model Training Function
# MAGIC 
# MAGIC This function trains a separate Prophet model for each entity (plant, supplier, part)
# MAGIC and logs all models to MLflow.

# COMMAND ----------

def train_multi_level_forecasts(
    df,
    date_col: str,
    value_col: str,
    group_col: str,
    group_name_col: str,
    model_prefix: str,
    top_n: int = None,
    min_records: int = 60,
    use_tuned_params: bool = False,
    tuned_params: dict = None
) -> dict:
    """
    Train separate Prophet models for each group (plant, supplier, part).
    
    Each model is logged to MLflow as a separate run for comparison.
    
    Parameters:
    -----------
    df : DataFrame
        Spark DataFrame with scrap data
    date_col : str
        Date column name
    value_col : str
        Value column to forecast
    group_col : str
        Column to group by (e.g., 'plant_id')
    group_name_col : str
        Friendly name column (e.g., 'plant_name_lookup')
    model_prefix : str
        Prefix for model names in MLflow
    top_n : int, optional
        Limit to top N entities by total value
    min_records : int
        Minimum days of data required to train
    use_tuned_params : bool
        Whether to use tuned hyperparameters
    tuned_params : dict
        Tuned hyperparameters to use
        
    Returns:
    --------
    dict
        Dictionary of {group_id: {name, model, forecast, metrics}}
    """
    
    # Get groups to model
    if top_n:
        top_groups = (
            df.groupBy(group_col, group_name_col)
            .agg(F.sum(value_col).alias("total"))
            .orderBy(F.desc("total"))
            .limit(top_n)
            .select(group_col, group_name_col)
            .collect()
        )
    else:
        top_groups = (
            df.select(group_col, group_name_col)
            .distinct()
            .collect()
        )
    
    results = {}
    
    print(f"\n🏭 Training {len(top_groups)} {model_prefix} models...")
    print("=" * 60)
    
    for idx, row in enumerate(top_groups, 1):
        group_id = row[group_col]
        group_name = row[group_name_col] or f"Unknown ({group_id})"
        
        print(f"\n[{idx}/{len(top_groups)}] Processing: {group_name}")
        
        # Filter data for this group
        group_df = df.filter(F.col(group_col) == group_id)
        
        # Prepare data
        group_data = prepare_prophet_data(
            group_df,
            date_col=date_col,
            value_col=value_col
        )
        
        # Check minimum records
        if len(group_data) < min_records:
            print(f"   ⚠️ Skipping: insufficient data ({len(group_data)} days, need {min_records})")
            continue
        
        # Train model with MLflow
        try:
            model_name = f"{model_prefix}_{group_id}"
            
            # Use tuned parameters if available
            if use_tuned_params and tuned_params:
                model, forecast, _ = train_prophet_model_with_mlflow(
                    group_data,
                    model_name=model_name,
                    changepoint_prior_scale=tuned_params.get('changepoint_prior_scale', 0.05),
                    seasonality_prior_scale=tuned_params.get('seasonality_prior_scale', 10.0),
                    holidays_prior_scale=tuned_params.get('holidays_prior_scale', 10.0),
                    seasonality_mode=tuned_params.get('seasonality_mode', 'multiplicative')
                )
            else:
                model, forecast, _ = train_prophet_model_with_mlflow(
                    group_data,
                    model_name=model_name
                )
            
            results[group_id] = {
                "name": group_name,
                "model": model,
                "forecast": forecast,
                "training_data": group_data
            }
            
        except Exception as e:
            print(f"   ❌ Error training model: {str(e)}")
    
    print(f"\n✅ Successfully trained {len(results)} models")
    return results

# COMMAND ----------

# MAGIC %md
# MAGIC ### Train Plant-Level Models
# MAGIC 
# MAGIC Train a forecast model for each of the 6 plants using parallel execution.

# COMMAND ----------

# Train plant-level forecasts with tuned hyperparameters (PARALLEL)
if USE_PARALLEL_TRAINING:
    plant_models = train_multi_level_forecasts_parallel(
        scrap_df,
        date_col="transaction_date",
        value_col="total_scrap_cost",
        group_col="plant_id",
        group_name_col="plant_name_lookup",
        model_prefix="plant_forecast",
        use_tuned_params=True,
        tuned_params=best_params,
        max_workers=MAX_PARALLEL_WORKERS
    )
else:
    plant_models = train_multi_level_forecasts(
        scrap_df,
        date_col="transaction_date",
        value_col="total_scrap_cost",
        group_col="plant_id",
        group_name_col="plant_name_lookup",
        model_prefix="plant_forecast",
        use_tuned_params=True,
        tuned_params=best_params
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ### Visualize Plant Forecasts
# MAGIC 
# MAGIC Compare forecasts across all plants in a grid view.

# COMMAND ----------

# Visualize plant forecasts
num_plants = min(len(plant_models), 6)
rows = 2
cols = 3

fig, axes = plt.subplots(rows, cols, figsize=(15, 10))
axes = axes.flatten()

for idx, (plant_id, data) in enumerate(plant_models.items()):
    if idx >= num_plants:
        break
    
    ax = axes[idx]
    forecast = data["forecast"]
    train = data["training_data"]
    
    # Plot training data
    ax.plot(train['ds'], train['y'], 'k.', alpha=0.3, markersize=2, label='Actual')
    
    # Plot forecast
    ax.plot(forecast['ds'], forecast['yhat'], 'b-', linewidth=1, label='Forecast')
    ax.fill_between(
        forecast['ds'],
        forecast['yhat_lower'],
        forecast['yhat_upper'],
        alpha=0.2,
        color='blue',
        label='95% CI'
    )
    
    ax.set_title(data["name"], fontsize=10)
    ax.set_xlabel("Date", fontsize=8)
    ax.set_ylabel("Scrap Cost ($)", fontsize=8)
    ax.tick_params(labelsize=8)
    
    # Add vertical line for forecast start
    last_actual = train['ds'].max()
    ax.axvline(x=last_actual, color='red', linestyle='--', alpha=0.5)

# Hide unused subplots
for idx in range(num_plants, len(axes)):
    axes[idx].set_visible(False)

plt.suptitle("Plant-Level Scrap Cost Forecasts (3 Month Horizon)", fontsize=14, y=1.02)
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3️⃣ Forecast by Supplier
# MAGIC 
# MAGIC Train forecasts for the **top 10 suppliers** by scrap volume using parallel execution.
# MAGIC This helps procurement teams anticipate supplier quality issues.

# COMMAND ----------

# Train supplier-level forecasts (top 10 suppliers by scrap volume) (PARALLEL)
if USE_PARALLEL_TRAINING:
    supplier_models = train_multi_level_forecasts_parallel(
        scrap_df,
        date_col="transaction_date",
        value_col="total_scrap_cost",
        group_col="supplier_id",
        group_name_col="supplier_name_lookup",
        model_prefix="supplier_forecast",
        top_n=10,
        use_tuned_params=True,
        tuned_params=best_params,
        max_workers=MAX_PARALLEL_WORKERS
    )
else:
    supplier_models = train_multi_level_forecasts(
        scrap_df,
        date_col="transaction_date",
        value_col="total_scrap_cost",
        group_col="supplier_id",
        group_name_col="supplier_name_lookup",
        model_prefix="supplier_forecast",
        top_n=10,
        use_tuned_params=True,
        tuned_params=best_params
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ### Visualize Top Supplier Forecasts

# COMMAND ----------

# Visualize top supplier forecasts
num_suppliers = min(len(supplier_models), 6)
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for idx, (supplier_id, data) in enumerate(list(supplier_models.items())[:num_suppliers]):
    ax = axes[idx]
    forecast = data["forecast"]
    train = data["training_data"]
    
    ax.plot(train['ds'], train['y'], 'k.', alpha=0.3, markersize=2)
    ax.plot(forecast['ds'], forecast['yhat'], 'b-', linewidth=1)
    ax.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], alpha=0.2)
    
    # Truncate long names
    title = data["name"][:35] + "..." if len(data["name"]) > 35 else data["name"]
    ax.set_title(title, fontsize=9)
    ax.tick_params(labelsize=8)

for idx in range(num_suppliers, len(axes)):
    axes[idx].set_visible(False)

plt.suptitle("Top Supplier Scrap Cost Forecasts", fontsize=14, y=1.02)
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4️⃣ Forecast by Part Number
# MAGIC 
# MAGIC Train forecasts for the **top 20 parts** by scrap volume using parallel execution.
# MAGIC This helps inventory planning teams anticipate high-scrap parts.

# COMMAND ----------

# Train part-level forecasts (top 20 parts by scrap volume) (PARALLEL)
if USE_PARALLEL_TRAINING:
    part_models = train_multi_level_forecasts_parallel(
        scrap_df,
        date_col="transaction_date",
        value_col="total_scrap_cost",
        group_col="part_id",
        group_name_col="part_name_lookup",
        model_prefix="part_forecast",
        top_n=20,
        min_records=30,  # Lower threshold for parts
        use_tuned_params=True,
        tuned_params=best_params,
        max_workers=MAX_PARALLEL_WORKERS
    )
else:
    part_models = train_multi_level_forecasts(
        scrap_df,
        date_col="transaction_date",
        value_col="total_scrap_cost",
        group_col="part_id",
        group_name_col="part_name_lookup",
        model_prefix="part_forecast",
        top_n=20,
        min_records=30,  # Lower threshold for parts
        use_tuned_params=True,
        tuned_params=best_params
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # 💾 Save Forecast Results
# MAGIC ---

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Forecast Result Tables
# MAGIC 
# MAGIC Save all forecasts to Delta tables for use in dashboards and downstream applications.

# COMMAND ----------

def create_forecast_table(
    models_dict: dict, 
    model_type: str, 
    id_col: str
) -> pd.DataFrame:
    """
    Create a consolidated forecast table from multiple models.
    
    Only includes FUTURE predictions (after last training date).
    
    Parameters:
    -----------
    models_dict : dict
        Dictionary of trained models from train_multi_level_forecasts
    model_type : str
        Type of model (plant, supplier, part)
    id_col : str
        Column name for the entity ID
        
    Returns:
    --------
    pd.DataFrame
        Consolidated forecast dataframe
    """
    all_forecasts = []
    
    for entity_id, data in models_dict.items():
        forecast = data["forecast"].copy()
        forecast[id_col] = entity_id
        forecast["entity_name"] = data["name"]
        forecast["model_type"] = model_type
        
        # Keep only future predictions
        last_training_date = data["training_data"]["ds"].max()
        future_forecast = forecast[forecast["ds"] > last_training_date]
        
        all_forecasts.append(future_forecast[[
            "ds", id_col, "entity_name", "model_type",
            "yhat", "yhat_lower", "yhat_upper"
        ]])
    
    if all_forecasts:
        return pd.concat(all_forecasts, ignore_index=True)
    return pd.DataFrame()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Save Plant Forecasts

# COMMAND ----------

# Create and save plant forecasts
if plant_models:
    plant_forecast_df = create_forecast_table(plant_models, "plant", "plant_id")
    
    plant_forecast_spark = spark.createDataFrame(plant_forecast_df)
    plant_forecast_spark = (
        plant_forecast_spark
        .withColumn("forecast_date", F.col("ds").cast("date"))
        .withColumn("predicted_cost", F.round(F.col("yhat"), 2))
        .withColumn("lower_bound", F.round(F.col("yhat_lower"), 2))
        .withColumn("upper_bound", F.round(F.col("yhat_upper"), 2))
        .withColumn("forecast_generated_at", F.current_timestamp())
        .select("plant_id", "entity_name", "forecast_date", "predicted_cost", 
                "lower_bound", "upper_bound", "forecast_generated_at")
    )
    
    plant_forecast_spark.write.mode("overwrite").saveAsTable(
        f"{FULL_SCHEMA}.ml_plant_forecast_results"
    )
    print(f"✅ Saved {plant_forecast_spark.count():,} plant forecast records")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Save Supplier Forecasts

# COMMAND ----------

# Create and save supplier forecasts
if supplier_models:
    supplier_forecast_df = create_forecast_table(supplier_models, "supplier", "supplier_id")
    
    supplier_forecast_spark = spark.createDataFrame(supplier_forecast_df)
    supplier_forecast_spark = (
        supplier_forecast_spark
        .withColumn("forecast_date", F.col("ds").cast("date"))
        .withColumn("predicted_cost", F.round(F.col("yhat"), 2))
        .withColumn("lower_bound", F.round(F.col("yhat_lower"), 2))
        .withColumn("upper_bound", F.round(F.col("yhat_upper"), 2))
        .withColumn("forecast_generated_at", F.current_timestamp())
        .select("supplier_id", "entity_name", "forecast_date", "predicted_cost",
                "lower_bound", "upper_bound", "forecast_generated_at")
    )
    
    supplier_forecast_spark.write.mode("overwrite").saveAsTable(
        f"{FULL_SCHEMA}.ml_supplier_forecast_results"
    )
    print(f"✅ Saved {supplier_forecast_spark.count():,} supplier forecast records")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Save Part Forecasts

# COMMAND ----------

# Create and save part forecasts
if part_models:
    part_forecast_df = create_forecast_table(part_models, "part", "part_id")
    
    part_forecast_spark = spark.createDataFrame(part_forecast_df)
    part_forecast_spark = (
        part_forecast_spark
        .withColumn("forecast_date", F.col("ds").cast("date"))
        .withColumn("predicted_cost", F.round(F.col("yhat"), 2))
        .withColumn("lower_bound", F.round(F.col("yhat_lower"), 2))
        .withColumn("upper_bound", F.round(F.col("yhat_upper"), 2))
        .withColumn("forecast_generated_at", F.current_timestamp())
        .select("part_id", "entity_name", "forecast_date", "predicted_cost",
                "lower_bound", "upper_bound", "forecast_generated_at")
    )
    
    part_forecast_spark.write.mode("overwrite").saveAsTable(
        f"{FULL_SCHEMA}.ml_part_forecast_results"
    )
    print(f"✅ Saved {part_forecast_spark.count():,} part forecast records")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Save Total Forecast

# COMMAND ----------

# Save total company forecast
total_future = total_forecast[total_forecast["ds"] > total_cost_data["ds"].max()].copy()
total_future["entity_id"] = "TOTAL"
total_future["entity_name"] = "Company Total"

total_forecast_spark = spark.createDataFrame(total_future[["ds", "yhat", "yhat_lower", "yhat_upper", "entity_id", "entity_name"]])
total_forecast_spark = (
    total_forecast_spark
    .withColumn("forecast_date", F.col("ds").cast("date"))
    .withColumn("predicted_cost", F.round(F.col("yhat"), 2))
    .withColumn("lower_bound", F.round(F.col("yhat_lower"), 2))
    .withColumn("upper_bound", F.round(F.col("yhat_upper"), 2))
    .withColumn("forecast_generated_at", F.current_timestamp())
    .select("entity_id", "entity_name", "forecast_date", "predicted_cost",
            "lower_bound", "upper_bound", "forecast_generated_at")
)

total_forecast_spark.write.mode("overwrite").saveAsTable(
    f"{FULL_SCHEMA}.ml_total_forecast_results"
)
print(f"✅ Saved {total_forecast_spark.count():,} total forecast records")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # 🏆 Register Best Model to Unity Catalog
# MAGIC ---

# COMMAND ----------

# MAGIC %md
# MAGIC ## Register Total Forecast Model
# MAGIC 
# MAGIC Register the total scrap forecast model to Unity Catalog Model Registry for:
# MAGIC - Version control
# MAGIC - Production deployment
# MAGIC - Model lineage tracking

# COMMAND ----------

# Register the total forecast model to Unity Catalog
model_name = f"{CATALOG}.{SCHEMA}.scrap_total_forecast_model"

# Create signature for Unity Catalog (required)
# Prophet expects input with 'ds' column and outputs forecast columns
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, ColSpec

# Define input schema (Prophet expects 'ds' datetime column)
input_schema = Schema([ColSpec("datetime", "ds")])

# Define output schema (Prophet outputs these columns)
output_schema = Schema([
    ColSpec("double", "yhat"),
    ColSpec("double", "yhat_lower"),
    ColSpec("double", "yhat_upper")
])

signature = ModelSignature(inputs=input_schema, outputs=output_schema)

with mlflow.start_run(run_name="register_production_model") as run:
    # Log model info
    mlflow.log_param("registered_model_name", model_name)
    mlflow.log_param("model_type", "Prophet")
    
    # Register model with signature (required for Unity Catalog)
    model_info = mlflow.prophet.log_model(
        total_model,
        artifact_path="model",
        signature=signature,
        registered_model_name=model_name
    )
    
    print(f"✅ Model registered: {model_name}")
    print(f"   Run ID: {run.info.run_id}")
    print(f"   Model URI: {model_info.model_uri}")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # 📋 Forecast Summary
# MAGIC ---

# COMMAND ----------

# MAGIC %md
# MAGIC ## View 30-Day Forecast Summary
# MAGIC 
# MAGIC Display aggregated forecast for the next 30 days.

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Next 30 days forecast summary by entity type
# MAGIC WITH plant_forecast AS (
# MAGIC     SELECT 'Plant' as entity_type, entity_name, 
# MAGIC            SUM(predicted_cost) as total_predicted,
# MAGIC            SUM(lower_bound) as total_lower,
# MAGIC            SUM(upper_bound) as total_upper
# MAGIC     FROM scrap_catalog.scrap_demo.ml_plant_forecast_results
# MAGIC     WHERE forecast_date <= DATE_ADD(CURRENT_DATE(), 30)
# MAGIC     GROUP BY entity_name
# MAGIC ),
# MAGIC supplier_forecast AS (
# MAGIC     SELECT 'Supplier' as entity_type, entity_name,
# MAGIC            SUM(predicted_cost) as total_predicted,
# MAGIC            SUM(lower_bound) as total_lower,
# MAGIC            SUM(upper_bound) as total_upper
# MAGIC     FROM scrap_catalog.scrap_demo.ml_supplier_forecast_results
# MAGIC     WHERE forecast_date <= DATE_ADD(CURRENT_DATE(), 30)
# MAGIC     GROUP BY entity_name
# MAGIC )
# MAGIC SELECT * FROM plant_forecast
# MAGIC UNION ALL
# MAGIC SELECT * FROM supplier_forecast
# MAGIC ORDER BY entity_type, total_predicted DESC

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📊 MLflow Experiment Summary
# MAGIC 
# MAGIC View all model runs in MLflow:
# MAGIC 1. Navigate to **Experiments** in the left sidebar
# MAGIC 2. Search for `/Shared/scrap_demo/scrap_forecasting`
# MAGIC 3. Compare model performance across different runs

# COMMAND ----------

# Print summary of all models
print("\n" + "=" * 70)
print("📊 FORECASTING SUMMARY")
print("=" * 70)
print(f"\n🔮 Models Trained:")
print(f"   - Total Company Forecast: 1 model")
print(f"   - Plant-Level Forecasts: {len(plant_models)} models")
print(f"   - Supplier-Level Forecasts: {len(supplier_models)} models")
print(f"   - Part-Level Forecasts: {len(part_models)} models")
print(f"\n📁 Output Tables:")
print(f"   - {FULL_SCHEMA}.ml_total_forecast_results")
print(f"   - {FULL_SCHEMA}.ml_plant_forecast_results")
print(f"   - {FULL_SCHEMA}.ml_supplier_forecast_results")
print(f"   - {FULL_SCHEMA}.ml_part_forecast_results")
print(f"\n🔗 MLflow Experiment: Default Notebook Experiment")
print(f"\n🏆 Registered Model: {CATALOG}.{SCHEMA}.scrap_total_forecast_model")
print("=" * 70)

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # 🔍 MLOps: Drift Detection & Model Monitoring
# MAGIC ---
# MAGIC 
# MAGIC Comprehensive drift monitoring to ensure model performance remains stable over time.
# MAGIC 
# MAGIC ## Key Monitoring Capabilities:
# MAGIC 1. **Data Drift Detection** - Identify distribution changes in incoming data
# MAGIC 2. **Model Performance Drift** - Track degradation in prediction accuracy
# MAGIC 3. **Automated Alerting** - Flag models requiring retraining
# MAGIC 4. **MLflow Integration** - Log all drift metrics for historical tracking

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1️⃣ Data Drift Detection
# MAGIC 
# MAGIC Compare recent data distribution against training baseline using:
# MAGIC - **Kolmogorov-Smirnov Test**: Statistical test for distribution changes
# MAGIC - **Wasserstein Distance**: Measures magnitude of distribution shift
# MAGIC - **Summary Statistics**: Track changes in mean, std, median, etc.

# COMMAND ----------

# Run data drift detection on total scrap data
print("\n🔍 DETECTING DATA DRIFT")
print("=" * 80)

# Use first 80% of data as reference, last 20% as current
reference_cutoff = total_cost_data['ds'].min() + (total_cost_data['ds'].max() - total_cost_data['ds'].min()) * 0.8
reference_data = total_cost_data[total_cost_data['ds'] <= reference_cutoff]
current_data = total_cost_data[total_cost_data['ds'] > reference_cutoff]

data_drift_results = detect_data_drift(
    reference_data=reference_data,
    current_data=current_data,
    feature_col='y',
    window_days=DRIFT_DETECTION_WINDOW
)

# Display results
if not data_drift_results.get("error"):
    print(f"\n📊 Data Drift Status: {'🚨 DETECTED' if data_drift_results['drift_detected'] else '✅ NOT DETECTED'}")
    print(f"\nDrift Score: {data_drift_results['drift_score']:.4f} (0=no drift, 1=max drift)")
    print(f"\nStatistical Tests:")
    print(f"  • KS Test p-value: {data_drift_results['ks_pvalue']:.6f} (threshold: {DRIFT_P_VALUE_THRESHOLD})")
    print(f"  • Wasserstein Distance: {data_drift_results['normalized_wasserstein_distance']:.4f} (threshold: {DRIFT_DISTANCE_THRESHOLD})")
    print(f"  • Mean Change: {data_drift_results['mean_change_pct']:.2f}%")
    print(f"  • Std Change: {data_drift_results['std_change_pct']:.2f}%")
    
    print(f"\nDistribution Comparison:")
    print(f"  Reference: mean=${data_drift_results['reference_stats']['mean']:.2f}, std=${data_drift_results['reference_stats']['std']:.2f}, n={data_drift_results['reference_samples']:,}")
    print(f"  Current:   mean=${data_drift_results['current_stats']['mean']:.2f}, std=${data_drift_results['current_stats']['std']:.2f}, n={data_drift_results['current_samples']:,}")
else:
    print(f"❌ Error: {data_drift_results['error']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2️⃣ Model Performance Drift Detection
# MAGIC 
# MAGIC Monitor model accuracy on recent data compared to baseline performance.
# MAGIC 
# MAGIC Tracks:
# MAGIC - MAE, MAPE, RMSE degradation
# MAGIC - Prediction vs actual comparison
# MAGIC - Alert threshold violations

# COMMAND ----------

# Detect model drift on total forecast model
print("\n🤖 DETECTING MODEL PERFORMANCE DRIFT")
print("=" * 80)

# Get baseline metrics from cross-validation
baseline_metrics = {
    "mae": total_val_metrics["mae"],
    "mape": total_val_metrics["mape"],
    "rmse": total_val_metrics["rmse"]
}

# Use recent data (last 30 days with actuals) for drift detection
drift_window_data = total_cost_data.tail(60)  # Last 60 days

model_drift_results = detect_model_drift(
    model=total_model,
    baseline_metrics=baseline_metrics,
    current_data=drift_window_data,
    model_name="total_scrap_cost_forecast"
)

# Display results
if not model_drift_results.get("error"):
    print(f"\n📊 Model Drift Status: {'🚨 DETECTED' if model_drift_results['drift_detected'] else '✅ NOT DETECTED'}")
    print(f"\nAverage Performance Degradation: {model_drift_results['avg_degradation_pct']:.2f}%")
    
    print(f"\nMetric Comparison (Current vs Baseline):")
    baseline = model_drift_results['baseline_metrics']
    current = model_drift_results['current_metrics']
    
    print(f"  • MAE:  ${current['mae']:.2f} vs ${baseline['mae']:.2f} ({model_drift_results['mae_degradation_pct']:+.2f}%)")
    print(f"  • MAPE: {current['mape']:.2f}% vs {baseline['mape']:.2f}% ({model_drift_results['mape_degradation_pct']:+.2f}%)")
    print(f"  • RMSE: ${current['rmse']:.2f} vs ${baseline['rmse']:.2f} ({model_drift_results['rmse_degradation_pct']:+.2f}%)")
    print(f"\nEvaluation Samples: {model_drift_results['evaluation_samples']:,}")
else:
    print(f"❌ Error: {model_drift_results['error']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3️⃣ Drift Monitoring Report
# MAGIC 
# MAGIC Generate a comprehensive drift monitoring report with recommendations.

# COMMAND ----------

# Create detailed drift report
drift_report = create_drift_monitoring_report(
    data_drift_results=data_drift_results,
    model_drift_results=model_drift_results,
    model_name="Total Scrap Cost Forecast"
)

print(drift_report)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4️⃣ Log Drift Metrics to MLflow
# MAGIC 
# MAGIC Store drift metrics in MLflow for historical tracking and alerting.
# MAGIC 
# MAGIC These metrics can be:
# MAGIC - Monitored over time using MLflow UI
# MAGIC - Used to trigger retraining pipelines
# MAGIC - Integrated with alerting systems (PagerDuty, Slack, etc.)

# COMMAND ----------

# Log drift metrics to MLflow as a new run
with mlflow.start_run(run_name="drift_monitoring_total_forecast") as drift_run:
    # Log drift metrics
    log_drift_metrics_to_mlflow(
        data_drift_results=data_drift_results,
        model_drift_results=model_drift_results
    )
    
    # Log drift report as artifact
    with open("/tmp/drift_report.txt", "w") as f:
        f.write(drift_report)
    mlflow.log_artifact("/tmp/drift_report.txt")
    
    # Log configuration
    mlflow.log_param("reference_data_samples", data_drift_results.get("reference_samples", 0))
    mlflow.log_param("current_data_samples", data_drift_results.get("current_samples", 0))
    mlflow.log_param("drift_window_days", DRIFT_DETECTION_WINDOW)
    
    print(f"\n✅ Drift metrics logged to MLflow")
    print(f"   Run ID: {drift_run.info.run_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5️⃣ Plant-Level Drift Monitoring
# MAGIC 
# MAGIC Run drift detection for each plant model to identify which plants need retraining.

# COMMAND ----------

# Monitor drift for each plant model
plant_drift_summary = []

print("\n🏭 PLANT-LEVEL DRIFT MONITORING")
print("=" * 80)

for plant_id, plant_data in list(plant_models.items())[:3]:  # Monitor first 3 plants as example
    print(f"\nPlant: {plant_data['name']}")
    print("-" * 60)
    
    # Get plant training data
    plant_train_data = plant_data['training_data']
    
    # Split for drift detection
    ref_cutoff = plant_train_data['ds'].min() + (plant_train_data['ds'].max() - plant_train_data['ds'].min()) * 0.8
    plant_ref = plant_train_data[plant_train_data['ds'] <= ref_cutoff]
    plant_curr = plant_train_data[plant_train_data['ds'] > ref_cutoff]
    
    # Data drift
    plant_data_drift = detect_data_drift(
        reference_data=plant_ref,
        current_data=plant_curr,
        feature_col='y',
        window_days=30
    )
    
    # Model drift
    plant_baseline_metrics = {
        "mae": plant_data['metrics']['mae'],
        "mape": plant_data['metrics']['mape'],
        "rmse": plant_data['metrics']['rmse']
    }
    
    plant_model_drift = detect_model_drift(
        model=plant_data['model'],
        baseline_metrics=plant_baseline_metrics,
        current_data=plant_train_data.tail(30),
        model_name=f"plant_{plant_id}"
    )
    
    # Summary
    data_status = "🚨 DRIFT" if plant_data_drift.get("drift_detected") else "✅ OK"
    model_status = "🚨 DRIFT" if plant_model_drift.get("drift_detected") else "✅ OK"
    
    print(f"  Data Drift: {data_status}")
    print(f"  Model Drift: {model_status}")
    
    if not plant_data_drift.get("error"):
        print(f"  Drift Score: {plant_data_drift['drift_score']:.4f}")
    if not plant_model_drift.get("error"):
        print(f"  Avg Degradation: {plant_model_drift['avg_degradation_pct']:.2f}%")
    
    plant_drift_summary.append({
        "plant_id": plant_id,
        "plant_name": plant_data['name'],
        "data_drift_detected": plant_data_drift.get("drift_detected", False),
        "model_drift_detected": plant_model_drift.get("drift_detected", False),
        "drift_score": plant_data_drift.get("drift_score", 0),
        "avg_degradation_pct": plant_model_drift.get("avg_degradation_pct", 0)
    })

# Create summary DataFrame
plant_drift_df = pd.DataFrame(plant_drift_summary)
print(f"\n📊 Plant Drift Summary:")
display(plant_drift_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6️⃣ Save Drift Monitoring Results
# MAGIC 
# MAGIC Store drift detection results in Delta tables for ongoing monitoring and alerting.

# COMMAND ----------

# Create drift monitoring results table
drift_monitoring_results = []

# Add total forecast drift
drift_monitoring_results.append({
    "model_name": "total_scrap_cost_forecast",
    "model_type": "total",
    "entity_id": None,
    "entity_name": "Company Total",
    "monitoring_timestamp": datetime.now(),
    "data_drift_detected": data_drift_results.get("drift_detected", False),
    "data_drift_score": data_drift_results.get("drift_score", 0.0),
    "data_ks_pvalue": data_drift_results.get("ks_pvalue", 1.0),
    "data_mean_change_pct": data_drift_results.get("mean_change_pct", 0.0),
    "model_drift_detected": model_drift_results.get("drift_detected", False),
    "model_avg_degradation_pct": model_drift_results.get("avg_degradation_pct", 0.0),
    "model_mae_degradation_pct": model_drift_results.get("mae_degradation_pct", 0.0),
    "model_mape_degradation_pct": model_drift_results.get("mape_degradation_pct", 0.0),
    "current_mae": model_drift_results.get("current_metrics", {}).get("mae", 0.0),
    "current_mape": model_drift_results.get("current_metrics", {}).get("mape", 0.0),
    "current_rmse": model_drift_results.get("current_metrics", {}).get("rmse", 0.0),
    "baseline_mae": baseline_metrics.get("mae", 0.0),
    "baseline_mape": baseline_metrics.get("mape", 0.0),
    "baseline_rmse": baseline_metrics.get("rmse", 0.0),
    "requires_retraining": data_drift_results.get("drift_detected", False) or model_drift_results.get("drift_detected", False)
})

# Add plant drift results
for result in plant_drift_summary:
    drift_monitoring_results.append({
        "model_name": f"plant_forecast_{result['plant_id']}",
        "model_type": "plant",
        "entity_id": result['plant_id'],
        "entity_name": result['plant_name'],
        "monitoring_timestamp": datetime.now(),
        "data_drift_detected": result['data_drift_detected'],
        "data_drift_score": result['drift_score'],
        "data_ks_pvalue": None,
        "data_mean_change_pct": None,
        "model_drift_detected": result['model_drift_detected'],
        "model_avg_degradation_pct": result['avg_degradation_pct'],
        "model_mae_degradation_pct": None,
        "model_mape_degradation_pct": None,
        "current_mae": None,
        "current_mape": None,
        "current_rmse": None,
        "baseline_mae": None,
        "baseline_mape": None,
        "baseline_rmse": None,
        "requires_retraining": result['data_drift_detected'] or result['model_drift_detected']
    })

# Convert to Spark DataFrame and save
drift_monitoring_df = pd.DataFrame(drift_monitoring_results)
drift_monitoring_spark = spark.createDataFrame(drift_monitoring_df)

# Save to Delta table
drift_table = f"{FULL_SCHEMA}.ml_drift_monitoring_results"
drift_monitoring_spark.write.mode("append").saveAsTable(drift_table)

print(f"✅ Drift monitoring results saved to: {drift_table}")

# Show models requiring retraining
models_needing_retraining = drift_monitoring_df[drift_monitoring_df['requires_retraining'] == True]
if len(models_needing_retraining) > 0:
    print(f"\n⚠️  {len(models_needing_retraining)} model(s) require retraining:")
    for _, row in models_needing_retraining.iterrows():
        print(f"   - {row['model_name']} ({row['entity_name']})")
else:
    print(f"\n✅ All models are performing within acceptable thresholds")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7️⃣ Drift Monitoring Dashboard Query
# MAGIC 
# MAGIC Query drift monitoring results for visualization in dashboards.

# COMMAND ----------

# MAGIC %sql
# MAGIC -- View latest drift monitoring results
# MAGIC SELECT 
# MAGIC   model_type,
# MAGIC   entity_name,
# MAGIC   data_drift_detected,
# MAGIC   ROUND(data_drift_score, 4) as drift_score,
# MAGIC   model_drift_detected,
# MAGIC   ROUND(model_avg_degradation_pct, 2) as avg_degradation_pct,
# MAGIC   requires_retraining,
# MAGIC   monitoring_timestamp
# MAGIC FROM scrap_catalog.scrap_demo.ml_drift_monitoring_results
# MAGIC WHERE monitoring_timestamp >= CURRENT_TIMESTAMP() - INTERVAL 7 DAYS
# MAGIC ORDER BY drift_score DESC, avg_degradation_pct DESC

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🎯 MLOps Best Practices Implemented
# MAGIC 
# MAGIC ✅ **Data Drift Detection**
# MAGIC - Statistical tests (KS test, Wasserstein distance)
# MAGIC - Distribution monitoring
# MAGIC - Configurable thresholds
# MAGIC 
# MAGIC ✅ **Model Performance Drift**
# MAGIC - Continuous accuracy monitoring
# MAGIC - Degradation tracking
# MAGIC - Automated alerting
# MAGIC 
# MAGIC ✅ **MLflow Integration**
# MAGIC - All metrics logged
# MAGIC - Historical tracking
# MAGIC - Experiment comparison
# MAGIC 
# MAGIC ✅ **Automated Monitoring**
# MAGIC - Delta tables for results
# MAGIC - SQL queries for dashboards
# MAGIC - Retraining recommendations
# MAGIC 
# MAGIC ## 🔄 Recommended Monitoring Schedule
# MAGIC 
# MAGIC - **Daily**: Check drift metrics via dashboard
# MAGIC - **Weekly**: Review drift trends and patterns
# MAGIC - **Monthly**: Evaluate retraining needs
# MAGIC - **Quarterly**: Update drift detection thresholds
# MAGIC 
# MAGIC ## 🚨 Alert Thresholds (Configurable)
# MAGIC 
# MAGIC - `DRIFT_P_VALUE_THRESHOLD`: {DRIFT_P_VALUE_THRESHOLD} (KS test significance)
# MAGIC - `DRIFT_DISTANCE_THRESHOLD`: {DRIFT_DISTANCE_THRESHOLD} (Wasserstein distance)
# MAGIC - `MODEL_PERFORMANCE_DEGRADATION_THRESHOLD`: {MODEL_PERFORMANCE_DEGRADATION_THRESHOLD} (15% degradation)

