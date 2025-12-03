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

# MAGIC %pip install prophet mlflow[databricks]


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

# COMMAND ----------

# MAGIC %md
# MAGIC ## ⚙️ Configuration
# MAGIC 
# MAGIC Define all configuration parameters in one place for easy modification:
# MAGIC - **CATALOG/SCHEMA**: Unity Catalog location
# MAGIC - **FORECAST_DAYS**: Prediction horizon (90 days = ~3 months)
# MAGIC - **EXPERIMENT_PATH**: MLflow experiment location

# COMMAND ----------

# Unity Catalog Configuration
CATALOG = "welch"
SCHEMA = "scrap_demo"
FULL_SCHEMA = f"{CATALOG}.{SCHEMA}"

# Forecasting Configuration
FORECAST_DAYS = 90  # 3 months forward
TRAIN_TEST_SPLIT_DAYS = 30  # Hold out last 30 days for validation

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
# MAGIC 2. **Model training** - Train Prophet with MLflow tracking
# MAGIC 3. **Metrics calculation** - MAE, MAPE, RMSE

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

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🧠 Define Model Training Function with Full MLflow Logging
# MAGIC 
# MAGIC This function:
# MAGIC 1. Creates train/test split for validation
# MAGIC 2. Trains Prophet model with seasonality
# MAGIC 3. Logs ALL artifacts to MLflow:
# MAGIC    - Model parameters
# MAGIC    - Training/validation metrics
# MAGIC    - Forecast plots
# MAGIC    - Component plots
# MAGIC    - Feature importance
# MAGIC    - Model artifacts

# COMMAND ----------

def train_prophet_model_with_mlflow(
    train_data: pd.DataFrame,
    model_name: str,
    forecast_days: int = FORECAST_DAYS,
    yearly_seasonality: bool = True,
    weekly_seasonality: bool = True,
    include_holidays: bool = True,
    seasonality_mode: str = 'multiplicative'
) -> tuple:
    """
    Train a Prophet model with comprehensive MLflow logging.
    
    All parameters, metrics, and artifacts are logged to MLflow for
    reproducibility and model comparison.
    
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
        mlflow.log_param("training_records", len(train_data))
        mlflow.log_param("training_start_date", str(train_data['ds'].min()))
        mlflow.log_param("training_end_date", str(train_data['ds'].max()))
        
        # ===== TRAIN/TEST SPLIT FOR VALIDATION =====
        # Hold out last N days for validation
        split_date = train_data['ds'].max() - timedelta(days=TRAIN_TEST_SPLIT_DAYS)
        train_set = train_data[train_data['ds'] <= split_date]
        test_set = train_data[train_data['ds'] > split_date]
        
        mlflow.log_param("train_size", len(train_set))
        mlflow.log_param("test_size", len(test_set))
        
        # ===== INITIALIZE PROPHET MODEL =====
        model = Prophet(
            yearly_seasonality=yearly_seasonality,
            weekly_seasonality=weekly_seasonality,
            daily_seasonality=False,
            seasonality_mode=seasonality_mode,
            interval_width=0.95,  # 95% confidence interval
            changepoint_prior_scale=0.05  # Flexibility of trend
        )
        
        # Add US holidays if requested
        if include_holidays:
            model.add_country_holidays(country_name='US')
        
        # ===== FIT MODEL =====
        model.fit(train_set)
        
        # ===== MAKE PREDICTIONS =====
        # Predict on full history + future
        future = model.make_future_dataframe(periods=forecast_days + TRAIN_TEST_SPLIT_DAYS)
        forecast = model.predict(future)
        
        # ===== CALCULATE VALIDATION METRICS =====
        # Merge test set with predictions
        test_forecast = forecast[forecast['ds'].isin(test_set['ds'])]
        test_merged = test_set.merge(test_forecast[['ds', 'yhat']], on='ds')
        
        val_metrics = calculate_metrics(test_merged['y'], test_merged['yhat'])
        
        # ===== CALCULATE TRAINING METRICS =====
        train_forecast = forecast[forecast['ds'].isin(train_set['ds'])]
        train_merged = train_set.merge(train_forecast[['ds', 'yhat']], on='ds')
        
        train_metrics = calculate_metrics(train_merged['y'], train_merged['yhat'])
        
        # ===== LOG METRICS =====
        mlflow.log_metric("train_mae", train_metrics['mae'])
        mlflow.log_metric("train_mape", train_metrics['mape'])
        mlflow.log_metric("train_rmse", train_metrics['rmse'])
        
        mlflow.log_metric("val_mae", val_metrics['mae'])
        mlflow.log_metric("val_mape", val_metrics['mape'])
        mlflow.log_metric("val_rmse", val_metrics['rmse'])
        
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
        
        # Actual vs Predicted plot
        fig3, ax = plt.subplots(figsize=(12, 6))
        ax.scatter(test_merged['y'], test_merged['yhat'], alpha=0.5)
        ax.plot([test_merged['y'].min(), test_merged['y'].max()],
                [test_merged['y'].min(), test_merged['y'].max()], 'r--')
        ax.set_xlabel('Actual')
        ax.set_ylabel('Predicted')
        ax.set_title(f'{model_name} - Actual vs Predicted (Validation Set)')
        mlflow.log_figure(fig3, "actual_vs_predicted.png")
        plt.close()
        
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
        print(f"   Training Records: {len(train_set):,}")
        print(f"   Validation Records: {len(test_set):,}")
        print(f"\n   📊 Validation Metrics:")
        print(f"      MAE:  ${val_metrics['mae']:,.2f}")
        print(f"      MAPE: {val_metrics['mape']:.2f}%")
        print(f"      RMSE: ${val_metrics['rmse']:,.2f}")
        print(f"\n   🔗 MLflow Run ID: {run.info.run_id}")
        
        return model, forecast

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
# MAGIC ### Train Total Scrap Model
# MAGIC 
# MAGIC Train Prophet model for total scrap cost with full MLflow logging.

# COMMAND ----------

# Train total scrap cost model
total_model, total_forecast = train_prophet_model_with_mlflow(
    total_cost_data,
    model_name="total_scrap_cost_forecast",
    yearly_seasonality=True,
    weekly_seasonality=True,
    include_holidays=True
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
    min_records: int = 60
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
            model, forecast = train_prophet_model_with_mlflow(
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
# MAGIC Train a forecast model for each of the 6 plants.

# COMMAND ----------

# Train plant-level forecasts
plant_models = train_multi_level_forecasts(
    scrap_df,
    date_col="transaction_date",
    value_col="total_scrap_cost",
    group_col="plant_id",
    group_name_col="plant_name_lookup",
    model_prefix="plant_forecast"
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
# MAGIC Train forecasts for the **top 10 suppliers** by scrap volume.
# MAGIC This helps procurement teams anticipate supplier quality issues.

# COMMAND ----------

# Train supplier-level forecasts (top 10 suppliers by scrap volume)
supplier_models = train_multi_level_forecasts(
    scrap_df,
    date_col="transaction_date",
    value_col="total_scrap_cost",
    group_col="supplier_id",
    group_name_col="supplier_name_lookup",
    model_prefix="supplier_forecast",
    top_n=10
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
# MAGIC Train forecasts for the **top 20 parts** by scrap volume.
# MAGIC This helps inventory planning teams anticipate high-scrap parts.

# COMMAND ----------

# Train part-level forecasts (top 20 parts by scrap volume)
part_models = train_multi_level_forecasts(
    scrap_df,
    date_col="transaction_date",
    value_col="total_scrap_cost",
    group_col="part_id",
    group_name_col="part_name_lookup",
    model_prefix="part_forecast",
    top_n=20,
    min_records=30  # Lower threshold for parts
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
# MAGIC     FROM welch.scrap_demo.ml_plant_forecast_results
# MAGIC     WHERE forecast_date <= DATE_ADD(CURRENT_DATE(), 30)
# MAGIC     GROUP BY entity_name
# MAGIC ),
# MAGIC supplier_forecast AS (
# MAGIC     SELECT 'Supplier' as entity_type, entity_name,
# MAGIC            SUM(predicted_cost) as total_predicted,
# MAGIC            SUM(lower_bound) as total_lower,
# MAGIC            SUM(upper_bound) as total_upper
# MAGIC     FROM welch.scrap_demo.ml_supplier_forecast_results
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
