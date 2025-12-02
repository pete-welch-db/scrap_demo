# Databricks notebook source
# MAGIC %md
# MAGIC # Scrap Forecasting Models
# MAGIC 
# MAGIC This notebook builds time series forecasting models to predict scrap 3 months into the future at multiple granularities:
# MAGIC 
# MAGIC 1. **Total Scrap Cost** - Company-wide forecast
# MAGIC 2. **By Plant** - Plant-level forecasts
# MAGIC 3. **By Supplier** - Supplier-level forecasts  
# MAGIC 4. **By Part Number** - Part-level forecasts (top volume parts)
# MAGIC 
# MAGIC **Models Used:**
# MAGIC - Facebook Prophet (primary)
# MAGIC - AutoML for comparison
# MAGIC 
# MAGIC **MLflow Tracking:** All experiments logged for comparison

# COMMAND ----------

# MAGIC %pip install prophet

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from prophet import Prophet
import mlflow
import mlflow.prophet
from pyspark.sql import functions as F
from pyspark.sql.window import Window
import matplotlib.pyplot as plt

# Set MLflow experiment
mlflow.set_experiment("/Shared/scrap_demo/scrap_forecasting")

# Configuration
CATALOG = "welch"
SCHEMA = "scrap_demo"
FORECAST_DAYS = 90  # 3 months

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Load and Prepare Data

# COMMAND ----------

# Load scrap transactions from silver layer
scrap_df = spark.table(f"{CATALOG}.{SCHEMA}.scrap_transactions_silver")

print(f"Total records: {scrap_df.count():,}")
print(f"Date range: {scrap_df.agg(F.min('transaction_date'), F.max('transaction_date')).collect()[0]}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Total Scrap Cost Forecast

# COMMAND ----------

def prepare_prophet_data(df, date_col, value_col, group_cols=None):
    """
    Prepare data for Prophet model.
    Prophet requires columns named 'ds' (date) and 'y' (value).
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
    
    return agg_df.orderBy("ds").toPandas()


def train_prophet_model(
    train_data: pd.DataFrame,
    model_name: str,
    yearly_seasonality: bool = True,
    weekly_seasonality: bool = True,
    include_holidays: bool = True
) -> tuple:
    """
    Train a Prophet model and log to MLflow.
    """
    with mlflow.start_run(run_name=model_name):
        # Initialize Prophet
        model = Prophet(
            yearly_seasonality=yearly_seasonality,
            weekly_seasonality=weekly_seasonality,
            daily_seasonality=False,
            seasonality_mode='multiplicative'
        )
        
        # Add US holidays if requested
        if include_holidays:
            model.add_country_holidays(country_name='US')
        
        # Fit model
        model.fit(train_data)
        
        # Make future dataframe
        future = model.make_future_dataframe(periods=FORECAST_DAYS)
        forecast = model.predict(future)
        
        # Calculate metrics on training data
        train_forecast = forecast[forecast['ds'].isin(train_data['ds'])]
        merged = train_data.merge(train_forecast[['ds', 'yhat']], on='ds')
        
        mae = np.mean(np.abs(merged['y'] - merged['yhat']))
        mape = np.mean(np.abs((merged['y'] - merged['yhat']) / merged['y'])) * 100
        rmse = np.sqrt(np.mean((merged['y'] - merged['yhat'])**2))
        
        # Log parameters
        mlflow.log_param("model_type", "Prophet")
        mlflow.log_param("forecast_days", FORECAST_DAYS)
        mlflow.log_param("yearly_seasonality", yearly_seasonality)
        mlflow.log_param("weekly_seasonality", weekly_seasonality)
        mlflow.log_param("training_records", len(train_data))
        
        # Log metrics
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("mape", mape)
        mlflow.log_metric("rmse", rmse)
        
        # Log model
        mlflow.prophet.log_model(model, "prophet_model")
        
        print(f"Model: {model_name}")
        print(f"  MAE: {mae:,.2f}")
        print(f"  MAPE: {mape:.2f}%")
        print(f"  RMSE: {rmse:,.2f}")
        
        return model, forecast

# COMMAND ----------

# Prepare total scrap cost data (daily aggregation)
total_cost_data = prepare_prophet_data(
    scrap_df, 
    date_col="transaction_date", 
    value_col="total_scrap_cost"
)

print(f"Training data shape: {total_cost_data.shape}")
total_cost_data.head()

# COMMAND ----------

# Train total scrap cost model
total_model, total_forecast = train_prophet_model(
    total_cost_data,
    model_name="total_scrap_cost_forecast"
)

# COMMAND ----------

# Plot forecast
fig = total_model.plot(total_forecast)
plt.title("Total Scrap Cost Forecast (3 Months)")
plt.ylabel("Daily Scrap Cost ($)")
plt.xlabel("Date")
plt.tight_layout()
plt.show()

# COMMAND ----------

# Plot components
fig = total_model.plot_components(total_forecast)
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Forecast by Plant

# COMMAND ----------

def train_multi_prophet_models(
    df,
    date_col: str,
    value_col: str,
    group_col: str,
    group_name_col: str,
    model_prefix: str,
    top_n: int = None
) -> dict:
    """
    Train separate Prophet models for each group (plant, supplier, part).
    Returns dictionary of {group_id: (model, forecast)}.
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
    
    for row in top_groups:
        group_id = row[group_col]
        group_name = row[group_name_col]
        
        # Filter data for this group
        group_df = df.filter(F.col(group_col) == group_id)
        
        # Prepare data
        group_data = prepare_prophet_data(
            group_df,
            date_col=date_col,
            value_col=value_col
        )
        
        # Need at least 2 data points
        if len(group_data) < 30:
            print(f"Skipping {group_name}: insufficient data ({len(group_data)} days)")
            continue
        
        # Train model
        model_name = f"{model_prefix}_{group_id}"
        try:
            model, forecast = train_prophet_model(
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
            print(f"Error training {group_name}: {e}")
    
    return results

# COMMAND ----------

# Train plant-level forecasts
plant_models = train_multi_prophet_models(
    scrap_df,
    date_col="transaction_date",
    value_col="total_scrap_cost",
    group_col="plant_id",
    group_name_col="plant_name_lookup",
    model_prefix="plant_forecast"
)

# COMMAND ----------

# Visualize plant forecasts
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for idx, (plant_id, data) in enumerate(plant_models.items()):
    if idx >= 6:
        break
    
    ax = axes[idx]
    forecast = data["forecast"]
    train = data["training_data"]
    
    # Plot training data
    ax.plot(train['ds'], train['y'], 'k.', alpha=0.5, label='Actual')
    
    # Plot forecast
    ax.plot(forecast['ds'], forecast['yhat'], 'b-', label='Forecast')
    ax.fill_between(
        forecast['ds'],
        forecast['yhat_lower'],
        forecast['yhat_upper'],
        alpha=0.3
    )
    
    ax.set_title(data["name"])
    ax.set_xlabel("Date")
    ax.set_ylabel("Scrap Cost ($)")
    ax.legend(loc='upper left')

plt.tight_layout()
plt.suptitle("Plant-Level Scrap Cost Forecasts", y=1.02)
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Forecast by Supplier

# COMMAND ----------

# Train supplier-level forecasts (top 10 suppliers by scrap volume)
supplier_models = train_multi_prophet_models(
    scrap_df,
    date_col="transaction_date",
    value_col="total_scrap_cost",
    group_col="supplier_id",
    group_name_col="supplier_name_lookup",
    model_prefix="supplier_forecast",
    top_n=10
)

# COMMAND ----------

# Visualize top supplier forecasts
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for idx, (supplier_id, data) in enumerate(list(supplier_models.items())[:6]):
    ax = axes[idx]
    forecast = data["forecast"]
    train = data["training_data"]
    
    ax.plot(train['ds'], train['y'], 'k.', alpha=0.5, label='Actual')
    ax.plot(forecast['ds'], forecast['yhat'], 'b-', label='Forecast')
    ax.fill_between(
        forecast['ds'],
        forecast['yhat_lower'],
        forecast['yhat_upper'],
        alpha=0.3
    )
    
    ax.set_title(data["name"][:30])
    ax.set_xlabel("Date")
    ax.set_ylabel("Scrap Cost ($)")

plt.tight_layout()
plt.suptitle("Top Supplier Scrap Cost Forecasts", y=1.02)
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Forecast by Part Number

# COMMAND ----------

# Train part-level forecasts (top 20 parts by scrap volume)
part_models = train_multi_prophet_models(
    scrap_df,
    date_col="transaction_date",
    value_col="total_scrap_cost",
    group_col="part_id",
    group_name_col="part_name_lookup",
    model_prefix="part_forecast",
    top_n=20
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Aggregate Forecasts and Save Results

# COMMAND ----------

def create_forecast_table(models_dict: dict, model_type: str, id_col: str) -> pd.DataFrame:
    """
    Create a consolidated forecast table from multiple models.
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

# Create consolidated forecast tables
plant_forecast_df = create_forecast_table(plant_models, "plant", "plant_id")
supplier_forecast_df = create_forecast_table(supplier_models, "supplier", "supplier_id")
part_forecast_df = create_forecast_table(part_models, "part", "part_id")

# Total forecast
total_future = total_forecast[total_forecast["ds"] > total_cost_data["ds"].max()].copy()
total_future["model_type"] = "total"
total_future["entity_id"] = "ALL"
total_future["entity_name"] = "Total Company"

# COMMAND ----------

# Save forecasts to Delta tables
from pyspark.sql.types import StructType, StructField, StringType, DateType, DoubleType

# Plant forecasts
if not plant_forecast_df.empty:
    plant_forecast_spark = spark.createDataFrame(plant_forecast_df)
    plant_forecast_spark = (
        plant_forecast_spark
        .withColumn("forecast_date", F.col("ds").cast("date"))
        .withColumn("forecast_generated_at", F.current_timestamp())
        .drop("ds")
    )
    plant_forecast_spark.write.mode("overwrite").saveAsTable(
        f"{CATALOG}.{SCHEMA}.ml_plant_forecast_results"
    )
    print(f"Saved {plant_forecast_spark.count()} plant forecast records")

# Supplier forecasts  
if not supplier_forecast_df.empty:
    supplier_forecast_spark = spark.createDataFrame(supplier_forecast_df)
    supplier_forecast_spark = (
        supplier_forecast_spark
        .withColumn("forecast_date", F.col("ds").cast("date"))
        .withColumn("forecast_generated_at", F.current_timestamp())
        .drop("ds")
    )
    supplier_forecast_spark.write.mode("overwrite").saveAsTable(
        f"{CATALOG}.{SCHEMA}.ml_supplier_forecast_results"
    )
    print(f"Saved {supplier_forecast_spark.count()} supplier forecast records")

# Part forecasts
if not part_forecast_df.empty:
    part_forecast_spark = spark.createDataFrame(part_forecast_df)
    part_forecast_spark = (
        part_forecast_spark
        .withColumn("forecast_date", F.col("ds").cast("date"))
        .withColumn("forecast_generated_at", F.current_timestamp())
        .drop("ds")
    )
    part_forecast_spark.write.mode("overwrite").saveAsTable(
        f"{CATALOG}.{SCHEMA}.ml_part_forecast_results"
    )
    print(f"Saved {part_forecast_spark.count()} part forecast records")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Forecast Summary Dashboard Query

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Next 30 days forecast summary
# MAGIC SELECT 
# MAGIC     model_type,
# MAGIC     entity_name,
# MAGIC     MIN(forecast_date) as forecast_start,
# MAGIC     MAX(forecast_date) as forecast_end,
# MAGIC     ROUND(SUM(yhat), 2) as total_predicted_cost,
# MAGIC     ROUND(SUM(yhat_lower), 2) as lower_bound,
# MAGIC     ROUND(SUM(yhat_upper), 2) as upper_bound
# MAGIC FROM (
# MAGIC     SELECT 'plant' as model_type, entity_name, forecast_date, yhat, yhat_lower, yhat_upper
# MAGIC     FROM welch.scrap_demo.ml_plant_forecast_results
# MAGIC     WHERE forecast_date <= DATE_ADD(CURRENT_DATE(), 30)
# MAGIC     UNION ALL
# MAGIC     SELECT 'supplier' as model_type, entity_name, forecast_date, yhat, yhat_lower, yhat_upper  
# MAGIC     FROM welch.scrap_demo.ml_supplier_forecast_results
# MAGIC     WHERE forecast_date <= DATE_ADD(CURRENT_DATE(), 30)
# MAGIC )
# MAGIC GROUP BY model_type, entity_name
# MAGIC ORDER BY total_predicted_cost DESC

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Model Registry - Register Best Models

# COMMAND ----------

# Register the total forecast model to Unity Catalog
model_name = f"{CATALOG}.{SCHEMA}.scrap_total_forecast_model"

with mlflow.start_run(run_name="register_total_model"):
    mlflow.prophet.log_model(
        total_model,
        "model",
        registered_model_name=model_name
    )
    
print(f"Model registered: {model_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next Steps
# MAGIC 
# MAGIC 1. **Schedule this notebook** to run weekly/monthly for updated forecasts
# MAGIC 2. **Add AutoML comparison** using Databricks AutoML for time series
# MAGIC 3. **Create alerts** when forecast exceeds threshold
# MAGIC 4. **Add external regressors** (production volume, new part launches, etc.)
# MAGIC 5. **Evaluate with holdout set** for model validation

