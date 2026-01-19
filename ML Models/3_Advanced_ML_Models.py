# Databricks notebook source
# MAGIC %md
# MAGIC # 🚀 Advanced ML Models for Scrap Reduction
# MAGIC
# MAGIC ## Business Objective
# MAGIC
# MAGIC This notebook implements additional ML models to support proactive scrap reduction:
# MAGIC
# MAGIC | Model | Purpose | Business Value |
# MAGIC |-------|---------|----------------|
# MAGIC | **Anomaly Detection** | Detect unusual scrap spikes | Early warning system |
# MAGIC | **Supplier Risk Scoring** | Predict supplier quality issues | Proactive sourcing |
# MAGIC | **Equipment Failure Prediction** | Identify high-scrap equipment | Maintenance prioritization |
# MAGIC | **Defect Pattern Clustering** | Find hidden defect patterns | Root cause discovery |
# MAGIC | **Corrective Action Recommender** | Suggest proven actions | Faster 8D resolution |
# MAGIC
# MAGIC ## ML Framework
# MAGIC - **Algorithms**: Isolation Forest, Random Forest, K-Means, TF-IDF
# MAGIC - **Tracking**: Full MLflow experiment tracking
# MAGIC - **Output**: Delta tables for dashboards and alerts

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📦 Install Dependencies

# COMMAND ----------

# MAGIC %pip install scikit-learn mlflow[databricks] prophet sentence-transformers scipy

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🔄 Restart Python Environment

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📚 Import Libraries
# MAGIC
# MAGIC Import all required libraries:
# MAGIC - **Anomaly Detection**: Isolation Forest
# MAGIC - **Classification**: Random Forest, Gradient Boosting
# MAGIC - **Clustering**: K-Means, DBSCAN
# MAGIC - **Preprocessing**: StandardScaler, LabelEncoder
# MAGIC - **MLflow**: Experiment tracking

# COMMAND ----------

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import warnings
warnings.filterwarnings('ignore')

# Prophet for time series forecasting
from prophet import Prophet

# Sentence Transformers for semantic similarity
from sentence_transformers import SentenceTransformer

# Scikit-learn
from sklearn.ensemble import (
    IsolationForest, 
    RandomForestClassifier, 
    GradientBoostingRegressor
)
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    classification_report, accuracy_score, 
    silhouette_score, roc_auc_score,
    precision_score, recall_score, f1_score,
    mean_absolute_error, mean_squared_error
)

# MLflow
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="mlflow")

# PySpark
from pyspark.sql import functions as F
from pyspark.sql.window import Window

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# COMMAND ----------

# MAGIC %md
# MAGIC ## ⚙️ Configuration
# MAGIC
# MAGIC Set up Unity Catalog paths and MLflow experiment.

# COMMAND ----------

# Unity Catalog Configuration - Use widget parameters from job, with defaults for interactive use
dbutils.widgets.text("catalog", "scrap_catalog", "Catalog Name")
dbutils.widgets.text("schema", "scrap_demo", "Schema Name")

CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
FULL_SCHEMA = f"{CATALOG}.{SCHEMA}"

print(f"Using catalog: {CATALOG}, schema: {SCHEMA}")

# MLflow Configuration
# MLflow will use the default notebook experiment when running from Git folders
# This avoids the error: "Cannot create workspace MLflow experiment in a Git folder"
# EXPERIMENT_PATH = "/Shared/scrap_demo/advanced_ml"
# mlflow.set_experiment(EXPERIMENT_PATH)  # Removed for Git folder compatibility

# Enable sklearn autologging
mlflow.sklearn.autolog(log_input_examples=True, log_model_signatures=True)

print(f"✅ Configuration loaded")
print(f"   Catalog: {CATALOG}.{SCHEMA}")
print(f"   MLflow Experiment: Default Notebook Experiment")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # 📊 Load Data
# MAGIC ---

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load All Required Tables
# MAGIC
# MAGIC Load scrap transactions, equipment, suppliers, and 8D reports from Silver layer.

# COMMAND ----------

# Helper function to check if table exists
def table_exists(table_name):
    try:
        spark.table(table_name)
        return True
    except Exception:
        return False

# Validate required tables exist
required_tables = [
    f"{FULL_SCHEMA}.scrap_transactions_silver",
    f"{FULL_SCHEMA}.equipment_silver",
    f"{FULL_SCHEMA}.suppliers_silver",
    f"{FULL_SCHEMA}.eight_d_reports_silver",
    f"{FULL_SCHEMA}.supplier_scenario_gold"  # Used for reallocation optimizer
]
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

# Load data from Silver layer
scrap_df = spark.table(f"{FULL_SCHEMA}.scrap_transactions_silver")
equipment_df = spark.table(f"{FULL_SCHEMA}.equipment_silver")
suppliers_df = spark.table(f"{FULL_SCHEMA}.suppliers_silver")
eight_d_df = spark.table(f"{FULL_SCHEMA}.eight_d_reports_silver")

print("📊 Data Loaded:")
print(f"   Scrap Transactions: {scrap_df.count():,}")
print(f"   Equipment: {equipment_df.count():,}")
print(f"   Suppliers: {suppliers_df.count():,}")
print(f"   8D Reports: {eight_d_df.count():,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # 🚨 Model 1: Anomaly Detection
# MAGIC ---
# MAGIC
# MAGIC ## Objective
# MAGIC Detect unusual daily scrap patterns for early intervention.
# MAGIC
# MAGIC ## Approach
# MAGIC Use **Isolation Forest** algorithm which:
# MAGIC - Identifies data points that are "isolated" from normal patterns
# MAGIC - Works well with high-dimensional data
# MAGIC - Doesn't require labeled anomaly examples

# COMMAND ----------

# MAGIC %md
# MAGIC ## Aggregate Daily Scrap Metrics
# MAGIC
# MAGIC Create daily aggregations with features that may indicate anomalies:
# MAGIC - Quantity and cost metrics
# MAGIC - Diversity metrics (unique defects, parts, plants)
# MAGIC - Quality indicators (equipment age, supplier quality)

# COMMAND ----------

# Aggregate daily scrap with features
daily_scrap = (
    scrap_df
    .groupBy("transaction_date")
    .agg(
        # Volume metrics
        F.sum("quantity_scrapped").alias("daily_qty"),
        F.sum("total_scrap_cost").alias("daily_cost"),
        F.count("*").alias("daily_records"),
        
        # Diversity metrics
        F.countDistinct("defect_code").alias("unique_defects"),
        F.countDistinct("part_id").alias("unique_parts"),
        F.countDistinct("plant_id").alias("plants_affected"),
        F.countDistinct("supplier_id").alias("suppliers_involved"),
        
        # Quality indicators
        F.avg("equipment_age_years").alias("avg_equipment_age"),
        F.avg("supplier_quality_score").alias("avg_supplier_quality"),
        
        # Shift patterns
        F.sum(F.when(F.col("shift") == 3, 1).otherwise(0)).alias("night_shift_records")
    )
    .orderBy("transaction_date")
    .toPandas()
)

print(f"📊 Daily Aggregation:")
print(f"   Days: {len(daily_scrap):,}")
print(f"   Date Range: {daily_scrap['transaction_date'].min()} to {daily_scrap['transaction_date'].max()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Calculate Rolling Features
# MAGIC
# MAGIC Add time-based features that capture trends and volatility:
# MAGIC - 7-day rolling averages
# MAGIC - Percentage change from previous day
# MAGIC - Day of week patterns

# COMMAND ----------

# Add rolling features
daily_scrap["qty_rolling_7d"] = daily_scrap["daily_qty"].rolling(7, min_periods=1).mean()
daily_scrap["cost_rolling_7d"] = daily_scrap["daily_cost"].rolling(7, min_periods=1).mean()
daily_scrap["qty_pct_change"] = daily_scrap["daily_qty"].pct_change().fillna(0)

# Clip extreme percentage changes
daily_scrap["qty_pct_change"] = daily_scrap["qty_pct_change"].clip(-5, 5)

# Date features
daily_scrap["day_of_week"] = pd.to_datetime(daily_scrap["transaction_date"]).dt.dayofweek
daily_scrap["month"] = pd.to_datetime(daily_scrap["transaction_date"]).dt.month
daily_scrap["is_monday"] = (daily_scrap["day_of_week"] == 0).astype(int)

print(f"✅ Added {6} rolling/date features")
daily_scrap.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train Anomaly Detection Model
# MAGIC
# MAGIC Train Isolation Forest with MLflow tracking:
# MAGIC - **contamination**: Expected proportion of anomalies (~5%)
# MAGIC - **n_estimators**: Number of isolation trees

# COMMAND ----------

# Define features for anomaly detection
ANOMALY_FEATURES = [
    "daily_qty", "daily_cost", "daily_records", 
    "unique_defects", "unique_parts", "plants_affected",
    "avg_equipment_age", "qty_pct_change", "night_shift_records"
]

# Prepare feature matrix
X_anomaly = daily_scrap[ANOMALY_FEATURES].fillna(0).values

# Scale features (important for Isolation Forest)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_anomaly)

print(f"📊 Feature Matrix Shape: {X_scaled.shape}")

# COMMAND ----------

# DBTITLE 1,Train Anomaly Detection Model
import tempfile
import os
import pickle

# Disable autolog for custom logging
mlflow.sklearn.autolog(disable=True)

# Train Isolation Forest
with mlflow.start_run(run_name="scrap_anomaly_detection") as run:
    
    # ===== LOG PARAMETERS =====
    contamination = 0.05  # Expect ~5% anomalies
    n_estimators = 100
    
    mlflow.log_param("model_type", "IsolationForest")
    mlflow.log_param("contamination", contamination)
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("features", json.dumps(ANOMALY_FEATURES))
    mlflow.log_param("dataset_size", len(daily_scrap))
    
    # ===== TRAIN MODEL =====
    print("🔄 Training Isolation Forest model...")
    
    iso_forest = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=42,
        n_jobs=-1
    )
    
    iso_forest.fit(X_scaled)
    
    # ===== PREDICT ANOMALIES =====
    # -1 = anomaly, 1 = normal
    daily_scrap["anomaly_prediction"] = iso_forest.predict(X_scaled)
    daily_scrap["anomaly_score"] = iso_forest.score_samples(X_scaled)
    daily_scrap["is_anomaly"] = (daily_scrap["anomaly_prediction"] == -1).astype(int)
    
    # Count anomalies
    anomalies = daily_scrap[daily_scrap["is_anomaly"] == 1]
    anomaly_rate = len(anomalies) / len(daily_scrap)
    
    # ===== LOG METRICS =====
    mlflow.log_metric("anomalies_detected", len(anomalies))
    mlflow.log_metric("anomaly_rate", anomaly_rate)
    mlflow.log_metric("normal_days", len(daily_scrap) - len(anomalies))
    mlflow.log_metric("avg_anomaly_score", daily_scrap["anomaly_score"].mean())
    
    # Calculate anomaly characteristics
    avg_normal_qty = daily_scrap[daily_scrap["is_anomaly"] == 0]["daily_qty"].mean()
    avg_anomaly_qty = anomalies["daily_qty"].mean()
    qty_increase = (avg_anomaly_qty - avg_normal_qty) / avg_normal_qty * 100
    
    mlflow.log_metric("avg_normal_qty", avg_normal_qty)
    mlflow.log_metric("avg_anomaly_qty", avg_anomaly_qty)
    mlflow.log_metric("anomaly_qty_increase_pct", qty_increase)
    
    # ===== LOG VISUALIZATIONS =====
    # Time series plot with anomalies
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    
    ax1 = axes[0]
    ax1.plot(daily_scrap["transaction_date"], daily_scrap["daily_qty"], 
             'b-', alpha=0.5, linewidth=0.5, label="Daily Qty")
    ax1.scatter(
        anomalies["transaction_date"], 
        anomalies["daily_qty"],
        c='red', s=50, label=f"Anomaly ({len(anomalies)})", zorder=5
    )
    ax1.set_ylabel("Daily Quantity Scrapped")
    ax1.legend()
    ax1.set_title("Scrap Anomaly Detection - Daily Quantity")
    
    ax2 = axes[1]
    ax2.plot(daily_scrap["transaction_date"], daily_scrap["anomaly_score"], 
             'g-', alpha=0.7, linewidth=0.5)
    threshold = daily_scrap[daily_scrap["is_anomaly"] == 1]["anomaly_score"].max()
    ax2.axhline(y=threshold, color='red', linestyle='--', 
                label=f"Anomaly Threshold ({threshold:.3f})")
    ax2.set_ylabel("Anomaly Score")
    ax2.set_xlabel("Date")
    ax2.legend()
    
    plt.tight_layout()
    mlflow.log_figure(fig, "anomaly_detection_plot.png")
    plt.close()
    
    # ===== LOG MODEL =====
    mlflow.sklearn.log_model(iso_forest, "model")
    
    # ===== LOG SCALER (needed for inference) =====
    # Use temporary directory with proper permissions
    with tempfile.TemporaryDirectory() as tmpdir:
        scaler_path = os.path.join(tmpdir, "scaler.pkl")
        with open(scaler_path, "wb") as f:
            pickle.dump(scaler, f)
        mlflow.log_artifact(scaler_path)
    
    # ===== PRINT RESULTS =====
    print(f"\n{'='*60}")
    print(f"✅ Model: Scrap Anomaly Detection")
    print(f"{'='*60}")
    print(f"\n📊 Results:")
    print(f"   Days Analyzed: {len(daily_scrap):,}")
    print(f"   Anomalies Detected: {len(anomalies):,} ({anomaly_rate:.1%})")
    print(f"   Avg Normal Day Qty: {avg_normal_qty:,.0f}")
    print(f"   Avg Anomaly Day Qty: {avg_anomaly_qty:,.0f} (+{qty_increase:.0f}%)")
    print(f"\n🔗 MLflow Run ID: {run.info.run_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Visualize Anomaly Patterns
# MAGIC
# MAGIC Analyze characteristics of detected anomalies.

# COMMAND ----------

# Display anomaly timeline
fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

ax1 = axes[0]
ax1.plot(daily_scrap["transaction_date"], daily_scrap["daily_qty"], 
         'b-', alpha=0.5, linewidth=0.5, label="Daily Qty")
ax1.scatter(anomalies["transaction_date"], anomalies["daily_qty"],
            c='red', s=50, label=f"Anomalies ({len(anomalies)})", zorder=5)
ax1.set_ylabel("Quantity Scrapped", fontsize=12)
ax1.legend(loc='upper right')
ax1.set_title("Scrap Anomaly Detection Results", fontsize=14)
ax1.grid(alpha=0.3)

ax2 = axes[1]
ax2.fill_between(daily_scrap["transaction_date"], 
                  daily_scrap["anomaly_score"], 
                  alpha=0.7, color='green')
ax2.axhline(y=daily_scrap[daily_scrap["is_anomaly"]==1]["anomaly_score"].max(), 
            color='red', linestyle='--', label='Anomaly Threshold')
ax2.set_ylabel("Anomaly Score", fontsize=12)
ax2.set_xlabel("Date", fontsize=12)
ax2.legend(loc='upper right')
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Compare Normal vs Anomaly Characteristics
# MAGIC
# MAGIC Statistical comparison of normal days vs anomalous days.

# COMMAND ----------

# Compare normal vs anomaly characteristics
print("📊 Normal vs Anomaly Day Comparison:")
print("=" * 60)

comparison = pd.DataFrame({
    "Metric": ANOMALY_FEATURES,
    "Normal Mean": daily_scrap[daily_scrap["is_anomaly"] == 0][ANOMALY_FEATURES].mean().values,
    "Anomaly Mean": anomalies[ANOMALY_FEATURES].mean().values
})
comparison["Difference %"] = (
    (comparison["Anomaly Mean"] - comparison["Normal Mean"]) / 
    comparison["Normal Mean"] * 100
).round(1)

print(comparison.to_string(index=False))

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # 📈 Model 1B: Time Series Anomaly Detection
# MAGIC ---
# MAGIC
# MAGIC ## Objective
# MAGIC Detect anomalies using time series forecasting to account for:
# MAGIC - **Trends**: Long-term increase/decrease in scrap
# MAGIC - **Seasonality**: Weekly, monthly patterns
# MAGIC - **Sequential Dependencies**: How today relates to recent history
# MAGIC
# MAGIC ## Approach
# MAGIC 1. Train Prophet forecast model on historical scrap data
# MAGIC 2. Predict expected scrap with uncertainty intervals
# MAGIC 3. Flag anomalies when actual values exceed prediction intervals
# MAGIC 4. Calculate anomaly severity scores

# COMMAND ----------

# MAGIC %md
# MAGIC ## Prepare Time Series Data
# MAGIC
# MAGIC Prophet requires columns named 'ds' (date) and 'y' (value).

# COMMAND ----------

# Prepare data for Prophet
ts_data = daily_scrap[['transaction_date', 'daily_qty', 'daily_cost']].copy()
ts_data.columns = ['ds', 'y_qty', 'y_cost']
ts_data['ds'] = pd.to_datetime(ts_data['ds'])

# Sort by date
ts_data = ts_data.sort_values('ds').reset_index(drop=True)

print(f"📊 Time Series Data:")
print(f"   Date Range: {ts_data['ds'].min()} to {ts_data['ds'].max()}")
print(f"   Days: {len(ts_data):,}")
print(f"   Mean Daily Qty: {ts_data['y_qty'].mean():,.0f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train Prophet Model for Quantity Forecast

# COMMAND ----------

# Prepare Prophet dataframe for quantity
prophet_df_qty = ts_data[['ds', 'y_qty']].copy()
prophet_df_qty.columns = ['ds', 'y']

# Train/test split - use 80% for training to establish pattern
split_idx = int(len(prophet_df_qty) * 0.8)
train_prophet = prophet_df_qty[:split_idx].copy()
test_prophet = prophet_df_qty[split_idx:].copy()

print(f"📊 Prophet Training Split:")
print(f"   Training: {len(train_prophet):,} days")
print(f"   Testing: {len(test_prophet):,} days")
print(f"   Split Date: {train_prophet['ds'].max().date()}")

# COMMAND ----------

# Train Prophet model
with mlflow.start_run(run_name="ts_anomaly_prophet_qty") as run:
    
    # Log parameters
    mlflow.log_param("model_type", "Prophet")
    mlflow.log_param("target", "daily_quantity")
    mlflow.log_param("train_days", len(train_prophet))
    mlflow.log_param("test_days", len(test_prophet))
    mlflow.log_param("seasonality_mode", "multiplicative")
    mlflow.log_param("interval_width", 0.95)
    
    print("🔄 Training Prophet time series model...")
    
    # Initialize Prophet with seasonality
    prophet_model = Prophet(
        interval_width=0.95,  # 95% prediction intervals
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode='multiplicative',  # Better for growing trends
        changepoint_prior_scale=0.05  # Flexibility in trend changes
    )
    
    # Fit on training data
    prophet_model.fit(train_prophet)
    
    # Make predictions for entire period (train + test)
    future = prophet_model.make_future_dataframe(periods=len(test_prophet))
    forecast = prophet_model.predict(future)
    
    # Calculate metrics on test set
    test_forecast = forecast[forecast['ds'] >= test_prophet['ds'].min()].copy()
    test_merged = test_prophet.merge(test_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], on='ds')
    
    mae = mean_absolute_error(test_merged['y'], test_merged['yhat'])
    rmse = np.sqrt(mean_squared_error(test_merged['y'], test_merged['yhat']))
    mape = np.mean(np.abs((test_merged['y'] - test_merged['yhat']) / test_merged['y'])) * 100
    
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("mape", mape)
    
    # Detect anomalies: actual value outside prediction interval
    forecast_full = prophet_df_qty.merge(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], on='ds')
    forecast_full['is_anomaly_prophet'] = (
        (forecast_full['y'] > forecast_full['yhat_upper']) |
        (forecast_full['y'] < forecast_full['yhat_lower'])
    ).astype(int)
    
    # Calculate anomaly severity (how many std devs from forecast)
    forecast_full['forecast_error'] = forecast_full['y'] - forecast_full['yhat']
    forecast_full['forecast_error_std'] = (
        forecast_full['forecast_error'] / (forecast_full['yhat_upper'] - forecast_full['yhat_lower']) * 4
    )  # Convert to approximate std devs (95% CI ≈ 4 std devs)
    
    forecast_full['anomaly_severity'] = np.abs(forecast_full['forecast_error_std'])
    
    # Count anomalies
    ts_anomalies = forecast_full[forecast_full['is_anomaly_prophet'] == 1]
    ts_anomaly_rate = len(ts_anomalies) / len(forecast_full)
    
    mlflow.log_metric("ts_anomalies_detected", len(ts_anomalies))
    mlflow.log_metric("ts_anomaly_rate", ts_anomaly_rate)
    mlflow.log_metric("avg_anomaly_severity", ts_anomalies['anomaly_severity'].mean() if len(ts_anomalies) > 0 else 0)
    
    # Log visualizations
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    
    # Plot 1: Forecast with anomalies
    ax1 = axes[0]
    ax1.plot(forecast_full['ds'], forecast_full['y'], 'k.', alpha=0.5, markersize=3, label='Actual')
    ax1.plot(forecast_full['ds'], forecast_full['yhat'], 'b-', linewidth=2, label='Forecast')
    ax1.fill_between(forecast_full['ds'], 
                       forecast_full['yhat_lower'], 
                       forecast_full['yhat_upper'],
                       alpha=0.2, color='blue', label='95% Interval')
    ax1.scatter(ts_anomalies['ds'], ts_anomalies['y'], 
                c='red', s=50, zorder=5, label=f'Anomalies ({len(ts_anomalies)})')
    ax1.axvline(x=train_prophet['ds'].max(), color='green', linestyle='--', 
                label='Train/Test Split', linewidth=2)
    ax1.set_ylabel('Daily Quantity Scrapped')
    ax1.set_title('Time Series Anomaly Detection with Prophet', fontsize=14)
    ax1.legend(loc='upper left')
    ax1.grid(alpha=0.3)
    
    # Plot 2: Forecast error
    ax2 = axes[1]
    ax2.bar(forecast_full['ds'], forecast_full['forecast_error'], 
            alpha=0.6, color=['red' if x == 1 else 'blue' for x in forecast_full['is_anomaly_prophet']])
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_ylabel('Forecast Error (Actual - Predicted)')
    ax2.set_title('Forecast Error Over Time')
    ax2.grid(alpha=0.3)
    
    # Plot 3: Anomaly severity
    ax3 = axes[2]
    ax3.plot(forecast_full['ds'], forecast_full['anomaly_severity'], 
             'g-', alpha=0.7, linewidth=1)
    ax3.axhline(y=2, color='orange', linestyle='--', label='Moderate (2σ)')
    ax3.axhline(y=3, color='red', linestyle='--', label='Severe (3σ)')
    ax3.scatter(ts_anomalies['ds'], ts_anomalies['anomaly_severity'],
                c='red', s=30, zorder=5)
    ax3.set_ylabel('Anomaly Severity (σ)')
    ax3.set_xlabel('Date')
    ax3.set_title('Anomaly Severity Score')
    ax3.legend()
    ax3.grid(alpha=0.3)
    
    plt.tight_layout()
    mlflow.log_figure(fig, "ts_anomaly_detection.png")
    plt.close()
    
    # Print results
    print(f"\n{'='*70}")
    print(f"✅ Model: Time Series Anomaly Detection (Prophet)")
    print(f"{'='*70}")
    print(f"\n📊 Forecast Performance:")
    print(f"   MAE: {mae:,.1f} units")
    print(f"   RMSE: {rmse:,.1f} units")
    print(f"   MAPE: {mape:.1f}%")
    
    print(f"\n📊 Anomaly Detection:")
    print(f"   Total Days: {len(forecast_full):,}")
    print(f"   Anomalies Detected: {len(ts_anomalies):,} ({ts_anomaly_rate:.1%})")
    print(f"   Avg Anomaly Severity: {ts_anomalies['anomaly_severity'].mean():.2f}σ" if len(ts_anomalies) > 0 else "N/A")
    
    print(f"\n🔗 MLflow Run ID: {run.info.run_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Analyze Time Series Anomalies
# MAGIC
# MAGIC Compare time series anomalies with Isolation Forest anomalies.

# COMMAND ----------

# DBTITLE 1,Compare Anomaly Detection Methods
# Compare TS anomalies with Isolation Forest anomalies
comparison_df = pd.DataFrame({
    'date': pd.to_datetime(forecast_full['ds']).dt.date,  # Convert to date only
    'daily_qty': forecast_full['y'],
    'prophet_forecast': forecast_full['yhat'],
    'forecast_lower': forecast_full['yhat_lower'],
    'forecast_upper': forecast_full['yhat_upper'],
    'ts_anomaly': forecast_full['is_anomaly_prophet'],
    'anomaly_severity': forecast_full['anomaly_severity']
})

# Convert transaction_date to date type for matching
daily_scrap_for_merge = daily_scrap[['transaction_date', 'is_anomaly']].copy()
daily_scrap_for_merge['transaction_date'] = pd.to_datetime(daily_scrap_for_merge['transaction_date']).dt.date

# Merge with Isolation Forest results
comparison_df = comparison_df.merge(
    daily_scrap_for_merge,
    left_on='date',
    right_on='transaction_date',
    how='left'
)
comparison_df.rename(columns={'is_anomaly': 'iso_forest_anomaly'}, inplace=True)
comparison_df.drop('transaction_date', axis=1, inplace=True)

# Fill NaN values in iso_forest_anomaly with 0 (for dates not in daily_scrap)
comparison_df['iso_forest_anomaly'] = comparison_df['iso_forest_anomaly'].fillna(0).astype(int)

# Calculate agreement
both_methods = (comparison_df['ts_anomaly'] == 1) & (comparison_df['iso_forest_anomaly'] == 1)
only_ts = (comparison_df['ts_anomaly'] == 1) & (comparison_df['iso_forest_anomaly'] == 0)
only_iso = (comparison_df['ts_anomaly'] == 0) & (comparison_df['iso_forest_anomaly'] == 1)
neither = (comparison_df['ts_anomaly'] == 0) & (comparison_df['iso_forest_anomaly'] == 0)

print("📊 ANOMALY DETECTION METHOD COMPARISON:")
print("=" * 70)
print(f"\n   Both Methods Agree (Anomaly): {both_methods.sum():,} days")
print(f"   Only Time Series Detected: {only_ts.sum():,} days")
print(f"   Only Isolation Forest Detected: {only_iso.sum():,} days")
print(f"   Both Methods Agree (Normal): {neither.sum():,} days")
print(f"\n   Agreement Rate: {(both_methods.sum() + neither.sum()) / len(comparison_df):.1%}")

# Visualize comparison
fig, ax = plt.subplots(figsize=(14, 6))

# Convert date back to datetime for plotting
comparison_df['date_plot'] = pd.to_datetime(comparison_df['date'])

# Plot actual values
ax.plot(comparison_df['date_plot'], comparison_df['daily_qty'], 
        'k-', alpha=0.3, linewidth=1, label='Daily Qty')

# Mark anomalies by method
both_dates = comparison_df[both_methods]['date_plot']
both_vals = comparison_df[both_methods]['daily_qty']
ax.scatter(both_dates, both_vals, c='red', s=100, marker='o', 
           label=f'Both Methods ({len(both_dates)})', zorder=5)

ts_only_dates = comparison_df[only_ts]['date_plot']
ts_only_vals = comparison_df[only_ts]['daily_qty']
ax.scatter(ts_only_dates, ts_only_vals, c='orange', s=80, marker='^',
           label=f'Time Series Only ({len(ts_only_dates)})', zorder=4)

iso_only_dates = comparison_df[only_iso]['date_plot']
iso_only_vals = comparison_df[only_iso]['daily_qty']
ax.scatter(iso_only_dates, iso_only_vals, c='blue', s=80, marker='s',
           label=f'Isolation Forest Only ({len(iso_only_dates)})', zorder=4)

ax.set_xlabel('Date', fontsize=12)
ax.set_ylabel('Daily Quantity Scrapped', fontsize=12)
ax.set_title('Anomaly Detection: Time Series vs Isolation Forest', fontsize=14)
ax.legend(loc='upper left')
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()

print("\n💡 Insights:")
print("   • Time Series: Catches deviations from expected trends/patterns")
print("   • Isolation Forest: Catches outliers based on feature distributions")
print("   • Both methods together provide comprehensive anomaly detection")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Categorize Anomalies by Severity

# COMMAND ----------

# Categorize anomalies by severity
severe_threshold = 3.0  # 3 standard deviations
moderate_threshold = 2.0  # 2 standard deviations

ts_anomalies_categorized = ts_anomalies.copy()
ts_anomalies_categorized['severity_category'] = pd.cut(
    ts_anomalies_categorized['anomaly_severity'],
    bins=[0, moderate_threshold, severe_threshold, float('inf')],
    labels=['Low', 'Moderate', 'Severe']
)

print("📊 ANOMALY SEVERITY BREAKDOWN:")
print("=" * 70)
severity_summary = ts_anomalies_categorized.groupby('severity_category').agg({
    'ds': 'count',
    'y': 'mean',
    'anomaly_severity': 'mean'
}).round(2)
severity_summary.columns = ['Count', 'Avg Daily Qty', 'Avg Severity (σ)']
print(severity_summary)

# Display most severe anomalies
print("\n🚨 TOP 10 MOST SEVERE ANOMALIES:")
print("=" * 70)
top_anomalies = ts_anomalies_categorized.nlargest(10, 'anomaly_severity')[
    ['ds', 'y', 'yhat', 'anomaly_severity']
]
top_anomalies.columns = ['Date', 'Actual Qty', 'Expected Qty', 'Severity (σ)']
print(top_anomalies.to_string(index=False))

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # 📊 Model 2: Supplier Risk Scoring
# MAGIC ---
# MAGIC
# MAGIC ## Objective
# MAGIC Predict which suppliers will have high scrap in the upcoming month.
# MAGIC
# MAGIC ## Approach
# MAGIC - Aggregate monthly supplier performance metrics
# MAGIC - Train classifier to predict "high scrap" next month
# MAGIC - Score all suppliers for current risk level

# COMMAND ----------

# MAGIC %md
# MAGIC ## Aggregate Monthly Supplier Metrics
# MAGIC
# MAGIC Create monthly aggregations of supplier performance.

# COMMAND ----------

# Aggregate supplier performance by month
supplier_metrics = (
    scrap_df
    .groupBy(
        "supplier_id", "supplier_name_lookup", "supplier_tier_lookup",
        F.year("transaction_date").alias("year"),
        F.month("transaction_date").alias("month")
    )
    .agg(
        # Volume metrics
        F.sum("quantity_scrapped").alias("monthly_qty_scrapped"),
        F.sum("total_scrap_cost").alias("monthly_scrap_cost"),
        F.count("*").alias("monthly_scrap_incidents"),
        
        # Diversity metrics
        F.countDistinct("defect_code").alias("unique_defects"),
        F.countDistinct("part_id").alias("parts_affected"),
        
        # Quality indicators
        F.avg("supplier_quality_score").alias("supplier_quality_score"),
        
        # Defect breakdown
        F.sum(F.when(F.col("defect_category") == "Material Defects", 1)
              .otherwise(0)).alias("material_defects"),
        F.sum(F.when(F.col("disposition") == "Return to Supplier", 1)
              .otherwise(0)).alias("supplier_returns")
    )
    .toPandas()
)

print(f"📊 Supplier Monthly Metrics:")
print(f"   Records: {len(supplier_metrics):,}")
print(f"   Unique Suppliers: {supplier_metrics['supplier_id'].nunique()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Target Variable
# MAGIC
# MAGIC Define "high scrap" as next month's quantity being >20% above supplier's median.

# COMMAND ----------

# Sort by supplier and time
supplier_metrics = supplier_metrics.sort_values(["supplier_id", "year", "month"])

# Create next month's quantity (target)
supplier_metrics["next_month_qty"] = (
    supplier_metrics.groupby("supplier_id")["monthly_qty_scrapped"].shift(-1)
)

# Calculate supplier-specific median
supplier_medians = supplier_metrics.groupby("supplier_id")["monthly_qty_scrapped"].median()
supplier_metrics["supplier_median"] = supplier_metrics["supplier_id"].map(supplier_medians)

# Define high scrap: >20% above supplier's median
supplier_metrics["high_scrap_next_month"] = (
    supplier_metrics["next_month_qty"] > supplier_metrics["supplier_median"] * 1.2
).astype(int)

# Remove rows without next month data
supplier_model_data = supplier_metrics.dropna(subset=["next_month_qty"])

print(f"📊 Training Data:")
print(f"   Records: {len(supplier_model_data):,}")
print(f"   High Scrap Months: {supplier_model_data['high_scrap_next_month'].sum():,} "
      f"({supplier_model_data['high_scrap_next_month'].mean():.1%})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Prepare Features and Train Model

# COMMAND ----------

# Define features
SUPPLIER_FEATURES = [
    "monthly_qty_scrapped", "monthly_scrap_cost", "monthly_scrap_incidents",
    "unique_defects", "parts_affected", "supplier_quality_score",
    "material_defects", "supplier_returns"
]

# Encode tier
tier_encoder = LabelEncoder()
supplier_model_data["tier_encoded"] = tier_encoder.fit_transform(
    supplier_model_data["supplier_tier_lookup"].astype(str)
)

feature_cols = SUPPLIER_FEATURES + ["tier_encoded"]

# Prepare X and y
X_supplier = supplier_model_data[feature_cols].fillna(0).values
y_supplier = supplier_model_data["high_scrap_next_month"].values

# Train/test split
X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
    X_supplier, y_supplier, test_size=0.2, random_state=42
)

print(f"📊 Data Split:")
print(f"   Training: {len(X_train_s):,}")
print(f"   Testing: {len(X_test_s):,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train Supplier Risk Model with MLflow

# COMMAND ----------

# Train supplier risk model
with mlflow.start_run(run_name="supplier_risk_model") as run:
    
    # Log parameters
    mlflow.log_param("model_type", "RandomForestClassifier")
    mlflow.log_param("features", json.dumps(feature_cols))
    mlflow.log_param("dataset_size", len(supplier_model_data))
    mlflow.log_param("high_scrap_threshold", "120% of median")
    
    # Define model
    supplier_risk_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )
    
    # Log hyperparameters
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 10)
    mlflow.log_param("class_weight", "balanced")
    
    # Train
    print("🔄 Training Supplier Risk model...")
    supplier_risk_model.fit(X_train_s, y_train_s)
    
    # Predict
    y_pred_s = supplier_risk_model.predict(X_test_s)
    y_proba_s = supplier_risk_model.predict_proba(X_test_s)[:, 1]
    
    # Metrics
    accuracy = accuracy_score(y_test_s, y_pred_s)
    precision = precision_score(y_test_s, y_pred_s, zero_division=0)
    recall = recall_score(y_test_s, y_pred_s, zero_division=0)
    f1 = f1_score(y_test_s, y_pred_s, zero_division=0)
    
    try:
        auc = roc_auc_score(y_test_s, y_proba_s)
        mlflow.log_metric("auc_roc", auc)
    except:
        auc = None
    
    # Log metrics
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)
    
    # Feature importance
    importance_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": supplier_risk_model.feature_importances_
    }).sort_values("importance", ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=importance_df, x="importance", y="feature", ax=ax, palette="rocket")
    ax.set_title("Supplier Risk - Feature Importance")
    plt.tight_layout()
    mlflow.log_figure(fig, "feature_importance.png")
    plt.close()
    
    # Log model
    mlflow.sklearn.log_model(supplier_risk_model, "model")
    
    # Print results
    print(f"\n{'='*60}")
    print(f"✅ Model: Supplier Risk Predictor")
    print(f"{'='*60}")
    print(f"\n📊 Test Metrics:")
    print(f"   Accuracy:  {accuracy:.4f}")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall:    {recall:.4f}")
    print(f"   F1 Score:  {f1:.4f}")
    if auc:
        print(f"   AUC-ROC:   {auc:.4f}")
    print(f"\n🔗 MLflow Run ID: {run.info.run_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Score Current Suppliers
# MAGIC
# MAGIC Apply model to score all suppliers for current risk level.

# COMMAND ----------

# Get most recent month for each supplier
latest_month = supplier_model_data.groupby("supplier_id").tail(1).copy()

# Score suppliers
X_latest = latest_month[feature_cols].fillna(0).values
risk_scores = supplier_risk_model.predict_proba(X_latest)[:, 1]

latest_month["risk_score"] = risk_scores
latest_month["risk_level"] = pd.cut(
    risk_scores, 
    bins=[0, 0.3, 0.6, 1.0], 
    labels=["Low", "Medium", "High"]
)

# Display high risk suppliers
print("🚨 HIGH RISK SUPPLIERS:")
print("=" * 70)
high_risk = latest_month[latest_month["risk_level"] == "High"].sort_values(
    "risk_score", ascending=False
)
print(high_risk[["supplier_name_lookup", "supplier_tier_lookup", 
                  "risk_score", "monthly_scrap_cost"]].head(10).to_string(index=False))

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # 🔧 Model 3: Equipment Failure Prediction
# MAGIC ---
# MAGIC
# MAGIC ## Objective
# MAGIC Identify equipment likely to cause high scrap based on age, OEE, and history.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Aggregate Equipment Scrap Metrics

# COMMAND ----------

# Aggregate scrap by equipment
equipment_scrap = (
    scrap_df
    .filter(F.col("equipment_id").isNotNull())
    .groupBy("equipment_id", "equipment_name_lookup", 
             "equipment_age_years", "equipment_oee")
    .agg(
        F.sum("quantity_scrapped").alias("total_qty_scrapped"),
        F.sum("total_scrap_cost").alias("total_scrap_cost"),
        F.count("*").alias("scrap_incidents"),
        F.countDistinct("defect_code").alias("unique_defects"),
        F.avg("defect_severity_weight").alias("avg_defect_severity")
    )
    .toPandas()
)

# Create high-scrap equipment flag (top 25%)
threshold = equipment_scrap["total_qty_scrapped"].quantile(0.75)
equipment_scrap["high_scrap_equipment"] = (
    equipment_scrap["total_qty_scrapped"] > threshold
).astype(int)

print(f"📊 Equipment Data:")
print(f"   Total Equipment: {len(equipment_scrap):,}")
print(f"   High-Scrap Equipment (top 25%): {equipment_scrap['high_scrap_equipment'].sum():,}")
print(f"   Scrap Threshold: {threshold:,.0f} units")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train Equipment Failure Model

# COMMAND ----------

# Define features
EQUIP_FEATURES = ["equipment_age_years", "equipment_oee", 
                   "scrap_incidents", "unique_defects"]

X_equip = equipment_scrap[EQUIP_FEATURES].fillna(0).values
y_equip = equipment_scrap["high_scrap_equipment"].values

# Train/test split
X_train_e, X_test_e, y_train_e, y_test_e = train_test_split(
    X_equip, y_equip, test_size=0.2, random_state=42
)

# Train model
with mlflow.start_run(run_name="equipment_scrap_predictor") as run:
    
    mlflow.log_param("model_type", "RandomForestClassifier")
    mlflow.log_param("features", json.dumps(EQUIP_FEATURES))
    
    equip_model = RandomForestClassifier(
        n_estimators=50, 
        max_depth=5, 
        random_state=42
    )
    equip_model.fit(X_train_e, y_train_e)
    
    y_pred_e = equip_model.predict(X_test_e)
    accuracy_e = accuracy_score(y_test_e, y_pred_e)
    
    mlflow.log_metric("accuracy", accuracy_e)
    mlflow.sklearn.log_model(equip_model, "model")
    
    print(f"\n✅ Equipment Scrap Prediction Accuracy: {accuracy_e:.4f}")

# Display feature importance
equip_importance = pd.DataFrame({
    "feature": EQUIP_FEATURES,
    "importance": equip_model.feature_importances_
}).sort_values("importance", ascending=False)

print("\n📊 Equipment Risk Factors:")
print(equip_importance.to_string(index=False))

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # 🔬 Model 4: Defect Pattern Clustering
# MAGIC ---
# MAGIC
# MAGIC ## Objective
# MAGIC Use unsupervised learning to discover hidden patterns in defect data.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Aggregate Defect Patterns

# COMMAND ----------

# Aggregate by defect code
defect_patterns = (
    scrap_df
    .groupBy("defect_code", "defect_category")
    .agg(
        F.sum("quantity_scrapped").alias("total_qty"),
        F.sum("total_scrap_cost").alias("total_cost"),
        F.count("*").alias("occurrences"),
        F.countDistinct("plant_id").alias("plants_affected"),
        F.countDistinct("supplier_id").alias("suppliers_involved"),
        F.countDistinct("part_id").alias("parts_affected"),
        F.avg("equipment_age_years").alias("avg_equipment_age"),
        F.avg("operator_experience_years").alias("avg_operator_experience"),
        F.sum(F.when(F.col("shift") == 3, 1).otherwise(0)).alias("night_shift_count"),
        F.avg("supplier_quality_score").alias("avg_supplier_quality")
    )
    .toPandas()
)

print(f"📊 Defect Codes: {len(defect_patterns):,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Find Optimal Number of Clusters
# MAGIC
# MAGIC Use silhouette score to determine optimal K for K-Means.

# COMMAND ----------

# Define clustering features
CLUSTER_FEATURES = [
    "total_qty", "total_cost", "occurrences", "plants_affected",
    "suppliers_involved", "parts_affected", "avg_equipment_age",
    "night_shift_count", "avg_supplier_quality"
]

X_cluster = defect_patterns[CLUSTER_FEATURES].fillna(0).values
X_cluster_scaled = StandardScaler().fit_transform(X_cluster)

# Find optimal clusters using silhouette score
silhouette_scores = []
K_range = range(2, 8)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_cluster_scaled)
    score = silhouette_score(X_cluster_scaled, labels)
    silhouette_scores.append(score)
    print(f"   K={k}: Silhouette Score = {score:.3f}")

best_k = K_range[np.argmax(silhouette_scores)]
print(f"\n✅ Optimal K: {best_k} (Silhouette: {max(silhouette_scores):.3f})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train Final Clustering Model

# COMMAND ----------

# Train final clustering model
with mlflow.start_run(run_name="defect_clustering") as run:
    
    mlflow.log_param("model_type", "KMeans")
    mlflow.log_param("n_clusters", best_k)
    mlflow.log_param("features", json.dumps(CLUSTER_FEATURES))
    
    kmeans_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    defect_patterns["cluster"] = kmeans_final.fit_predict(X_cluster_scaled)
    
    mlflow.log_metric("silhouette_score", max(silhouette_scores))
    mlflow.log_metric("inertia", kmeans_final.inertia_)
    mlflow.sklearn.log_model(kmeans_final, "model")
    
    print(f"\n✅ Clustering complete: {best_k} clusters")
    print(f"🔗 MLflow Run ID: {run.info.run_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Analyze Clusters
# MAGIC
# MAGIC Understand characteristics of each defect cluster.

# COMMAND ----------

# Analyze each cluster
print("\n📊 DEFECT CLUSTER ANALYSIS")
print("=" * 70)

for cluster_id in range(best_k):
    cluster_data = defect_patterns[defect_patterns["cluster"] == cluster_id]
    
    print(f"\n🔵 Cluster {cluster_id}: {len(cluster_data)} defect codes")
    print("-" * 50)
    print(f"   Categories: {cluster_data['defect_category'].unique().tolist()}")
    print(f"   Avg Total Cost: ${cluster_data['total_cost'].mean():,.0f}")
    print(f"   Avg Occurrences: {cluster_data['occurrences'].mean():,.0f}")
    print(f"   Avg Plants Affected: {cluster_data['plants_affected'].mean():.1f}")
    print(f"   Avg Equipment Age: {cluster_data['avg_equipment_age'].mean():.1f} years")
    print(f"   Avg Supplier Quality: {cluster_data['avg_supplier_quality'].mean():.3f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Visualize Clusters

# COMMAND ----------

# Visualize clusters
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Scatter plot
ax1 = axes[0]
scatter = ax1.scatter(
    defect_patterns["total_cost"],
    defect_patterns["occurrences"],
    c=defect_patterns["cluster"],
    cmap="viridis",
    s=defect_patterns["parts_affected"] * 5,
    alpha=0.6
)
ax1.set_xlabel("Total Scrap Cost ($)", fontsize=12)
ax1.set_ylabel("Occurrences", fontsize=12)
ax1.set_title("Defect Clusters by Cost and Frequency", fontsize=14)
plt.colorbar(scatter, ax=ax1, label="Cluster")

# Cluster summary bar chart
ax2 = axes[1]
cluster_summary = defect_patterns.groupby("cluster").agg({
    "total_cost": "sum",
    "occurrences": "sum"
}).reset_index()
bars = ax2.bar(cluster_summary["cluster"], cluster_summary["total_cost"], 
               color=plt.cm.viridis(cluster_summary["cluster"]/best_k))
ax2.set_xlabel("Cluster", fontsize=12)
ax2.set_ylabel("Total Scrap Cost ($)", fontsize=12)
ax2.set_title("Total Cost by Defect Cluster", fontsize=14)

plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # 💡 Model 5: Enhanced Corrective Action Recommender
# MAGIC ---
# MAGIC
# MAGIC ## Objective
# MAGIC Recommend corrective actions based on defect patterns and historical effectiveness.
# MAGIC
# MAGIC ## Enhancements
# MAGIC - **Contextual Filtering**: Filter by plant, supplier tier, cost
# MAGIC - **BERT Similarity**: Find similar defects when exact match fails
# MAGIC - **Confidence Intervals**: Show statistical confidence in effectiveness
# MAGIC - **Usage Tracking**: Log recommendations for future improvement

# COMMAND ----------

# MAGIC %md
# MAGIC ## Build Enhanced Action-Defect Mappings

# COMMAND ----------

# DBTITLE 1,Create Action Patterns
# Load 8D data with additional context
eight_d_pd = eight_d_df.toPandas()

# Create comprehensive action patterns with context
# Note: Removed 'supplier_tier_lookup' as it doesn't exist in the table
action_patterns = eight_d_pd.groupby([
    "defect_category", 
    "root_cause_category",
    "plant_id"
]).agg({
    "corrective_actions": lambda x: list(x),
    "verification_result": lambda x: list(x),
    "eight_d_id": "count",
    "total_scrap_cost": "mean"
}).reset_index()

# Calculate effectiveness rate and confidence interval
def calculate_effectiveness_with_ci(results_list):
    """
    Calculate effectiveness rate with 95% confidence interval.
    """
    if len(results_list) == 0:
        return 0, 0, 0
    
    effective_count = sum(1 for r in results_list if r == "Effective")
    total_count = len(results_list)
    rate = effective_count / total_count if total_count > 0 else 0
    
    # Wilson score interval (better for small samples)
    from scipy import stats
    if total_count > 0:
        # Binomial proportion confidence interval
        ci_low, ci_high = stats.binom.interval(0.95, total_count, rate)
        ci_low = ci_low / total_count
        ci_high = ci_high / total_count
    else:
        ci_low, ci_high = 0, 0
    
    return rate, ci_low, ci_high

# Apply to each row (returns 3 values: rate, ci_low, ci_high)
action_patterns[["effectiveness_rate", "ci_low", "ci_high"]] = action_patterns.apply(
    lambda row: pd.Series(calculate_effectiveness_with_ci(row["verification_result"])),
    axis=1
)

# Drop verification_result and rename columns
action_patterns.drop("verification_result", axis=1, inplace=True)

# Debug: print current columns before renaming
print(f"\n🔍 Current columns ({len(action_patterns.columns)}): {list(action_patterns.columns)}")

# Rename columns to be more descriptive
action_patterns.columns = [
    "defect_category", 
    "root_cause_category", 
    "plant_id",
    "actions",              # was: corrective_actions
    "sample_size",          # was: eight_d_id (count)
    "avg_scrap_cost",       # was: total_scrap_cost (mean)
    "effectiveness_rate",   # new column
    "ci_low",               # new column
    "ci_high"               # new column
]

print(f"\n📊 Enhanced Action Patterns: {len(action_patterns):,} combinations")
print(f"   Grouped by: defect_category, root_cause_category, plant_id")
print(f"   Avg actions per combination: {action_patterns['sample_size'].mean():.1f}")
print(f"   Avg effectiveness rate: {action_patterns['effectiveness_rate'].mean():.1%}")
print(f"   Total 8D reports analyzed: {action_patterns['sample_size'].sum():.0f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load BERT Model for Semantic Similarity

# COMMAND ----------

print("🔄 Loading BERT model for semantic defect matching...")
bert_model = SentenceTransformer('all-MiniLM-L6-v2')
print("✅ BERT model loaded")

# Create embeddings for all unique defect categories
unique_defects = action_patterns['defect_category'].unique()
defect_embeddings = bert_model.encode(unique_defects.tolist())

print(f"📊 Created embeddings for {len(unique_defects)} defect categories")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Enhanced Recommendation Function

# COMMAND ----------

def recommend_corrective_actions_enhanced(
    defect_category: str,
    root_cause: str = None,
    plant_id: str = None,
    supplier_tier: str = None,
    max_cost: float = None,
    top_n: int = 5,
    similarity_threshold: float = 0.7
) -> list:
    """
    Enhanced corrective action recommender with contextual filtering and BERT similarity.
    
    Parameters:
    -----------
    defect_category : str
        Category of defect (e.g., "Dimensional", "Surface Defects")
    root_cause : str, optional
        Root cause category (e.g., "Machine", "Material")
    plant_id : str, optional
        Filter recommendations by plant
    supplier_tier : str, optional
        Filter by supplier tier ("Tier 1", "Tier 2", "Tier 3")
    max_cost : float, optional
        Maximum average scrap cost for filtering
    top_n : int
        Number of recommendations to return
    similarity_threshold : float
        Minimum BERT similarity score for fuzzy matching (0-1)
        
    Returns:
    --------
    list of dict
        Recommendations with effectiveness metrics, confidence intervals, and metadata
    """
    
    # Step 1: Try exact match with filters
    matches = action_patterns[action_patterns["defect_category"] == defect_category].copy()
    
    # Apply optional filters
    if root_cause:
        matches = matches[matches["root_cause_category"] == root_cause]
    
    if plant_id:
        matches = matches[matches["plant_id"] == plant_id]
    
    # Only filter by supplier_tier if the column exists
    if supplier_tier and "supplier_tier" in action_patterns.columns:
        matches = matches[matches["supplier_tier"] == supplier_tier]
    
    if max_cost:
        matches = matches[matches["avg_scrap_cost"] <= max_cost]
    
    # Step 2: If no exact matches, use BERT similarity for fuzzy matching
    if len(matches) == 0:
        print(f"   No exact match for '{defect_category}', trying semantic similarity...")
        
        # Find similar defect categories using BERT
        query_embedding = bert_model.encode([defect_category])
        
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity(query_embedding, defect_embeddings)[0]
        
        # Get defects above similarity threshold
        similar_indices = np.where(similarities >= similarity_threshold)[0]
        
        if len(similar_indices) > 0:
            similar_defects = unique_defects[similar_indices]
            matches = action_patterns[
                action_patterns["defect_category"].isin(similar_defects)
            ].copy()
            
            # Add similarity score
            defect_to_similarity = dict(zip(unique_defects[similar_indices], 
                                           similarities[similar_indices]))
            matches["similarity_score"] = matches["defect_category"].map(defect_to_similarity)
            
            print(f"   Found {len(similar_defects)} similar defects via BERT")
        else:
            print(f"   No similar defects found (threshold: {similarity_threshold})")
    else:
        # Add perfect similarity score for exact matches
        matches["similarity_score"] = 1.0
    
    # Step 3: If still no matches, broaden search (remove filters)
    if len(matches) == 0:
        print(f"   Broadening search: removing contextual filters...")
        matches = action_patterns[
            action_patterns["defect_category"] == defect_category
        ].copy()
        matches["similarity_score"] = 1.0
    
    # Step 4: If still nothing, return message
    if len(matches) == 0:
        return [{
            "message": f"No historical actions found for '{defect_category}'",
            "suggestion": "Consider similar defect categories or consult 8D database"
        }]
    
    # Step 5: Score and rank recommendations
    # Score = effectiveness_rate * similarity_score * sqrt(sample_size)
    # Square root of sample size balances confidence with not over-weighting popular actions
    matches["recommendation_score"] = (
        matches["effectiveness_rate"] * 
        matches["similarity_score"] * 
        np.sqrt(matches["sample_size"])
    )
    
    matches = matches.sort_values("recommendation_score", ascending=False)
    
    # Step 6: Build recommendations
    recommendations = []
    for _, row in matches.head(top_n).iterrows():
        if row["actions"] and len(row["actions"]) > 0:
            # Get a sample action (first one)
            sample_action = str(row["actions"][0])[:300]
            
            # Build recommendation dict - handle missing supplier_tier gracefully
            rec_dict = {
                "defect_category": row["defect_category"],
                "root_cause": row["root_cause_category"],
                "plant": row["plant_id"],
                "sample_action": sample_action,
                "effectiveness_rate": f"{row['effectiveness_rate']:.0%}",
                "confidence_interval": f"({row['ci_low']:.0%} - {row['ci_high']:.0%})",
                "historical_cases": int(row["sample_size"]),
                "avg_cost": f"${row['avg_scrap_cost']:,.0f}",
                "similarity": f"{row['similarity_score']:.0%}",
                "recommendation_score": f"{row['recommendation_score']:.2f}",
                "match_type": "exact" if row["similarity_score"] == 1.0 else "similar"
            }
            
            # Add supplier_tier only if it exists in the row
            if "supplier_tier" in row.index and pd.notna(row["supplier_tier"]):
                rec_dict["supplier_tier"] = row["supplier_tier"]
            
            recommendations.append(rec_dict)
    
    return recommendations

print("✅ Enhanced recommendation function defined")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test Enhanced Recommender

# COMMAND ----------

# DBTITLE 1,Test Recommendation System
# Load 8D data with additional context
eight_d_pd = eight_d_df.toPandas()

# Print available columns for debugging
print(f"Available columns in eight_d_pd: {list(eight_d_pd.columns)}")

# Create comprehensive action patterns with context
# NOTE: Removed supplier_tier_lookup as it doesn't exist in this dataset
action_patterns = eight_d_pd.groupby([
    "defect_category", 
    "root_cause_category",
    "plant_id"
]).agg({
    "corrective_actions": lambda x: list(x),
    "verification_result": lambda x: list(x),
    "eight_d_id": "count",
    "total_scrap_cost": "mean"
}).reset_index()

# Calculate effectiveness rate and confidence interval
def calculate_effectiveness_with_ci(results_list):
    """Calculate effectiveness rate with 95% confidence interval."""
    if len(results_list) == 0:
        return 0, 0, 0
    
    effective_count = sum(1 for r in results_list if r == "Effective")
    total_count = len(results_list)
    rate = effective_count / total_count
    
    # Wilson score interval (better for small samples)
    from scipy import stats
    if total_count > 0:
        # Binomial proportion confidence interval
        ci_low, ci_high = stats.binom.interval(0.95, total_count, rate)
        ci_low = ci_low / total_count
        ci_high = ci_high / total_count
    else:
        ci_low, ci_high = 0, 0
    
    return rate, ci_low, ci_high

# Apply to each row
action_patterns[["effectiveness_rate", "ci_low", "ci_high"]] = action_patterns.apply(
    lambda row: pd.Series(calculate_effectiveness_with_ci(row["verification_result"])),
    axis=1
)

# Rename columns (including total_scrap_cost -> avg_scrap_cost)
action_patterns.rename(columns={
    "corrective_actions": "actions",
    "eight_d_id": "sample_size",
    "total_scrap_cost": "avg_scrap_cost"
}, inplace=True)

# Drop verification_result
action_patterns.drop("verification_result", axis=1, inplace=True)

# Reorder columns (without supplier_tier)
action_patterns = action_patterns[[
    "defect_category", "root_cause_category", "plant_id",
    "actions", "sample_size", "avg_scrap_cost", 
    "effectiveness_rate", "ci_low", "ci_high"
]]

print(f"📊 Enhanced Action Patterns: {len(action_patterns):,} combinations")
print(f"   Columns: {list(action_patterns.columns)}")
print(f"   Avg actions per combination: {action_patterns['sample_size'].mean():.1f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Recommendation Usage Tracking Table

# COMMAND ----------

# Create schema for tracking recommendations
from pyspark.sql.types import StructType, StructField, StringType, TimestampType, DoubleType, IntegerType, BooleanType

recommendation_log_schema = StructType([
    StructField("recommendation_id", StringType(), True),
    StructField("timestamp", TimestampType(), True),
    StructField("defect_category", StringType(), True),
    StructField("root_cause_category", StringType(), True),
    StructField("plant_id", StringType(), True),
    StructField("supplier_tier", StringType(), True),
    StructField("max_cost", DoubleType(), True),
    StructField("num_recommendations", IntegerType(), True),
    StructField("top_recommendation_action", StringType(), True),
    StructField("top_recommendation_effectiveness", DoubleType(), True),
    StructField("match_type", StringType(), True),
    StructField("user_selected", BooleanType(), True),
    StructField("selected_recommendation_index", IntegerType(), True),
    StructField("feedback_effective", BooleanType(), True),
    StructField("feedback_timestamp", TimestampType(), True)
])

# Initialize empty tracking table
tracking_df = spark.createDataFrame([], schema=recommendation_log_schema)
tracking_df.write.mode("overwrite").saveAsTable(
    f"{FULL_SCHEMA}.ml_action_recommendation_log"
)

print(f"✅ Created recommendation tracking table: {FULL_SCHEMA}.ml_action_recommendation_log")
print("   This table will log:")
print("   - Recommendation requests and parameters")
print("   - Which recommendations were shown")
print("   - Which recommendations were selected by users")
print("   - Feedback on effectiveness")
print("\n💡 Usage tracking enables future ML improvements:")
print("   - Identify which recommendations are most useful")
print("   - Measure adoption rates")
print("   - A/B test new recommendation strategies")
print("   - Train collaborative filtering when data accumulates")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Helper Function: Log Recommendations

# COMMAND ----------

def log_recommendation(
    defect_category: str,
    root_cause: str,
    recommendations: list,
    plant_id: str = None,
    supplier_tier: str = None,
    max_cost: float = None,
    user_selected: bool = False,
    selected_index: int = None
):
    """
    Log recommendation requests for future analysis and ML improvement.
    
    Parameters:
    -----------
    defect_category : str
        Defect category requested
    root_cause : str
        Root cause category
    recommendations : list
        List of recommendations returned
    plant_id : str, optional
        Plant filter used
    supplier_tier : str, optional
        Supplier tier filter used
    max_cost : float, optional
        Cost filter used
    user_selected : bool
        Whether user selected a recommendation
    selected_index : int, optional
        Which recommendation was selected (0-indexed)
    """
    import uuid
    from datetime import datetime
    
    # Create log entry
    log_entry = {
        "recommendation_id": str(uuid.uuid4()),
        "timestamp": datetime.now(),
        "defect_category": defect_category,
        "root_cause_category": root_cause,
        "plant_id": plant_id,
        "supplier_tier": supplier_tier,
        "max_cost": max_cost,
        "num_recommendations": len(recommendations),
        "top_recommendation_action": recommendations[0].get("sample_action", "")[:200] if recommendations else None,
        "top_recommendation_effectiveness": float(recommendations[0].get("effectiveness_rate", "0%").strip('%')) / 100 if recommendations and "effectiveness_rate" in recommendations[0] else None,
        "match_type": recommendations[0].get("match_type", "unknown") if recommendations else None,
        "user_selected": user_selected,
        "selected_recommendation_index": selected_index,
        "feedback_effective": None,  # To be updated later based on 8D verification
        "feedback_timestamp": None
    }
    
    # Append to tracking table - use explicit schema to avoid type inference issues
    log_df = spark.createDataFrame([log_entry], schema=recommendation_log_schema)
    log_df.write.mode("append").saveAsTable(
        f"{FULL_SCHEMA}.ml_action_recommendation_log"
    )
    
    return log_entry["recommendation_id"]

print("✅ Recommendation logging function defined")
print("   Call log_recommendation() after generating recommendations to track usage")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Demo: Log Sample Recommendations

# COMMAND ----------

# Demo logging
demo_recs = recommend_corrective_actions_enhanced(
    defect_category="Dimensional",
    root_cause="Machine",
    plant_id="P001",
    top_n=3
)

rec_id = log_recommendation(
    defect_category="Dimensional",
    root_cause="Machine",
    recommendations=demo_recs,
    plant_id="P001",
    supplier_tier=None,  # Explicitly pass None
    max_cost=None,       # Explicitly pass None
    user_selected=True,  # Simulate user selection
    selected_index=0
)

print(f"✅ Demo recommendation logged with ID: {rec_id}")
print(f"\n📊 View logs:")
print(f"   SELECT * FROM {FULL_SCHEMA}.ml_action_recommendation_log")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # 💾 Save Model Outputs
# MAGIC ---

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save Anomaly Detection Results
# MAGIC
# MAGIC Save both Isolation Forest and Time Series anomaly results.

# COMMAND ----------

# Combine both anomaly detection methods
anomaly_combined = comparison_df.copy()
anomaly_combined['combined_anomaly'] = (
    (anomaly_combined['ts_anomaly'] == 1) | 
    (anomaly_combined['iso_forest_anomaly'] == 1)
).astype(int)

anomaly_combined['anomaly_methods'] = ''
anomaly_combined.loc[anomaly_combined['ts_anomaly'] == 1, 'anomaly_methods'] = 'time_series'
anomaly_combined.loc[anomaly_combined['iso_forest_anomaly'] == 1, 'anomaly_methods'] += ',isolation_forest'
anomaly_combined['anomaly_methods'] = anomaly_combined['anomaly_methods'].str.strip(',')

# Save anomaly detection results
anomaly_spark = spark.createDataFrame(anomaly_combined)
anomaly_spark = (
    anomaly_spark
    .withColumn("detection_date", F.current_timestamp())
    .select(
        F.col("date").alias("transaction_date"),
        "daily_qty",
        "prophet_forecast",
        "forecast_lower",
        "forecast_upper",
        "ts_anomaly",
        "anomaly_severity",
        "iso_forest_anomaly",
        "combined_anomaly",
        "anomaly_methods",
        "detection_date"
    )
)

# Add overwriteSchema option to replace the old table schema
anomaly_spark.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(
    f"{FULL_SCHEMA}.ml_scrap_anomalies"
)
print(f"✅ Saved anomaly results: {anomaly_spark.count():,} days")
print(f"   Time Series Anomalies: {anomaly_spark.filter('ts_anomaly = 1').count():,}")
print(f"   Isolation Forest Anomalies: {anomaly_spark.filter('iso_forest_anomaly = 1').count():,}")
print(f"   Combined (Either Method): {anomaly_spark.filter('combined_anomaly = 1').count():,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save Supplier Risk Scores

# COMMAND ----------

# Save supplier risk scores
risk_output = latest_month[["supplier_id", "supplier_name_lookup", 
                             "risk_score", "risk_level", "monthly_scrap_cost"]].copy()

risk_spark = spark.createDataFrame(risk_output)
risk_spark = risk_spark.withColumn("scored_at", F.current_timestamp())

risk_spark.write.mode("overwrite").saveAsTable(
    f"{FULL_SCHEMA}.ml_supplier_risk_scores"
)
print(f"✅ Saved supplier risk scores: {risk_spark.count():,} suppliers")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save Defect Clusters

# COMMAND ----------

# Save defect clusters
cluster_spark = spark.createDataFrame(defect_patterns)
cluster_spark = cluster_spark.withColumn("clustered_at", F.current_timestamp())

cluster_spark.write.mode("overwrite").saveAsTable(
    f"{FULL_SCHEMA}.ml_defect_clusters"
)
print(f"✅ Saved defect clusters: {cluster_spark.count():,} defect codes")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # 📊 Summary
# MAGIC ---

# COMMAND ----------

# Print comprehensive summary
print("=" * 70)
print("📊 ADVANCED ML MODELS - SUMMARY")
print("=" * 70)

print(f"""
🚨 ANOMALY DETECTION
   • Algorithm: Isolation Forest + Prophet Time Series
   • Days Analyzed: {len(daily_scrap):,}
   • Isolation Forest Anomalies: {len(anomalies):,} ({len(anomalies)/len(daily_scrap):.1%})
   • Time Series Anomalies: {len(ts_anomalies):,} ({ts_anomaly_rate:.1%})
   • Combined Anomalies: {anomaly_combined['combined_anomaly'].sum():,}
   • Forecast MAPE: {mape:.1f}%
   • Output Table: {FULL_SCHEMA}.ml_scrap_anomalies

📊 SUPPLIER RISK SCORING  
   • Algorithm: Random Forest Classifier
   • Suppliers Scored: {len(latest_month):,}
   • High Risk Suppliers: {len(high_risk):,}
   • Output Table: {FULL_SCHEMA}.ml_supplier_risk_scores

🔧 EQUIPMENT PREDICTION
   • Algorithm: Random Forest Classifier
   • Equipment Analyzed: {len(equipment_scrap):,}
   • Accuracy: {accuracy_e:.1%}

🔬 DEFECT CLUSTERING
   • Algorithm: K-Means
   • Optimal Clusters: {best_k}
   • Silhouette Score: {max(silhouette_scores):.3f}
   • Output Table: {FULL_SCHEMA}.ml_defect_clusters

💡 ACTION RECOMMENDER
   • Action Patterns: {len(action_patterns):,}
   • Method: Effectiveness-based ranking

🔗 MLflow Experiment: Default Notebook Experiment
""")

print("=" * 70)
print("💡 RECOMMENDED ACTIONS:")
print("=" * 70)
print("""
1. SET UP ANOMALY ALERTS
   - Create workflow to check ml_scrap_anomalies daily
   - Alert when combined_anomaly = 1
   - Prioritize by anomaly_severity (time series score)
   - Use anomaly_methods to understand detection type

2. INTEGRATE SUPPLIER RISK INTO PROCUREMENT
   - Review high-risk suppliers before orders
   - Add risk_score to supplier dashboard

3. PRIORITIZE EQUIPMENT MAINTENANCE
   - Focus on old equipment with high defect counts
   - Use model to plan preventive maintenance

4. INVESTIGATE DEFECT CLUSTERS
   - Each cluster represents a pattern
   - High-cost clusters need immediate attention
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # 🏢 Model 6: Supplier Reallocation Optimizer
# MAGIC ---
# MAGIC
# MAGIC **Purpose**: Optimize supplier allocation to minimize scrap while respecting capacity constraints.
# MAGIC
# MAGIC This model supports the **What-If Scenarios** in the Streamlit app by:
# MAGIC 1. Identifying optimal suppliers for each part category
# MAGIC 2. Predicting scrap impact of supplier changes
# MAGIC 3. Recommending allocation shifts based on quality and capacity

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Supplier Scenario Data

# COMMAND ----------

# Load supplier scenario data
supplier_scenario_df = spark.table(f"{FULL_SCHEMA}.supplier_scenario_gold")

# Also get historical supplier-scrap correlation
supplier_scrap_history = spark.sql(f"""
    SELECT 
        supplier_id,
        supplier_name_lookup as supplier_name,
        supplier_tier_lookup as tier,
        AVG(supplier_quality_score) as avg_quality_score,
        SUM(total_scrap_cost) as total_scrap_cost,
        SUM(quantity_scrapped) as total_qty_scrapped,
        COUNT(DISTINCT part_id) as parts_affected,
        COUNT(DISTINCT plant_id) as plants_served,
        COUNT(*) as scrap_records
    FROM {FULL_SCHEMA}.scrap_transactions_silver
    GROUP BY supplier_id, supplier_name_lookup, supplier_tier_lookup
""")

print(f"Supplier scenario records: {supplier_scenario_df.count()}")
print(f"Supplier history records: {supplier_scrap_history.count()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train Scrap Prediction Model for Supplier Changes
# MAGIC
# MAGIC **Goal**: Predict expected scrap cost change when reallocating volume between suppliers.

# COMMAND ----------

# Prepare features for supplier scrap prediction
supplier_features = supplier_scrap_history.toPandas()

# Calculate scrap rate (scrap per record)
supplier_features['scrap_rate'] = supplier_features['total_qty_scrapped'] / (supplier_features['scrap_records'] + 1)
supplier_features['scrap_cost_per_record'] = supplier_features['total_scrap_cost'] / (supplier_features['scrap_records'] + 1)

# Create tier encoding
tier_map = {'Tier 1': 3, 'Tier 2': 2, 'Tier 3': 1}
supplier_features['tier_score'] = supplier_features['tier'].map(tier_map).fillna(1)

# Features for prediction
feature_cols = ['avg_quality_score', 'tier_score', 'parts_affected', 'plants_served']
target_col = 'scrap_rate'

X = supplier_features[feature_cols].fillna(0)
y = supplier_features[target_col].fillna(supplier_features[target_col].median())

print(f"Training supplier scrap predictor on {len(X)} suppliers")

# COMMAND ----------

# Train regression model to predict scrap rate based on supplier characteristics
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
import mlflow
import mlflow.sklearn

with mlflow.start_run(run_name="supplier_scrap_predictor"):
    # Log parameters
    mlflow.log_param("model_type", "GradientBoostingRegressor")
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("target", "scrap_rate")
    mlflow.log_param("n_suppliers", len(X))
    
    # Train model
    supplier_model = GradientBoostingRegressor(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        random_state=42
    )
    
    # Cross-validation
    cv_scores = cross_val_score(supplier_model, X, y, cv=5, scoring='r2')
    
    mlflow.log_metric("cv_r2_mean", cv_scores.mean())
    mlflow.log_metric("cv_r2_std", cv_scores.std())
    
    # Fit final model
    supplier_model.fit(X, y)
    
    # Feature importance
    for feat, imp in zip(feature_cols, supplier_model.feature_importances_):
        mlflow.log_metric(f"importance_{feat}", imp)
    
    # Log model
    mlflow.sklearn.log_model(supplier_model, "supplier_scrap_predictor")
    
    print(f"Supplier Scrap Predictor CV R²: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    print("\nFeature Importance:")
    for feat, imp in sorted(zip(feature_cols, supplier_model.feature_importances_), key=lambda x: -x[1]):
        print(f"  {feat}: {imp:.3f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Generate Supplier Reallocation Recommendations
# MAGIC
# MAGIC For each specialization/category, identify optimal supplier and potential reallocations.

# COMMAND ----------

# Get supplier scenario data with all metrics
supplier_scenario_pd = supplier_scenario_df.toPandas()

# Function to find optimal allocation within a category
def optimize_category_allocation(category_df):
    """Find optimal supplier allocation for a category."""
    
    if len(category_df) < 2:
        return None
    
    # Score each supplier (higher = better)
    # Score = quality * (1 - scrap_rate) * available_capacity_factor
    category_df = category_df.copy()
    category_df['allocation_score'] = (
        category_df['quality_score'] * 
        (1 - category_df['scrap_rate'].clip(0, 0.5)) * 
        (1 + category_df['available_capacity'].clip(0, 0.5))
    )
    
    # Find best and worst performers
    best = category_df.loc[category_df['allocation_score'].idxmax()]
    worst = category_df.loc[category_df['allocation_score'].idxmin()]
    
    # Calculate potential savings if we shift 20% from worst to best
    shift_amount = worst['estimated_annual_spend'] * 0.20 if 'estimated_annual_spend' in category_df.columns else 100000
    
    # Scrap savings estimate
    scrap_reduction = (worst['scrap_rate'] - best['scrap_rate']) * shift_amount
    
    return {
        'category': category_df['specialization'].iloc[0],
        'best_supplier': best['supplier_name'],
        'best_quality': best['quality_score'],
        'best_available_capacity': best['available_capacity'],
        'worst_supplier': worst['supplier_name'],
        'worst_quality': worst['quality_score'],
        'recommended_shift_pct': 20,
        'estimated_scrap_savings': max(0, scrap_reduction)
    }

# Generate recommendations by category
recommendations = []
for category in supplier_scenario_pd['specialization'].unique():
    category_df = supplier_scenario_pd[supplier_scenario_pd['specialization'] == category]
    rec = optimize_category_allocation(category_df)
    if rec:
        recommendations.append(rec)

recommendations_df = pd.DataFrame(recommendations)
print("Supplier Reallocation Recommendations:")
display(recommendations_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save Supplier Optimization Results

# COMMAND ----------

# Convert to Spark and save
if len(recommendations_df) > 0:
    recommendations_spark = spark.createDataFrame(recommendations_df)
    recommendations_spark = recommendations_spark.withColumn("generated_at", F.current_timestamp())
    
    (recommendations_spark
     .write
     .mode("overwrite")
     .saveAsTable(f"{FULL_SCHEMA}.ml_supplier_reallocation_recommendations"))
    
    print(f"Saved {len(recommendations_df)} supplier reallocation recommendations")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Supplier-Part Allocation Matrix
# MAGIC
# MAGIC For use in what-if scenario analysis - shows which suppliers can produce which parts.

# COMMAND ----------

# Get supplier-part relationships from scrap data
# Note: part_category is available via part_category_lookup in scrap_transactions_silver
supplier_part_matrix = spark.sql(f"""
    SELECT 
        supplier_id,
        supplier_name_lookup as supplier_name,
        supplier_tier_lookup as tier,
        part_category_lookup as part_category,
        COUNT(DISTINCT part_id) as parts_count,
        SUM(total_scrap_cost) as total_scrap_cost,
        AVG(supplier_quality_score) as avg_quality
    FROM {FULL_SCHEMA}.scrap_transactions_silver
    GROUP BY supplier_id, supplier_name_lookup, supplier_tier_lookup, part_category_lookup
    HAVING COUNT(DISTINCT part_id) >= 2
    ORDER BY part_category_lookup, total_scrap_cost DESC
""")

# Save matrix
(supplier_part_matrix
 .withColumn("generated_at", F.current_timestamp())
 .write
 .mode("overwrite")
 .saveAsTable(f"{FULL_SCHEMA}.ml_supplier_part_capability_matrix"))

print(f"Saved supplier-part capability matrix: {supplier_part_matrix.count()} relationships")
display(supplier_part_matrix.limit(20))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Log Final Summary with Supplier Optimization

# COMMAND ----------

print("\n" + "=" * 70)
print("🎯 MODEL 6: SUPPLIER REALLOCATION OPTIMIZER - COMPLETE")
print("=" * 70)
print(f"""
📊 SUPPLIER OPTIMIZATION RESULTS
   • Suppliers Analyzed: {len(supplier_features)}
   • Categories with Recommendations: {len(recommendations_df)}
   • Predictor CV R²: {cv_scores.mean():.3f}
   
📈 KEY OUTPUTS
   • Scrap Rate Predictor Model (in MLflow)
   • Reallocation Recommendations: {FULL_SCHEMA}.ml_supplier_reallocation_recommendations
   • Capability Matrix: {FULL_SCHEMA}.ml_supplier_part_capability_matrix

🎯 TOP RECOMMENDATIONS
""")

# Show top 5 recommendations by savings
if len(recommendations_df) > 0:
    top_recs = recommendations_df.nlargest(5, 'estimated_scrap_savings')
    for _, rec in top_recs.iterrows():
        print(f"   • {rec['category']}: Shift 20% from {rec['worst_supplier'][:20]} → {rec['best_supplier'][:20]}")
        print(f"     Estimated Annual Savings: ${rec['estimated_scrap_savings']:,.0f}")