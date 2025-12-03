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

# MAGIC %pip install scikit-learn

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
    precision_score, recall_score, f1_score
)

# MLflow
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

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

# Unity Catalog Configuration
CATALOG = "welch"
SCHEMA = "scrap_demo"
FULL_SCHEMA = f"{CATALOG}.{SCHEMA}"

# MLflow Configuration
EXPERIMENT_PATH = "/Shared/scrap_demo/advanced_ml"
mlflow.set_experiment(EXPERIMENT_PATH)

# Enable sklearn autologging
mlflow.sklearn.autolog(log_input_examples=True, log_model_signatures=True)

print(f"✅ Configuration loaded")
print(f"   Catalog: {CATALOG}.{SCHEMA}")
print(f"   MLflow Experiment: {EXPERIMENT_PATH}")

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
    import pickle
    with open("/tmp/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    mlflow.log_artifact("/tmp/scaler.pkl")
    
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
# MAGIC # 💡 Model 5: Corrective Action Recommender
# MAGIC ---
# MAGIC 
# MAGIC ## Objective
# MAGIC Recommend corrective actions based on defect patterns and historical effectiveness.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Build Action-Defect Mappings

# COMMAND ----------

# Load 8D data
eight_d_pd = eight_d_df.toPandas()

# Create action patterns by defect and root cause
action_patterns = eight_d_pd.groupby(["defect_category", "root_cause_category"]).agg({
    "corrective_actions": lambda x: list(x),
    "verification_result": lambda x: (x == "Effective").mean() if len(x) > 0 else 0,
    "eight_d_id": "count"
}).reset_index()

action_patterns.columns = ["defect_category", "root_cause_category", 
                            "actions", "effectiveness_rate", "count"]

print(f"📊 Action Patterns: {len(action_patterns):,} combinations")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Define Recommendation Function

# COMMAND ----------

def recommend_corrective_actions(
    defect_category: str, 
    root_cause: str = None, 
    top_n: int = 3
) -> list:
    """
    Recommend corrective actions based on defect category and root cause.
    
    Returns actions sorted by historical effectiveness rate.
    
    Parameters:
    -----------
    defect_category : str
        Category of defect (e.g., "Dimensional", "Surface Defects")
    root_cause : str, optional
        Root cause category (e.g., "Machine", "Material")
    top_n : int
        Number of recommendations to return
        
    Returns:
    --------
    list of dict
        Recommendations with effectiveness metrics
    """
    # Filter patterns
    if root_cause:
        matches = action_patterns[
            (action_patterns["defect_category"] == defect_category) &
            (action_patterns["root_cause_category"] == root_cause)
        ]
    else:
        matches = action_patterns[
            action_patterns["defect_category"] == defect_category
        ]
    
    if len(matches) == 0:
        # Fallback to just defect category
        matches = action_patterns[
            action_patterns["defect_category"] == defect_category
        ]
    
    if len(matches) == 0:
        return [{"message": "No historical actions found for this pattern"}]
    
    # Sort by effectiveness
    matches = matches.sort_values("effectiveness_rate", ascending=False)
    
    # Get top actions
    recommendations = []
    for _, row in matches.head(top_n).iterrows():
        if row["actions"] and len(row["actions"]) > 0:
            sample_action = str(row["actions"][0])[:300]
            recommendations.append({
                "root_cause": row["root_cause_category"],
                "sample_action": sample_action,
                "effectiveness_rate": f"{row['effectiveness_rate']:.0%}",
                "historical_cases": row["count"]
            })
    
    return recommendations

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test Recommender

# COMMAND ----------

# Test recommendations
test_cases = [
    ("Dimensional", "Machine"),
    ("Surface Defects", "Material"),
    ("Welding Defects", "Method"),
    ("Material Defects", None)
]

print("💡 CORRECTIVE ACTION RECOMMENDATIONS")
print("=" * 70)

for defect_cat, root_cause in test_cases:
    print(f"\n📋 Defect: {defect_cat} | Root Cause: {root_cause or 'Any'}")
    print("-" * 60)
    
    recs = recommend_corrective_actions(defect_cat, root_cause, top_n=2)
    
    for i, rec in enumerate(recs, 1):
        if "message" in rec:
            print(f"   {rec['message']}")
        else:
            print(f"   {i}. Root Cause: {rec['root_cause']}")
            print(f"      Effectiveness: {rec['effectiveness_rate']} "
                  f"({rec['historical_cases']} cases)")
            print(f"      Action: {rec['sample_action'][:100]}...")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # 💾 Save Model Outputs
# MAGIC ---

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save Anomaly Detection Results

# COMMAND ----------

# Save anomaly detection results
anomaly_spark = spark.createDataFrame(daily_scrap)
anomaly_spark = (
    anomaly_spark
    .withColumn("detection_date", F.current_timestamp())
    .select(
        "transaction_date", "daily_qty", "daily_cost", "daily_records",
        "anomaly_score", "is_anomaly", "detection_date"
    )
)

anomaly_spark.write.mode("overwrite").saveAsTable(
    f"{FULL_SCHEMA}.ml_scrap_anomalies"
)
print(f"✅ Saved anomaly results: {anomaly_spark.count():,} days")

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
   • Algorithm: Isolation Forest
   • Days Analyzed: {len(daily_scrap):,}
   • Anomalies Detected: {len(anomalies):,} ({len(anomalies)/len(daily_scrap):.1%})
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

🔗 MLflow Experiment: {EXPERIMENT_PATH}
""")

print("=" * 70)
print("💡 RECOMMENDED ACTIONS:")
print("=" * 70)
print("""
1. SET UP ANOMALY ALERTS
   - Create workflow to check ml_scrap_anomalies daily
   - Alert when is_anomaly = 1

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
