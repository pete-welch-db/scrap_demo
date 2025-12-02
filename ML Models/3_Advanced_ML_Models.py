# Databricks notebook source
# MAGIC %md
# MAGIC # Advanced ML Models for Scrap Reduction
# MAGIC 
# MAGIC This notebook contains additional ML models for scrap reduction:
# MAGIC 
# MAGIC 1. **Anomaly Detection** - Detect unusual scrap spikes for early intervention
# MAGIC 2. **Scrap Risk Scoring** - Score production orders for scrap probability
# MAGIC 3. **Supplier Risk Model** - Predict supplier quality issues
# MAGIC 4. **Equipment Failure Prediction** - Predict equipment-related scrap
# MAGIC 5. **Defect Clustering** - Find hidden defect patterns
# MAGIC 6. **Corrective Action Recommender** - Suggest actions based on patterns

# COMMAND ----------

# MAGIC %pip install scikit-learn mlflow

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier, GradientBoostingRegressor
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, silhouette_score
import mlflow
import mlflow.sklearn
from pyspark.sql import functions as F
from pyspark.sql.window import Window
import matplotlib.pyplot as plt
import seaborn as sns

# Set MLflow experiment
mlflow.set_experiment("/Shared/scrap_demo/advanced_ml")

# Configuration
CATALOG = "welch"
SCHEMA = "scrap_demo"

# COMMAND ----------

# Load data
scrap_df = spark.table(f"{CATALOG}.{SCHEMA}.scrap_transactions_silver")
equipment_df = spark.table(f"{CATALOG}.{SCHEMA}.equipment_silver")
suppliers_df = spark.table(f"{CATALOG}.{SCHEMA}.suppliers_silver")
eight_d_df = spark.table(f"{CATALOG}.{SCHEMA}.eight_d_reports_silver")

print(f"Scrap records: {scrap_df.count():,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Anomaly Detection - Scrap Spike Detection

# COMMAND ----------

# MAGIC %md
# MAGIC ### Detect Unusual Daily Scrap Patterns
# MAGIC Use Isolation Forest to identify days with anomalous scrap behavior.

# COMMAND ----------

# Aggregate daily scrap with features
daily_scrap = (
    scrap_df
    .groupBy("transaction_date")
    .agg(
        F.sum("quantity_scrapped").alias("daily_qty"),
        F.sum("total_scrap_cost").alias("daily_cost"),
        F.count("*").alias("daily_records"),
        F.countDistinct("defect_code").alias("unique_defects"),
        F.countDistinct("part_id").alias("unique_parts"),
        F.countDistinct("plant_id").alias("plants_affected"),
        F.avg("equipment_age_years").alias("avg_equipment_age"),
        F.avg("supplier_quality_score").alias("avg_supplier_quality"),
        F.sum(F.when(F.col("shift") == 3, 1).otherwise(0)).alias("night_shift_records")
    )
    .orderBy("transaction_date")
    .toPandas()
)

# Add rolling features
daily_scrap["qty_rolling_7d"] = daily_scrap["daily_qty"].rolling(7, min_periods=1).mean()
daily_scrap["cost_rolling_7d"] = daily_scrap["daily_cost"].rolling(7, min_periods=1).mean()
daily_scrap["qty_pct_change"] = daily_scrap["daily_qty"].pct_change().fillna(0)

# Extract date features
daily_scrap["day_of_week"] = pd.to_datetime(daily_scrap["transaction_date"]).dt.dayofweek
daily_scrap["month"] = pd.to_datetime(daily_scrap["transaction_date"]).dt.month

print(f"Daily records: {len(daily_scrap):,}")
daily_scrap.head()

# COMMAND ----------

# Prepare features for anomaly detection
anomaly_features = [
    "daily_qty", "daily_cost", "daily_records", "unique_defects",
    "unique_parts", "plants_affected", "avg_equipment_age",
    "qty_pct_change", "night_shift_records"
]

X_anomaly = daily_scrap[anomaly_features].fillna(0).values

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_anomaly)

# Train Isolation Forest
with mlflow.start_run(run_name="scrap_anomaly_detection"):
    iso_forest = IsolationForest(
        n_estimators=100,
        contamination=0.05,  # Expect ~5% anomalies
        random_state=42
    )
    
    # Predict anomalies (-1 = anomaly, 1 = normal)
    daily_scrap["anomaly_prediction"] = iso_forest.fit_predict(X_scaled)
    daily_scrap["anomaly_score"] = iso_forest.score_samples(X_scaled)
    
    # Identify anomalies
    anomalies = daily_scrap[daily_scrap["anomaly_prediction"] == -1]
    
    mlflow.log_param("model_type", "IsolationForest")
    mlflow.log_param("contamination", 0.05)
    mlflow.log_metric("anomalies_detected", len(anomalies))
    mlflow.log_metric("anomaly_rate", len(anomalies) / len(daily_scrap))
    mlflow.sklearn.log_model(iso_forest, "model")
    
    print(f"Anomalies detected: {len(anomalies)} ({len(anomalies)/len(daily_scrap):.1%})")

# COMMAND ----------

# Visualize anomalies
fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

# Daily scrap with anomalies highlighted
ax1 = axes[0]
ax1.plot(daily_scrap["transaction_date"], daily_scrap["daily_qty"], 'b-', alpha=0.5, label="Daily Qty")
ax1.scatter(
    anomalies["transaction_date"], 
    anomalies["daily_qty"],
    c='red', s=50, label="Anomaly", zorder=5
)
ax1.set_ylabel("Daily Quantity Scrapped")
ax1.legend()
ax1.set_title("Scrap Anomaly Detection")

# Anomaly score
ax2 = axes[1]
ax2.plot(daily_scrap["transaction_date"], daily_scrap["anomaly_score"], 'g-', alpha=0.7)
ax2.axhline(y=daily_scrap[daily_scrap["anomaly_prediction"] == -1]["anomaly_score"].max(), 
            color='red', linestyle='--', label="Anomaly Threshold")
ax2.set_ylabel("Anomaly Score")
ax2.set_xlabel("Date")
ax2.legend()

plt.tight_layout()
plt.show()

# COMMAND ----------

# Analyze anomaly characteristics
print("\nAnomalous Days - Key Statistics:")
print("-" * 60)

anomaly_comparison = pd.DataFrame({
    "Metric": anomaly_features,
    "Normal Mean": daily_scrap[daily_scrap["anomaly_prediction"] == 1][anomaly_features].mean().values,
    "Anomaly Mean": anomalies[anomaly_features].mean().values
})
anomaly_comparison["Difference %"] = (
    (anomaly_comparison["Anomaly Mean"] - anomaly_comparison["Normal Mean"]) / 
    anomaly_comparison["Normal Mean"] * 100
).round(1)

print(anomaly_comparison.to_string(index=False))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Supplier Risk Scoring Model

# COMMAND ----------

# MAGIC %md
# MAGIC Predict which suppliers are likely to have quality issues in the next month.

# COMMAND ----------

# Aggregate supplier performance metrics
supplier_metrics = (
    scrap_df
    .groupBy(
        "supplier_id", "supplier_name_lookup", "supplier_tier_lookup",
        F.year("transaction_date").alias("year"),
        F.month("transaction_date").alias("month")
    )
    .agg(
        F.sum("quantity_scrapped").alias("monthly_qty_scrapped"),
        F.sum("total_scrap_cost").alias("monthly_scrap_cost"),
        F.count("*").alias("monthly_scrap_incidents"),
        F.countDistinct("defect_code").alias("unique_defects"),
        F.countDistinct("part_id").alias("parts_affected"),
        F.avg("supplier_quality_score").alias("supplier_quality_score"),
        F.sum(F.when(F.col("defect_category") == "Material Defects", 1).otherwise(0)).alias("material_defects"),
        F.sum(F.when(F.col("disposition") == "Return to Supplier", 1).otherwise(0)).alias("supplier_returns")
    )
    .toPandas()
)

# Create target: Is next month a "high scrap" month?
# High scrap = above median for that supplier
supplier_metrics = supplier_metrics.sort_values(["supplier_id", "year", "month"])
supplier_metrics["next_month_qty"] = supplier_metrics.groupby("supplier_id")["monthly_qty_scrapped"].shift(-1)

# Calculate supplier-specific threshold
supplier_medians = supplier_metrics.groupby("supplier_id")["monthly_qty_scrapped"].median()
supplier_metrics["supplier_median"] = supplier_metrics["supplier_id"].map(supplier_medians)
supplier_metrics["high_scrap_next_month"] = (
    supplier_metrics["next_month_qty"] > supplier_metrics["supplier_median"] * 1.2
).astype(int)

# Remove rows without next month data
supplier_model_data = supplier_metrics.dropna(subset=["next_month_qty"])

print(f"Training samples: {len(supplier_model_data):,}")
print(f"High scrap months: {supplier_model_data['high_scrap_next_month'].sum():,} ({supplier_model_data['high_scrap_next_month'].mean():.1%})")

# COMMAND ----------

# Prepare features
supplier_features = [
    "monthly_qty_scrapped", "monthly_scrap_cost", "monthly_scrap_incidents",
    "unique_defects", "parts_affected", "supplier_quality_score",
    "material_defects", "supplier_returns"
]

# Encode tier
tier_encoder = LabelEncoder()
supplier_model_data["tier_encoded"] = tier_encoder.fit_transform(
    supplier_model_data["supplier_tier_lookup"].astype(str)
)

feature_cols = supplier_features + ["tier_encoded"]

X_supplier = supplier_model_data[feature_cols].fillna(0).values
y_supplier = supplier_model_data["high_scrap_next_month"].values

# Split
X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
    X_supplier, y_supplier, test_size=0.2, random_state=42
)

# COMMAND ----------

# Train supplier risk model
with mlflow.start_run(run_name="supplier_risk_model"):
    supplier_risk_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        class_weight="balanced"
    )
    supplier_risk_model.fit(X_train_s, y_train_s)
    
    y_pred_s = supplier_risk_model.predict(X_test_s)
    y_proba_s = supplier_risk_model.predict_proba(X_test_s)[:, 1]
    
    from sklearn.metrics import accuracy_score, roc_auc_score
    accuracy = accuracy_score(y_test_s, y_pred_s)
    auc = roc_auc_score(y_test_s, y_proba_s)
    
    mlflow.log_param("model_type", "RandomForest")
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("auc_roc", auc)
    mlflow.sklearn.log_model(supplier_risk_model, "model")
    
    print(f"Supplier Risk Model:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  AUC-ROC: {auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test_s, y_pred_s, target_names=["Low Risk", "High Risk"]))

# COMMAND ----------

# Score current suppliers for risk
latest_month = supplier_model_data.groupby("supplier_id").tail(1)
X_latest = latest_month[feature_cols].fillna(0).values
risk_scores = supplier_risk_model.predict_proba(X_latest)[:, 1]

latest_month = latest_month.copy()
latest_month["risk_score"] = risk_scores
latest_month["risk_level"] = pd.cut(
    risk_scores, 
    bins=[0, 0.3, 0.6, 1.0], 
    labels=["Low", "Medium", "High"]
)

print("\nCurrent Supplier Risk Scores:")
print("-" * 60)
high_risk = latest_month[latest_month["risk_level"] == "High"].sort_values("risk_score", ascending=False)
print(high_risk[["supplier_name_lookup", "supplier_tier_lookup", "risk_score", "monthly_scrap_cost"]].head(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Equipment Failure Prediction

# COMMAND ----------

# MAGIC %md
# MAGIC Predict which equipment will cause scrap based on age, OEE, and maintenance status.

# COMMAND ----------

# Aggregate scrap by equipment
equipment_scrap = (
    scrap_df
    .filter(F.col("equipment_id").isNotNull())
    .groupBy("equipment_id", "equipment_name_lookup", "equipment_age_years", "equipment_oee")
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
equipment_scrap["high_scrap_equipment"] = (equipment_scrap["total_qty_scrapped"] > threshold).astype(int)

print(f"Equipment records: {len(equipment_scrap):,}")
print(f"High-scrap equipment: {equipment_scrap['high_scrap_equipment'].sum():,}")

# COMMAND ----------

# Train equipment failure model
equip_features = ["equipment_age_years", "equipment_oee", "scrap_incidents", "unique_defects"]

X_equip = equipment_scrap[equip_features].fillna(0).values
y_equip = equipment_scrap["high_scrap_equipment"].values

X_train_e, X_test_e, y_train_e, y_test_e = train_test_split(
    X_equip, y_equip, test_size=0.2, random_state=42
)

with mlflow.start_run(run_name="equipment_scrap_predictor"):
    equip_model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
    equip_model.fit(X_train_e, y_train_e)
    
    y_pred_e = equip_model.predict(X_test_e)
    accuracy_e = accuracy_score(y_test_e, y_pred_e)
    
    mlflow.log_metric("accuracy", accuracy_e)
    mlflow.sklearn.log_model(equip_model, "model")
    
    print(f"Equipment Scrap Prediction Accuracy: {accuracy_e:.4f}")

# Equipment feature importance
equip_importance = pd.DataFrame({
    "feature": equip_features,
    "importance": equip_model.feature_importances_
}).sort_values("importance", ascending=False)

print("\nKey factors for equipment-related scrap:")
print(equip_importance.to_string(index=False))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Defect Pattern Clustering

# COMMAND ----------

# MAGIC %md
# MAGIC Use clustering to find hidden patterns in defect data.

# COMMAND ----------

# Prepare defect pattern features
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

print(f"Defect codes: {len(defect_patterns):,}")

# COMMAND ----------

# Cluster defects
cluster_features = [
    "total_qty", "total_cost", "occurrences", "plants_affected",
    "suppliers_involved", "parts_affected", "avg_equipment_age",
    "night_shift_count", "avg_supplier_quality"
]

X_cluster = defect_patterns[cluster_features].fillna(0).values
X_cluster_scaled = StandardScaler().fit_transform(X_cluster)

# Find optimal clusters
silhouette_scores = []
K_range = range(2, 8)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_cluster_scaled)
    score = silhouette_score(X_cluster_scaled, labels)
    silhouette_scores.append(score)

best_k = K_range[np.argmax(silhouette_scores)]
print(f"Optimal clusters: {best_k}")

# COMMAND ----------

# Train final clustering model
with mlflow.start_run(run_name="defect_clustering"):
    kmeans_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    defect_patterns["cluster"] = kmeans_final.fit_predict(X_cluster_scaled)
    
    mlflow.log_param("n_clusters", best_k)
    mlflow.log_metric("silhouette_score", max(silhouette_scores))
    mlflow.sklearn.log_model(kmeans_final, "model")

# COMMAND ----------

# Analyze clusters
print("\nDefect Cluster Analysis:")
print("=" * 80)

for cluster_id in range(best_k):
    cluster_data = defect_patterns[defect_patterns["cluster"] == cluster_id]
    print(f"\nCluster {cluster_id}: {len(cluster_data)} defect codes")
    print("-" * 40)
    print(f"  Categories: {cluster_data['defect_category'].unique().tolist()}")
    print(f"  Avg Total Cost: ${cluster_data['total_cost'].mean():,.0f}")
    print(f"  Avg Occurrences: {cluster_data['occurrences'].mean():,.0f}")
    print(f"  Avg Plants Affected: {cluster_data['plants_affected'].mean():.1f}")
    print(f"  Avg Equipment Age: {cluster_data['avg_equipment_age'].mean():.1f} years")
    print(f"  Avg Supplier Quality: {cluster_data['avg_supplier_quality'].mean():.3f}")

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
ax1.set_xlabel("Total Scrap Cost ($)")
ax1.set_ylabel("Occurrences")
ax1.set_title("Defect Clusters by Cost and Frequency")
plt.colorbar(scatter, ax=ax1, label="Cluster")

# Cluster summary
ax2 = axes[1]
cluster_summary = defect_patterns.groupby("cluster").agg({
    "total_cost": "sum",
    "occurrences": "sum"
}).reset_index()
ax2.bar(cluster_summary["cluster"], cluster_summary["total_cost"])
ax2.set_xlabel("Cluster")
ax2.set_ylabel("Total Scrap Cost ($)")
ax2.set_title("Total Cost by Defect Cluster")

plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Corrective Action Recommender

# COMMAND ----------

# MAGIC %md
# MAGIC Build a recommender that suggests corrective actions based on defect patterns.

# COMMAND ----------

# Load 8D corrective actions
eight_d_pd = eight_d_df.toPandas()

# Create action-defect mappings
action_patterns = eight_d_pd.groupby(["defect_category", "root_cause_category"]).agg({
    "corrective_actions": lambda x: list(x),
    "verification_result": lambda x: (x == "Effective").mean(),
    "eight_d_id": "count"
}).reset_index()

action_patterns.columns = ["defect_category", "root_cause_category", "actions", "effectiveness_rate", "count"]

print(f"Action patterns: {len(action_patterns)}")

# COMMAND ----------

def recommend_actions(defect_category: str, root_cause: str = None, top_n: int = 3) -> list:
    """
    Recommend corrective actions based on defect category and optional root cause.
    Returns actions sorted by historical effectiveness.
    """
    # Filter patterns
    if root_cause:
        matches = action_patterns[
            (action_patterns["defect_category"] == defect_category) &
            (action_patterns["root_cause_category"] == root_cause)
        ]
    else:
        matches = action_patterns[action_patterns["defect_category"] == defect_category]
    
    if len(matches) == 0:
        # Fallback to just defect category
        matches = action_patterns[action_patterns["defect_category"] == defect_category]
    
    if len(matches) == 0:
        return ["No historical actions found for this pattern"]
    
    # Sort by effectiveness
    matches = matches.sort_values("effectiveness_rate", ascending=False)
    
    # Get top actions
    recommendations = []
    for _, row in matches.head(top_n).iterrows():
        if row["actions"]:
            sample_action = row["actions"][0][:300] if row["actions"][0] else "N/A"
            recommendations.append({
                "root_cause": row["root_cause_category"],
                "action_sample": sample_action,
                "effectiveness": row["effectiveness_rate"],
                "historical_count": row["count"]
            })
    
    return recommendations

# COMMAND ----------

# Example recommendations
print("Corrective Action Recommendations")
print("=" * 80)

test_cases = [
    ("Dimensional", "Machine"),
    ("Surface Defects", "Material"),
    ("Welding Defects", "Method"),
    ("Material Defects", None)
]

for defect_cat, root_cause in test_cases:
    print(f"\nDefect: {defect_cat}, Root Cause: {root_cause or 'Any'}")
    print("-" * 60)
    recs = recommend_actions(defect_cat, root_cause)
    for i, rec in enumerate(recs, 1):
        print(f"{i}. Root Cause: {rec['root_cause']}")
        print(f"   Effectiveness: {rec['effectiveness']:.0%} ({rec['historical_count']} cases)")
        print(f"   Sample Action: {rec['action_sample'][:150]}...")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Save All Model Outputs

# COMMAND ----------

# Save anomaly detection results
anomaly_spark = spark.createDataFrame(daily_scrap)
anomaly_spark = anomaly_spark.withColumn("detection_date", F.current_timestamp())
anomaly_spark.write.mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.ml_scrap_anomalies")
print(f"Saved anomaly detection results: {anomaly_spark.count()} days")

# Save supplier risk scores
risk_spark = spark.createDataFrame(latest_month[["supplier_id", "supplier_name_lookup", "risk_score", "risk_level"]])
risk_spark = risk_spark.withColumn("scored_at", F.current_timestamp())
risk_spark.write.mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.ml_supplier_risk_scores")
print(f"Saved supplier risk scores: {risk_spark.count()} suppliers")

# Save defect clusters
cluster_spark = spark.createDataFrame(defect_patterns)
cluster_spark = cluster_spark.withColumn("clustered_at", F.current_timestamp())
cluster_spark.write.mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.ml_defect_clusters")
print(f"Saved defect clusters: {cluster_spark.count()} defect codes")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary of Advanced ML Models
# MAGIC 
# MAGIC | Model | Purpose | Key Output |
# MAGIC |-------|---------|------------|
# MAGIC | Anomaly Detection | Early warning for scrap spikes | Daily anomaly flags |
# MAGIC | Supplier Risk Scoring | Predict supplier quality issues | Risk scores by supplier |
# MAGIC | Equipment Prediction | Identify high-scrap equipment | Equipment risk flags |
# MAGIC | Defect Clustering | Find hidden defect patterns | Defect cluster assignments |
# MAGIC | Action Recommender | Suggest corrective actions | Action recommendations |
# MAGIC 
# MAGIC ### Next Steps:
# MAGIC 1. **Integrate alerts** for anomaly detection
# MAGIC 2. **Add supplier risk to procurement workflow**
# MAGIC 3. **Create equipment maintenance triggers**
# MAGIC 4. **Build Genie space** for action recommendations

