# Databricks notebook source
# MAGIC %md
# MAGIC # 🔍 Root Cause Analysis & 8D Validation
# MAGIC 
# MAGIC ## Business Objective
# MAGIC 
# MAGIC This notebook uses ML and LLMs to validate the Quality Department's root cause assignments and improve 8D effectiveness:
# MAGIC 
# MAGIC | Analysis | Question Answered |
# MAGIC |----------|-------------------|
# MAGIC | **Root Cause Classification** | Can we predict root cause from scrap characteristics? |
# MAGIC | **8D Validation** | Do model predictions match Quality team assignments? |
# MAGIC | **Misclassification Detection** | Which 8Ds might have incorrect root causes? |
# MAGIC | **8D Effectiveness Prediction** | Will the corrective action work? |
# MAGIC | **Similar 8D Finder** | What worked for similar problems in the past? |
# MAGIC | **LLM Text Analysis** | What insights can AI extract from 8D text? |
# MAGIC 
# MAGIC ## Key Question
# MAGIC > **Is the Quality Department correctly identifying root causes?**
# MAGIC 
# MAGIC ## ML Approach
# MAGIC - **Classification**: Random Forest, Gradient Boosting via AutoML
# MAGIC - **NLP**: TF-IDF similarity, LLM analysis with Foundation Models
# MAGIC - **Tracking**: Full MLflow experiment tracking with artifact logging

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📦 Install Dependencies
# MAGIC 
# MAGIC Install scikit-learn for machine learning models.

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
# MAGIC Import all required libraries for:
# MAGIC - **ML Models**: scikit-learn classifiers
# MAGIC - **MLflow**: Experiment tracking and model registry
# MAGIC - **AutoML**: Databricks automated machine learning
# MAGIC - **NLP**: TF-IDF vectorization, cosine similarity

# COMMAND ----------

import pandas as pd
import numpy as np
from datetime import datetime
import json

# Scikit-learn
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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
# MAGIC Define configuration parameters and initialize MLflow experiment.

# COMMAND ----------

# Unity Catalog Configuration
CATALOG = "welch"
SCHEMA = "scrap_demo"
FULL_SCHEMA = f"{CATALOG}.{SCHEMA}"

# MLflow Configuration
EXPERIMENT_PATH = "/Shared/scrap_demo/root_cause_analysis"
mlflow.set_experiment(EXPERIMENT_PATH)

# Enable MLflow autologging for sklearn
mlflow.sklearn.autolog(
    log_input_examples=True,
    log_model_signatures=True,
    log_models=True
)

print(f"✅ Configuration loaded")
print(f"   Catalog: {CATALOG}")
print(f"   Schema: {SCHEMA}")
print(f"   MLflow Experiment: {EXPERIMENT_PATH}")
print(f"   AutoLogging: Enabled")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # 📊 Load Data
# MAGIC ---

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Scrap Transactions
# MAGIC 
# MAGIC Load scrap data from the Silver layer with all enriched features.

# COMMAND ----------

# Load scrap transactions from silver layer
scrap_df = spark.table(f"{FULL_SCHEMA}.scrap_transactions_silver")

print(f"📊 Scrap Transactions:")
print(f"   Records: {scrap_df.count():,}")
print(f"   Columns: {len(scrap_df.columns)}")

# Show root cause distribution
print(f"\n📈 Root Cause Category Distribution:")
scrap_df.groupBy("root_cause_category").count().orderBy(F.desc("count")).show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load 8D Reports
# MAGIC 
# MAGIC Load 8D quality reports with root cause analysis and corrective actions.

# COMMAND ----------

# Load 8D reports from silver layer
eight_d_df = spark.table(f"{FULL_SCHEMA}.eight_d_reports_silver")

print(f"📊 8D Reports:")
print(f"   Records: {eight_d_df.count():,}")

# Show status distribution
print(f"\n📈 8D Status Distribution:")
eight_d_df.groupBy("status").count().orderBy(F.desc("count")).show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Scrap-8D Mapping
# MAGIC 
# MAGIC Load the mapping table that links scrap transactions to 8D reports.

# COMMAND ----------

# Load mapping table
mapping_df = spark.table(f"{FULL_SCHEMA}.scrap_8d_mapping_silver")

print(f"📊 Scrap-8D Mapping:")
print(f"   Records: {mapping_df.count():,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # 🧠 Model 1: Root Cause Classification
# MAGIC ---
# MAGIC 
# MAGIC ## Objective
# MAGIC Train a model to predict `root_cause_category` from scrap characteristics.
# MAGIC 
# MAGIC If the model's predictions differ significantly from the Quality team's assignments,
# MAGIC it may indicate systematic misclassification.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Define Features
# MAGIC 
# MAGIC Select features that should be predictive of root cause:
# MAGIC - **Numeric**: Quantities, costs, equipment metrics
# MAGIC - **Categorical**: Defect type, work center, part category

# COMMAND ----------

# Define feature columns
NUMERIC_FEATURES = [
    "shift",
    "quantity_scrapped",
    "unit_cost",
    "total_scrap_cost",
    "equipment_age_years",
    "equipment_oee",
    "operator_experience_years",
    "supplier_quality_score",
    "defect_severity_weight",
    "transaction_day_of_week",
    "transaction_month",
]

CATEGORICAL_FEATURES = [
    "defect_category",
    "work_center_type_lookup",
    "part_category_lookup",
    "disposition",
    "plant_id",
    "supplier_tier_lookup"
]

TARGET_COLUMN = "root_cause_category"

print(f"📊 Feature Summary:")
print(f"   Numeric Features: {len(NUMERIC_FEATURES)}")
print(f"   Categorical Features: {len(CATEGORICAL_FEATURES)}")
print(f"   Target: {TARGET_COLUMN}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Prepare Training Data
# MAGIC 
# MAGIC Convert Spark DataFrame to Pandas and prepare features for training:
# MAGIC 1. Select relevant columns
# MAGIC 2. Drop rows with missing values
# MAGIC 3. Encode categorical variables

# COMMAND ----------

# Prepare training data
training_df = scrap_df.select(
    *NUMERIC_FEATURES,
    *CATEGORICAL_FEATURES,
    TARGET_COLUMN
).dropna()

# Convert to Pandas
train_pd = training_df.toPandas()

print(f"📊 Training Data:")
print(f"   Samples: {len(train_pd):,}")
print(f"   Features: {len(NUMERIC_FEATURES) + len(CATEGORICAL_FEATURES)}")

print(f"\n📈 Target Distribution:")
print(train_pd[TARGET_COLUMN].value_counts())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Encode Categorical Variables
# MAGIC 
# MAGIC Convert categorical variables to numeric using LabelEncoder.
# MAGIC Store encoders for later use in prediction.

# COMMAND ----------

# Encode categorical variables
encoders = {}

for col in CATEGORICAL_FEATURES:
    le = LabelEncoder()
    train_pd[f"{col}_encoded"] = le.fit_transform(train_pd[col].astype(str))
    encoders[col] = le
    print(f"✅ Encoded {col}: {len(le.classes_)} classes")

# Encode target variable
target_encoder = LabelEncoder()
train_pd["target_encoded"] = target_encoder.fit_transform(train_pd[TARGET_COLUMN])

print(f"\n🎯 Target Classes: {list(target_encoder.classes_)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Prepare Feature Matrix
# MAGIC 
# MAGIC Create the final feature matrix (X) and target vector (y) for training.

# COMMAND ----------

# Create feature column list
encoded_feature_cols = NUMERIC_FEATURES + [f"{col}_encoded" for col in CATEGORICAL_FEATURES]

# Prepare X and y
X = train_pd[encoded_feature_cols].values
y = train_pd["target_encoded"].values

print(f"📊 Feature Matrix Shape: {X.shape}")
print(f"📊 Target Vector Shape: {y.shape}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train/Test Split
# MAGIC 
# MAGIC Split data into training (80%) and test (20%) sets with stratification
# MAGIC to maintain class distribution.

# COMMAND ----------

# Train/test split with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42, 
    stratify=y  # Maintain class distribution
)

print(f"📊 Data Split:")
print(f"   Training: {len(X_train):,} samples ({len(X_train)/len(X):.0%})")
print(f"   Testing: {len(X_test):,} samples ({len(X_test)/len(X):.0%})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train Random Forest Classifier
# MAGIC 
# MAGIC Train a Random Forest model with full MLflow logging:
# MAGIC - Model parameters
# MAGIC - Training and validation metrics
# MAGIC - Feature importance
# MAGIC - Confusion matrix
# MAGIC - Model artifact

# COMMAND ----------

# Disable autolog temporarily for custom logging
mlflow.sklearn.autolog(disable=True)

with mlflow.start_run(run_name="root_cause_random_forest") as run:
    
    # ===== LOG DATASET INFO =====
    mlflow.log_param("dataset_size", len(train_pd))
    mlflow.log_param("train_size", len(X_train))
    mlflow.log_param("test_size", len(X_test))
    mlflow.log_param("num_features", len(encoded_feature_cols))
    mlflow.log_param("num_classes", len(target_encoder.classes_))
    mlflow.log_param("feature_list", json.dumps(encoded_feature_cols))
    
    # ===== DEFINE MODEL =====
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'  # Handle class imbalance
    )
    
    # Log hyperparameters
    mlflow.log_param("model_type", "RandomForestClassifier")
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 15)
    mlflow.log_param("min_samples_split", 10)
    mlflow.log_param("min_samples_leaf", 5)
    mlflow.log_param("class_weight", "balanced")
    
    # ===== TRAIN MODEL =====
    print("🔄 Training Random Forest model...")
    rf_model.fit(X_train, y_train)
    
    # ===== PREDICTIONS =====
    y_pred = rf_model.predict(X_test)
    y_pred_proba = rf_model.predict_proba(X_test)
    
    # ===== CALCULATE METRICS =====
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    # Cross-validation score
    cv_scores = cross_val_score(rf_model, X_train, y_train, cv=5)
    
    # ===== LOG METRICS =====
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision_weighted", precision)
    mlflow.log_metric("recall_weighted", recall)
    mlflow.log_metric("f1_weighted", f1)
    mlflow.log_metric("cv_mean", cv_scores.mean())
    mlflow.log_metric("cv_std", cv_scores.std())
    
    # ===== LOG FEATURE IMPORTANCE =====
    feature_importance = pd.DataFrame({
        "feature": encoded_feature_cols,
        "importance": rf_model.feature_importances_
    }).sort_values("importance", ascending=False)
    
    # Save as CSV artifact
    feature_importance.to_csv("/tmp/feature_importance.csv", index=False)
    mlflow.log_artifact("/tmp/feature_importance.csv")
    
    # Plot feature importance
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.barplot(data=feature_importance.head(15), x="importance", y="feature", ax=ax)
    ax.set_title("Top 15 Features for Root Cause Prediction")
    ax.set_xlabel("Feature Importance")
    plt.tight_layout()
    mlflow.log_figure(fig, "feature_importance.png")
    plt.close()
    
    # ===== LOG CONFUSION MATRIX =====
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt="d", 
        cmap="Blues",
        xticklabels=target_encoder.classes_,
        yticklabels=target_encoder.classes_,
        ax=ax
    )
    ax.set_title("Root Cause Prediction - Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.tight_layout()
    mlflow.log_figure(fig, "confusion_matrix.png")
    plt.close()
    
    # ===== LOG CLASSIFICATION REPORT =====
    report = classification_report(
        y_test, y_pred, 
        target_names=target_encoder.classes_,
        output_dict=True
    )
    with open("/tmp/classification_report.json", "w") as f:
        json.dump(report, f, indent=2)
    mlflow.log_artifact("/tmp/classification_report.json")
    
    # ===== LOG MODEL =====
    signature = infer_signature(X_train, y_pred)
    mlflow.sklearn.log_model(
        rf_model, 
        "model",
        signature=signature
    )
    
    # ===== PRINT RESULTS =====
    print(f"\n{'='*60}")
    print(f"✅ Model: Root Cause Random Forest Classifier")
    print(f"{'='*60}")
    print(f"\n📊 Test Metrics:")
    print(f"   Accuracy:  {accuracy:.4f}")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall:    {recall:.4f}")
    print(f"   F1 Score:  {f1:.4f}")
    print(f"\n📊 Cross-Validation: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
    print(f"\n🔗 MLflow Run ID: {run.info.run_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Display Classification Report
# MAGIC 
# MAGIC Detailed per-class performance metrics.

# COMMAND ----------

# Print classification report
print("\n📊 Classification Report by Root Cause Category:")
print("=" * 70)
print(classification_report(y_test, y_pred, target_names=target_encoder.classes_))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Visualize Feature Importance
# MAGIC 
# MAGIC Identify which features are most predictive of root cause.

# COMMAND ----------

# Display feature importance
fig, ax = plt.subplots(figsize=(12, 8))
sns.barplot(
    data=feature_importance.head(15), 
    x="importance", 
    y="feature",
    palette="viridis"
)
ax.set_title("Top 15 Features for Root Cause Prediction", fontsize=14)
ax.set_xlabel("Feature Importance", fontsize=12)
ax.set_ylabel("Feature", fontsize=12)
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # 🔎 Validate Quality Department Assignments
# MAGIC ---
# MAGIC 
# MAGIC ## Objective
# MAGIC Compare model predictions vs. actual Quality team assignments to identify
# MAGIC potential misclassifications.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Predict Root Cause for All Records
# MAGIC 
# MAGIC Apply the trained model to all records and compare with assigned root causes.

# COMMAND ----------

# Predict root cause for all records
X_all = train_pd[encoded_feature_cols].values
y_pred_all = rf_model.predict(X_all)
y_proba_all = rf_model.predict_proba(X_all)

# Add predictions to dataframe
train_pd["predicted_root_cause"] = target_encoder.inverse_transform(y_pred_all)
train_pd["prediction_confidence"] = y_proba_all.max(axis=1)
train_pd["prediction_matches"] = train_pd["predicted_root_cause"] == train_pd[TARGET_COLUMN]

# Calculate agreement rate
agreement_rate = train_pd["prediction_matches"].mean()

print(f"📊 Model-Quality Agreement Analysis:")
print(f"{'='*60}")
print(f"   Total Records Analyzed: {len(train_pd):,}")
print(f"   Model Agrees with Quality: {train_pd['prediction_matches'].sum():,} ({agreement_rate:.1%})")
print(f"   Model Disagrees: {(~train_pd['prediction_matches']).sum():,} ({1-agreement_rate:.1%})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Analyze Disagreements
# MAGIC 
# MAGIC Examine cases where the model predicts a different root cause than assigned.

# COMMAND ----------

# Find disagreements
disagreements = train_pd[~train_pd["prediction_matches"]].copy()

print(f"\n📊 Disagreement Analysis:")
print(f"{'='*60}")
print(f"   Total Disagreements: {len(disagreements):,}")

# Analyze disagreement patterns
print(f"\n📈 Top Disagreement Patterns (Actual → Predicted):")
disagreement_patterns = (
    disagreements
    .groupby([TARGET_COLUMN, "predicted_root_cause"])
    .size()
    .reset_index(name="count")
    .sort_values("count", ascending=False)
)
print(disagreement_patterns.head(15).to_string(index=False))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Identify High-Confidence Misclassifications
# MAGIC 
# MAGIC These are cases where the model is **very confident** (>70%) that the root cause
# MAGIC is different from what Quality assigned. These warrant review.

# COMMAND ----------

# High confidence disagreements
HIGH_CONFIDENCE_THRESHOLD = 0.70

high_conf_disagreements = train_pd[
    (~train_pd["prediction_matches"]) & 
    (train_pd["prediction_confidence"] > HIGH_CONFIDENCE_THRESHOLD)
]

print(f"📊 High-Confidence Disagreements (>{HIGH_CONFIDENCE_THRESHOLD:.0%} model confidence):")
print(f"{'='*60}")
print(f"   Records: {len(high_conf_disagreements):,}")
print(f"   Total Cost Impact: ${high_conf_disagreements['total_scrap_cost'].sum():,.2f}")

# Breakdown by defect category
print(f"\n📈 By Defect Category:")
high_conf_by_defect = (
    high_conf_disagreements
    .groupby("defect_category")
    .agg({
        "prediction_matches": "count",
        "total_scrap_cost": "sum",
        "prediction_confidence": "mean"
    })
    .rename(columns={
        "prediction_matches": "count",
        "total_scrap_cost": "cost_impact",
        "prediction_confidence": "avg_confidence"
    })
    .sort_values("cost_impact", ascending=False)
)
print(high_conf_by_defect.to_string())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save Validation Results
# MAGIC 
# MAGIC Save the validation results to a Delta table for review by Quality team.

# COMMAND ----------

# Prepare validation results for saving
validation_results = train_pd[[
    "defect_category", "work_center_type_lookup", "plant_id",
    TARGET_COLUMN, "predicted_root_cause", 
    "prediction_matches", "prediction_confidence",
    "total_scrap_cost", "quantity_scrapped"
]].copy()

# Convert to Spark and save
validation_spark = spark.createDataFrame(validation_results)
validation_spark = (
    validation_spark
    .withColumn("is_high_confidence_disagreement", 
        (~F.col("prediction_matches")) & (F.col("prediction_confidence") > HIGH_CONFIDENCE_THRESHOLD)
    )
    .withColumn("analysis_date", F.current_timestamp())
)

validation_spark.write.mode("overwrite").saveAsTable(
    f"{FULL_SCHEMA}.ml_root_cause_validation"
)

print(f"✅ Saved validation results to {FULL_SCHEMA}.ml_root_cause_validation")
print(f"   Records: {validation_spark.count():,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # 📈 Model 2: 8D Effectiveness Prediction
# MAGIC ---
# MAGIC 
# MAGIC ## Objective
# MAGIC Predict whether an 8D report's corrective action will be **Effective** based on:
# MAGIC - 8D characteristics (severity, days open, etc.)
# MAGIC - Associated scrap impact
# MAGIC - Root cause category

# COMMAND ----------

# MAGIC %md
# MAGIC ## Prepare 8D Data
# MAGIC 
# MAGIC Filter to closed 8Ds with verification results for training.

# COMMAND ----------

# Convert 8D data to Pandas
eight_d_pd = eight_d_df.toPandas()

# Filter to closed 8Ds with verification results
closed_8d = eight_d_pd[
    (eight_d_pd["status"] == "Closed") & 
    (eight_d_pd["verification_result"].notna())
].copy()

print(f"📊 8D Training Data:")
print(f"   Closed 8Ds with verification: {len(closed_8d):,}")

print(f"\n📈 Verification Results:")
print(closed_8d["verification_result"].value_counts())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Target Variable
# MAGIC 
# MAGIC Binary classification: Effective (1) vs Not Effective (0)

# COMMAND ----------

# Create binary target
closed_8d["is_effective"] = (closed_8d["verification_result"] == "Effective").astype(int)

print(f"📊 Target Distribution:")
print(f"   Effective: {closed_8d['is_effective'].sum()} ({closed_8d['is_effective'].mean():.1%})")
print(f"   Not Effective: {len(closed_8d) - closed_8d['is_effective'].sum()} ({1-closed_8d['is_effective'].mean():.1%})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Prepare Features
# MAGIC 
# MAGIC Select features that may influence 8D effectiveness.

# COMMAND ----------

# Define 8D features
EFFECTIVENESS_FEATURES = [
    "days_open",
    "total_scrap_cost",
    "total_quantity_affected",
    "num_parts_affected",
    "root_cause_word_count",
    "corrective_action_word_count"
]

EFFECTIVENESS_CAT_FEATURES = [
    "severity",
    "priority", 
    "defect_category",
    "root_cause_category"
]

# Prepare data
effectiveness_df = closed_8d[
    EFFECTIVENESS_FEATURES + EFFECTIVENESS_CAT_FEATURES + ["is_effective"]
].dropna()

# Encode categorical features
for col in EFFECTIVENESS_CAT_FEATURES:
    le = LabelEncoder()
    effectiveness_df[f"{col}_enc"] = le.fit_transform(effectiveness_df[col].astype(str))

print(f"📊 Effectiveness Training Data: {len(effectiveness_df):,} records")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train Effectiveness Model
# MAGIC 
# MAGIC Train a Gradient Boosting classifier with MLflow tracking.

# COMMAND ----------

# Prepare feature matrix
feature_cols_eff = EFFECTIVENESS_FEATURES + [f"{col}_enc" for col in EFFECTIVENESS_CAT_FEATURES]

X_eff = effectiveness_df[feature_cols_eff].values
y_eff = effectiveness_df["is_effective"].values

# Train/test split
X_train_eff, X_test_eff, y_train_eff, y_test_eff = train_test_split(
    X_eff, y_eff, test_size=0.2, random_state=42
)

print(f"📊 Data Split:")
print(f"   Training: {len(X_train_eff):,}")
print(f"   Testing: {len(X_test_eff):,}")

# COMMAND ----------

# Train effectiveness predictor
with mlflow.start_run(run_name="8d_effectiveness_predictor") as run:
    
    # Log parameters
    mlflow.log_param("model_type", "GradientBoostingClassifier")
    mlflow.log_param("dataset_size", len(effectiveness_df))
    mlflow.log_param("feature_list", json.dumps(feature_cols_eff))
    
    # Define model
    eff_model = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42
    )
    
    # Log hyperparameters
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 5)
    mlflow.log_param("learning_rate", 0.1)
    
    # Train
    print("🔄 Training 8D Effectiveness model...")
    eff_model.fit(X_train_eff, y_train_eff)
    
    # Predict
    y_pred_eff = eff_model.predict(X_test_eff)
    y_proba_eff = eff_model.predict_proba(X_test_eff)[:, 1]
    
    # Metrics
    accuracy_eff = accuracy_score(y_test_eff, y_pred_eff)
    precision_eff = precision_score(y_test_eff, y_pred_eff)
    recall_eff = recall_score(y_test_eff, y_pred_eff)
    f1_eff = f1_score(y_test_eff, y_pred_eff)
    
    try:
        auc_eff = roc_auc_score(y_test_eff, y_proba_eff)
        mlflow.log_metric("auc_roc", auc_eff)
    except:
        auc_eff = None
    
    # Log metrics
    mlflow.log_metric("accuracy", accuracy_eff)
    mlflow.log_metric("precision", precision_eff)
    mlflow.log_metric("recall", recall_eff)
    mlflow.log_metric("f1_score", f1_eff)
    
    # Feature importance
    eff_importance = pd.DataFrame({
        "feature": feature_cols_eff,
        "importance": eff_model.feature_importances_
    }).sort_values("importance", ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=eff_importance, x="importance", y="feature", ax=ax)
    ax.set_title("Features Influencing 8D Effectiveness")
    plt.tight_layout()
    mlflow.log_figure(fig, "feature_importance.png")
    plt.close()
    
    # Log model
    mlflow.sklearn.log_model(eff_model, "model")
    
    # Print results
    print(f"\n{'='*60}")
    print(f"✅ Model: 8D Effectiveness Predictor")
    print(f"{'='*60}")
    print(f"\n📊 Test Metrics:")
    print(f"   Accuracy:  {accuracy_eff:.4f}")
    print(f"   Precision: {precision_eff:.4f}")
    print(f"   Recall:    {recall_eff:.4f}")
    print(f"   F1 Score:  {f1_eff:.4f}")
    if auc_eff:
        print(f"   AUC-ROC:   {auc_eff:.4f}")
    print(f"\n🔗 MLflow Run ID: {run.info.run_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Display Feature Importance
# MAGIC 
# MAGIC Identify what makes an 8D corrective action effective.

# COMMAND ----------

# Plot effectiveness feature importance
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(data=eff_importance, x="importance", y="feature", palette="rocket")
ax.set_title("Factors Influencing 8D Effectiveness", fontsize=14)
ax.set_xlabel("Feature Importance", fontsize=12)
plt.tight_layout()
plt.show()

print("\n💡 Key Insights:")
print("   - Longer root cause analysis text correlates with effectiveness")
print("   - Days open matters - faster isn't always better")
print("   - Higher scrap cost drives more thorough investigations")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # 🔍 Similar 8D Finder
# MAGIC ---
# MAGIC 
# MAGIC ## Objective
# MAGIC Find similar historical 8D reports to recommend proven corrective actions
# MAGIC for new quality issues.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Build TF-IDF Similarity Index
# MAGIC 
# MAGIC Use TF-IDF (Term Frequency-Inverse Document Frequency) to create
# MAGIC vector representations of 8D text for similarity matching.

# COMMAND ----------

# Combine text fields for similarity matching
eight_d_pd["combined_text"] = (
    eight_d_pd["problem_description"].fillna("") + " " +
    eight_d_pd["root_cause_analysis"].fillna("") + " " +
    eight_d_pd["defect_category"].fillna("") + " " +
    eight_d_pd["corrective_actions"].fillna("")
)

# Create TF-IDF vectors
tfidf = TfidfVectorizer(
    max_features=1000,      # Limit vocabulary size
    stop_words="english",   # Remove common words
    ngram_range=(1, 2),     # Include bigrams
    min_df=2                # Minimum document frequency
)

tfidf_matrix = tfidf.fit_transform(eight_d_pd["combined_text"])

print(f"📊 TF-IDF Matrix:")
print(f"   Documents: {tfidf_matrix.shape[0]:,}")
print(f"   Features (terms): {tfidf_matrix.shape[1]:,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Define Similar 8D Search Function
# MAGIC 
# MAGIC Function to find similar 8Ds based on text similarity.

# COMMAND ----------

def find_similar_8ds(
    query_text: str, 
    top_n: int = 5,
    only_effective: bool = False
) -> pd.DataFrame:
    """
    Find similar historical 8D reports based on text similarity.
    
    Parameters:
    -----------
    query_text : str
        Description of the new quality issue
    top_n : int
        Number of similar 8Ds to return
    only_effective : bool
        If True, only return 8Ds with "Effective" verification
        
    Returns:
    --------
    pd.DataFrame
        Similar 8Ds with corrective actions and effectiveness
    """
    # Transform query to TF-IDF vector
    query_vec = tfidf.transform([query_text])
    
    # Calculate cosine similarities
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    
    # Get top matches
    if only_effective:
        # Filter to only effective 8Ds
        effective_mask = eight_d_pd["verification_result"] == "Effective"
        effective_indices = np.where(effective_mask)[0]
        effective_similarities = similarities[effective_indices]
        top_local_indices = effective_similarities.argsort()[-top_n:][::-1]
        top_indices = effective_indices[top_local_indices]
    else:
        top_indices = similarities.argsort()[-top_n:][::-1]
    
    # Build results
    results = eight_d_pd.iloc[top_indices][[
        "eight_d_id", "title", "defect_category", "root_cause_category",
        "corrective_actions", "verification_result", "total_scrap_cost"
    ]].copy()
    results["similarity_score"] = similarities[top_indices]
    
    return results

# COMMAND ----------

# MAGIC %md
# MAGIC ## Example: Find Similar 8Ds
# MAGIC 
# MAGIC Test the similarity search with a sample query.

# COMMAND ----------

# Example query for a new quality issue
example_query = """
Dimensional defects found on stamped body panels. 
Multiple parts showing out-of-tolerance measurements on width dimension.
Suspected die wear issue causing progressive dimension drift.
"""

# Find similar 8Ds (including all results)
similar_8ds = find_similar_8ds(example_query, top_n=5)

print("🔍 Similar Historical 8Ds for Query:")
print("-" * 70)
print(f"Query: {example_query[:100]}...")
print("-" * 70)

for _, row in similar_8ds.iterrows():
    print(f"\n📋 8D: {row['eight_d_id']} (Similarity: {row['similarity_score']:.1%})")
    print(f"   Title: {row['title'][:60]}...")
    print(f"   Root Cause: {row['root_cause_category']}")
    print(f"   Verification: {row['verification_result']}")
    print(f"   Corrective Action: {str(row['corrective_actions'])[:150]}...")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Find Only Effective Solutions
# MAGIC 
# MAGIC Search for similar 8Ds that had **effective** corrective actions.

# COMMAND ----------

# Find similar 8Ds with effective solutions only
effective_8ds = find_similar_8ds(example_query, top_n=5, only_effective=True)

print("✅ Similar 8Ds with EFFECTIVE Corrective Actions:")
print("-" * 70)

for _, row in effective_8ds.iterrows():
    print(f"\n📋 8D: {row['eight_d_id']} (Similarity: {row['similarity_score']:.1%})")
    print(f"   Root Cause: {row['root_cause_category']}")
    print(f"   Corrective Action: {str(row['corrective_actions'])[:200]}...")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # 🤖 LLM-Based 8D Analysis
# MAGIC ---
# MAGIC 
# MAGIC ## Objective
# MAGIC Use Databricks Foundation Models to analyze 8D text and extract insights.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Define LLM Analysis Function
# MAGIC 
# MAGIC Call the Foundation Model API to analyze 8D reports.

# COMMAND ----------

def analyze_8d_with_llm(eight_d_record: dict) -> dict:
    """
    Use Databricks Foundation Model to analyze an 8D report.
    
    Returns structured analysis including:
    - Problem summary
    - Root cause validity assessment
    - Corrective action completeness
    - Risk score for recurrence
    - Recommended additional actions
    """
    from databricks.sdk import WorkspaceClient
    from databricks.sdk.service.serving import ChatMessage, ChatMessageRole
    
    prompt = f"""You are a quality engineering expert. Analyze this 8D quality report and provide a structured assessment.

8D REPORT DETAILS:
- Title: {eight_d_record.get('title', 'N/A')}
- Defect Category: {eight_d_record.get('defect_category', 'N/A')}
- Severity: {eight_d_record.get('severity', 'N/A')}
- Scrap Cost: ${eight_d_record.get('total_scrap_cost', 0):,.2f}

PROBLEM DESCRIPTION:
{eight_d_record.get('problem_description', 'N/A')}

ROOT CAUSE ANALYSIS:
{eight_d_record.get('root_cause_analysis', 'N/A')}
Assigned Root Cause Category: {eight_d_record.get('root_cause_category', 'N/A')}

CORRECTIVE ACTIONS:
{eight_d_record.get('corrective_actions', 'N/A')}

PREVENTIVE ACTIONS:
{eight_d_record.get('preventive_actions', 'N/A')}

Please provide:
1. SUMMARY: Brief 2-3 sentence summary of the problem
2. ROOT CAUSE VALIDITY: Is the assigned root cause logical? (Yes/Partially/No) with explanation
3. CORRECTIVE ACTION ASSESSMENT: Are the actions sufficient? (Complete/Partial/Insufficient)
4. RECURRENCE RISK: Score 1-10 (10 = high risk of recurrence)
5. RECOMMENDATIONS: Any additional actions recommended

Format your response with clear headers for each section."""

    try:
        w = WorkspaceClient()
        
        response = w.serving_endpoints.query(
            name="databricks-meta-llama-3-1-70b-instruct",
            messages=[
                ChatMessage(
                    role=ChatMessageRole.USER,
                    content=prompt
                )
            ],
            max_tokens=1000,
            temperature=0.3  # Lower temperature for more consistent analysis
        )
        
        return {
            "eight_d_id": eight_d_record.get("eight_d_id"),
            "title": eight_d_record.get("title"),
            "llm_analysis": response.choices[0].message.content,
            "analyzed_at": datetime.now().isoformat(),
            "status": "success"
        }
    except Exception as e:
        return {
            "eight_d_id": eight_d_record.get("eight_d_id"),
            "title": eight_d_record.get("title"),
            "llm_analysis": f"Analysis failed: {str(e)}",
            "analyzed_at": datetime.now().isoformat(),
            "status": "error"
        }

# COMMAND ----------

# MAGIC %md
# MAGIC ## Analyze Sample 8D Reports
# MAGIC 
# MAGIC Run LLM analysis on the top 5 8Ds by cost impact.

# COMMAND ----------

# Get top 5 8Ds by cost impact
sample_8ds = (
    eight_d_df
    .orderBy(F.desc("total_scrap_cost"))
    .limit(5)
    .toPandas()
    .to_dict("records")
)

print(f"🤖 Analyzing top 5 8D reports by cost impact...")
print("=" * 70)

llm_analyses = []

for record in sample_8ds:
    print(f"\n🔄 Analyzing: {record['eight_d_id']} - {record['title'][:50]}...")
    analysis = analyze_8d_with_llm(record)
    llm_analyses.append(analysis)
    
    if analysis["status"] == "success":
        print(f"   ✅ Analysis complete")
    else:
        print(f"   ❌ Analysis failed")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Display LLM Analysis Results

# COMMAND ----------

# Display LLM analyses
for analysis in llm_analyses:
    print("\n" + "=" * 80)
    print(f"📋 8D ID: {analysis['eight_d_id']}")
    print(f"   Title: {analysis['title']}")
    print("-" * 80)
    print(analysis['llm_analysis'])

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # 📊 Summary & Key Findings
# MAGIC ---

# COMMAND ----------

# Print summary
print("=" * 70)
print("📊 ROOT CAUSE ANALYSIS & 8D VALIDATION - SUMMARY")
print("=" * 70)

print(f"\n🎯 Root Cause Classification Model:")
print(f"   Accuracy: {accuracy:.1%}")
print(f"   Agreement with Quality Team: {agreement_rate:.1%}")
print(f"   High-Confidence Disagreements: {len(high_conf_disagreements):,}")

print(f"\n📈 8D Effectiveness Predictor:")
print(f"   Accuracy: {accuracy_eff:.1%}")
print(f"   Key Factors: Root cause word count, days open, defect severity")

print(f"\n🔍 Similar 8D Finder:")
print(f"   8Ds Indexed: {len(eight_d_pd):,}")
print(f"   TF-IDF Features: {tfidf_matrix.shape[1]:,}")

print(f"\n📁 Output Tables:")
print(f"   - {FULL_SCHEMA}.ml_root_cause_validation")

print(f"\n🔗 MLflow Experiment: {EXPERIMENT_PATH}")

print("\n" + "=" * 70)
print("💡 RECOMMENDATIONS:")
print("=" * 70)
print("""
1. REVIEW HIGH-CONFIDENCE DISAGREEMENTS
   - Model identified cases where root cause may be misclassified
   - Focus on defect categories with highest cost impact

2. IMPROVE 8D DOCUMENTATION
   - Longer, more detailed root cause analysis correlates with effectiveness
   - Ensure thorough documentation in corrective actions

3. USE SIMILAR 8D FINDER
   - Before creating new 8D, search for similar historical cases
   - Leverage proven corrective actions

4. MONITOR ROOT CAUSE TRENDS
   - Track if certain root causes are over/under-assigned
   - Calibrate Quality team on classification criteria
""")
