# Databricks notebook source
# MAGIC %md
# MAGIC # Root Cause Analysis & 8D Validation
# MAGIC 
# MAGIC This notebook uses ML and LLMs to:
# MAGIC 
# MAGIC 1. **Predict Root Cause** - Classify root cause from scrap characteristics
# MAGIC 2. **Validate 8D Reports** - Check if assigned root causes match data patterns
# MAGIC 3. **Extract Insights from 8D Text** - Use LLM to summarize and analyze
# MAGIC 4. **Find Similar 8Ds** - Recommend actions based on historical cases
# MAGIC 5. **8D Effectiveness Prediction** - Predict if corrective actions will work
# MAGIC 
# MAGIC **Key Question:** Is the Quality Department correctly identifying root causes?

# COMMAND ----------

# MAGIC %pip install scikit-learn mlflow

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import mlflow
import mlflow.sklearn
from pyspark.sql import functions as F
from pyspark.sql.window import Window
import matplotlib.pyplot as plt
import seaborn as sns

# Set MLflow experiment
mlflow.set_experiment("/Shared/scrap_demo/root_cause_analysis")

# Configuration
CATALOG = "welch"
SCHEMA = "scrap_demo"

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Load Data

# COMMAND ----------

# Load scrap transactions with all features
scrap_df = spark.table(f"{CATALOG}.{SCHEMA}.scrap_transactions_silver")

# Load 8D reports
eight_d_df = spark.table(f"{CATALOG}.{SCHEMA}.eight_d_reports_silver")

# Load mapping
mapping_df = spark.table(f"{CATALOG}.{SCHEMA}.scrap_8d_mapping_silver")

print(f"Scrap records: {scrap_df.count():,}")
print(f"8D reports: {eight_d_df.count():,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Root Cause Classification Model
# MAGIC 
# MAGIC **Objective:** Train a model to predict root cause category from scrap characteristics.
# MAGIC If the model's predictions differ from quality team's assignments, it may indicate misclassification.

# COMMAND ----------

# Prepare features for root cause prediction
feature_cols = [
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
    "is_weekend"
]

categorical_cols = [
    "defect_category",
    "work_center_type_lookup",
    "part_category_lookup",
    "disposition",
    "plant_id",
    "supplier_tier_lookup"
]

target_col = "root_cause_category"

# COMMAND ----------

# Prepare training data
training_df = scrap_df.select(
    *feature_cols,
    *categorical_cols,
    target_col
).dropna()

# Convert to Pandas
train_pd = training_df.toPandas()

print(f"Training samples: {len(train_pd):,}")
print(f"\nRoot cause distribution:")
print(train_pd[target_col].value_counts())

# COMMAND ----------

# Encode categorical variables
encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    train_pd[f"{col}_encoded"] = le.fit_transform(train_pd[col].astype(str))
    encoders[col] = le

# Encode target
target_encoder = LabelEncoder()
train_pd["target_encoded"] = target_encoder.fit_transform(train_pd[target_col])

# COMMAND ----------

# Prepare feature matrix
encoded_feature_cols = feature_cols + [f"{col}_encoded" for col in categorical_cols]

X = train_pd[encoded_feature_cols].values
y = train_pd["target_encoded"].values

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set: {len(X_train):,}")
print(f"Test set: {len(X_test):,}")

# COMMAND ----------

# Train Random Forest model
with mlflow.start_run(run_name="root_cause_classifier_rf"):
    # Train model
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_split=10,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    
    # Predict
    y_pred = rf_model.predict(X_test)
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    
    # Log to MLflow
    mlflow.log_param("model_type", "RandomForest")
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 15)
    mlflow.log_metric("accuracy", accuracy)
    
    mlflow.sklearn.log_model(rf_model, "model")
    
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(
        y_test, y_pred, 
        target_names=target_encoder.classes_
    ))

# COMMAND ----------

# Feature importance
feature_importance = pd.DataFrame({
    "feature": encoded_feature_cols,
    "importance": rf_model.feature_importances_
}).sort_values("importance", ascending=False)

plt.figure(figsize=(10, 8))
sns.barplot(data=feature_importance.head(15), x="importance", y="feature")
plt.title("Top 15 Features for Root Cause Prediction")
plt.xlabel("Feature Importance")
plt.tight_layout()
plt.show()

# COMMAND ----------

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(
    cm, 
    annot=True, 
    fmt="d", 
    cmap="Blues",
    xticklabels=target_encoder.classes_,
    yticklabels=target_encoder.classes_
)
plt.title("Root Cause Prediction Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Validate Quality Department's Root Cause Assignments
# MAGIC 
# MAGIC Compare model predictions vs. actual assignments to find potential misclassifications.

# COMMAND ----------

# Predict root cause for all records
X_all = train_pd[encoded_feature_cols].values
train_pd["predicted_root_cause"] = target_encoder.inverse_transform(rf_model.predict(X_all))
train_pd["prediction_matches"] = train_pd["predicted_root_cause"] == train_pd[target_col]

# Calculate agreement rate
agreement_rate = train_pd["prediction_matches"].mean()
print(f"Model agrees with Quality Department: {agreement_rate:.1%}")

# COMMAND ----------

# Find disagreements
disagreements = train_pd[~train_pd["prediction_matches"]].copy()
print(f"\nDisagreements: {len(disagreements):,} records ({len(disagreements)/len(train_pd):.1%})")

# Analyze disagreement patterns
print("\nDisagreement patterns (Actual vs Predicted):")
disagreement_analysis = disagreements.groupby(
    [target_col, "predicted_root_cause"]
).size().reset_index(name="count").sort_values("count", ascending=False)
print(disagreement_analysis.head(20))

# COMMAND ----------

# Identify high-confidence misclassifications
# These are cases where the model is very confident the root cause is different

# Get prediction probabilities
proba = rf_model.predict_proba(X_all)
train_pd["prediction_confidence"] = proba.max(axis=1)

# High confidence disagreements
high_conf_disagreements = train_pd[
    (~train_pd["prediction_matches"]) & 
    (train_pd["prediction_confidence"] > 0.7)
]

print(f"\nHigh-confidence disagreements (>70% model confidence): {len(high_conf_disagreements):,}")
print("\nThese records may warrant review by Quality team:")

# COMMAND ----------

# Summary of potential misclassifications by defect category
print("\nPotential misclassifications by defect category:")
misclass_by_defect = (
    high_conf_disagreements
    .groupby("defect_category")
    .agg({
        "prediction_matches": "count",
        "total_scrap_cost": "sum"
    })
    .rename(columns={
        "prediction_matches": "disagreement_count",
        "total_scrap_cost": "total_cost_impact"
    })
    .sort_values("total_cost_impact", ascending=False)
)
print(misclass_by_defect)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. 8D Effectiveness Prediction
# MAGIC 
# MAGIC Predict whether an 8D's corrective action will be effective based on characteristics.

# COMMAND ----------

# Prepare 8D data for effectiveness prediction
eight_d_pd = eight_d_df.toPandas()

# Filter to closed 8Ds with verification results
closed_8d = eight_d_pd[
    (eight_d_pd["status"] == "Closed") & 
    (eight_d_pd["verification_result"].notna())
].copy()

print(f"Closed 8Ds with verification: {len(closed_8d):,}")
print(f"\nVerification results:")
print(closed_8d["verification_result"].value_counts())

# COMMAND ----------

# Create binary target: Effective vs Not Effective
closed_8d["is_effective"] = closed_8d["verification_result"] == "Effective"

# Features for effectiveness prediction
effectiveness_features = [
    "severity",
    "priority", 
    "defect_category",
    "root_cause_category",
    "days_open",
    "total_scrap_cost",
    "total_quantity_affected",
    "num_parts_affected",
    "root_cause_word_count",
    "corrective_action_word_count"
]

# Prepare features
effectiveness_df = closed_8d[effectiveness_features + ["is_effective"]].dropna()

# Encode categorical
for col in ["severity", "priority", "defect_category", "root_cause_category"]:
    le = LabelEncoder()
    effectiveness_df[f"{col}_enc"] = le.fit_transform(effectiveness_df[col].astype(str))

# COMMAND ----------

# Train effectiveness model
feature_cols_eff = [
    "days_open", "total_scrap_cost", "total_quantity_affected", 
    "num_parts_affected", "root_cause_word_count", "corrective_action_word_count",
    "severity_enc", "priority_enc", "defect_category_enc", "root_cause_category_enc"
]

X_eff = effectiveness_df[feature_cols_eff].values
y_eff = effectiveness_df["is_effective"].values

X_train_eff, X_test_eff, y_train_eff, y_test_eff = train_test_split(
    X_eff, y_eff, test_size=0.2, random_state=42
)

with mlflow.start_run(run_name="8d_effectiveness_predictor"):
    eff_model = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=42
    )
    eff_model.fit(X_train_eff, y_train_eff)
    
    y_pred_eff = eff_model.predict(X_test_eff)
    accuracy_eff = accuracy_score(y_test_eff, y_pred_eff)
    
    mlflow.log_param("model_type", "GradientBoosting")
    mlflow.log_metric("accuracy", accuracy_eff)
    mlflow.sklearn.log_model(eff_model, "model")
    
    print(f"8D Effectiveness Prediction Accuracy: {accuracy_eff:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test_eff, y_pred_eff, target_names=["Not Effective", "Effective"]))

# COMMAND ----------

# Feature importance for effectiveness
eff_importance = pd.DataFrame({
    "feature": feature_cols_eff,
    "importance": eff_model.feature_importances_
}).sort_values("importance", ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(data=eff_importance, x="importance", y="feature")
plt.title("Features Influencing 8D Effectiveness")
plt.xlabel("Feature Importance")
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. LLM-Based 8D Analysis
# MAGIC 
# MAGIC Use Foundation Models to analyze 8D text and extract insights.

# COMMAND ----------

# Function to call Databricks Foundation Model for 8D analysis
def analyze_8d_with_llm(eight_d_record: dict) -> dict:
    """
    Use LLM to analyze an 8D report and provide insights.
    """
    from databricks.sdk import WorkspaceClient
    from databricks.sdk.service.serving import ChatMessage, ChatMessageRole
    
    prompt = f"""Analyze this 8D quality report and provide:
1. A brief summary of the problem (2-3 sentences)
2. Assessment of root cause validity (is this root cause logical given the problem?)
3. Assessment of corrective action completeness (are the actions sufficient?)
4. Risk score (1-10) for recurrence
5. Recommended additional actions if any

8D Report:
- Title: {eight_d_record.get('title', 'N/A')}
- Defect Category: {eight_d_record.get('defect_category', 'N/A')}
- Severity: {eight_d_record.get('severity', 'N/A')}
- Problem Description: {eight_d_record.get('problem_description', 'N/A')}
- Root Cause Analysis: {eight_d_record.get('root_cause_analysis', 'N/A')}
- Root Cause Category: {eight_d_record.get('root_cause_category', 'N/A')}
- Corrective Actions: {eight_d_record.get('corrective_actions', 'N/A')}
- Preventive Actions: {eight_d_record.get('preventive_actions', 'N/A')}

Provide your analysis in a structured format."""

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
            max_tokens=1000
        )
        
        return {
            "eight_d_id": eight_d_record.get("eight_d_id"),
            "llm_analysis": response.choices[0].message.content,
            "analyzed_at": pd.Timestamp.now()
        }
    except Exception as e:
        return {
            "eight_d_id": eight_d_record.get("eight_d_id"),
            "llm_analysis": f"Error: {str(e)}",
            "analyzed_at": pd.Timestamp.now()
        }

# COMMAND ----------

# Analyze sample 8D reports (top 5 by cost)
sample_8ds = (
    eight_d_df
    .orderBy(F.desc("total_scrap_cost"))
    .limit(5)
    .toPandas()
    .to_dict("records")
)

print("Analyzing top 5 8D reports by cost impact...")
llm_analyses = []

for record in sample_8ds:
    print(f"\nAnalyzing: {record['eight_d_id']} - {record['title'][:50]}...")
    analysis = analyze_8d_with_llm(record)
    llm_analyses.append(analysis)
    print(f"Analysis complete.")

# COMMAND ----------

# Display LLM analyses
for analysis in llm_analyses:
    print("=" * 80)
    print(f"8D ID: {analysis['eight_d_id']}")
    print("-" * 80)
    print(analysis['llm_analysis'])
    print()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Similar 8D Finder
# MAGIC 
# MAGIC Find similar historical 8Ds to recommend actions for new quality issues.

# COMMAND ----------

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Combine text fields for similarity matching
eight_d_pd["combined_text"] = (
    eight_d_pd["problem_description"].fillna("") + " " +
    eight_d_pd["root_cause_analysis"].fillna("") + " " +
    eight_d_pd["defect_category"].fillna("")
)

# Create TF-IDF vectors
tfidf = TfidfVectorizer(
    max_features=1000,
    stop_words="english",
    ngram_range=(1, 2)
)
tfidf_matrix = tfidf.fit_transform(eight_d_pd["combined_text"])

print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")

# COMMAND ----------

def find_similar_8ds(query_text: str, top_n: int = 5) -> pd.DataFrame:
    """
    Find similar historical 8D reports based on text similarity.
    """
    # Transform query
    query_vec = tfidf.transform([query_text])
    
    # Calculate similarities
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    
    # Get top matches
    top_indices = similarities.argsort()[-top_n:][::-1]
    
    results = eight_d_pd.iloc[top_indices][[
        "eight_d_id", "title", "defect_category", "root_cause_category",
        "corrective_actions", "verification_result", "total_scrap_cost"
    ]].copy()
    results["similarity_score"] = similarities[top_indices]
    
    return results

# COMMAND ----------

# Example: Find similar 8Ds for a new quality issue
new_issue = """
Dimensional defects found on stamped body panels. 
Multiple parts showing out-of-tolerance measurements on width dimension.
Suspected die wear issue.
"""

similar_8ds = find_similar_8ds(new_issue, top_n=5)
print("Similar Historical 8Ds:")
print("-" * 80)
for _, row in similar_8ds.iterrows():
    print(f"\n8D: {row['eight_d_id']} (Similarity: {row['similarity_score']:.2%})")
    print(f"Title: {row['title']}")
    print(f"Root Cause: {row['root_cause_category']}")
    print(f"Verification: {row['verification_result']}")
    print(f"Corrective Action: {row['corrective_actions'][:200]}...")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Save Model Outputs

# COMMAND ----------

# Save root cause validation results
validation_results = train_pd[[
    "defect_category", "work_center_type_lookup", "plant_id",
    "root_cause_category", "predicted_root_cause", 
    "prediction_matches", "prediction_confidence",
    "total_scrap_cost", "quantity_scrapped"
]].copy()

validation_spark = spark.createDataFrame(validation_results)
validation_spark = validation_spark.withColumn("analysis_date", F.current_timestamp())

validation_spark.write.mode("overwrite").saveAsTable(
    f"{CATALOG}.{SCHEMA}.ml_root_cause_validation"
)

print(f"Saved {validation_spark.count():,} root cause validation records")

# COMMAND ----------

# Summary statistics
print("\n" + "=" * 60)
print("ROOT CAUSE VALIDATION SUMMARY")
print("=" * 60)

summary_stats = validation_spark.agg(
    F.count("*").alias("total_records"),
    F.sum(F.when(F.col("prediction_matches"), 1).otherwise(0)).alias("agreements"),
    F.sum(F.when(~F.col("prediction_matches"), 1).otherwise(0)).alias("disagreements"),
    F.round(F.avg(F.col("prediction_confidence")), 3).alias("avg_confidence")
).collect()[0]

print(f"Total Records Analyzed: {summary_stats['total_records']:,}")
print(f"Model-Quality Agreement: {summary_stats['agreements']:,} ({summary_stats['agreements']/summary_stats['total_records']:.1%})")
print(f"Potential Misclassifications: {summary_stats['disagreements']:,} ({summary_stats['disagreements']/summary_stats['total_records']:.1%})")
print(f"Average Model Confidence: {summary_stats['avg_confidence']:.1%}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Key Findings & Recommendations
# MAGIC 
# MAGIC ### Model Insights:
# MAGIC 1. **Top predictive features** for root cause: Equipment age, defect category, supplier quality
# MAGIC 2. **Agreement rate** between model and quality team assignments
# MAGIC 3. **High-confidence disagreements** that warrant review
# MAGIC 
# MAGIC ### Recommendations:
# MAGIC 1. Review cases where model has >70% confidence in different root cause
# MAGIC 2. Focus on defect categories with highest misclassification rates
# MAGIC 3. Use similar 8D finder to leverage historical solutions
# MAGIC 4. Monitor 8D effectiveness predictions for early intervention

