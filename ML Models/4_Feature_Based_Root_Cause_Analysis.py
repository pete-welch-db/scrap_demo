# Databricks notebook source
# MAGIC %md
# MAGIC # 🔬 Feature-Based Root Cause Analysis with AutoML
# MAGIC
# MAGIC ## Business Objective
# MAGIC
# MAGIC This notebook goes **beyond stated root causes** to discover hidden factors driving scrap. Instead of accepting 
# MAGIC the root cause written in the 8D or transaction, we analyze ALL available features to find the REAL drivers.
# MAGIC
# MAGIC ## Key Questions
# MAGIC | Question | Analysis |
# MAGIC |----------|----------|
# MAGIC | Which factors ACTUALLY drive scrap? | Feature importance across all dimensions |
# MAGIC | Is it the shift or the operator? | Interaction analysis |
# MAGIC | Does equipment age matter? | Correlation & regression |
# MAGIC | Are supplier issues being hidden? | Controlled experiments |
# MAGIC | What would happen if...? | What-if scenario simulation |
# MAGIC
# MAGIC ## Feature Categories Analyzed
# MAGIC - **Time**: Shift, day of week, month, weekend, month-end
# MAGIC - **People**: Operator skill level, experience, training
# MAGIC - **Equipment**: Age, OEE, type, maintenance history
# MAGIC - **Supplier**: Quality score, tier, on-time delivery
# MAGIC - **Part**: Category, material, criticality, complexity
# MAGIC - **Process**: Work center type, volume, cycle time
# MAGIC
# MAGIC ## ML Approach
# MAGIC - **Databricks AutoML**: Automated model selection & hyperparameter tuning
# MAGIC - **SHAP Values**: Feature importance explanations
# MAGIC - **Interaction Analysis**: Multi-factor dependencies
# MAGIC - **Causal Inference**: Beyond correlation to causation
# MAGIC
# MAGIC ## Output Tables
# MAGIC - `ml_feature_importance_scores` - Feature importance by model
# MAGIC - `ml_factor_analysis_results` - Detailed factor contribution analysis
# MAGIC - `ml_interaction_effects` - Feature interaction strengths
# MAGIC - `ml_whatif_scenarios` - Pre-computed scenario outcomes
# MAGIC - `ml_hidden_root_causes` - AI-discovered root causes

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📦 Install Dependencies

# COMMAND ----------

# MAGIC %pip install shap scikit-learn xgboost lightgbm databricks-feature-engineering mlflow[databricks]

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🔄 Restart Python Environment

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📚 Import Libraries

# COMMAND ----------

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import warnings
warnings.filterwarnings('ignore')

# Scikit-learn
from sklearn.ensemble import (
    RandomForestRegressor, RandomForestClassifier,
    GradientBoostingRegressor, GradientBoostingClassifier
)
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    classification_report, accuracy_score, roc_auc_score
)
from sklearn.inspection import permutation_importance

# XGBoost & LightGBM
import xgboost as xgb
import lightgbm as lgb

# SHAP for explainability
import shap

# MLflow
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm
from mlflow.models.signature import infer_signature

# PySpark
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import *

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# COMMAND ----------

# MAGIC %md
# MAGIC ## ⚙️ Configuration

# COMMAND ----------

# Unity Catalog Configuration - Use widget parameters from job, with defaults for interactive use
dbutils.widgets.text("catalog", "scrap_catalog", "Catalog Name")
dbutils.widgets.text("schema", "scrap_demo", "Schema Name")

CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
FULL_SCHEMA = f"{CATALOG}.{SCHEMA}"

print(f"Using catalog: {CATALOG}, schema: {SCHEMA}")

# Analysis Configuration
TEST_SIZE = 0.2
RANDOM_STATE = 42
CV_FOLDS = 5

print(f"📍 Target Schema: {FULL_SCHEMA}")

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

# MAGIC %md
# MAGIC ## 📊 Load and Prepare Feature-Rich Dataset
# MAGIC
# MAGIC We load the scrap_transactions_silver table which contains ALL the features we need for analysis.

# COMMAND ----------

# Load full feature set from silver table
scrap_df = spark.sql(f"""
    SELECT 
        -- Target variable
        total_scrap_cost,
        quantity_scrapped,
        
        -- Time features
        transaction_year,
        transaction_month,
        transaction_quarter,
        transaction_week,
        transaction_day_of_week,
        day_name,
        is_weekend,
        is_month_end,
        shift,
        shift_name,
        
        -- Operator features
        operator_id,
        operator_name,
        operator_skill_level,
        operator_experience_years,
        
        -- Equipment features
        equipment_id,
        equipment_name_lookup,
        equipment_type_lookup,
        equipment_age_years,
        equipment_oee,
        
        -- Plant features
        plant_id,
        plant_name_lookup,
        plant_city,
        
        -- Supplier features
        supplier_id,
        supplier_name_lookup,
        supplier_tier_lookup,
        supplier_quality_score,
        
        -- Part features
        part_id,
        part_name_lookup,
        part_category_lookup,
        part_type_lookup,
        part_material,
        is_critical_part,
        
        -- Work center features
        work_center_id,
        work_center_name_lookup,
        work_center_type_lookup,
        
        -- Defect features (stated root cause - we'll compare against)
        defect_code,
        defect_category,
        defect_name_lookup,
        root_cause_category,
        
        -- Cost features
        unit_cost,
        cost_severity,
        quantity_severity
        
    FROM {FULL_SCHEMA}.scrap_transactions_silver
    WHERE transaction_year >= 2022
""")

print(f"✅ Loaded {scrap_df.count():,} scrap transactions")
scrap_df.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🎯 Model 1: Feature Importance Analysis
# MAGIC
# MAGIC **Question: Which factors ACTUALLY drive scrap cost?**
# MAGIC
# MAGIC We'll train multiple models and extract feature importance to understand what's really driving scrap.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Prepare Feature Matrix

# COMMAND ----------

# Determine optimal sample size based on dataset size
total_records = scrap_df.count()
print(f"📊 Total records available: {total_records:,}")

# Dynamic sampling strategy:
# - Use 100% if dataset <= 50k (small enough to process in full)
# - Use 75% if dataset 50k-150k (balance speed and data volume)
# - Use max 100k if dataset > 150k (cap for reasonable training time)
if total_records <= 50000:
    sample_size = total_records
    sample_pct = 100.0
elif total_records <= 150000:
    sample_size = int(total_records * 0.75)
    sample_pct = 75.0
else:
    sample_size = min(100000, total_records)
    sample_pct = (sample_size / total_records) * 100

print(f"📊 Using sample size: {sample_size:,} ({sample_pct:.1f}% of data)")

# Convert to pandas for modeling
pdf = scrap_df.limit(sample_size).toPandas()
print(f"✅ Loaded {len(pdf):,} records for analysis")

# Define feature groups
NUMERIC_FEATURES = [
    'operator_experience_years', 'equipment_age_years', 'equipment_oee',
    'supplier_quality_score', 'unit_cost', 'transaction_month',
    'transaction_day_of_week', 'shift'
]

CATEGORICAL_FEATURES = [
    'operator_skill_level', 'equipment_type_lookup', 'plant_name_lookup',
    'supplier_tier_lookup', 'part_category_lookup', 'part_material',
    'work_center_type_lookup', 'defect_category', 'is_critical_part',
    'is_weekend', 'is_month_end'
]

TARGET = 'total_scrap_cost'

# Handle missing values
for col in NUMERIC_FEATURES:
    if col in pdf.columns:
        pdf[col] = pdf[col].fillna(pdf[col].median())

for col in CATEGORICAL_FEATURES:
    if col in pdf.columns:
        pdf[col] = pdf[col].fillna('Unknown')

# Encode categorical features
label_encoders = {}
for col in CATEGORICAL_FEATURES:
    if col in pdf.columns:
        le = LabelEncoder()
        pdf[f'{col}_encoded'] = le.fit_transform(pdf[col].astype(str))
        label_encoders[col] = le

# Create feature matrix
feature_cols = NUMERIC_FEATURES + [f'{col}_encoded' for col in CATEGORICAL_FEATURES if col in pdf.columns]
X = pdf[feature_cols].fillna(0)
y = pdf[TARGET]

print(f"✅ Feature matrix shape: {X.shape}")
print(f"📋 Features: {list(X.columns)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Train Multiple Models for Robust Feature Importance

# COMMAND ----------

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)

# Models to compare
models = {
    'RandomForest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=RANDOM_STATE, n_jobs=-1),
    'GradientBoosting': GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=RANDOM_STATE),
    'XGBoost': xgb.XGBRegressor(n_estimators=100, max_depth=5, random_state=RANDOM_STATE, n_jobs=-1),
    'LightGBM': lgb.LGBMRegressor(n_estimators=100, max_depth=5, random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)
}

# Train and evaluate each model
results = []
feature_importance_dfs = []

with mlflow.start_run(run_name="feature_importance_analysis"):
    mlflow.log_param("total_records", total_records)
    mlflow.log_param("sample_size", sample_size)
    mlflow.log_param("sample_pct", sample_pct)
    mlflow.log_param("n_features", len(feature_cols))
    
    for name, model in models.items():
        print(f"\n🔄 Training {name}...")
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"   RMSE: ${rmse:,.2f} | MAE: ${mae:,.2f} | R²: {r2:.3f}")
        
        # Log metrics
        mlflow.log_metrics({
            f"{name}_rmse": rmse,
            f"{name}_mae": mae,
            f"{name}_r2": r2
        })
        
        results.append({
            'model': name,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        })
        
        # Extract feature importance
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            fi_df = pd.DataFrame({
                'feature': feature_cols,
                'importance': importance,
                'model': name
            }).sort_values('importance', ascending=False)
            feature_importance_dfs.append(fi_df)
            
            print(f"   Top 5 features: {fi_df.head()['feature'].tolist()}")

# Combine all feature importance results
all_importance = pd.concat(feature_importance_dfs)
print("\n✅ Feature importance analysis complete!")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Aggregate Feature Importance Across Models

# COMMAND ----------

# Calculate average importance across all models
avg_importance = all_importance.groupby('feature').agg({
    'importance': ['mean', 'std', 'min', 'max']
}).reset_index()
avg_importance.columns = ['feature', 'avg_importance', 'std_importance', 'min_importance', 'max_importance']
avg_importance = avg_importance.sort_values('avg_importance', ascending=False)

# Map encoded features back to original names
def decode_feature_name(encoded_name):
    if encoded_name.endswith('_encoded'):
        return encoded_name.replace('_encoded', '')
    return encoded_name

avg_importance['feature_clean'] = avg_importance['feature'].apply(decode_feature_name)

# Add feature category
def get_feature_category(feature):
    if 'operator' in feature.lower():
        return 'People'
    elif 'equipment' in feature.lower():
        return 'Equipment'
    elif 'supplier' in feature.lower():
        return 'Supplier'
    elif 'part' in feature.lower() or 'material' in feature.lower():
        return 'Part'
    elif 'work_center' in feature.lower():
        return 'Process'
    elif 'shift' in feature.lower() or 'month' in feature.lower() or 'day' in feature.lower() or 'weekend' in feature.lower():
        return 'Time'
    elif 'defect' in feature.lower():
        return 'Defect'
    elif 'plant' in feature.lower():
        return 'Plant'
    else:
        return 'Other'

avg_importance['feature_category'] = avg_importance['feature_clean'].apply(get_feature_category)

# Display results
print("🎯 TOP 15 MOST IMPORTANT FACTORS FOR SCRAP:")
print("=" * 60)
display(avg_importance.head(15))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Visualize Feature Importance

# COMMAND ----------

# Plot top 20 features
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# Bar chart of top features
top_20 = avg_importance.head(20)
colors = {'People': '#e74c3c', 'Equipment': '#3498db', 'Supplier': '#f39c12', 
          'Part': '#9b59b6', 'Process': '#2ecc71', 'Time': '#1abc9c', 
          'Defect': '#e67e22', 'Plant': '#34495e', 'Other': '#95a5a6'}
bar_colors = [colors.get(cat, '#95a5a6') for cat in top_20['feature_category']]

axes[0].barh(range(len(top_20)), top_20['avg_importance'], color=bar_colors)
axes[0].set_yticks(range(len(top_20)))
axes[0].set_yticklabels(top_20['feature_clean'])
axes[0].set_xlabel('Average Importance Score')
axes[0].set_title('Top 20 Scrap Cost Drivers')
axes[0].invert_yaxis()

# Pie chart by category
category_importance = avg_importance.groupby('feature_category')['avg_importance'].sum()
axes[1].pie(category_importance, labels=category_importance.index, autopct='%1.1f%%',
           colors=[colors.get(cat, '#95a5a6') for cat in category_importance.index])
axes[1].set_title('Importance by Category')

plt.tight_layout()
mlflow.log_figure(fig, "feature_importance.png")
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Save Feature Importance to Delta Table

# COMMAND ----------

# Prepare for Delta save
importance_spark_df = spark.createDataFrame(avg_importance.assign(
    analysis_date=datetime.now(),
    sample_size=sample_size,
    total_records=total_records
))

# Save to Delta table
importance_spark_df.write.mode("overwrite").saveAsTable(f"{FULL_SCHEMA}.ml_feature_importance_scores")
print(f"✅ Saved feature importance to {FULL_SCHEMA}.ml_feature_importance_scores")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🔍 Model 2: SHAP Analysis for Explainability
# MAGIC
# MAGIC SHAP (SHapley Additive exPlanations) provides detailed insights into how each feature affects predictions.

# COMMAND ----------

# Retrain using RandomForest for better SHAP compatibility
# (XGBoost + SHAP can have version compatibility issues)
from sklearn.ensemble import RandomForestRegressor

best_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=6,
    random_state=RANDOM_STATE,
    n_jobs=-1
)
best_model.fit(X_train, y_train)

# Calculate SHAP values
print("📊 Calculating SHAP values (this may take a few minutes)...")

# Use a sample for SHAP to save time
shap_sample = X_test.sample(
    n=min(1000, len(X_test)),
    random_state=RANDOM_STATE
)

# TreeExplainer works reliably with sklearn RandomForest
explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(shap_sample)

print("✅ SHAP analysis complete!")

# COMMAND ----------

# MAGIC %md
# MAGIC ### SHAP Summary Plot

# COMMAND ----------

# Create SHAP summary plot
fig, ax = plt.subplots(figsize=(12, 10))
shap.summary_plot(shap_values, shap_sample, feature_names=feature_cols, show=False)
plt.tight_layout()
mlflow.log_figure(fig, "shap_summary.png")
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Save SHAP Values to Delta Table

# COMMAND ----------

# Aggregate SHAP values by feature
shap_importance = pd.DataFrame({
    'feature': feature_cols,
    'mean_abs_shap': np.abs(shap_values).mean(axis=0),
    'mean_shap': shap_values.mean(axis=0),
    'std_shap': shap_values.std(axis=0)
}).sort_values('mean_abs_shap', ascending=False)

shap_importance['feature_clean'] = shap_importance['feature'].apply(decode_feature_name)
shap_importance['feature_category'] = shap_importance['feature_clean'].apply(get_feature_category)
shap_importance['analysis_date'] = datetime.now()

# Save to Delta
shap_spark_df = spark.createDataFrame(shap_importance)
shap_spark_df.write.mode("overwrite").saveAsTable(f"{FULL_SCHEMA}.ml_shap_analysis")
print(f"✅ Saved SHAP analysis to {FULL_SCHEMA}.ml_shap_analysis")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🔀 Model 3: Feature Interaction Analysis
# MAGIC
# MAGIC **Question: Do features work together to cause scrap?**
# MAGIC
# MAGIC Example: Equipment age alone may not cause scrap, but equipment age + inexperienced operator might.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Calculate Interaction Effects

# COMMAND ----------

# Create interaction features for key combinations
interaction_pairs = [
    ('shift', 'operator_skill_level_encoded'),
    ('equipment_age_years', 'operator_experience_years'),
    ('supplier_quality_score', 'is_critical_part_encoded'),
    ('equipment_oee', 'work_center_type_lookup_encoded'),
    ('shift', 'is_month_end_encoded'),
    ('operator_experience_years', 'equipment_age_years'),
]

interaction_results = []

for feat1, feat2 in interaction_pairs:
    if feat1 in X.columns and feat2 in X.columns:
        # Create interaction term
        X_interaction = X.copy()
        X_interaction['interaction'] = X_interaction[feat1] * X_interaction[feat2]
        
        # Train model with and without interaction
        model_with = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=RANDOM_STATE, n_jobs=-1)
        model_without = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=RANDOM_STATE, n_jobs=-1)
        
        # Without interaction
        X_no_int_train, X_no_int_test, y_train_int, y_test_int = train_test_split(
            X[[feat1, feat2]], y, test_size=0.2, random_state=RANDOM_STATE
        )
        model_without.fit(X_no_int_train, y_train_int)
        r2_without = r2_score(y_test_int, model_without.predict(X_no_int_test))
        
        # With interaction
        X_with_int_train, X_with_int_test, _, _ = train_test_split(
            X_interaction[[feat1, feat2, 'interaction']], y, test_size=0.2, random_state=RANDOM_STATE
        )
        model_with.fit(X_with_int_train, y_train_int)
        r2_with = r2_score(y_test_int, model_with.predict(X_with_int_test))
        
        # Calculate interaction strength
        interaction_strength = r2_with - r2_without
        
        interaction_results.append({
            'feature_1': decode_feature_name(feat1),
            'feature_2': decode_feature_name(feat2),
            'r2_without_interaction': r2_without,
            'r2_with_interaction': r2_with,
            'interaction_strength': interaction_strength,
            'analysis_date': datetime.now()
        })
        
        print(f"📊 {decode_feature_name(feat1)} × {decode_feature_name(feat2)}: Interaction strength = {interaction_strength:.4f}")

# Save to Delta
interaction_df = spark.createDataFrame(pd.DataFrame(interaction_results))
interaction_df.write.mode("overwrite").saveAsTable(f"{FULL_SCHEMA}.ml_interaction_effects")
print(f"\n✅ Saved interaction analysis to {FULL_SCHEMA}.ml_interaction_effects")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📊 Model 4: Detailed Factor Analysis by Dimension
# MAGIC
# MAGIC Analyze each factor category in detail to understand contribution to scrap.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Operator Analysis

# COMMAND ----------

# Analyze operators
operator_analysis = pdf.groupby(['operator_skill_level', 'operator_name']).agg({
    'total_scrap_cost': ['sum', 'mean', 'count'],
    'quantity_scrapped': 'sum',
    'operator_experience_years': 'first'
}).reset_index()
operator_analysis.columns = ['skill_level', 'operator', 'total_cost', 'avg_cost', 'transaction_count', 'total_qty', 'experience_years']

# Calculate scrap rate per operator
operator_analysis['cost_per_transaction'] = operator_analysis['total_cost'] / operator_analysis['transaction_count']

# Rank operators by scrap cost
operator_analysis = operator_analysis.sort_values('total_cost', ascending=False)

print("🧑‍🔧 TOP 10 OPERATORS BY SCRAP COST:")
display(operator_analysis.head(10))

# Skill level analysis
skill_analysis = operator_analysis.groupby('skill_level').agg({
    'total_cost': 'sum',
    'avg_cost': 'mean',
    'transaction_count': 'sum',
    'experience_years': 'mean'
}).reset_index()
skill_analysis['pct_of_total'] = skill_analysis['total_cost'] / skill_analysis['total_cost'].sum() * 100

print("\n📊 SCRAP BY OPERATOR SKILL LEVEL:")
display(skill_analysis)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Shift Analysis

# COMMAND ----------

# Analyze shifts
shift_analysis = pdf.groupby(['shift', 'shift_name']).agg({
    'total_scrap_cost': ['sum', 'mean', 'count'],
    'quantity_scrapped': 'sum'
}).reset_index()
shift_analysis.columns = ['shift_num', 'shift_name', 'total_cost', 'avg_cost', 'transaction_count', 'total_qty']
shift_analysis['pct_of_total'] = shift_analysis['total_cost'] / shift_analysis['total_cost'].sum() * 100

print("⏰ SCRAP BY SHIFT:")
display(shift_analysis)

# Shift + Day of week interaction
shift_day_analysis = pdf.groupby(['shift_name', 'day_name']).agg({
    'total_scrap_cost': 'sum'
}).reset_index()
shift_day_pivot = shift_day_analysis.pivot(index='shift_name', columns='day_name', values='total_scrap_cost')

print("\n📅 SHIFT × DAY OF WEEK HEATMAP:")
display(shift_day_pivot)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Equipment Analysis

# COMMAND ----------

# Analyze equipment
equipment_analysis = pdf.groupby(['equipment_type_lookup', 'equipment_name_lookup']).agg({
    'total_scrap_cost': ['sum', 'mean', 'count'],
    'equipment_age_years': 'first',
    'equipment_oee': 'first'
}).reset_index()
equipment_analysis.columns = ['equip_type', 'equipment', 'total_cost', 'avg_cost', 'transaction_count', 'age_years', 'oee']
equipment_analysis = equipment_analysis.sort_values('total_cost', ascending=False)

print("🔧 TOP 10 EQUIPMENT BY SCRAP COST:")
display(equipment_analysis.head(10))

# Age correlation
age_correlation = pdf[['equipment_age_years', 'total_scrap_cost']].corr().iloc[0, 1]
oee_correlation = pdf[['equipment_oee', 'total_scrap_cost']].corr().iloc[0, 1]

print(f"\n📈 Equipment Age vs Scrap Cost Correlation: {age_correlation:.3f}")
print(f"📉 Equipment OEE vs Scrap Cost Correlation: {oee_correlation:.3f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Compile Factor Analysis Results

# COMMAND ----------

# Compile all factor analysis into a unified table
factor_results = []

# Skill level factors
for _, row in skill_analysis.iterrows():
    factor_results.append({
        'factor_category': 'Operator',
        'factor_name': f"Skill Level: {row['skill_level']}",
        'factor_value': row['skill_level'],
        'total_scrap_cost': row['total_cost'],
        'avg_scrap_cost': row['avg_cost'],
        'transaction_count': row['transaction_count'],
        'pct_of_total': row['pct_of_total'],
        'analysis_date': datetime.now()
    })

# Shift factors
for _, row in shift_analysis.iterrows():
    factor_results.append({
        'factor_category': 'Shift',
        'factor_name': f"Shift: {row['shift_name']}",
        'factor_value': str(row['shift_num']),
        'total_scrap_cost': row['total_cost'],
        'avg_scrap_cost': row['avg_cost'],
        'transaction_count': row['transaction_count'],
        'pct_of_total': row['pct_of_total'],
        'analysis_date': datetime.now()
    })

# Equipment type factors
equip_type_analysis = equipment_analysis.groupby('equip_type').agg({
    'total_cost': 'sum',
    'avg_cost': 'mean',
    'transaction_count': 'sum'
}).reset_index()
equip_type_analysis['pct_of_total'] = equip_type_analysis['total_cost'] / equip_type_analysis['total_cost'].sum() * 100

for _, row in equip_type_analysis.iterrows():
    factor_results.append({
        'factor_category': 'Equipment',
        'factor_name': f"Equipment Type: {row['equip_type']}",
        'factor_value': row['equip_type'],
        'total_scrap_cost': row['total_cost'],
        'avg_scrap_cost': row['avg_cost'],
        'transaction_count': row['transaction_count'],
        'pct_of_total': row['pct_of_total'],
        'analysis_date': datetime.now()
    })

# Save factor analysis
factor_df = spark.createDataFrame(pd.DataFrame(factor_results))
factor_df.write.mode("overwrite").saveAsTable(f"{FULL_SCHEMA}.ml_factor_analysis_results")
print(f"✅ Saved factor analysis to {FULL_SCHEMA}.ml_factor_analysis_results")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🎯 Model 5: Hidden Root Cause Discovery
# MAGIC
# MAGIC **Question: What factors predict scrap BETTER than the stated root cause?**
# MAGIC
# MAGIC We compare models that use stated root causes vs. models that use operational factors.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Train Model with Stated Root Cause Only

# COMMAND ----------

# Model using ONLY stated root cause
X_stated = pdf[['defect_category', 'root_cause_category']].copy()
for col in X_stated.columns:
    le = LabelEncoder()
    X_stated[col] = le.fit_transform(X_stated[col].astype(str))

X_train_stated, X_test_stated, y_train_rc, y_test_rc = train_test_split(
    X_stated, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)

model_stated = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=RANDOM_STATE, n_jobs=-1)
model_stated.fit(X_train_stated, y_train_rc)
r2_stated = r2_score(y_test_rc, model_stated.predict(X_test_stated))

print(f"📋 Model with STATED root cause only: R² = {r2_stated:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Train Model with Operational Factors Only

# COMMAND ----------

# Model using operational factors (no stated root cause)
operational_features = [
    'shift', 'operator_experience_years', 'equipment_age_years', 'equipment_oee',
    'supplier_quality_score', 'is_critical_part_encoded', 'operator_skill_level_encoded',
    'work_center_type_lookup_encoded', 'is_weekend_encoded', 'is_month_end_encoded'
]
X_operational = X[[col for col in operational_features if col in X.columns]]

X_train_ops, X_test_ops, y_train_ops, y_test_ops = train_test_split(
    X_operational, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)

model_operational = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=RANDOM_STATE, n_jobs=-1)
model_operational.fit(X_train_ops, y_train_ops)
r2_operational = r2_score(y_test_ops, model_operational.predict(X_test_ops))

print(f"🔧 Model with OPERATIONAL factors only: R² = {r2_operational:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Compare: Are Operational Factors Better Predictors?

# COMMAND ----------

improvement = r2_operational - r2_stated
print("\n" + "=" * 60)
print("🎯 ROOT CAUSE DISCOVERY RESULTS")
print("=" * 60)
print(f"\n📋 Stated Root Cause Model R²:     {r2_stated:.4f}")
print(f"🔧 Operational Factors Model R²:   {r2_operational:.4f}")
print(f"\n📊 Improvement:                    {improvement:+.4f} ({improvement/max(r2_stated, 0.001)*100:+.1f}%)")

if r2_operational > r2_stated:
    print("\n✅ FINDING: Operational factors predict scrap BETTER than stated root causes!")
    print("   This suggests the Quality team may be missing the real drivers.")
else:
    print("\n📋 FINDING: Stated root causes predict scrap well.")
    print("   The Quality team's assessments appear accurate.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Identify Hidden Root Causes

# COMMAND ----------

# Get feature importance from operational model
ops_importance = pd.DataFrame({
    'feature': X_operational.columns,
    'importance': model_operational.feature_importances_
}).sort_values('importance', ascending=False)

ops_importance['feature_clean'] = ops_importance['feature'].apply(decode_feature_name)
ops_importance['rank'] = range(1, len(ops_importance) + 1)

# These are the "hidden" root causes - factors that predict scrap but aren't in stated root causes
print("\n🔍 HIDDEN ROOT CAUSES (factors not typically in 8D reports):")
print("=" * 60)
for _, row in ops_importance.head(10).iterrows():
    print(f"  {row['rank']}. {row['feature_clean']}: {row['importance']:.4f}")

# Save hidden root causes
hidden_causes = ops_importance.assign(
    discovery_type='operational_factor',
    stated_r2=r2_stated,
    operational_r2=r2_operational,
    improvement=improvement,
    analysis_date=datetime.now()
)

hidden_spark_df = spark.createDataFrame(hidden_causes)
hidden_spark_df.write.mode("overwrite").saveAsTable(f"{FULL_SCHEMA}.ml_hidden_root_causes")
print(f"\n✅ Saved hidden root causes to {FULL_SCHEMA}.ml_hidden_root_causes")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🎮 Model 6: What-If Scenario Simulator
# MAGIC
# MAGIC Pre-compute scenarios for the Streamlit app to enable instant what-if analysis.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Generate Scenario Predictions

# COMMAND ----------

# Train production model for scenario simulation
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

scenario_model = xgb.XGBRegressor(n_estimators=200, max_depth=8, random_state=RANDOM_STATE, n_jobs=-1)
scenario_model.fit(X_scaled, y)

# Baseline predictions
baseline_pred = scenario_model.predict(X_scaled).mean()
print(f"📊 Baseline Average Scrap Cost: ${baseline_pred:,.2f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Simulate Various Scenarios

# COMMAND ----------

scenarios = []

# Scenario 1: Improve operator experience
X_improved_ops = X.copy()
if 'operator_experience_years' in X.columns:
    X_improved_ops['operator_experience_years'] = X_improved_ops['operator_experience_years'] * 1.5
    pred = scenario_model.predict(scaler.transform(X_improved_ops)).mean()
    scenarios.append({
        'scenario_name': 'Increase Operator Experience 50%',
        'scenario_type': 'People',
        'baseline_cost': baseline_pred,
        'predicted_cost': pred,
        'cost_change': pred - baseline_pred,
        'pct_change': (pred - baseline_pred) / baseline_pred * 100
    })

# Scenario 2: Reduce equipment age (newer equipment)
X_new_equip = X.copy()
if 'equipment_age_years' in X.columns:
    X_new_equip['equipment_age_years'] = X_new_equip['equipment_age_years'] * 0.5
    pred = scenario_model.predict(scaler.transform(X_new_equip)).mean()
    scenarios.append({
        'scenario_name': 'Reduce Equipment Age 50%',
        'scenario_type': 'Equipment',
        'baseline_cost': baseline_pred,
        'predicted_cost': pred,
        'cost_change': pred - baseline_pred,
        'pct_change': (pred - baseline_pred) / baseline_pred * 100
    })

# Scenario 3: Improve supplier quality scores
X_better_suppliers = X.copy()
if 'supplier_quality_score' in X.columns:
    X_better_suppliers['supplier_quality_score'] = np.minimum(
        X_better_suppliers['supplier_quality_score'] * 1.1, 1.0
    )
    pred = scenario_model.predict(scaler.transform(X_better_suppliers)).mean()
    scenarios.append({
        'scenario_name': 'Improve Supplier Quality 10%',
        'scenario_type': 'Supplier',
        'baseline_cost': baseline_pred,
        'predicted_cost': pred,
        'cost_change': pred - baseline_pred,
        'pct_change': (pred - baseline_pred) / baseline_pred * 100
    })

# Scenario 4: Improve equipment OEE
X_better_oee = X.copy()
if 'equipment_oee' in X.columns:
    X_better_oee['equipment_oee'] = np.minimum(X_better_oee['equipment_oee'] * 1.15, 1.0)
    pred = scenario_model.predict(scaler.transform(X_better_oee)).mean()
    scenarios.append({
        'scenario_name': 'Improve Equipment OEE 15%',
        'scenario_type': 'Equipment',
        'baseline_cost': baseline_pred,
        'predicted_cost': pred,
        'cost_change': pred - baseline_pred,
        'pct_change': (pred - baseline_pred) / baseline_pred * 100
    })

# Scenario 5: Eliminate weekend production
X_no_weekend = X.copy()
if 'is_weekend_encoded' in X.columns:
    X_no_weekend['is_weekend_encoded'] = 0
    pred = scenario_model.predict(scaler.transform(X_no_weekend)).mean()
    scenarios.append({
        'scenario_name': 'Eliminate Weekend Production',
        'scenario_type': 'Time',
        'baseline_cost': baseline_pred,
        'predicted_cost': pred,
        'cost_change': pred - baseline_pred,
        'pct_change': (pred - baseline_pred) / baseline_pred * 100
    })

# Scenario 6: All shifts to Shift 1 (day shift)
X_day_shift = X.copy()
if 'shift' in X.columns:
    X_day_shift['shift'] = 1
    pred = scenario_model.predict(scaler.transform(X_day_shift)).mean()
    scenarios.append({
        'scenario_name': 'Move All Production to Day Shift',
        'scenario_type': 'Time',
        'baseline_cost': baseline_pred,
        'predicted_cost': pred,
        'cost_change': pred - baseline_pred,
        'pct_change': (pred - baseline_pred) / baseline_pred * 100
    })

# Display scenarios
scenarios_df = pd.DataFrame(scenarios).sort_values('pct_change')
scenarios_df['analysis_date'] = datetime.now()

print("\n🎮 WHAT-IF SCENARIO ANALYSIS:")
print("=" * 80)
for _, row in scenarios_df.iterrows():
    direction = "📉" if row['pct_change'] < 0 else "📈"
    print(f"{direction} {row['scenario_name']}: {row['pct_change']:+.1f}% (${row['cost_change']:+,.0f})")

display(scenarios_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Save Scenarios to Delta Table

# COMMAND ----------

# Save what-if scenarios
scenarios_spark_df = spark.createDataFrame(scenarios_df)
scenarios_spark_df.write.mode("overwrite").saveAsTable(f"{FULL_SCHEMA}.ml_whatif_scenarios")
print(f"✅ Saved what-if scenarios to {FULL_SCHEMA}.ml_whatif_scenarios")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📋 Summary: Output Tables Created
# MAGIC
# MAGIC | Table | Description | Use Case |
# MAGIC |-------|-------------|----------|
# MAGIC | `ml_feature_importance_scores` | Feature importance from multiple models | Dashboard: Top factors |
# MAGIC | `ml_shap_analysis` | SHAP values for explainability | Dashboard: Detailed explanations |
# MAGIC | `ml_interaction_effects` | Feature interaction strengths | Dashboard: Combined effects |
# MAGIC | `ml_factor_analysis_results` | Detailed analysis by factor | Dashboard: Drill-down |
# MAGIC | `ml_hidden_root_causes` | AI-discovered root causes | 8D Improvement |
# MAGIC | `ml_whatif_scenarios` | Pre-computed scenarios | What-If Simulator |

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🔗 Model Registration for Production

# COMMAND ----------

# Register best model to MLflow
# End any active runs first
mlflow.end_run()

with mlflow.start_run(run_name="production_scenario_model"):
    # Log model (use X_train_ops since model_operational was trained on operational features)
    signature = infer_signature(X_train_ops, model_operational.predict(X_train_ops))
    mlflow.sklearn.log_model(
        model_operational,
        "scrap_predictor",
        signature=signature
    )
    
    # Log metrics
    mlflow.log_metrics({
        'r2_score': r2_operational,
        'improvement_over_stated': improvement
    })
    
    # Log feature importance
    for i, row in ops_importance.iterrows():
        mlflow.log_metric(f"importance_{row['feature_clean']}", row['importance'])
    
    print(f"✅ Model registered to MLflow")
    print(f"   R² Score: {r2_operational:.4f}")
    print(f"   Improvement over stated causes: {improvement:+.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## ✅ Notebook Complete
# MAGIC
# MAGIC ### Key Findings:
# MAGIC 1. **Top Factors**: Review `ml_feature_importance_scores` for the real drivers
# MAGIC 2. **Hidden Causes**: Check `ml_hidden_root_causes` for factors Quality team may be missing
# MAGIC 3. **Interactions**: See `ml_interaction_effects` for combined effects
# MAGIC 4. **What-If**: Use `ml_whatif_scenarios` for scenario planning
# MAGIC
# MAGIC ### Next Steps:
# MAGIC 1. Review the hidden root causes with the Quality team
# MAGIC 2. Implement targeted improvements based on what-if scenarios
# MAGIC 3. Monitor feature importance trends over time
# MAGIC 4. Connect scenarios to Streamlit app for interactive analysis
# MAGIC
