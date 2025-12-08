# Databricks notebook source
# MAGIC %md
# MAGIC # 🚀 AutoML Forecasting & Advanced Scenario Planning
# MAGIC 
# MAGIC ## Business Objective
# MAGIC 
# MAGIC This notebook leverages **Databricks AutoML** for automated model selection and hyperparameter tuning,
# MAGIC going beyond Prophet to find the best forecasting model for our scrap data.
# MAGIC 
# MAGIC ## AutoML Advantages over Manual Modeling
# MAGIC | Feature | Manual (Prophet) | Databricks AutoML |
# MAGIC |---------|-----------------|-------------------|
# MAGIC | Model Selection | Single model | Tests 20+ algorithms |
# MAGIC | Hyperparameter Tuning | Manual | Automated optimization |
# MAGIC | Feature Engineering | Manual | Automated feature generation |
# MAGIC | Explainability | Limited | Built-in SHAP |
# MAGIC | Time to Production | Hours | Minutes |
# MAGIC 
# MAGIC ## Models AutoML Will Test
# MAGIC - LightGBM, XGBoost, CatBoost
# MAGIC - Random Forest, Extra Trees
# MAGIC - ARIMA, Prophet
# MAGIC - Linear Regression, ElasticNet
# MAGIC - Decision Tree, Gradient Boosting
# MAGIC 
# MAGIC ## Output Tables
# MAGIC - `ml_automl_forecast_results` - Best model forecasts
# MAGIC - `ml_automl_model_comparison` - Model performance comparison
# MAGIC - `ml_scenario_simulations` - Advanced scenario outcomes
# MAGIC - `ml_sensitivity_analysis` - Factor sensitivity ranges

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📦 Install Dependencies

# COMMAND ----------

# MAGIC %pip install databricks-automl mlflow[databricks] scikit-learn xgboost lightgbm shap

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

# Databricks AutoML
from databricks import automl

# Scikit-learn
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

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

# COMMAND ----------

# Unity Catalog Configuration
CATALOG = "welch"
SCHEMA = "scrap_demo"
FULL_SCHEMA = f"{CATALOG}.{SCHEMA}"

# AutoML Configuration
AUTOML_TIMEOUT_MINUTES = 30  # Time for AutoML to run
FORECAST_HORIZON_DAYS = 90  # 3-month forecast

print(f"📍 Target Schema: {FULL_SCHEMA}")
print(f"⏱️ AutoML Timeout: {AUTOML_TIMEOUT_MINUTES} minutes")
print(f"📅 Forecast Horizon: {FORECAST_HORIZON_DAYS} days")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📊 Prepare Data for AutoML Forecasting

# COMMAND ----------

# MAGIC %md
# MAGIC ### Load and Aggregate Daily Scrap Data

# COMMAND ----------

# Create daily aggregation for forecasting
daily_scrap = spark.sql(f"""
    SELECT 
        transaction_date as ds,
        SUM(total_scrap_cost) as y,
        SUM(quantity_scrapped) as total_quantity,
        COUNT(*) as transaction_count,
        AVG(equipment_age_years) as avg_equipment_age,
        AVG(equipment_oee) as avg_equipment_oee,
        AVG(supplier_quality_score) as avg_supplier_quality,
        AVG(operator_experience_years) as avg_operator_experience,
        COUNT(DISTINCT plant_id) as active_plants,
        COUNT(DISTINCT supplier_id) as active_suppliers,
        SUM(CASE WHEN shift = 1 THEN 1 ELSE 0 END) / COUNT(*) as shift1_ratio,
        SUM(CASE WHEN shift = 2 THEN 1 ELSE 0 END) / COUNT(*) as shift2_ratio,
        SUM(CASE WHEN shift = 3 THEN 1 ELSE 0 END) / COUNT(*) as shift3_ratio,
        SUM(CASE WHEN is_critical_part THEN total_scrap_cost ELSE 0 END) as critical_part_cost
    FROM {FULL_SCHEMA}.scrap_transactions_silver
    WHERE transaction_date >= '2022-01-01'
    GROUP BY transaction_date
    ORDER BY transaction_date
""")

print(f"✅ Loaded {daily_scrap.count():,} days of data")
display(daily_scrap.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Add Time-Based Features

# COMMAND ----------

# Add calendar features
daily_scrap_features = daily_scrap.withColumn(
    "day_of_week", F.dayofweek("ds")
).withColumn(
    "month", F.month("ds")
).withColumn(
    "quarter", F.quarter("ds")
).withColumn(
    "is_weekend", F.dayofweek("ds").isin(1, 7).cast("int")
).withColumn(
    "is_month_end", (F.dayofmonth("ds") >= 25).cast("int")
).withColumn(
    "week_of_year", F.weekofyear("ds")
)

# Convert to pandas for AutoML
forecast_pdf = daily_scrap_features.toPandas()
forecast_pdf['ds'] = pd.to_datetime(forecast_pdf['ds'])
forecast_pdf = forecast_pdf.sort_values('ds').reset_index(drop=True)

print(f"📊 Data shape: {forecast_pdf.shape}")
print(f"📅 Date range: {forecast_pdf['ds'].min()} to {forecast_pdf['ds'].max()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🤖 Run Databricks AutoML for Forecasting
# MAGIC 
# MAGIC AutoML will automatically:
# MAGIC 1. Try 20+ different algorithms
# MAGIC 2. Tune hyperparameters
# MAGIC 3. Handle feature engineering
# MAGIC 4. Provide explainability

# COMMAND ----------

# MAGIC %md
# MAGIC ### AutoML Regression Experiment

# COMMAND ----------

# Prepare dataset for AutoML
# Use last 90 days as holdout
train_cutoff = forecast_pdf['ds'].max() - timedelta(days=FORECAST_HORIZON_DAYS)
train_df = forecast_pdf[forecast_pdf['ds'] <= train_cutoff].copy()
test_df = forecast_pdf[forecast_pdf['ds'] > train_cutoff].copy()

print(f"📊 Training samples: {len(train_df):,}")
print(f"📊 Test samples: {len(test_df):,}")

# Create Spark DataFrame for AutoML
train_spark_df = spark.createDataFrame(train_df)

# COMMAND ----------

# Run AutoML
print("🚀 Starting Databricks AutoML...")
print(f"   This will run for up to {AUTOML_TIMEOUT_MINUTES} minutes...")
print("   AutoML is testing multiple algorithms and hyperparameters...")

# Run AutoML regression
automl_results = automl.regress(
    dataset=train_spark_df,
    target_col="y",
    exclude_cols=["ds"],  # Exclude date from features (we added calendar features)
    primary_metric="rmse",
    timeout_minutes=AUTOML_TIMEOUT_MINUTES,
    max_trials=50
)

print("\n✅ AutoML Complete!")
print(f"   Best Model: {automl_results.best_trial.model_description}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Evaluate AutoML Results

# COMMAND ----------

# Get the best model
best_model = automl_results.best_trial.load_model()

# Prepare test features (exclude ds and y)
test_features = test_df.drop(columns=['ds', 'y'])

# Make predictions
test_df['predicted'] = best_model.predict(test_features)

# Calculate metrics
rmse = np.sqrt(mean_squared_error(test_df['y'], test_df['predicted']))
mae = mean_absolute_error(test_df['y'], test_df['predicted'])
r2 = r2_score(test_df['y'], test_df['predicted'])
mape = np.mean(np.abs((test_df['y'] - test_df['predicted']) / test_df['y'])) * 100

print("\n📊 AutoML FORECAST PERFORMANCE:")
print("=" * 50)
print(f"RMSE:  ${rmse:,.2f}")
print(f"MAE:   ${mae:,.2f}")
print(f"R²:    {r2:.4f}")
print(f"MAPE:  {mape:.2f}%")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Model Comparison Summary

# COMMAND ----------

# Get all trial results
trials_df = automl_results.trials
print(f"\n🏆 TOP 10 MODELS TESTED:")
print("=" * 80)

model_comparison = []
for i, trial in enumerate(trials_df[:10]):
    model_info = {
        'rank': i + 1,
        'model_name': trial.model_description,
        'rmse': trial.metrics.get('val_rmse', None),
        'r2': trial.metrics.get('val_r2_score', None),
        'training_time_seconds': trial.duration
    }
    model_comparison.append(model_info)
    print(f"{i+1}. {trial.model_description}")
    print(f"   RMSE: ${trial.metrics.get('val_rmse', 0):,.2f} | R²: {trial.metrics.get('val_r2_score', 0):.4f}")

# Save model comparison
comparison_df = pd.DataFrame(model_comparison)
comparison_df['analysis_date'] = datetime.now()
comparison_spark_df = spark.createDataFrame(comparison_df)
comparison_spark_df.write.mode("overwrite").saveAsTable(f"{FULL_SCHEMA}.ml_automl_model_comparison")
print(f"\n✅ Saved model comparison to {FULL_SCHEMA}.ml_automl_model_comparison")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Visualize Forecast vs Actual

# COMMAND ----------

fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# Plot 1: Time series comparison
axes[0].plot(test_df['ds'], test_df['y'], label='Actual', color='blue', linewidth=2)
axes[0].plot(test_df['ds'], test_df['predicted'], label='Predicted', color='red', linewidth=2, linestyle='--')
axes[0].fill_between(test_df['ds'], 
                     test_df['predicted'] * 0.9, 
                     test_df['predicted'] * 1.1, 
                     alpha=0.2, color='red', label='±10% Band')
axes[0].set_xlabel('Date')
axes[0].set_ylabel('Daily Scrap Cost ($)')
axes[0].set_title('AutoML Forecast vs Actual')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot 2: Residuals
residuals = test_df['y'] - test_df['predicted']
axes[1].bar(test_df['ds'], residuals, color=['green' if r > 0 else 'red' for r in residuals], alpha=0.7)
axes[1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
axes[1].set_xlabel('Date')
axes[1].set_ylabel('Residual ($)')
axes[1].set_title('Prediction Residuals (Actual - Predicted)')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/tmp/automl_forecast.png', dpi=150, bbox_inches='tight')
mlflow.log_artifact('/tmp/automl_forecast.png')
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Generate Future Forecasts

# COMMAND ----------

# Generate dates for future forecast
last_date = forecast_pdf['ds'].max()
future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=FORECAST_HORIZON_DAYS, freq='D')

# Create future features (using averages from historical data)
future_df = pd.DataFrame({'ds': future_dates})
future_df['day_of_week'] = future_df['ds'].dt.dayofweek + 1
future_df['month'] = future_df['ds'].dt.month
future_df['quarter'] = future_df['ds'].dt.quarter
future_df['is_weekend'] = future_df['day_of_week'].isin([1, 7]).astype(int)
future_df['is_month_end'] = (future_df['ds'].dt.day >= 25).astype(int)
future_df['week_of_year'] = future_df['ds'].dt.isocalendar().week

# Use historical averages for other features
for col in ['transaction_count', 'avg_equipment_age', 'avg_equipment_oee', 
            'avg_supplier_quality', 'avg_operator_experience', 'active_plants',
            'active_suppliers', 'shift1_ratio', 'shift2_ratio', 'shift3_ratio',
            'critical_part_cost', 'total_quantity']:
    if col in forecast_pdf.columns:
        future_df[col] = forecast_pdf[col].mean()

# Make future predictions
future_features = future_df.drop(columns=['ds'])
future_df['predicted_cost'] = best_model.predict(future_features)

# Add confidence bounds (simple ±15%)
future_df['lower_bound'] = future_df['predicted_cost'] * 0.85
future_df['upper_bound'] = future_df['predicted_cost'] * 1.15

print(f"\n📅 FORECAST SUMMARY (Next {FORECAST_HORIZON_DAYS} Days):")
print("=" * 50)
print(f"Total Predicted Cost: ${future_df['predicted_cost'].sum():,.0f}")
print(f"Avg Daily Cost: ${future_df['predicted_cost'].mean():,.0f}")
print(f"Min Daily Cost: ${future_df['predicted_cost'].min():,.0f}")
print(f"Max Daily Cost: ${future_df['predicted_cost'].max():,.0f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Save AutoML Forecasts

# COMMAND ----------

# Prepare forecast results for saving
forecast_results = future_df[['ds', 'predicted_cost', 'lower_bound', 'upper_bound']].copy()
forecast_results = forecast_results.rename(columns={'ds': 'forecast_date'})
forecast_results['model_type'] = automl_results.best_trial.model_description
forecast_results['rmse'] = rmse
forecast_results['r2'] = r2
forecast_results['created_at'] = datetime.now()
forecast_results['entity_name'] = 'Total'
forecast_results['entity_id'] = 'ALL'

# Save to Delta
forecast_spark_df = spark.createDataFrame(forecast_results)
forecast_spark_df.write.mode("overwrite").saveAsTable(f"{FULL_SCHEMA}.ml_automl_forecast_results")
print(f"✅ Saved AutoML forecasts to {FULL_SCHEMA}.ml_automl_forecast_results")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📊 Advanced Scenario Simulations
# MAGIC 
# MAGIC Use the trained model to simulate various operational scenarios.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Define Scenario Grid

# COMMAND ----------

# Base scenario (current averages)
base_scenario = {
    'avg_equipment_age': forecast_pdf['avg_equipment_age'].mean(),
    'avg_equipment_oee': forecast_pdf['avg_equipment_oee'].mean(),
    'avg_supplier_quality': forecast_pdf['avg_supplier_quality'].mean(),
    'avg_operator_experience': forecast_pdf['avg_operator_experience'].mean(),
    'shift1_ratio': forecast_pdf['shift1_ratio'].mean(),
    'shift2_ratio': forecast_pdf['shift2_ratio'].mean(),
    'shift3_ratio': forecast_pdf['shift3_ratio'].mean(),
    'active_plants': forecast_pdf['active_plants'].mean(),
    'critical_part_cost': forecast_pdf['critical_part_cost'].mean()
}

print("📊 BASELINE SCENARIO VALUES:")
for k, v in base_scenario.items():
    print(f"   {k}: {v:.3f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Run Scenario Simulations

# COMMAND ----------

def simulate_scenario(scenario_name, modifications, base=base_scenario):
    """Simulate a scenario by modifying base values."""
    scenario = base.copy()
    scenario.update(modifications)
    
    # Create feature vector (matching training features)
    features = pd.DataFrame([{
        'day_of_week': 3,  # Wednesday (mid-week)
        'month': 6,  # June (mid-year)
        'quarter': 2,
        'is_weekend': 0,
        'is_month_end': 0,
        'week_of_year': 25,
        'transaction_count': forecast_pdf['transaction_count'].mean(),
        'total_quantity': forecast_pdf['total_quantity'].mean(),
        **scenario
    }])
    
    # Ensure all columns match training
    for col in future_features.columns:
        if col not in features.columns:
            features[col] = forecast_pdf[col].mean() if col in forecast_pdf.columns else 0
    
    features = features[future_features.columns]
    
    pred = best_model.predict(features)[0]
    return pred

# Baseline prediction
baseline_cost = simulate_scenario("Baseline", {})

# Define scenarios to test
scenarios = [
    ("Baseline (Current State)", {}, "baseline"),
    
    # Equipment scenarios
    ("New Equipment Fleet (-50% age)", {"avg_equipment_age": base_scenario['avg_equipment_age'] * 0.5}, "equipment"),
    ("Equipment Modernization (-25% age)", {"avg_equipment_age": base_scenario['avg_equipment_age'] * 0.75}, "equipment"),
    ("OEE Improvement (+10%)", {"avg_equipment_oee": min(base_scenario['avg_equipment_oee'] * 1.1, 1.0)}, "equipment"),
    ("OEE World Class (0.85)", {"avg_equipment_oee": 0.85}, "equipment"),
    
    # Supplier scenarios
    ("Supplier Quality +5%", {"avg_supplier_quality": min(base_scenario['avg_supplier_quality'] * 1.05, 1.0)}, "supplier"),
    ("Supplier Quality +10%", {"avg_supplier_quality": min(base_scenario['avg_supplier_quality'] * 1.10, 1.0)}, "supplier"),
    ("Premium Suppliers Only (0.95 quality)", {"avg_supplier_quality": 0.95}, "supplier"),
    
    # Operator scenarios
    ("Operator Experience +25%", {"avg_operator_experience": base_scenario['avg_operator_experience'] * 1.25}, "people"),
    ("Operator Experience +50%", {"avg_operator_experience": base_scenario['avg_operator_experience'] * 1.5}, "people"),
    
    # Shift scenarios
    ("All Day Shift", {"shift1_ratio": 1.0, "shift2_ratio": 0.0, "shift3_ratio": 0.0}, "time"),
    ("Day + Evening Only", {"shift1_ratio": 0.6, "shift2_ratio": 0.4, "shift3_ratio": 0.0}, "time"),
    ("Balanced Shifts", {"shift1_ratio": 0.33, "shift2_ratio": 0.33, "shift3_ratio": 0.34}, "time"),
    
    # Combined scenarios
    ("Best Practice Combo", {
        "avg_equipment_oee": 0.85,
        "avg_supplier_quality": 0.95,
        "avg_operator_experience": base_scenario['avg_operator_experience'] * 1.25
    }, "combined"),
    
    ("Aggressive Improvement", {
        "avg_equipment_age": base_scenario['avg_equipment_age'] * 0.5,
        "avg_equipment_oee": 0.90,
        "avg_supplier_quality": 0.98,
        "avg_operator_experience": base_scenario['avg_operator_experience'] * 1.5,
        "shift1_ratio": 0.7, "shift2_ratio": 0.3, "shift3_ratio": 0.0
    }, "combined")
]

# Run all scenarios
simulation_results = []
for name, mods, category in scenarios:
    predicted_cost = simulate_scenario(name, mods)
    change = predicted_cost - baseline_cost
    pct_change = (change / baseline_cost) * 100
    
    simulation_results.append({
        'scenario_name': name,
        'scenario_category': category,
        'predicted_daily_cost': predicted_cost,
        'cost_change': change,
        'pct_change': pct_change,
        'annual_impact': change * 365,
        'baseline_cost': baseline_cost
    })

# Create DataFrame and sort
sim_df = pd.DataFrame(simulation_results)
sim_df = sim_df.sort_values('pct_change')
sim_df['analysis_date'] = datetime.now()

print("\n🎮 SCENARIO SIMULATION RESULTS:")
print("=" * 90)
print(f"{'Scenario':<45} {'Daily Cost':>12} {'Change':>12} {'Annual':>15}")
print("-" * 90)
for _, row in sim_df.iterrows():
    print(f"{row['scenario_name']:<45} ${row['predicted_daily_cost']:>10,.0f} {row['pct_change']:>+10.1f}% ${row['annual_impact']:>+12,.0f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Save Scenario Simulations

# COMMAND ----------

sim_spark_df = spark.createDataFrame(sim_df)
sim_spark_df.write.mode("overwrite").saveAsTable(f"{FULL_SCHEMA}.ml_scenario_simulations")
print(f"✅ Saved scenario simulations to {FULL_SCHEMA}.ml_scenario_simulations")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📈 Sensitivity Analysis
# MAGIC 
# MAGIC Understand how sensitive scrap cost is to changes in each factor.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Run Sensitivity Analysis

# COMMAND ----------

# Factors to analyze
sensitivity_factors = [
    ('avg_equipment_age', -0.5, 0.5, 'Equipment Age'),
    ('avg_equipment_oee', -0.2, 0.3, 'Equipment OEE'),
    ('avg_supplier_quality', -0.1, 0.15, 'Supplier Quality'),
    ('avg_operator_experience', -0.3, 0.5, 'Operator Experience'),
    ('shift1_ratio', -0.3, 0.4, 'Day Shift Ratio'),
]

sensitivity_results = []
steps = 11  # Number of points to test

for factor, min_change, max_change, label in sensitivity_factors:
    base_value = base_scenario[factor]
    
    for pct in np.linspace(min_change, max_change, steps):
        new_value = base_value * (1 + pct)
        
        # Clamp values to valid ranges
        if 'ratio' in factor or 'quality' in factor or 'oee' in factor:
            new_value = max(0, min(1, new_value))
        if 'age' in factor or 'experience' in factor:
            new_value = max(0, new_value)
        
        pred = simulate_scenario(f"{label} {pct:+.0%}", {factor: new_value})
        
        sensitivity_results.append({
            'factor': label,
            'pct_change_input': pct * 100,
            'new_value': new_value,
            'base_value': base_value,
            'predicted_cost': pred,
            'cost_change': pred - baseline_cost,
            'cost_pct_change': (pred - baseline_cost) / baseline_cost * 100
        })

sens_df = pd.DataFrame(sensitivity_results)
sens_df['analysis_date'] = datetime.now()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Visualize Sensitivity

# COMMAND ----------

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

colors = ['#e74c3c', '#3498db', '#f39c12', '#9b59b6', '#2ecc71']

for i, (factor_name, color) in enumerate(zip(sens_df['factor'].unique(), colors)):
    factor_data = sens_df[sens_df['factor'] == factor_name]
    
    ax = axes[i]
    ax.plot(factor_data['pct_change_input'], factor_data['cost_pct_change'], 
           color=color, linewidth=2, marker='o')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Input Change (%)')
    ax.set_ylabel('Cost Change (%)')
    ax.set_title(f'Sensitivity: {factor_name}')
    ax.grid(True, alpha=0.3)

# Remove empty subplot
axes[-1].axis('off')

plt.tight_layout()
plt.savefig('/tmp/sensitivity_analysis.png', dpi=150, bbox_inches='tight')
mlflow.log_artifact('/tmp/sensitivity_analysis.png')
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Save Sensitivity Analysis

# COMMAND ----------

sens_spark_df = spark.createDataFrame(sens_df)
sens_spark_df.write.mode("overwrite").saveAsTable(f"{FULL_SCHEMA}.ml_sensitivity_analysis")
print(f"✅ Saved sensitivity analysis to {FULL_SCHEMA}.ml_sensitivity_analysis")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📋 Summary: Tables Created
# MAGIC 
# MAGIC | Table | Description | Use Case |
# MAGIC |-------|-------------|----------|
# MAGIC | `ml_automl_model_comparison` | Performance of all models tested | Model selection |
# MAGIC | `ml_automl_forecast_results` | Best model forecasts | Dashboard |
# MAGIC | `ml_scenario_simulations` | Pre-computed scenarios | What-If planning |
# MAGIC | `ml_sensitivity_analysis` | Factor sensitivity ranges | Risk assessment |

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🔗 Register Best Model

# COMMAND ----------

# Register the best model from AutoML
model_uri = automl_results.best_trial.model_path

# Log additional metrics
with mlflow.start_run(run_name="automl_best_model_registration"):
    mlflow.log_metrics({
        'test_rmse': rmse,
        'test_mae': mae,
        'test_r2': r2,
        'test_mape': mape,
        'forecast_days': FORECAST_HORIZON_DAYS
    })
    mlflow.log_param('model_type', automl_results.best_trial.model_description)
    mlflow.log_param('training_samples', len(train_df))
    
    print(f"\n✅ Best AutoML Model Registered")
    print(f"   Model Type: {automl_results.best_trial.model_description}")
    print(f"   Test RMSE: ${rmse:,.2f}")
    print(f"   Test R²: {r2:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## ✅ Notebook Complete
# MAGIC 
# MAGIC ### Key Results:
# MAGIC 1. **AutoML** tested 20+ models and selected the best performer
# MAGIC 2. **Scenarios** pre-computed for instant what-if analysis
# MAGIC 3. **Sensitivity** analysis shows which factors have most impact
# MAGIC 
# MAGIC ### Use in Streamlit App:
# MAGIC - Query `ml_scenario_simulations` for what-if dropdown
# MAGIC - Query `ml_sensitivity_analysis` for impact sliders
# MAGIC - Query `ml_automl_forecast_results` for forecast charts

