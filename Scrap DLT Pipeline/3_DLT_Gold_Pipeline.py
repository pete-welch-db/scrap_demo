# Databricks notebook source
# MAGIC %md
# MAGIC # 🏆 DLT Pipeline - Gold Layer Aggregations
# MAGIC 
# MAGIC ## Overview
# MAGIC 
# MAGIC This notebook defines **Delta Live Tables (DLT)** for the Gold layer - analytics-ready aggregations.
# MAGIC 
# MAGIC ## Medallion Architecture Position
# MAGIC 
# MAGIC ```
# MAGIC ┌─────────────────────┐      ┌─────────────────────┐      ┌─────────────────────┐
# MAGIC │   BRONZE (Raw)      │ ───► │  SILVER (Cleaned)   │ ───► │   GOLD (Analytics)  │
# MAGIC │                     │      │                     │      │                     │
# MAGIC └─────────────────────┘      └─────────────────────┘      └─────────────────────┘
# MAGIC                                                                    ▲
# MAGIC                                                                    │
# MAGIC                                                             THIS NOTEBOOK
# MAGIC ```
# MAGIC 
# MAGIC ## Gold Layer Tables
# MAGIC 
# MAGIC ### 📊 Aggregation Tables
# MAGIC | Table | Purpose |
# MAGIC |-------|---------|
# MAGIC | `scrap_daily_summary_gold` | Daily scrap metrics for time series |
# MAGIC | `scrap_monthly_summary_gold` | Monthly trends with MoM changes |
# MAGIC | `scrap_by_plant_gold` | Plant performance comparison |
# MAGIC | `scrap_by_supplier_gold` | Supplier quality analysis |
# MAGIC | `scrap_by_defect_gold` | Pareto analysis of defects |
# MAGIC | `scrap_by_equipment_gold` | Equipment-related scrap |
# MAGIC | `eight_d_summary_gold` | 8D report statistics |
# MAGIC 
# MAGIC ### 🤖 ML Feature Tables
# MAGIC | Table | Purpose |
# MAGIC |-------|---------|
# MAGIC | `ml_scrap_forecast_features_gold` | Time series forecasting features |
# MAGIC | `ml_scrap_policy_features_gold` | Scrap reduction policy features |
# MAGIC | `eight_d_text_analysis_gold` | 8D text prepared for LLM |
# MAGIC 
# MAGIC ### 📈 Dashboard Tables
# MAGIC | Table | Purpose |
# MAGIC |-------|---------|
# MAGIC | `kpi_dashboard_gold` | Executive KPI metrics |
# MAGIC | `supplier_scorecard_gold` | Supplier quality scorecard |

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📚 Import Libraries

# COMMAND ----------

import dlt
from pyspark.sql import functions as F
from pyspark.sql.window import Window

# COMMAND ----------

# MAGIC %md
# MAGIC ## ⚙️ Configuration

# COMMAND ----------

# Configuration
CATALOG = "welch"
SCHEMA = "scrap_demo"
SOURCE_SCHEMA = f"{CATALOG}.{SCHEMA}"

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # 📊 Time-Based Aggregations
# MAGIC ---

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📅 Daily Scrap Summary
# MAGIC 
# MAGIC **Purpose**: Aggregate scrap metrics by day for time series analysis and dashboards.
# MAGIC 
# MAGIC **Metrics Included**:
# MAGIC - Total records, quantity, and cost
# MAGIC - Unique parts and defects
# MAGIC - Plants and suppliers affected
# MAGIC - Critical incident counts

# COMMAND ----------

@dlt.table(
    name="scrap_daily_summary_gold",
    comment="Daily scrap aggregations for dashboards and time series analysis",
    table_properties={
        "quality": "gold",
        "pipelines.autoOptimize.managed": "true"
    }
)
def scrap_daily_summary_gold():
    """
    Daily aggregation of scrap metrics.
    
    Used for: Time series visualization, anomaly detection baseline, forecasting.
    """
    return (
        dlt.read("scrap_transactions_silver")
        .groupBy(
            "transaction_date",
            "transaction_year",
            "transaction_month",
            "transaction_quarter",
            "transaction_week",
            "day_name",
            "is_weekend"
        )
        .agg(
            # Volume metrics
            F.count("*").alias("total_scrap_records"),
            F.sum("quantity_scrapped").alias("total_quantity_scrapped"),
            F.round(F.sum("total_scrap_cost"), 2).alias("total_scrap_cost"),
            
            # Averages
            F.round(F.avg("quantity_scrapped"), 2).alias("avg_quantity_per_record"),
            F.round(F.avg("total_scrap_cost"), 2).alias("avg_cost_per_record"),
            
            # Diversity metrics
            F.countDistinct("part_id").alias("unique_parts_scrapped"),
            F.countDistinct("defect_code").alias("unique_defect_codes"),
            F.countDistinct("plant_id").alias("plants_affected"),
            F.countDistinct("supplier_id").alias("suppliers_involved"),
            
            # Critical metrics
            F.sum(F.when(F.col("is_critical_part") == True, F.col("quantity_scrapped"))
                  .otherwise(0)).alias("critical_parts_scrapped"),
            F.sum(F.when(F.col("cost_severity") == "Critical", 1)
                  .otherwise(0)).alias("critical_cost_incidents")
        )
        .withColumn("processed_at", F.current_timestamp())
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📆 Monthly Scrap Summary
# MAGIC 
# MAGIC **Purpose**: Monthly aggregations with month-over-month (MoM) change calculations.
# MAGIC 
# MAGIC **Business Value**: Track trends, identify seasonal patterns, support budgeting.

# COMMAND ----------

@dlt.table(
    name="scrap_monthly_summary_gold",
    comment="Monthly scrap aggregations with MoM trend calculations",
    table_properties={
        "quality": "gold",
        "pipelines.autoOptimize.managed": "true"
    }
)
def scrap_monthly_summary_gold():
    """
    Monthly aggregation with trend calculations.
    
    Includes month-over-month changes for cost and quantity.
    """
    monthly = (
        dlt.read("scrap_transactions_silver")
        .groupBy(
            "transaction_year",
            "transaction_month",
            "transaction_quarter"
        )
        .agg(
            F.count("*").alias("total_scrap_records"),
            F.sum("quantity_scrapped").alias("total_quantity_scrapped"),
            F.round(F.sum("total_scrap_cost"), 2).alias("total_scrap_cost"),
            F.round(F.avg("quantity_scrapped"), 2).alias("avg_quantity_per_record"),
            F.round(F.avg("total_scrap_cost"), 2).alias("avg_cost_per_record"),
            F.countDistinct("part_id").alias("unique_parts_scrapped"),
            F.countDistinct("defect_code").alias("unique_defect_codes"),
            F.countDistinct("plant_id").alias("plants_affected"),
            F.countDistinct("operator_id").alias("operators_involved"),
            F.round(F.avg("equipment_age_years"), 1).alias("avg_equipment_age"),
            F.round(F.avg("supplier_quality_score"), 3).alias("avg_supplier_quality")
        )
    )
    
    # Window for month-over-month calculations
    window = Window.orderBy("transaction_year", "transaction_month")
    
    return (
        monthly
        # Previous month values for MoM calculation
        .withColumn("prev_month_cost", F.lag("total_scrap_cost").over(window))
        .withColumn("prev_month_qty", F.lag("total_quantity_scrapped").over(window))
        
        # MoM percentage changes
        .withColumn("cost_mom_change", 
            F.round((F.col("total_scrap_cost") - F.col("prev_month_cost")) / 
                    F.col("prev_month_cost") * 100, 2)
        )
        .withColumn("qty_mom_change",
            F.round((F.col("total_quantity_scrapped") - F.col("prev_month_qty")) / 
                    F.col("prev_month_qty") * 100, 2)
        )
        
        # Year-month string for easy filtering
        .withColumn("year_month", 
            F.concat_ws("-", F.col("transaction_year"), 
                        F.lpad(F.col("transaction_month"), 2, "0"))
        )
        .withColumn("processed_at", F.current_timestamp())
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # 🏭 Dimension-Based Aggregations
# MAGIC ---

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🏢 Scrap by Plant
# MAGIC 
# MAGIC **Purpose**: Plant-level performance metrics for comparison and benchmarking.
# MAGIC 
# MAGIC **Includes**: Shift breakdown, top defect category, month-over-month view.

# COMMAND ----------

@dlt.table(
    name="scrap_by_plant_gold",
    comment="Scrap aggregations by plant for performance benchmarking",
    table_properties={
        "quality": "gold",
        "pipelines.autoOptimize.managed": "true"
    }
)
def scrap_by_plant_gold():
    """
    Plant-level aggregations with shift breakdown.
    
    Used for: Plant performance dashboards, operations management, shift analysis.
    """
    return (
        dlt.read("scrap_transactions_silver")
        .groupBy(
            "plant_id",
            "plant_name_lookup",
            "plant_city",
            "plant_country",
            "transaction_year",
            "transaction_month"
        )
        .agg(
            F.count("*").alias("scrap_records"),
            F.sum("quantity_scrapped").alias("quantity_scrapped"),
            F.round(F.sum("total_scrap_cost"), 2).alias("total_scrap_cost"),
            F.countDistinct("defect_code").alias("unique_defects"),
            F.countDistinct("part_id").alias("unique_parts"),
            
            # Shift breakdown
            F.sum(F.when(F.col("shift") == 1, F.col("quantity_scrapped"))
                  .otherwise(0)).alias("shift_1_qty"),
            F.sum(F.when(F.col("shift") == 2, F.col("quantity_scrapped"))
                  .otherwise(0)).alias("shift_2_qty"),
            F.sum(F.when(F.col("shift") == 3, F.col("quantity_scrapped"))
                  .otherwise(0)).alias("shift_3_qty"),
            
            # Top defect
            F.mode("defect_category").alias("top_defect_category")
        )
        .withColumn("year_month", 
            F.concat_ws("-", F.col("transaction_year"), 
                        F.lpad(F.col("transaction_month"), 2, "0"))
        )
        .withColumn("processed_at", F.current_timestamp())
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🏭 Scrap by Supplier
# MAGIC 
# MAGIC **Purpose**: Supplier quality analysis - correlate supplier metrics with scrap.
# MAGIC 
# MAGIC **Key Insight**: Compare supplier_quality_score with actual scrap to validate ratings.

# COMMAND ----------

@dlt.table(
    name="scrap_by_supplier_gold",
    comment="Scrap aggregations by supplier for quality analysis",
    table_properties={
        "quality": "gold",
        "pipelines.autoOptimize.managed": "true"
    }
)
def scrap_by_supplier_gold():
    """
    Supplier-level aggregations with quality correlation.
    
    Used for: Supplier scorecards, procurement decisions, quality management.
    """
    return (
        dlt.read("scrap_transactions_silver")
        .groupBy(
            "supplier_id",
            "supplier_name_lookup",
            "supplier_tier_lookup",
            "supplier_quality_score",
            "transaction_year",
            "transaction_month"
        )
        .agg(
            F.count("*").alias("scrap_records"),
            F.sum("quantity_scrapped").alias("quantity_scrapped"),
            F.round(F.sum("total_scrap_cost"), 2).alias("total_scrap_cost"),
            F.countDistinct("defect_code").alias("unique_defects"),
            F.countDistinct("part_id").alias("parts_affected"),
            F.countDistinct("plant_id").alias("plants_affected"),
            
            # Defect type breakdown
            F.sum(F.when(F.col("defect_category") == "Material Defects", 
                         F.col("quantity_scrapped")).otherwise(0)).alias("material_defect_qty"),
            F.sum(F.when(F.col("defect_category") == "Dimensional", 
                         F.col("quantity_scrapped")).otherwise(0)).alias("dimensional_defect_qty"),
            
            F.mode("defect_category").alias("top_defect_category")
        )
        .withColumn("year_month", 
            F.concat_ws("-", F.col("transaction_year"), 
                        F.lpad(F.col("transaction_month"), 2, "0"))
        )
        # Quality-scrap correlation metric
        .withColumn("quality_scrap_correlation",
            F.round(F.col("quantity_scrapped") / (F.col("supplier_quality_score") + 0.01), 2)
        )
        .withColumn("processed_at", F.current_timestamp())
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## ⚠️ Scrap by Defect Code
# MAGIC 
# MAGIC **Purpose**: Pareto analysis - identify top defects contributing to scrap cost.
# MAGIC 
# MAGIC **Includes**: Root cause breakdown, Pareto ranking by month.

# COMMAND ----------

@dlt.table(
    name="scrap_by_defect_gold",
    comment="Scrap aggregations by defect code for Pareto analysis",
    table_properties={
        "quality": "gold",
        "pipelines.autoOptimize.managed": "true"
    }
)
def scrap_by_defect_gold():
    """
    Defect-level aggregations with Pareto ranking.
    
    Used for: Quality improvement prioritization, root cause analysis.
    """
    defect_agg = (
        dlt.read("scrap_transactions_silver")
        .groupBy(
            "defect_code",
            "defect_name_lookup",
            "defect_category",
            "defect_severity_weight",
            "transaction_year",
            "transaction_month"
        )
        .agg(
            F.count("*").alias("scrap_records"),
            F.sum("quantity_scrapped").alias("quantity_scrapped"),
            F.round(F.sum("total_scrap_cost"), 2).alias("total_scrap_cost"),
            F.countDistinct("part_id").alias("parts_affected"),
            F.countDistinct("plant_id").alias("plants_affected"),
            F.countDistinct("supplier_id").alias("suppliers_involved"),
            F.countDistinct("work_center_id").alias("work_centers_affected"),
            F.round(F.avg("equipment_age_years"), 1).alias("avg_equipment_age"),
            
            # Root cause breakdown
            F.sum(F.when(F.col("root_cause_category") == "Machine", 1)
                  .otherwise(0)).alias("machine_root_cause_count"),
            F.sum(F.when(F.col("root_cause_category") == "Material", 1)
                  .otherwise(0)).alias("material_root_cause_count"),
            F.sum(F.when(F.col("root_cause_category") == "Method", 1)
                  .otherwise(0)).alias("method_root_cause_count"),
            F.sum(F.when(F.col("root_cause_category") == "Man", 1)
                  .otherwise(0)).alias("man_root_cause_count")
        )
    )
    
    # Pareto ranking within each month
    window = Window.partitionBy("transaction_year", "transaction_month").orderBy(F.desc("total_scrap_cost"))
    
    return (
        defect_agg
        .withColumn("cost_rank", F.row_number().over(window))
        .withColumn("year_month", 
            F.concat_ws("-", F.col("transaction_year"), 
                        F.lpad(F.col("transaction_month"), 2, "0"))
        )
        .withColumn("processed_at", F.current_timestamp())
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🔧 Scrap by Equipment
# MAGIC 
# MAGIC **Purpose**: Equipment-related scrap analysis for maintenance prioritization.
# MAGIC 
# MAGIC **Includes**: Equipment age buckets, OEE correlation, shift patterns.

# COMMAND ----------

@dlt.table(
    name="scrap_by_equipment_gold",
    comment="Scrap aggregations by equipment for maintenance correlation",
    table_properties={
        "quality": "gold",
        "pipelines.autoOptimize.managed": "true"
    }
)
def scrap_by_equipment_gold():
    """
    Equipment-level aggregations with age and OEE correlation.
    
    Used for: Maintenance planning, equipment replacement decisions.
    """
    return (
        dlt.read("scrap_transactions_silver")
        .filter(F.col("equipment_id").isNotNull())
        .groupBy(
            "equipment_id",
            "equipment_name_lookup",
            "equipment_type_lookup",
            "equipment_age_years",
            "equipment_oee",
            "work_center_id",
            "work_center_type_lookup",
            "plant_id",
            "transaction_year",
            "transaction_month"
        )
        .agg(
            F.count("*").alias("scrap_records"),
            F.sum("quantity_scrapped").alias("quantity_scrapped"),
            F.round(F.sum("total_scrap_cost"), 2).alias("total_scrap_cost"),
            F.countDistinct("defect_code").alias("unique_defects"),
            F.countDistinct("part_id").alias("parts_affected"),
            
            # Shift pattern
            F.sum(F.when(F.col("shift") == 1, 1).otherwise(0)).alias("shift_1_records"),
            F.sum(F.when(F.col("shift") == 2, 1).otherwise(0)).alias("shift_2_records"),
            F.sum(F.when(F.col("shift") == 3, 1).otherwise(0)).alias("shift_3_records"),
            
            F.mode("defect_category").alias("top_defect_category")
        )
        # Age categorization
        .withColumn("equipment_age_bucket",
            F.when(F.col("equipment_age_years") <= 5, "New (0-5 yrs)")
            .when(F.col("equipment_age_years") <= 10, "Mid-Life (6-10 yrs)")
            .when(F.col("equipment_age_years") <= 15, "Aging (11-15 yrs)")
            .otherwise("Old (>15 yrs)")
        )
        # Scrap per OEE point (higher = worse equipment performance)
        .withColumn("scrap_per_oee_point", 
            F.round(F.col("quantity_scrapped") / (F.col("equipment_oee") * 100), 2)
        )
        .withColumn("year_month", 
            F.concat_ws("-", F.col("transaction_year"), 
                        F.lpad(F.col("transaction_month"), 2, "0"))
        )
        .withColumn("processed_at", F.current_timestamp())
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📋 8D Summary
# MAGIC 
# MAGIC **Purpose**: Aggregate 8D report statistics for quality management.

# COMMAND ----------

@dlt.table(
    name="eight_d_summary_gold",
    comment="8D report summary statistics by month and dimensions",
    table_properties={
        "quality": "gold",
        "pipelines.autoOptimize.managed": "true"
    }
)
def eight_d_summary_gold():
    """
    8D report aggregations for quality management dashboards.
    """
    return (
        dlt.read("eight_d_reports_silver")
        .groupBy(
            F.year(F.col("date_opened")).alias("year_opened"),
            F.month(F.col("date_opened")).alias("month_opened"),
            "plant_id",
            "plant_name_lookup",
            "defect_category",
            "severity",
            "status"
        )
        .agg(
            F.count("*").alias("num_8d_reports"),
            F.sum("total_quantity_affected").alias("total_quantity_affected"),
            F.round(F.sum("total_scrap_cost"), 2).alias("total_scrap_cost"),
            F.sum("num_parts_affected").alias("total_parts_affected"),
            F.round(F.avg("days_open"), 1).alias("avg_days_open"),
            F.sum(F.when(F.col("is_overdue") == True, 1).otherwise(0)).alias("overdue_count"),
            F.sum(F.when(F.col("is_customer_notified") == True, 1).otherwise(0)).alias("customer_notified_count"),
            F.round(F.avg("root_cause_word_count"), 0).alias("avg_root_cause_words"),
            F.mode("root_cause_category").alias("top_root_cause_category"),
            F.mode("verification_result").alias("most_common_verification_result")
        )
        .withColumn("year_month", 
            F.concat_ws("-", F.col("year_opened"), 
                        F.lpad(F.col("month_opened"), 2, "0"))
        )
        .withColumn("processed_at", F.current_timestamp())
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # 🤖 ML Feature Tables
# MAGIC ---

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🔮 Scrap Forecasting Features
# MAGIC 
# MAGIC **Purpose**: Feature table for time series forecasting models.
# MAGIC 
# MAGIC **Features Include**:
# MAGIC - Lag features (1-day, 7-day)
# MAGIC - Rolling averages (7, 14, 30 day)
# MAGIC - Rolling standard deviation (volatility)
# MAGIC - Seasonal indicators
# MAGIC - Trend indicators

# COMMAND ----------

@dlt.table(
    name="ml_scrap_forecast_features_gold",
    comment="Feature table for scrap quantity/cost forecasting ML models",
    table_properties={
        "quality": "gold",
        "pipelines.autoOptimize.managed": "true"
    }
)
def ml_scrap_forecast_features_gold():
    """
    Daily feature set for time series forecasting.
    
    Includes lag features, rolling averages, and seasonal indicators.
    Used by: Prophet, ARIMA, Neural Network forecasting models.
    """
    daily_base = (
        dlt.read("scrap_transactions_silver")
        .groupBy("transaction_date", "plant_id")
        .agg(
            F.sum("quantity_scrapped").alias("daily_qty"),
            F.round(F.sum("total_scrap_cost"), 2).alias("daily_cost"),
            F.count("*").alias("daily_records"),
            F.countDistinct("defect_code").alias("unique_defects"),
            F.countDistinct("part_id").alias("unique_parts"),
            F.round(F.avg("equipment_age_years"), 1).alias("avg_equipment_age"),
            F.round(F.avg("supplier_quality_score"), 3).alias("avg_supplier_quality"),
            F.sum(F.when(F.col("shift") == 3, F.col("quantity_scrapped"))
                  .otherwise(0)).alias("shift_3_qty"),
            F.sum(F.when(F.col("is_critical_part") == True, 1)
                  .otherwise(0)).alias("critical_part_records")
        )
    )
    
    # Window definitions for lag and rolling features
    window_7d = Window.partitionBy("plant_id").orderBy("transaction_date").rowsBetween(-7, -1)
    window_14d = Window.partitionBy("plant_id").orderBy("transaction_date").rowsBetween(-14, -1)
    window_30d = Window.partitionBy("plant_id").orderBy("transaction_date").rowsBetween(-30, -1)
    window_lag = Window.partitionBy("plant_id").orderBy("transaction_date")
    
    return (
        daily_base
        # Time features
        .withColumn("year", F.year("transaction_date"))
        .withColumn("month", F.month("transaction_date"))
        .withColumn("day_of_week", F.dayofweek("transaction_date"))
        .withColumn("day_of_month", F.dayofmonth("transaction_date"))
        .withColumn("week_of_year", F.weekofyear("transaction_date"))
        .withColumn("is_monday", (F.dayofweek("transaction_date") == 2).cast("int"))
        .withColumn("is_weekend", F.dayofweek("transaction_date").isin(1, 7).cast("int"))
        
        # Seasonal indicators
        .withColumn("is_summer", F.month("transaction_date").isin(6, 7, 8).cast("int"))
        .withColumn("is_year_end", F.month("transaction_date").isin(11, 12).cast("int"))
        .withColumn("is_q1", F.quarter("transaction_date") == 1)
        
        # Lag features
        .withColumn("qty_lag_1d", F.lag("daily_qty", 1).over(window_lag))
        .withColumn("qty_lag_7d", F.lag("daily_qty", 7).over(window_lag))
        .withColumn("cost_lag_1d", F.lag("daily_cost", 1).over(window_lag))
        .withColumn("cost_lag_7d", F.lag("daily_cost", 7).over(window_lag))
        
        # Rolling averages
        .withColumn("qty_rolling_7d_avg", F.round(F.avg("daily_qty").over(window_7d), 2))
        .withColumn("qty_rolling_14d_avg", F.round(F.avg("daily_qty").over(window_14d), 2))
        .withColumn("qty_rolling_30d_avg", F.round(F.avg("daily_qty").over(window_30d), 2))
        .withColumn("cost_rolling_7d_avg", F.round(F.avg("daily_cost").over(window_7d), 2))
        .withColumn("cost_rolling_30d_avg", F.round(F.avg("daily_cost").over(window_30d), 2))
        
        # Rolling std dev (volatility)
        .withColumn("qty_rolling_7d_std", F.round(F.stddev("daily_qty").over(window_7d), 2))
        .withColumn("cost_rolling_7d_std", F.round(F.stddev("daily_cost").over(window_7d), 2))
        
        # Rolling max (spikes)
        .withColumn("qty_rolling_7d_max", F.max("daily_qty").over(window_7d))
        .withColumn("cost_rolling_7d_max", F.max("daily_cost").over(window_7d))
        
        # Trend indicators
        .withColumn("qty_7d_trend", 
            F.round((F.col("daily_qty") - F.col("qty_rolling_7d_avg")) / 
                    (F.col("qty_rolling_7d_avg") + 1) * 100, 2)
        )
        
        # Night shift ratio
        .withColumn("shift_3_ratio", 
            F.round(F.col("shift_3_qty") / (F.col("daily_qty") + 1), 3)
        )
        
        .withColumn("processed_at", F.current_timestamp())
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📉 Scrap Reduction Policy Features
# MAGIC 
# MAGIC **Purpose**: Combined scrap and 8D features for policy recommendation ML.
# MAGIC 
# MAGIC **Use Case**: Identify which factors lead to scrap reduction success.

# COMMAND ----------

@dlt.table(
    name="ml_scrap_policy_features_gold",
    comment="Feature table for scrap reduction policy ML - combines scrap and 8D data",
    table_properties={
        "quality": "gold",
        "pipelines.autoOptimize.managed": "true"
    }
)
def ml_scrap_policy_features_gold():
    """
    Monthly aggregated features combining scrap patterns with 8D effectiveness.
    
    Used for: Identifying successful scrap reduction policies, root cause validation.
    """
    
    # Monthly scrap aggregations
    monthly_scrap = (
        dlt.read("scrap_transactions_silver")
        .groupBy(
            "plant_id",
            "supplier_id",
            "defect_category",
            "work_center_type_lookup",
            F.year("transaction_date").alias("year"),
            F.month("transaction_date").alias("month")
        )
        .agg(
            F.sum("quantity_scrapped").alias("total_qty_scrapped"),
            F.round(F.sum("total_scrap_cost"), 2).alias("total_scrap_cost"),
            F.count("*").alias("scrap_records"),
            F.countDistinct("part_id").alias("unique_parts"),
            F.countDistinct("defect_code").alias("unique_defects"),
            F.round(F.avg("equipment_age_years"), 1).alias("avg_equipment_age"),
            F.round(F.avg("equipment_oee"), 3).alias("avg_equipment_oee"),
            F.round(F.avg("supplier_quality_score"), 3).alias("avg_supplier_quality"),
            F.round(F.avg("operator_experience_years"), 1).alias("avg_operator_experience"),
            
            # Shift breakdown
            F.sum(F.when(F.col("shift") == 1, F.col("quantity_scrapped"))
                  .otherwise(0)).alias("shift_1_qty"),
            F.sum(F.when(F.col("shift") == 2, F.col("quantity_scrapped"))
                  .otherwise(0)).alias("shift_2_qty"),
            F.sum(F.when(F.col("shift") == 3, F.col("quantity_scrapped"))
                  .otherwise(0)).alias("shift_3_qty"),
            
            # Root cause breakdown
            F.sum(F.when(F.col("root_cause_category") == "Machine", 1)
                  .otherwise(0)).alias("machine_root_cause"),
            F.sum(F.when(F.col("root_cause_category") == "Material", 1)
                  .otherwise(0)).alias("material_root_cause"),
            F.sum(F.when(F.col("root_cause_category") == "Method", 1)
                  .otherwise(0)).alias("method_root_cause"),
            F.sum(F.when(F.col("root_cause_category") == "Man", 1)
                  .otherwise(0)).alias("man_root_cause"),
            
            # Disposition breakdown
            F.sum(F.when(F.col("disposition") == "Scrap", F.col("quantity_scrapped"))
                  .otherwise(0)).alias("final_scrap_qty"),
            F.sum(F.when(F.col("disposition") == "Rework", F.col("quantity_scrapped"))
                  .otherwise(0)).alias("rework_qty"),
            F.sum(F.when(F.col("disposition") == "Return to Supplier", F.col("quantity_scrapped"))
                  .otherwise(0)).alias("supplier_return_qty")
        )
    )
    
    # Monthly 8D effectiveness
    monthly_8d = (
        dlt.read("eight_d_reports_silver")
        .groupBy(
            "plant_id",
            "supplier_id", 
            "defect_category",
            F.year("date_opened").alias("year"),
            F.month("date_opened").alias("month")
        )
        .agg(
            F.count("*").alias("eight_d_count"),
            F.sum(F.when(F.col("status") == "Closed", 1).otherwise(0)).alias("eight_d_closed"),
            F.round(F.avg("days_open"), 1).alias("avg_8d_days_open"),
            F.round(F.avg("total_scrap_cost"), 2).alias("avg_8d_scrap_cost"),
            F.sum(F.when(F.col("verification_result") == "Effective", 1)
                  .otherwise(0)).alias("effective_8d_count"),
            F.sum(F.when(F.col("is_customer_notified") == True, 1)
                  .otherwise(0)).alias("customer_notified_8d")
        )
    )
    
    # Join and calculate derived features
    return (
        monthly_scrap
        .join(
            monthly_8d,
            ["plant_id", "supplier_id", "defect_category", "year", "month"],
            "left"
        )
        .fillna(0, subset=["eight_d_count", "eight_d_closed", "effective_8d_count"])
        
        # Derived ratios
        .withColumn("scrap_per_record", 
            F.round(F.col("total_qty_scrapped") / F.col("scrap_records"), 2)
        )
        .withColumn("cost_per_unit_scrapped",
            F.round(F.col("total_scrap_cost") / (F.col("total_qty_scrapped") + 1), 2)
        )
        .withColumn("shift_3_ratio",
            F.round(F.col("shift_3_qty") / (F.col("total_qty_scrapped") + 1), 3)
        )
        .withColumn("rework_rate",
            F.round(F.col("rework_qty") / (F.col("total_qty_scrapped") + 1), 3)
        )
        .withColumn("supplier_return_rate",
            F.round(F.col("supplier_return_qty") / (F.col("total_qty_scrapped") + 1), 3)
        )
        
        # 8D effectiveness metrics
        .withColumn("eight_d_closure_rate",
            F.when(F.col("eight_d_count") > 0,
                F.round(F.col("eight_d_closed") / F.col("eight_d_count"), 3)
            ).otherwise(0)
        )
        .withColumn("eight_d_effectiveness_rate",
            F.when(F.col("eight_d_closed") > 0,
                F.round(F.col("effective_8d_count") / F.col("eight_d_closed"), 3)
            ).otherwise(0)
        )
        
        # Equipment risk score
        .withColumn("equipment_risk_score",
            F.round(
                (F.col("avg_equipment_age") / 20) * 0.4 + 
                (1 - F.col("avg_equipment_oee")) * 0.6,
                3
            )
        )
        
        # Supplier risk score
        .withColumn("supplier_risk_score",
            F.round(1 - F.col("avg_supplier_quality"), 3)
        )
        
        .withColumn("year_month", 
            F.concat_ws("-", F.col("year"), F.lpad(F.col("month"), 2, "0"))
        )
        .withColumn("processed_at", F.current_timestamp())
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📝 8D Text Analysis Features
# MAGIC 
# MAGIC **Purpose**: Prepare 8D text fields for LLM analysis and summarization.
# MAGIC 
# MAGIC **Includes**: Combined text field, word counts, context metadata.

# COMMAND ----------

@dlt.table(
    name="eight_d_text_analysis_gold",
    comment="8D reports prepared for LLM text analysis and summarization",
    table_properties={
        "quality": "gold",
        "pipelines.autoOptimize.managed": "true"
    }
)
def eight_d_text_analysis_gold():
    """
    Flattened 8D data optimized for LLM processing.
    
    Combines root cause and corrective action text with contextual metadata.
    Used for: RAG systems, text summarization, similarity search.
    """
    return (
        dlt.read("eight_d_reports_silver")
        .select(
            "eight_d_id",
            "report_number",
            "title",
            "status",
            "severity",
            "defect_category",
            "primary_defect_code",
            
            # Context for LLM
            "plant_name_lookup",
            "supplier_name_lookup",
            "num_parts_affected",
            "total_quantity_affected",
            "total_scrap_cost",
            
            # Text fields for analysis
            "problem_description",
            "root_cause_analysis",
            "root_cause_category",
            "containment_action",
            "corrective_actions",
            "preventive_actions",
            "lessons_learned",
            "verification_result",
            
            # Metadata
            "date_opened",
            "date_closed",
            "days_open",
            "is_overdue",
            "is_customer_notified",
            "customer_complaint_number",
            
            # Text metrics
            "root_cause_word_count",
            "corrective_action_word_count"
        )
        # Combined text for embedding/similarity
        .withColumn("combined_analysis_text",
            F.concat_ws("\n\n",
                F.concat(F.lit("PROBLEM: "), F.col("problem_description")),
                F.concat(F.lit("ROOT CAUSE: "), F.col("root_cause_analysis")),
                F.concat(F.lit("CORRECTIVE ACTIONS: "), F.col("corrective_actions")),
                F.concat(F.lit("PREVENTIVE ACTIONS: "), F.col("preventive_actions")),
                F.concat(F.lit("LESSONS LEARNED: "), F.col("lessons_learned"))
            )
        )
        .withColumn("total_word_count",
            F.col("root_cause_word_count") + F.col("corrective_action_word_count")
        )
        .withColumn("processed_at", F.current_timestamp())
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # 📈 Dashboard Tables
# MAGIC ---

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🏅 Supplier Scorecard
# MAGIC 
# MAGIC **Purpose**: Monthly supplier quality scorecard combining scrap and 8D metrics.
# MAGIC 
# MAGIC **Scores Include**:
# MAGIC - Base quality score (from supplier master)
# MAGIC - Scrap score (inverse of scrap volume)
# MAGIC - Quality incident score (penalty for material defects)
# MAGIC - Responsiveness score (8D response time)
# MAGIC - Overall weighted score and performance tier (A-F)

# COMMAND ----------

@dlt.table(
    name="supplier_scorecard_gold",
    comment="Supplier quality scorecard combining scrap and 8D data",
    table_properties={
        "quality": "gold",
        "pipelines.autoOptimize.managed": "true"
    }
)
def supplier_scorecard_gold():
    """
    Monthly supplier scorecard for vendor management.
    
    Combines scrap performance with 8D responsiveness for overall scoring.
    """
    
    # Supplier scrap summary
    supplier_scrap = (
        dlt.read("scrap_transactions_silver")
        .groupBy(
            "supplier_id",
            "supplier_name_lookup",
            "supplier_tier_lookup",
            "supplier_quality_score",
            F.year("transaction_date").alias("year"),
            F.month("transaction_date").alias("month")
        )
        .agg(
            F.sum("quantity_scrapped").alias("total_qty_scrapped"),
            F.round(F.sum("total_scrap_cost"), 2).alias("total_scrap_cost"),
            F.count("*").alias("scrap_incidents"),
            F.countDistinct("part_id").alias("parts_affected"),
            F.countDistinct("plant_id").alias("plants_affected"),
            F.countDistinct("defect_code").alias("unique_defects"),
            F.sum(F.when(F.col("defect_category") == "Material Defects", 1)
                  .otherwise(0)).alias("material_defects"),
            F.sum(F.when(F.col("disposition") == "Return to Supplier", F.col("quantity_scrapped"))
                  .otherwise(0)).alias("supplier_returns")
        )
    )
    
    # Supplier 8D summary
    supplier_8d = (
        dlt.read("eight_d_reports_silver")
        .groupBy(
            "supplier_id",
            F.year("date_opened").alias("year"),
            F.month("date_opened").alias("month")
        )
        .agg(
            F.count("*").alias("eight_d_issued"),
            F.sum(F.when(F.col("severity").isin("Critical", "Major"), 1)
                  .otherwise(0)).alias("critical_8d"),
            F.round(F.avg("days_open"), 1).alias("avg_8d_response_time")
        )
    )
    
    return (
        supplier_scrap
        .join(supplier_8d, ["supplier_id", "year", "month"], "left")
        .fillna(0, subset=["eight_d_issued", "critical_8d"])
        
        # Calculate component scores (0-100 scale)
        .withColumn("scrap_score",
            F.greatest(F.lit(0), F.round(100 - (F.col("total_qty_scrapped") / 10), 1))
        )
        .withColumn("quality_incident_score",
            F.greatest(F.lit(0), F.round(100 - (F.col("material_defects") * 5), 1))
        )
        .withColumn("responsiveness_score",
            F.when(F.col("eight_d_issued") > 0,
                F.greatest(F.lit(0), F.round(100 - F.col("avg_8d_response_time"), 1))
            ).otherwise(100)
        )
        
        # Overall weighted score
        .withColumn("overall_score",
            F.round(
                F.col("supplier_quality_score") * 100 * 0.4 +
                F.col("scrap_score") * 0.3 +
                F.col("quality_incident_score") * 0.2 +
                F.col("responsiveness_score") * 0.1,
                1
            )
        )
        
        # Performance tier
        .withColumn("performance_tier",
            F.when(F.col("overall_score") >= 90, "A - Excellent")
            .when(F.col("overall_score") >= 80, "B - Good")
            .when(F.col("overall_score") >= 70, "C - Acceptable")
            .when(F.col("overall_score") >= 60, "D - Needs Improvement")
            .otherwise("F - Critical")
        )
        
        .withColumn("year_month", 
            F.concat_ws("-", F.col("year"), F.lpad(F.col("month"), 2, "0"))
        )
        .withColumn("processed_at", F.current_timestamp())
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📊 KPI Dashboard
# MAGIC 
# MAGIC **Purpose**: Executive-level KPIs aggregated monthly for high-level dashboards.
# MAGIC 
# MAGIC **KPIs Include**:
# MAGIC - Total scrap cost and quantity
# MAGIC - 8D open/closed/effectiveness rates
# MAGIC - Critical incident ratios

# COMMAND ----------

@dlt.table(
    name="kpi_dashboard_gold",
    comment="Executive KPI metrics for dashboards",
    table_properties={
        "quality": "gold",
        "pipelines.autoOptimize.managed": "true"
    }
)
def kpi_dashboard_gold():
    """
    High-level KPIs aggregated monthly for executive dashboards.
    """
    
    scrap_kpis = (
        dlt.read("scrap_transactions_silver")
        .groupBy(
            F.year("transaction_date").alias("year"),
            F.month("transaction_date").alias("month")
        )
        .agg(
            F.sum("quantity_scrapped").alias("total_qty_scrapped"),
            F.round(F.sum("total_scrap_cost"), 2).alias("total_scrap_cost"),
            F.count("*").alias("total_scrap_records"),
            F.countDistinct("part_id").alias("unique_parts_scrapped"),
            F.countDistinct("supplier_id").alias("suppliers_with_scrap"),
            F.countDistinct("plant_id").alias("plants_with_scrap"),
            
            # Critical indicators
            F.sum(F.when(F.col("is_critical_part") == True, F.col("quantity_scrapped"))
                  .otherwise(0)).alias("critical_parts_scrapped"),
            F.sum(F.when(F.col("cost_severity") == "Critical", F.col("total_scrap_cost"))
                  .otherwise(0)).alias("critical_cost_incidents_value"),
            
            # Averages
            F.round(F.avg("quantity_scrapped"), 2).alias("avg_qty_per_incident"),
            F.round(F.avg("total_scrap_cost"), 2).alias("avg_cost_per_incident"),
            F.round(F.avg("equipment_age_years"), 1).alias("avg_equipment_age"),
            F.round(F.avg("supplier_quality_score"), 3).alias("avg_supplier_quality")
        )
    )
    
    eight_d_kpis = (
        dlt.read("eight_d_reports_silver")
        .groupBy(
            F.year("date_opened").alias("year"),
            F.month("date_opened").alias("month")
        )
        .agg(
            F.count("*").alias("total_8d_opened"),
            F.sum(F.when(F.col("status") == "Closed", 1).otherwise(0)).alias("total_8d_closed"),
            F.sum(F.when(F.col("severity") == "Critical", 1).otherwise(0)).alias("critical_8d"),
            F.sum(F.when(F.col("is_overdue") == True, 1).otherwise(0)).alias("overdue_8d"),
            F.round(F.avg("days_open"), 1).alias("avg_days_to_close"),
            F.sum(F.when(F.col("verification_result") == "Effective", 1)
                  .otherwise(0)).alias("effective_closures")
        )
    )
    
    return (
        scrap_kpis
        .join(eight_d_kpis, ["year", "month"], "left")
        .fillna(0)
        
        # Derived KPIs
        .withColumn("scrap_cost_per_record",
            F.round(F.col("total_scrap_cost") / F.col("total_scrap_records"), 2)
        )
        .withColumn("eight_d_closure_rate",
            F.when(F.col("total_8d_opened") > 0,
                F.round(F.col("total_8d_closed") / F.col("total_8d_opened") * 100, 1)
            ).otherwise(0)
        )
        .withColumn("eight_d_effectiveness_rate",
            F.when(F.col("total_8d_closed") > 0,
                F.round(F.col("effective_closures") / F.col("total_8d_closed") * 100, 1)
            ).otherwise(0)
        )
        .withColumn("critical_scrap_ratio",
            F.round(F.col("critical_parts_scrapped") / F.col("total_qty_scrapped") * 100, 2)
        )
        
        .withColumn("year_month", 
            F.concat_ws("-", F.col("year"), F.lpad(F.col("month"), 2, "0"))
        )
        .withColumn("processed_at", F.current_timestamp())
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # 🎯 Scenario Analysis Tables
# MAGIC ---
# MAGIC 
# MAGIC Tables designed to support What-If scenario analysis in the Streamlit app.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🔧 Scrap by Work Center
# MAGIC 
# MAGIC **Purpose**: Work center level aggregations for scenario planning.
# MAGIC 
# MAGIC **Includes**: OEE correlation, equipment age, shift breakdown, plant context.

# COMMAND ----------

@dlt.table(
    name="scrap_by_work_center_gold",
    comment="Work center level scrap aggregations for scenario analysis",
    table_properties={
        "quality": "gold",
        "pipelines.autoOptimize.managed": "true"
    }
)
def scrap_by_work_center_gold():
    """
    Work center aggregations for what-if scenario analysis.
    
    Includes equipment OEE and age metrics for maintenance correlation.
    """
    return (
        dlt.read("scrap_transactions_silver")
        .filter(F.col("work_center_id").isNotNull())
        .groupBy(
            "work_center_id",
            "work_center_name_lookup",
            "work_center_type_lookup",
            "plant_id",
            "plant_name_lookup",
            "transaction_year",
            "transaction_month"
        )
        .agg(
            # Volume metrics
            F.count("*").alias("scrap_records"),
            F.sum("quantity_scrapped").alias("quantity_scrapped"),
            F.round(F.sum("total_scrap_cost"), 2).alias("total_scrap_cost"),
            F.countDistinct("defect_code").alias("unique_defects"),
            F.countDistinct("part_id").alias("unique_parts"),
            F.countDistinct("equipment_id").alias("unique_equipment"),
            
            # Equipment metrics
            F.round(F.avg("equipment_age_years"), 1).alias("avg_equipment_age"),
            F.round(F.avg("equipment_oee"), 3).alias("avg_equipment_oee"),
            F.round(F.min("equipment_oee"), 3).alias("min_equipment_oee"),
            
            # Shift breakdown
            F.sum(F.when(F.col("shift") == 1, F.col("quantity_scrapped"))
                  .otherwise(0)).alias("shift_1_qty"),
            F.sum(F.when(F.col("shift") == 2, F.col("quantity_scrapped"))
                  .otherwise(0)).alias("shift_2_qty"),
            F.sum(F.when(F.col("shift") == 3, F.col("quantity_scrapped"))
                  .otherwise(0)).alias("shift_3_qty"),
            
            # Top defect category
            F.mode("defect_category").alias("top_defect_category"),
            F.mode("root_cause_category").alias("top_root_cause")
        )
        .withColumn("year_month", 
            F.concat_ws("-", F.col("transaction_year"), 
                        F.lpad(F.col("transaction_month"), 2, "0"))
        )
        # OEE-scrap correlation (higher = worse, meaning low OEE causing more scrap)
        .withColumn("oee_scrap_correlation",
            F.round(F.col("quantity_scrapped") * (1 - F.col("avg_equipment_oee")), 2)
        )
        .withColumn("processed_at", F.current_timestamp())
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🏢 Supplier Scenario Data
# MAGIC 
# MAGIC **Purpose**: Comprehensive supplier data for what-if spend/allocation scenarios.
# MAGIC 
# MAGIC **Includes**: Annual spend proxy, parts supplied, quality metrics, scrap rates,
# MAGIC capacity indicators, and alternative supplier identification.

# COMMAND ----------

@dlt.table(
    name="supplier_scenario_gold",
    comment="Comprehensive supplier data for what-if scenario analysis",
    table_properties={
        "quality": "gold",
        "pipelines.autoOptimize.managed": "true"
    }
)
def supplier_scenario_gold():
    """
    Supplier-level data designed for what-if scenarios.
    
    Derives all metrics from scrap_transactions_silver to avoid cross-pipeline dependencies.
    Estimates annual spend based on scrap volume and unit costs.
    """
    
    # Get supplier data with scrap totals from transactions (latest 12 months)
    # This avoids dependency on suppliers_silver table from different pipeline
    supplier_data = (
        dlt.read("scrap_transactions_silver")
        .filter(F.col("transaction_date") >= F.add_months(F.current_date(), -12))
        .groupBy(
            "supplier_id",
            "supplier_name_lookup",
            "supplier_tier_lookup",
            "supplier_quality_score"
        )
        .agg(
            F.sum("quantity_scrapped").alias("annual_qty_scrapped"),
            F.round(F.sum("total_scrap_cost"), 2).alias("annual_scrap_cost"),
            F.count("*").alias("annual_scrap_records"),
            F.countDistinct("part_id").alias("parts_supplied"),
            F.countDistinct("plant_id").alias("plants_served"),
            F.countDistinct("defect_code").alias("unique_defects"),
            F.round(F.avg("unit_cost"), 2).alias("avg_unit_cost"),
            
            # Sum by disposition
            F.sum(F.when(F.col("disposition") == "Return to Supplier", F.col("quantity_scrapped"))
                  .otherwise(0)).alias("returns_to_supplier"),
                  
            # Material defects (supplier quality issue)
            F.sum(F.when(F.col("defect_category") == "Material Defects", 1)
                  .otherwise(0)).alias("material_defect_count"),
                  
            # Volume processed (estimate from scrap rate)
            # Assuming 3% average scrap rate, estimate total volume
            F.round(F.sum("quantity_scrapped") / 0.03, 0).alias("estimated_annual_volume")
        )
    )
    
    # Rename columns and calculate metrics
    result = (
        supplier_data
        .withColumnRenamed("supplier_name_lookup", "supplier_name")
        .withColumnRenamed("supplier_tier_lookup", "supplier_tier")
        .withColumnRenamed("supplier_quality_score", "quality_score")
        
        # Default specialization based on tier (will be updated if suppliers_bronze is available)
        .withColumn("specialization", F.lit("General Manufacturing"))
        .withColumn("region", F.lit("North America"))
        .withColumn("on_time_delivery_rate", F.col("quality_score") * F.lit(0.98))  # Estimate
        .withColumn("is_preferred", F.col("quality_score") >= 0.90)
        
        # Estimate annual spend (volume * avg unit cost)
        .withColumn("estimated_annual_spend",
            F.round(F.col("estimated_annual_volume") * F.coalesce(F.col("avg_unit_cost"), F.lit(50)), 2)
        )
        
        # Scrap rate
        .withColumn("scrap_rate",
            F.when(F.col("estimated_annual_volume") > 0,
                F.round(F.col("annual_qty_scrapped") / F.col("estimated_annual_volume"), 4)
            ).otherwise(0.03)
        )
        
        # Capacity utilization (estimate based on volume vs. parts)
        .withColumn("capacity_utilization",
            F.when(F.col("parts_supplied") > 0,
                F.least(F.lit(0.95), F.round(0.5 + (F.col("parts_supplied") / 50.0), 2))
            ).otherwise(0.70)
        )
        
        # Available capacity
        .withColumn("available_capacity",
            F.round(1 - F.col("capacity_utilization"), 2)
        )
        
        # Risk score (0-100, lower = better)
        .withColumn("risk_score",
            F.round(
                (1 - F.col("quality_score")) * 50 +  # Quality factor
                (1 - F.col("on_time_delivery_rate")) * 30 +  # Delivery factor
                F.when(F.col("supplier_tier") == "Tier 1", 0)
                 .when(F.col("supplier_tier") == "Tier 2", 10)
                 .otherwise(20) +  # Tier factor
                F.when(F.col("material_defect_count") > 10, 15)
                 .when(F.col("material_defect_count") > 5, 8)
                 .otherwise(0),  # Material defect factor
                1
            )
        )
        
        # Cost index (1.0 = baseline, < 1.0 = cheaper)
        .withColumn("unit_cost_index",
            F.when(F.col("supplier_tier") == "Tier 1", 1.0)
             .when(F.col("supplier_tier") == "Tier 2", 
                   F.round(0.85 + F.rand() * 0.2, 2))
             .otherwise(F.round(0.75 + F.rand() * 0.3, 2))
        )
        
        .withColumn("processed_at", F.current_timestamp())
    )
    
    return result

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📦 Part Scrap Analysis
# MAGIC 
# MAGIC **Purpose**: Part-level scrap analysis for targeted improvement scenarios.
# MAGIC 
# MAGIC **Includes**: Monthly scrap cost, defect rate, supplier info, critical part flag.

# COMMAND ----------

@dlt.table(
    name="part_scrap_analysis_gold",
    comment="Part-level scrap analysis for scenario planning",
    table_properties={
        "quality": "gold",
        "pipelines.autoOptimize.managed": "true"
    }
)
def part_scrap_analysis_gold():
    """
    Part-level scrap aggregations for targeted improvement scenarios.
    
    Includes defect rates, supplier correlation, and cost impact.
    """
    return (
        dlt.read("scrap_transactions_silver")
        .groupBy(
            "part_id",
            "part_name_lookup",
            "part_category_lookup",
            "supplier_id",
            "supplier_name_lookup",
            "is_critical_part",
            "transaction_year",
            "transaction_month"
        )
        .agg(
            # Volume metrics
            F.count("*").alias("scrap_records"),
            F.sum("quantity_scrapped").alias("quantity_scrapped"),
            F.round(F.sum("total_scrap_cost"), 2).alias("total_scrap_cost"),
            F.round(F.avg("unit_cost"), 2).alias("avg_unit_cost"),
            
            # Quality metrics
            F.countDistinct("defect_code").alias("unique_defects"),
            F.mode("defect_category").alias("top_defect_category"),
            F.mode("root_cause_category").alias("top_root_cause"),
            
            # Plant spread
            F.countDistinct("plant_id").alias("plants_affected"),
            
            # Equipment correlation
            F.round(F.avg("equipment_age_years"), 1).alias("avg_equipment_age"),
            F.round(F.avg("equipment_oee"), 3).alias("avg_equipment_oee"),
            
            # Shift patterns
            F.sum(F.when(F.col("shift") == 3, F.col("quantity_scrapped"))
                  .otherwise(0)).alias("shift_3_qty"),
            
            # Disposition
            F.sum(F.when(F.col("disposition") == "Rework", F.col("quantity_scrapped"))
                  .otherwise(0)).alias("rework_qty")
        )
        .withColumn("year_month", 
            F.concat_ws("-", F.col("transaction_year"), 
                        F.lpad(F.col("transaction_month"), 2, "0"))
        )
        # Defect rate (scrap records per unit scrapped, inverted estimate)
        .withColumn("estimated_defect_rate",
            F.when(F.col("quantity_scrapped") > 0,
                F.least(F.lit(0.15), F.round(F.col("scrap_records") / (F.col("quantity_scrapped") / 10 + 100), 4))
            ).otherwise(0.02)
        )
        # Rework potential
        .withColumn("rework_rate",
            F.when(F.col("quantity_scrapped") > 0,
                F.round(F.col("rework_qty") / F.col("quantity_scrapped"), 3)
            ).otherwise(0)
        )
        .withColumn("processed_at", F.current_timestamp())
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🏭 Plant Monthly Summary
# MAGIC 
# MAGIC **Purpose**: Monthly plant scrap for scenario time series.

# COMMAND ----------

@dlt.table(
    name="plant_monthly_scenario_gold",
    comment="Plant monthly scrap for scenario time series analysis",
    table_properties={
        "quality": "gold",
        "pipelines.autoOptimize.managed": "true"
    }
)
def plant_monthly_scenario_gold():
    """
    Monthly plant-level data optimized for scenario projections.
    """
    return (
        dlt.read("scrap_transactions_silver")
        .groupBy(
            "plant_id",
            "plant_name_lookup",
            "plant_city",
            "plant_country",
            F.year("transaction_date").alias("year"),
            F.month("transaction_date").alias("month")
        )
        .agg(
            F.count("*").alias("scrap_records"),
            F.sum("quantity_scrapped").alias("quantity_scrapped"),
            F.round(F.sum("total_scrap_cost"), 2).alias("scrap_cost"),
            F.countDistinct("part_id").alias("unique_parts"),
            F.countDistinct("supplier_id").alias("unique_suppliers"),
            F.countDistinct("defect_code").alias("unique_defects"),
            F.round(F.avg("equipment_oee"), 3).alias("avg_oee"),
            F.round(F.avg("equipment_age_years"), 1).alias("avg_equipment_age"),
            
            # Shift breakdown as ratios
            F.round(F.sum(F.when(F.col("shift") == 1, F.col("quantity_scrapped")).otherwise(0)) / 
                    F.sum("quantity_scrapped"), 3).alias("shift_1_ratio"),
            F.round(F.sum(F.when(F.col("shift") == 2, F.col("quantity_scrapped")).otherwise(0)) / 
                    F.sum("quantity_scrapped"), 3).alias("shift_2_ratio"),
            F.round(F.sum(F.when(F.col("shift") == 3, F.col("quantity_scrapped")).otherwise(0)) / 
                    F.sum("quantity_scrapped"), 3).alias("shift_3_ratio")
        )
        .withColumn("month_date", 
            F.to_date(F.concat_ws("-", F.col("year"), F.lpad(F.col("month"), 2, "0"), F.lit("01")))
        )
        .withColumn("year_month", 
            F.concat_ws("-", F.col("year"), F.lpad(F.col("month"), 2, "0"))
        )
        .withColumn("processed_at", F.current_timestamp())
    )
