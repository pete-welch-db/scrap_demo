# Databricks notebook source
# MAGIC %md
# MAGIC # DLT Pipeline - Gold Layer Aggregations
# MAGIC 
# MAGIC This notebook defines Delta Live Tables (DLT) pipeline for Gold layer aggregations.
# MAGIC 
# MAGIC **Aggregations:**
# MAGIC - Scrap summary by time period
# MAGIC - Scrap by plant/supplier/defect
# MAGIC - ML feature tables for forecasting
# MAGIC - 8D analytics
# MAGIC 
# MAGIC **Source:** `welch.scrap_demo.*_silver`  
# MAGIC **Target:** `welch.scrap_demo.*_gold`

# COMMAND ----------

import dlt
from pyspark.sql import functions as F
from pyspark.sql.window import Window

# Configuration
CATALOG = "welch"
SCHEMA = "scrap_demo"
SOURCE_SCHEMA = f"{CATALOG}.{SCHEMA}"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Daily Scrap Summary Gold

# COMMAND ----------

@dlt.table(
    name="scrap_daily_summary_gold",
    comment="Daily scrap aggregations for dashboards and reporting",
    table_properties={
        "quality": "gold",
        "pipelines.autoOptimize.managed": "true"
    }
)
def scrap_daily_summary_gold():
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
            F.count("*").alias("total_scrap_records"),
            F.sum("quantity_scrapped").alias("total_quantity_scrapped"),
            F.round(F.sum("total_scrap_cost"), 2).alias("total_scrap_cost"),
            F.round(F.avg("quantity_scrapped"), 2).alias("avg_quantity_per_record"),
            F.round(F.avg("total_scrap_cost"), 2).alias("avg_cost_per_record"),
            F.countDistinct("part_id").alias("unique_parts_scrapped"),
            F.countDistinct("defect_code").alias("unique_defect_codes"),
            F.countDistinct("plant_id").alias("plants_affected"),
            F.countDistinct("supplier_id").alias("suppliers_involved"),
            F.sum(F.when(F.col("is_critical_part") == True, F.col("quantity_scrapped")).otherwise(0)).alias("critical_parts_scrapped"),
            F.sum(F.when(F.col("cost_severity") == "Critical", 1).otherwise(0)).alias("critical_cost_incidents")
        )
        .withColumn("processed_at", F.current_timestamp())
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Monthly Scrap Summary Gold

# COMMAND ----------

@dlt.table(
    name="scrap_monthly_summary_gold",
    comment="Monthly scrap aggregations for trend analysis",
    table_properties={
        "quality": "gold",
        "pipelines.autoOptimize.managed": "true"
    }
)
def scrap_monthly_summary_gold():
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
    
    # Add month-over-month calculations
    window = Window.orderBy("transaction_year", "transaction_month")
    
    return (
        monthly
        .withColumn("prev_month_cost", F.lag("total_scrap_cost").over(window))
        .withColumn("prev_month_qty", F.lag("total_quantity_scrapped").over(window))
        .withColumn("cost_mom_change", 
            F.round((F.col("total_scrap_cost") - F.col("prev_month_cost")) / F.col("prev_month_cost") * 100, 2)
        )
        .withColumn("qty_mom_change",
            F.round((F.col("total_quantity_scrapped") - F.col("prev_month_qty")) / F.col("prev_month_qty") * 100, 2)
        )
        .withColumn("year_month", 
            F.concat_ws("-", F.col("transaction_year"), F.lpad(F.col("transaction_month"), 2, "0"))
        )
        .withColumn("processed_at", F.current_timestamp())
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Scrap by Plant Gold

# COMMAND ----------

@dlt.table(
    name="scrap_by_plant_gold",
    comment="Scrap aggregations by plant for performance comparison",
    table_properties={
        "quality": "gold",
        "pipelines.autoOptimize.managed": "true"
    }
)
def scrap_by_plant_gold():
    total_cost = dlt.read("scrap_transactions_silver").agg(F.sum("total_scrap_cost")).collect()[0][0]
    
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
            
            # Shift analysis
            F.sum(F.when(F.col("shift") == 1, F.col("quantity_scrapped")).otherwise(0)).alias("shift_1_qty"),
            F.sum(F.when(F.col("shift") == 2, F.col("quantity_scrapped")).otherwise(0)).alias("shift_2_qty"),
            F.sum(F.when(F.col("shift") == 3, F.col("quantity_scrapped")).otherwise(0)).alias("shift_3_qty"),
            
            # Top defect category
            F.mode("defect_category").alias("top_defect_category")
        )
        .withColumn("year_month", 
            F.concat_ws("-", F.col("transaction_year"), F.lpad(F.col("transaction_month"), 2, "0"))
        )
        .withColumn("processed_at", F.current_timestamp())
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Scrap by Supplier Gold

# COMMAND ----------

@dlt.table(
    name="scrap_by_supplier_gold",
    comment="Scrap aggregations by supplier for supplier quality analysis",
    table_properties={
        "quality": "gold",
        "pipelines.autoOptimize.managed": "true"
    }
)
def scrap_by_supplier_gold():
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
            
            # Defect breakdown
            F.sum(F.when(F.col("defect_category") == "Material Defects", F.col("quantity_scrapped")).otherwise(0)).alias("material_defect_qty"),
            F.sum(F.when(F.col("defect_category") == "Dimensional", F.col("quantity_scrapped")).otherwise(0)).alias("dimensional_defect_qty"),
            
            F.mode("defect_category").alias("top_defect_category")
        )
        .withColumn("year_month", 
            F.concat_ws("-", F.col("transaction_year"), F.lpad(F.col("transaction_month"), 2, "0"))
        )
        .withColumn("quality_scrap_correlation",
            F.round(F.col("quantity_scrapped") / (F.col("supplier_quality_score") + 0.01), 2)
        )
        .withColumn("processed_at", F.current_timestamp())
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Scrap by Defect Code Gold

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
            F.sum(F.when(F.col("root_cause_category") == "Machine", 1).otherwise(0)).alias("machine_root_cause_count"),
            F.sum(F.when(F.col("root_cause_category") == "Material", 1).otherwise(0)).alias("material_root_cause_count"),
            F.sum(F.when(F.col("root_cause_category") == "Method", 1).otherwise(0)).alias("method_root_cause_count"),
            F.sum(F.when(F.col("root_cause_category") == "Man", 1).otherwise(0)).alias("man_root_cause_count")
        )
    )
    
    # Calculate Pareto ranking within each month
    window = Window.partitionBy("transaction_year", "transaction_month").orderBy(F.desc("total_scrap_cost"))
    
    return (
        defect_agg
        .withColumn("cost_rank", F.row_number().over(window))
        .withColumn("year_month", 
            F.concat_ws("-", F.col("transaction_year"), F.lpad(F.col("transaction_month"), 2, "0"))
        )
        .withColumn("processed_at", F.current_timestamp())
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Scrap by Equipment Gold

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
        .withColumn("equipment_age_bucket",
            F.when(F.col("equipment_age_years") <= 5, "New (0-5 yrs)")
            .when(F.col("equipment_age_years") <= 10, "Mid-Life (6-10 yrs)")
            .when(F.col("equipment_age_years") <= 15, "Aging (11-15 yrs)")
            .otherwise("Old (>15 yrs)")
        )
        .withColumn("scrap_per_oee_point", 
            F.round(F.col("quantity_scrapped") / (F.col("equipment_oee") * 100), 2)
        )
        .withColumn("year_month", 
            F.concat_ws("-", F.col("transaction_year"), F.lpad(F.col("transaction_month"), 2, "0"))
        )
        .withColumn("processed_at", F.current_timestamp())
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8D Summary Gold

# COMMAND ----------

@dlt.table(
    name="eight_d_summary_gold",
    comment="8D report summary statistics",
    table_properties={
        "quality": "gold",
        "pipelines.autoOptimize.managed": "true"
    }
)
def eight_d_summary_gold():
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
            F.concat_ws("-", F.col("year_opened"), F.lpad(F.col("month_opened"), 2, "0"))
        )
        .withColumn("processed_at", F.current_timestamp())
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## ML Feature Table - Scrap Forecasting

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
            F.sum(F.when(F.col("shift") == 3, F.col("quantity_scrapped")).otherwise(0)).alias("shift_3_qty"),
            F.sum(F.when(F.col("is_critical_part") == True, 1).otherwise(0)).alias("critical_part_records")
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
            F.round((F.col("daily_qty") - F.col("qty_rolling_7d_avg")) / (F.col("qty_rolling_7d_avg") + 1) * 100, 2)
        )
        
        # Night shift ratio
        .withColumn("shift_3_ratio", 
            F.round(F.col("shift_3_qty") / (F.col("daily_qty") + 1), 3)
        )
        
        .withColumn("processed_at", F.current_timestamp())
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## ML Feature Table - Scrap Reduction Policy

# COMMAND ----------

@dlt.table(
    name="ml_scrap_policy_features_gold",
    comment="Feature table for scrap reduction policy ML models - combines scrap and 8D data",
    table_properties={
        "quality": "gold",
        "pipelines.autoOptimize.managed": "true"
    }
)
def ml_scrap_policy_features_gold():
    """
    Monthly aggregated features combining scrap patterns with 8D effectiveness.
    Used for identifying scrap reduction policies.
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
            F.sum(F.when(F.col("shift") == 1, F.col("quantity_scrapped")).otherwise(0)).alias("shift_1_qty"),
            F.sum(F.when(F.col("shift") == 2, F.col("quantity_scrapped")).otherwise(0)).alias("shift_2_qty"),
            F.sum(F.when(F.col("shift") == 3, F.col("quantity_scrapped")).otherwise(0)).alias("shift_3_qty"),
            
            # Root cause breakdown
            F.sum(F.when(F.col("root_cause_category") == "Machine", 1).otherwise(0)).alias("machine_root_cause"),
            F.sum(F.when(F.col("root_cause_category") == "Material", 1).otherwise(0)).alias("material_root_cause"),
            F.sum(F.when(F.col("root_cause_category") == "Method", 1).otherwise(0)).alias("method_root_cause"),
            F.sum(F.when(F.col("root_cause_category") == "Man", 1).otherwise(0)).alias("man_root_cause"),
            
            # Disposition breakdown
            F.sum(F.when(F.col("disposition") == "Scrap", F.col("quantity_scrapped")).otherwise(0)).alias("final_scrap_qty"),
            F.sum(F.when(F.col("disposition") == "Rework", F.col("quantity_scrapped")).otherwise(0)).alias("rework_qty"),
            F.sum(F.when(F.col("disposition") == "Return to Supplier", F.col("quantity_scrapped")).otherwise(0)).alias("supplier_return_qty")
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
            F.sum(F.when(F.col("verification_result") == "Effective", 1).otherwise(0)).alias("effective_8d_count"),
            F.sum(F.when(F.col("is_customer_notified") == True, 1).otherwise(0)).alias("customer_notified_8d")
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
# MAGIC ## 8D Text Analysis Ready Table

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
# MAGIC ## Supplier Scorecard Gold

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
    Monthly supplier scorecard for quality performance tracking.
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
            
            F.sum(F.when(F.col("defect_category") == "Material Defects", 1).otherwise(0)).alias("material_defects"),
            F.sum(F.when(F.col("disposition") == "Return to Supplier", F.col("quantity_scrapped")).otherwise(0)).alias("supplier_returns")
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
            F.sum(F.when(F.col("severity").isin("Critical", "Major"), 1).otherwise(0)).alias("critical_8d"),
            F.round(F.avg("days_open"), 1).alias("avg_8d_response_time")
        )
    )
    
    return (
        supplier_scrap
        .join(supplier_8d, ["supplier_id", "year", "month"], "left")
        .fillna(0, subset=["eight_d_issued", "critical_8d"])
        
        # Calculate scores
        .withColumn("scrap_score",
            F.greatest(
                F.lit(0),
                F.round(100 - (F.col("total_qty_scrapped") / 10), 1)
            )
        )
        .withColumn("quality_incident_score",
            F.greatest(
                F.lit(0),
                F.round(100 - (F.col("material_defects") * 5), 1)
            )
        )
        .withColumn("responsiveness_score",
            F.when(F.col("eight_d_issued") > 0,
                F.greatest(F.lit(0), F.round(100 - F.col("avg_8d_response_time"), 1))
            ).otherwise(100)
        )
        .withColumn("overall_score",
            F.round(
                F.col("supplier_quality_score") * 100 * 0.4 +
                F.col("scrap_score") * 0.3 +
                F.col("quality_incident_score") * 0.2 +
                F.col("responsiveness_score") * 0.1,
                1
            )
        )
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
# MAGIC ## KPI Dashboard Gold

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
    High-level KPIs aggregated at monthly level.
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
            F.sum(F.when(F.col("is_critical_part") == True, F.col("quantity_scrapped")).otherwise(0)).alias("critical_parts_scrapped"),
            F.sum(F.when(F.col("cost_severity") == "Critical", F.col("total_scrap_cost")).otherwise(0)).alias("critical_cost_incidents_value"),
            
            # Average metrics
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
            F.sum(F.when(F.col("verification_result") == "Effective", 1).otherwise(0)).alias("effective_closures")
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

