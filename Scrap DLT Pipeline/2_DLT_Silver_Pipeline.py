# Databricks notebook source
# MAGIC %md
# MAGIC # DLT Pipeline - Silver Layer Transformations
# MAGIC
# MAGIC This notebook defines Delta Live Tables (DLT) pipeline for Silver layer transformations.
# MAGIC
# MAGIC **Transformations:**
# MAGIC - Data quality validation and cleansing
# MAGIC - Standardized data types
# MAGIC - Denormalized dimension lookups
# MAGIC - Business rule enforcement
# MAGIC
# MAGIC **Source:** `welch.scrap_demo.*_bronze`  
# MAGIC **Target:** `welch.scrap_demo.*_silver`

# COMMAND ----------

import dlt
from pyspark.sql import functions as F
from pyspark.sql.types import *

# Configuration
CATALOG = "welch"
SCHEMA = "scrap_demo"
SOURCE_SCHEMA = f"{CATALOG}.{SCHEMA}"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Suppliers Silver

# COMMAND ----------

@dlt.table(
    name="suppliers_silver",
    comment="Cleaned and validated supplier master data",
    table_properties={
        "quality": "silver",
        "pipelines.autoOptimize.managed": "true"
    }
)
@dlt.expect_or_drop("valid_supplier_id", "supplier_id IS NOT NULL")
@dlt.expect_or_drop("valid_quality_score", "quality_score BETWEEN 0 AND 1")
@dlt.expect_or_drop("valid_delivery_rate", "on_time_delivery_rate BETWEEN 0 AND 1")
@dlt.expect("valid_tier", "supplier_tier IN ('Tier 1', 'Tier 2', 'Tier 3')")
def suppliers_silver():
    return (
        spark.table(f"{SOURCE_SCHEMA}.suppliers_bronze")
        .withColumn("supplier_id", F.upper(F.trim(F.col("supplier_id"))))
        .withColumn("supplier_name", F.initcap(F.trim(F.col("supplier_name"))))
        .withColumn("quality_score", F.round(F.col("quality_score"), 3))
        .withColumn("on_time_delivery_rate", F.round(F.col("on_time_delivery_rate"), 3))
        .withColumn("quality_rating", 
            F.when(F.col("quality_score") >= 0.95, "A")
            .when(F.col("quality_score") >= 0.90, "B")
            .when(F.col("quality_score") >= 0.80, "C")
            .otherwise("D")
        )
        .withColumn("delivery_rating",
            F.when(F.col("on_time_delivery_rate") >= 0.95, "A")
            .when(F.col("on_time_delivery_rate") >= 0.90, "B")
            .when(F.col("on_time_delivery_rate") >= 0.80, "C")
            .otherwise("D")
        )
        .withColumn("overall_supplier_score", 
            F.round((F.col("quality_score") * 0.6 + F.col("on_time_delivery_rate") * 0.4), 3)
        )
        .withColumn("risk_level",
            F.when(F.col("quality_score") < 0.80, "High")
            .when(F.col("quality_score") < 0.90, "Medium")
            .otherwise("Low")
        )
        .withColumn("processed_at", F.current_timestamp())
        .drop("is_problematic_flag")  # Remove internal flag from silver layer
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Plants Silver

# COMMAND ----------

@dlt.table(
    name="plants_silver",
    comment="Cleaned and validated plant/facility master data",
    table_properties={
        "quality": "silver",
        "pipelines.autoOptimize.managed": "true"
    }
)
@dlt.expect_or_drop("valid_plant_id", "plant_id IS NOT NULL")
@dlt.expect("valid_capacity", "annual_capacity_units > 0")
@dlt.expect("valid_year", "year_opened >= 1950 AND year_opened <= 2025")
def plants_silver():
    return (
        spark.table(f"{SOURCE_SCHEMA}.plants_bronze")
        .withColumn("plant_id", F.upper(F.trim(F.col("plant_id"))))
        .withColumn("plant_age_years", F.year(F.current_date()) - F.col("year_opened"))
        .withColumn("size_category",
            F.when(F.col("square_footage") >= 1500000, "Large")
            .when(F.col("square_footage") >= 800000, "Medium")
            .otherwise("Small")
        )
        .withColumn("region",
            F.when(F.col("country") == "USA", 
                F.when(F.col("state_province").isin("MI", "OH", "IN", "IL"), "US-Midwest")
                .when(F.col("state_province").isin("TX", "AZ", "NM"), "US-Southwest")
                .when(F.col("state_province").isin("KY", "TN", "AL", "GA", "SC"), "US-Southeast")
                .otherwise("US-Other")
            )
            .when(F.col("country") == "Canada", "Canada")
            .when(F.col("country") == "Mexico", "Mexico")
            .otherwise("Other")
        )
        .withColumn("processed_at", F.current_timestamp())
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Work Centers Silver

# COMMAND ----------

@dlt.table(
    name="work_centers_silver",
    comment="Cleaned work center data with plant enrichment",
    table_properties={
        "quality": "silver",
        "pipelines.autoOptimize.managed": "true"
    }
)
@dlt.expect_or_drop("valid_work_center_id", "work_center_id IS NOT NULL")
@dlt.expect_or_drop("valid_plant_id", "plant_id IS NOT NULL")
@dlt.expect("valid_capacity", "capacity_per_hour > 0")
def work_centers_silver():
    plants = spark.table(f"{SOURCE_SCHEMA}.plants_bronze").select(
        "plant_id", 
        F.col("plant_name").alias("plant_name_lookup"),
        F.col("city").alias("plant_city"),
        F.col("country").alias("plant_country")
    )
    
    return (
        spark.table(f"{SOURCE_SCHEMA}.work_centers_bronze")
        .withColumn("work_center_id", F.upper(F.trim(F.col("work_center_id"))))
        .withColumn("plant_id", F.upper(F.trim(F.col("plant_id"))))
        .withColumn("days_since_maintenance", 
            F.datediff(F.current_date(), F.col("last_maintenance_date"))
        )
        .withColumn("maintenance_status",
            F.when(F.col("days_since_maintenance") > 30, "Overdue")
            .when(F.col("days_since_maintenance") > 21, "Due Soon")
            .otherwise("Current")
        )
        .join(plants, "plant_id", "left")
        .withColumn("processed_at", F.current_timestamp())
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Equipment Silver

# COMMAND ----------

@dlt.table(
    name="equipment_silver",
    comment="Cleaned equipment data with health metrics",
    table_properties={
        "quality": "silver",
        "pipelines.autoOptimize.managed": "true"
    }
)
@dlt.expect_or_drop("valid_equipment_id", "equipment_id IS NOT NULL")
@dlt.expect_or_drop("valid_work_center", "work_center_id IS NOT NULL")
@dlt.expect("valid_oee", "oee_rating BETWEEN 0 AND 1")
def equipment_silver():
    work_centers = spark.table(f"{SOURCE_SCHEMA}.work_centers_bronze").select(
        "work_center_id",
        F.col("work_center_name").alias("work_center_name_lookup"),
        F.col("work_center_type").alias("work_center_type_lookup")
    )
    
    return (
        spark.table(f"{SOURCE_SCHEMA}.equipment_bronze")
        .withColumn("equipment_id", F.upper(F.trim(F.col("equipment_id"))))
        .withColumn("work_center_id", F.upper(F.trim(F.col("work_center_id"))))
        .withColumn("plant_id", F.upper(F.trim(F.col("plant_id"))))
        .withColumn("days_until_maintenance",
            F.datediff(F.col("next_maintenance_date"), F.current_date())
        )
        .withColumn("days_since_calibration",
            F.datediff(F.current_date(), F.col("last_calibration_date"))
        )
        .withColumn("equipment_health_score",
            F.round(
                F.col("oee_rating") * 0.5 + 
                F.least(F.col("mtbf_hours") / 1500, F.lit(1)) * 0.3 +
                F.when(F.col("age_years") < 5, 1.0)
                 .when(F.col("age_years") < 10, 0.8)
                 .when(F.col("age_years") < 15, 0.6)
                 .otherwise(0.4) * 0.2,
                3
            )
        )
        .withColumn("replacement_priority",
            F.when((F.col("age_years") > 15) & (F.col("oee_rating") < 0.75), "High")
            .when((F.col("age_years") > 10) & (F.col("oee_rating") < 0.80), "Medium")
            .otherwise("Low")
        )
        .withColumn("calibration_status",
            F.when(F.col("days_since_calibration") > 90, "Overdue")
            .when(F.col("days_since_calibration") > 60, "Due Soon")
            .otherwise("Current")
        )
        .join(work_centers, "work_center_id", "left")
        .withColumn("processed_at", F.current_timestamp())
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Operators Silver

# COMMAND ----------

@dlt.table(
    name="operators_silver",
    comment="Cleaned operator data with skill assessments",
    table_properties={
        "quality": "silver",
        "pipelines.autoOptimize.managed": "true"
    }
)
@dlt.expect_or_drop("valid_operator_id", "operator_id IS NOT NULL")
@dlt.expect_or_drop("valid_plant_id", "plant_id IS NOT NULL")
@dlt.expect("valid_shift", "primary_shift IN (1, 2, 3)")
def operators_silver():
    plants = spark.table(f"{SOURCE_SCHEMA}.plants_bronze").select(
        "plant_id",
        F.col("plant_name").alias("plant_name_lookup"),
        F.col("city").alias("plant_city")
    )
    
    return (
        spark.table(f"{SOURCE_SCHEMA}.operators_bronze")
        .withColumn("operator_id", F.upper(F.trim(F.col("operator_id"))))
        .withColumn("plant_id", F.upper(F.trim(F.col("plant_id"))))
        .withColumn("full_name", F.concat_ws(" ", F.col("first_name"), F.col("last_name")))
        .withColumn("tenure_years", 
            F.round(F.datediff(F.current_date(), F.col("hire_date")) / 365.25, 1)
        )
        .withColumn("skill_score",
            F.when(F.col("skill_level") == "Expert", 5)
            .when(F.col("skill_level") == "Senior", 4)
            .when(F.col("skill_level") == "Intermediate", 3)
            .when(F.col("skill_level") == "Junior", 2)
            .otherwise(1)
        )
        .withColumn("shift_name",
            F.when(F.col("primary_shift") == 1, "Day Shift")
            .when(F.col("primary_shift") == 2, "Evening Shift")
            .otherwise("Night Shift")
        )
        .join(plants, "plant_id", "left")
        .drop("is_high_error_flag")  # Remove internal flag
        .withColumn("processed_at", F.current_timestamp())
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cost Centers Silver

# COMMAND ----------

@dlt.table(
    name="cost_centers_silver",
    comment="Cleaned cost center master data",
    table_properties={
        "quality": "silver",
        "pipelines.autoOptimize.managed": "true"
    }
)
@dlt.expect_or_drop("valid_cost_center_id", "cost_center_id IS NOT NULL")
@dlt.expect("valid_budget", "budget_annual > 0")
def cost_centers_silver():
    return (
        spark.table(f"{SOURCE_SCHEMA}.cost_centers_bronze")
        .withColumn("cost_center_id", F.upper(F.trim(F.col("cost_center_id"))))
        .withColumn("budget_monthly", F.round(F.col("budget_annual") / 12, 2))
        .withColumn("budget_category",
            F.when(F.col("budget_annual") >= 2000000, "High")
            .when(F.col("budget_annual") >= 500000, "Medium")
            .otherwise("Low")
        )
        .withColumn("processed_at", F.current_timestamp())
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Defect Codes Silver

# COMMAND ----------

@dlt.table(
    name="defect_codes_silver",
    comment="Cleaned defect code hierarchy",
    table_properties={
        "quality": "silver",
        "pipelines.autoOptimize.managed": "true"
    }
)
@dlt.expect_or_drop("valid_defect_code", "defect_code IS NOT NULL")
def defect_codes_silver():
    return (
        spark.table(f"{SOURCE_SCHEMA}.defect_codes_bronze")
        .withColumn("defect_code", F.upper(F.trim(F.col("defect_code"))))
        .withColumn("parent_code", F.upper(F.trim(F.col("parent_code"))))
        .withColumn("is_leaf_code", F.col("hierarchy_level") == 2)
        .withColumn("severity_category",
            F.when(F.col("severity_weight") >= 0.7, "High")
            .when(F.col("severity_weight") >= 0.4, "Medium")
            .otherwise("Low")
        )
        .withColumn("processed_at", F.current_timestamp())
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Parts Master Silver

# COMMAND ----------

@dlt.table(
    name="parts_master_silver",
    comment="Cleaned parts master with supplier enrichment",
    table_properties={
        "quality": "silver",
        "pipelines.autoOptimize.managed": "true"
    }
)
@dlt.expect_or_drop("valid_part_id", "part_id IS NOT NULL")
@dlt.expect_or_drop("valid_supplier_id", "supplier_id IS NOT NULL")
@dlt.expect("valid_cost", "unit_cost > 0")
@dlt.expect("valid_lead_time", "lead_time_days > 0")
def parts_master_silver():
    suppliers = spark.table(f"{SOURCE_SCHEMA}.suppliers_bronze").select(
        "supplier_id",
        F.col("supplier_name").alias("supplier_name_lookup"),
        F.col("supplier_tier").alias("supplier_tier_lookup"),
        "quality_score",
        "specialization"
    )
    
    return (
        spark.table(f"{SOURCE_SCHEMA}.parts_master_bronze")
        .withColumn("part_id", F.upper(F.trim(F.col("part_id"))))
        .withColumn("supplier_id", F.upper(F.trim(F.col("supplier_id"))))
        .withColumn("part_number", F.upper(F.trim(F.col("part_number"))))
        .withColumn("cost_category",
            F.when(F.col("unit_cost") >= 100, "High Value")
            .when(F.col("unit_cost") >= 25, "Medium Value")
            .otherwise("Low Value")
        )
        .withColumn("lead_time_category",
            F.when(F.col("lead_time_days") >= 30, "Long Lead")
            .when(F.col("lead_time_days") >= 14, "Medium Lead")
            .otherwise("Short Lead")
        )
        .withColumn("inventory_criticality",
            F.when(F.col("is_critical") & (F.col("lead_time_days") >= 21), "Critical-Long Lead")
            .when(F.col("is_critical"), "Critical")
            .when(F.col("lead_time_days") >= 30, "Long Lead Risk")
            .otherwise("Standard")
        )
        .join(suppliers, "supplier_id", "left")
        .withColumn("processed_at", F.current_timestamp())
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Scrap Transactions Silver (Main Fact Table)

# COMMAND ----------

@dlt.table(
    name="scrap_transactions_silver",
    comment="Cleaned and enriched scrap transaction data - main fact table",
    table_properties={
        "quality": "silver",
        "pipelines.autoOptimize.managed": "true"
    },
    partition_cols=["transaction_year", "transaction_month"]
)
@dlt.expect_or_drop("valid_transaction_id", "scrap_transaction_id IS NOT NULL")
@dlt.expect_or_drop("valid_date", "transaction_date IS NOT NULL")
@dlt.expect_or_drop("valid_quantity", "quantity_scrapped > 0")
@dlt.expect("valid_cost", "total_scrap_cost >= 0")
@dlt.expect("valid_shift", "shift IN (1, 2, 3)")
def scrap_transactions_silver():
    # Lookup tables
    parts = spark.table(f"{SOURCE_SCHEMA}.parts_master_bronze").select(
        "part_id",
        F.col("part_name").alias("part_name_lookup"),
        F.col("part_category").alias("part_category_lookup"),
        F.col("part_type").alias("part_type_lookup"),
        F.col("material").alias("part_material"),
        F.col("is_critical").alias("is_critical_part")
    )
    
    work_centers = spark.table(f"{SOURCE_SCHEMA}.work_centers_bronze").select(
        "work_center_id",
        F.col("work_center_name").alias("work_center_name_lookup"),
        F.col("work_center_type").alias("work_center_type_lookup"),
        F.col("work_center_category").alias("work_center_category_lookup")
    )
    
    equipment = spark.table(f"{SOURCE_SCHEMA}.equipment_bronze").select(
        "equipment_id",
        F.col("equipment_name").alias("equipment_name_lookup"),
        F.col("equipment_type").alias("equipment_type_lookup"),
        F.col("age_years").alias("equipment_age_years"),
        F.col("oee_rating").alias("equipment_oee")
    )
    
    operators = spark.table(f"{SOURCE_SCHEMA}.operators_bronze").select(
        "operator_id",
        F.concat_ws(" ", F.col("first_name"), F.col("last_name")).alias("operator_name"),
        F.col("skill_level").alias("operator_skill_level"),
        F.col("years_experience").alias("operator_experience_years")
    )
    
    plants = spark.table(f"{SOURCE_SCHEMA}.plants_bronze").select(
        "plant_id",
        F.col("plant_name").alias("plant_name_lookup"),
        F.col("city").alias("plant_city"),
        F.col("country").alias("plant_country")
    )
    
    suppliers = spark.table(f"{SOURCE_SCHEMA}.suppliers_bronze").select(
        "supplier_id",
        F.col("supplier_name").alias("supplier_name_lookup"),
        F.col("supplier_tier").alias("supplier_tier_lookup"),
        F.col("quality_score").alias("supplier_quality_score")
    )
    
    defect_codes = spark.table(f"{SOURCE_SCHEMA}.defect_codes_bronze").select(
        "defect_code",
        F.col("defect_name").alias("defect_name_lookup"),
        F.col("severity_weight").alias("defect_severity_weight")
    )
    
    return (
        spark.table(f"{SOURCE_SCHEMA}.scrap_transactions_bronze")
        # Clean IDs
        .withColumn("scrap_transaction_id", F.upper(F.trim(F.col("scrap_transaction_id"))))
        .withColumn("part_id", F.upper(F.trim(F.col("part_id"))))
        .withColumn("work_center_id", F.upper(F.trim(F.col("work_center_id"))))
        .withColumn("equipment_id", F.upper(F.trim(F.col("equipment_id"))))
        .withColumn("operator_id", F.upper(F.trim(F.col("operator_id"))))
        .withColumn("plant_id", F.upper(F.trim(F.col("plant_id"))))
        .withColumn("supplier_id", F.upper(F.trim(F.col("supplier_id"))))
        .withColumn("defect_code", F.upper(F.trim(F.col("defect_code"))))
        
        # Time dimensions
        .withColumn("transaction_year", F.year(F.col("transaction_date")))
        .withColumn("transaction_month", F.month(F.col("transaction_date")))
        .withColumn("transaction_quarter", F.quarter(F.col("transaction_date")))
        .withColumn("transaction_week", F.weekofyear(F.col("transaction_date")))
        .withColumn("transaction_day_of_week", F.dayofweek(F.col("transaction_date")))
        .withColumn("day_name", F.date_format(F.col("transaction_date"), "EEEE"))
        .withColumn("is_weekend", F.dayofweek(F.col("transaction_date")).isin(1, 7))
        .withColumn("is_month_end", F.dayofmonth(F.col("transaction_date")) >= 25)
        
        # Shift enrichment
        .withColumn("shift_name",
            F.when(F.col("shift") == 1, "Day Shift")
            .when(F.col("shift") == 2, "Evening Shift")
            .otherwise("Night Shift")
        )
        
        # Severity classification
        .withColumn("cost_severity",
            F.when(F.col("total_scrap_cost") >= 1000, "Critical")
            .when(F.col("total_scrap_cost") >= 500, "High")
            .when(F.col("total_scrap_cost") >= 100, "Medium")
            .otherwise("Low")
        )
        
        .withColumn("quantity_severity",
            F.when(F.col("quantity_scrapped") >= 50, "Critical")
            .when(F.col("quantity_scrapped") >= 20, "High")
            .when(F.col("quantity_scrapped") >= 5, "Medium")
            .otherwise("Low")
        )
        
        # Join lookups
        .join(parts, "part_id", "left")
        .join(work_centers, "work_center_id", "left")
        .join(equipment, "equipment_id", "left")
        .join(operators, "operator_id", "left")
        .join(plants, "plant_id", "left")
        .join(suppliers, "supplier_id", "left")
        .join(defect_codes, "defect_code", "left")
        
        .withColumn("processed_at", F.current_timestamp())
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8D Reports Silver

# COMMAND ----------

@dlt.table(
    name="eight_d_reports_silver",
    comment="Cleaned and enriched 8D quality reports",
    table_properties={
        "quality": "silver",
        "pipelines.autoOptimize.managed": "true"
    }
)
@dlt.expect_or_drop("valid_8d_id", "eight_d_id IS NOT NULL")
@dlt.expect_or_drop("valid_date_opened", "date_opened IS NOT NULL")
@dlt.expect("has_root_cause", "root_cause_analysis IS NOT NULL")
@dlt.expect("has_corrective_action", "corrective_actions IS NOT NULL")
def eight_d_reports_silver():
    plants = spark.table(f"{SOURCE_SCHEMA}.plants_bronze").select(
        "plant_id",
        F.col("plant_name").alias("plant_name_lookup"),
        F.col("city").alias("plant_city")
    )
    
    suppliers = spark.table(f"{SOURCE_SCHEMA}.suppliers_bronze").select(
        "supplier_id",
        F.col("supplier_name").alias("supplier_name_lookup")
    )
    
    return (
        spark.table(f"{SOURCE_SCHEMA}.eight_d_reports_bronze")
        .withColumn("eight_d_id", F.upper(F.trim(F.col("eight_d_id"))))
        .withColumn("plant_id", F.upper(F.trim(F.col("plant_id"))))
        .withColumn("supplier_id", F.upper(F.trim(F.col("supplier_id"))))
        
        # Time metrics
        .withColumn("days_open",
            F.when(F.col("date_closed").isNotNull(),
                F.datediff(F.col("date_closed"), F.col("date_opened"))
            ).otherwise(
                F.datediff(F.current_date(), F.col("date_opened"))
            )
        )
        .withColumn("days_until_target",
            F.datediff(F.col("target_close_date"), F.current_date())
        )
        .withColumn("is_overdue",
            (F.col("status") != "Closed") & (F.col("days_until_target") < 0)
        )
        
        # Text analysis metrics (for LLM prep)
        .withColumn("root_cause_word_count", 
            F.size(F.split(F.col("root_cause_analysis"), " "))
        )
        .withColumn("corrective_action_word_count",
            F.size(F.split(F.col("corrective_actions"), " "))
        )
        
        # Cost per part affected
        .withColumn("cost_per_part_affected",
            F.when(F.col("num_parts_affected") > 0,
                F.round(F.col("total_scrap_cost") / F.col("num_parts_affected"), 2)
            ).otherwise(0)
        )
        
        # Resolution time category
        .withColumn("resolution_time_category",
            F.when(F.col("status") == "Closed",
                F.when(F.col("days_open") <= 14, "Fast (<14 days)")
                .when(F.col("days_open") <= 30, "Normal (14-30 days)")
                .when(F.col("days_open") <= 60, "Slow (30-60 days)")
                .otherwise("Extended (>60 days)")
            ).otherwise("In Progress")
        )
        
        .join(plants, "plant_id", "left")
        .join(suppliers, "supplier_id", "left")
        .withColumn("processed_at", F.current_timestamp())
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Scrap to 8D Mapping Silver

# COMMAND ----------

@dlt.table(
    name="scrap_8d_mapping_silver",
    comment="Cleaned scrap to 8D report mapping",
    table_properties={
        "quality": "silver",
        "pipelines.autoOptimize.managed": "true"
    }
)
@dlt.expect_or_drop("valid_scrap_id", "scrap_transaction_id IS NOT NULL")
@dlt.expect_or_drop("valid_8d_id", "eight_d_id IS NOT NULL")
def scrap_8d_mapping_silver():
    return (
        spark.table(f"{SOURCE_SCHEMA}.scrap_8d_mapping_bronze")
        .withColumn("scrap_transaction_id", F.upper(F.trim(F.col("scrap_transaction_id"))))
        .withColumn("eight_d_id", F.upper(F.trim(F.col("eight_d_id"))))
        .withColumn("processed_at", F.current_timestamp())
    )

