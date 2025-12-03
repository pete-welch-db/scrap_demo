# Databricks notebook source
# MAGIC %md
# MAGIC # 🔄 DLT Pipeline - Silver Layer Transformations
# MAGIC 
# MAGIC ## Overview
# MAGIC 
# MAGIC This notebook defines **Delta Live Tables (DLT)** pipeline for transforming Bronze data into the Silver layer.
# MAGIC 
# MAGIC ## Medallion Architecture
# MAGIC 
# MAGIC ```
# MAGIC ┌─────────────────────┐      ┌─────────────────────┐      ┌─────────────────────┐
# MAGIC │   BRONZE (Raw)      │ ───► │  SILVER (Cleaned)   │ ───► │   GOLD (Analytics)  │
# MAGIC │   - Raw ingestion   │      │  - Validated        │      │   - Aggregated      │
# MAGIC │   - Schema applied  │      │  - Enriched         │      │   - ML-ready        │
# MAGIC │   - No transforms   │      │  - Standardized     │      │   - Dashboard-ready │
# MAGIC └─────────────────────┘      └─────────────────────┘      └─────────────────────┘
# MAGIC                                      ▲
# MAGIC                                      │
# MAGIC                               THIS NOTEBOOK
# MAGIC ```
# MAGIC 
# MAGIC ## Silver Layer Responsibilities
# MAGIC 
# MAGIC | Transformation | Description |
# MAGIC |----------------|-------------|
# MAGIC | **Data Quality** | Validate records, drop/quarantine bad data |
# MAGIC | **Cleansing** | Standardize IDs, trim whitespace, fix casing |
# MAGIC | **Enrichment** | Add calculated fields, business classifications |
# MAGIC | **Denormalization** | Join dimension lookups for easier analysis |
# MAGIC | **Type Standardization** | Ensure consistent data types |
# MAGIC 
# MAGIC ## Source/Target
# MAGIC - **Source**: `welch.scrap_demo.*_bronze`
# MAGIC - **Target**: `welch.scrap_demo.*_silver`

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📚 Import Libraries
# MAGIC 
# MAGIC Import DLT module and PySpark functions for transformations.

# COMMAND ----------

import dlt
from pyspark.sql import functions as F
from pyspark.sql.types import *

# COMMAND ----------

# MAGIC %md
# MAGIC ## ⚙️ Configuration
# MAGIC 
# MAGIC Define source schema for bronze tables.

# COMMAND ----------

# Configuration
CATALOG = "welch"
SCHEMA = "scrap_demo"
SOURCE_SCHEMA = f"{CATALOG}.{SCHEMA}"

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # 📊 Dimension Tables
# MAGIC ---

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🏭 Suppliers Silver
# MAGIC 
# MAGIC **Purpose**: Clean supplier master data and add quality ratings.
# MAGIC 
# MAGIC **Transformations**:
# MAGIC - Validate supplier_id is not null
# MAGIC - Validate quality scores are between 0 and 1
# MAGIC - Calculate quality/delivery ratings (A/B/C/D)
# MAGIC - Calculate overall supplier score (weighted average)
# MAGIC - Assign risk level based on quality score
# MAGIC - Remove internal flags from silver layer
# MAGIC 
# MAGIC **Data Quality Expectations**:
# MAGIC - `valid_supplier_id`: Drop if NULL
# MAGIC - `valid_quality_score`: Drop if not between 0-1
# MAGIC - `valid_delivery_rate`: Drop if not between 0-1
# MAGIC - `valid_tier`: Warn if not Tier 1/2/3

# COMMAND ----------

@dlt.table(
    name="suppliers_silver",
    comment="Cleaned and validated supplier master data with quality ratings",
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
    """
    Transform suppliers_bronze to suppliers_silver.
    
    Adds quality ratings, risk levels, and overall scores.
    """
    return (
        spark.table(f"{SOURCE_SCHEMA}.suppliers_bronze")
        # Standardize IDs and names
        .withColumn("supplier_id", F.upper(F.trim(F.col("supplier_id"))))
        .withColumn("supplier_name", F.initcap(F.trim(F.col("supplier_name"))))
        
        # Round numeric fields
        .withColumn("quality_score", F.round(F.col("quality_score"), 3))
        .withColumn("on_time_delivery_rate", F.round(F.col("on_time_delivery_rate"), 3))
        
        # Calculate quality rating (A/B/C/D)
        .withColumn("quality_rating", 
            F.when(F.col("quality_score") >= 0.95, "A")
            .when(F.col("quality_score") >= 0.90, "B")
            .when(F.col("quality_score") >= 0.80, "C")
            .otherwise("D")
        )
        
        # Calculate delivery rating
        .withColumn("delivery_rating",
            F.when(F.col("on_time_delivery_rate") >= 0.95, "A")
            .when(F.col("on_time_delivery_rate") >= 0.90, "B")
            .when(F.col("on_time_delivery_rate") >= 0.80, "C")
            .otherwise("D")
        )
        
        # Calculate weighted overall score (60% quality, 40% delivery)
        .withColumn("overall_supplier_score", 
            F.round((F.col("quality_score") * 0.6 + F.col("on_time_delivery_rate") * 0.4), 3)
        )
        
        # Assign risk level
        .withColumn("risk_level",
            F.when(F.col("quality_score") < 0.80, "High")
            .when(F.col("quality_score") < 0.90, "Medium")
            .otherwise("Low")
        )
        
        # Add processing timestamp
        .withColumn("processed_at", F.current_timestamp())
        
        # Remove internal ML flag from silver layer
        .drop("is_problematic_flag")
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🏢 Plants Silver
# MAGIC 
# MAGIC **Purpose**: Clean plant/facility data and add derived attributes.
# MAGIC 
# MAGIC **Transformations**:
# MAGIC - Calculate plant age from year opened
# MAGIC - Classify plant size (Small/Medium/Large)
# MAGIC - Assign region based on location
# MAGIC 
# MAGIC **Data Quality Expectations**:
# MAGIC - `valid_plant_id`: Drop if NULL
# MAGIC - `valid_capacity`: Warn if capacity <= 0
# MAGIC - `valid_year`: Warn if year_opened is unreasonable

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
    """
    Transform plants_bronze to plants_silver.
    
    Adds plant age, size category, and regional classification.
    """
    return (
        spark.table(f"{SOURCE_SCHEMA}.plants_bronze")
        # Standardize IDs
        .withColumn("plant_id", F.upper(F.trim(F.col("plant_id"))))
        
        # Calculate plant age
        .withColumn("plant_age_years", F.year(F.current_date()) - F.col("year_opened"))
        
        # Classify plant size
        .withColumn("size_category",
            F.when(F.col("square_footage") >= 1500000, "Large")
            .when(F.col("square_footage") >= 800000, "Medium")
            .otherwise("Small")
        )
        
        # Assign region
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
# MAGIC ## 🔧 Work Centers Silver
# MAGIC 
# MAGIC **Purpose**: Clean work center data and add maintenance status.
# MAGIC 
# MAGIC **Transformations**:
# MAGIC - Join with plants for location enrichment
# MAGIC - Calculate days since last maintenance
# MAGIC - Assign maintenance status (Current/Due Soon/Overdue)

# COMMAND ----------

@dlt.table(
    name="work_centers_silver",
    comment="Cleaned work center data with plant enrichment and maintenance status",
    table_properties={
        "quality": "silver",
        "pipelines.autoOptimize.managed": "true"
    }
)
@dlt.expect_or_drop("valid_work_center_id", "work_center_id IS NOT NULL")
@dlt.expect_or_drop("valid_plant_id", "plant_id IS NOT NULL")
@dlt.expect("valid_capacity", "capacity_per_hour > 0")
def work_centers_silver():
    """
    Transform work_centers_bronze to work_centers_silver.
    
    Adds plant lookup, maintenance tracking, and operational status.
    """
    # Get plant lookup data
    plants = spark.table(f"{SOURCE_SCHEMA}.plants_bronze").select(
        "plant_id", 
        F.col("plant_name").alias("plant_name_lookup"),
        F.col("city").alias("plant_city"),
        F.col("country").alias("plant_country")
    )
    
    return (
        spark.table(f"{SOURCE_SCHEMA}.work_centers_bronze")
        # Standardize IDs
        .withColumn("work_center_id", F.upper(F.trim(F.col("work_center_id"))))
        .withColumn("plant_id", F.upper(F.trim(F.col("plant_id"))))
        
        # Calculate maintenance metrics
        .withColumn("days_since_maintenance", 
            F.datediff(F.current_date(), F.col("last_maintenance_date"))
        )
        .withColumn("maintenance_status",
            F.when(F.col("days_since_maintenance") > 30, "Overdue")
            .when(F.col("days_since_maintenance") > 21, "Due Soon")
            .otherwise("Current")
        )
        
        # Join plant lookup
        .join(plants, "plant_id", "left")
        
        .withColumn("processed_at", F.current_timestamp())
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## ⚙️ Equipment Silver
# MAGIC 
# MAGIC **Purpose**: Clean equipment data and calculate health metrics.
# MAGIC 
# MAGIC **Transformations**:
# MAGIC - Calculate equipment health score (composite of OEE, MTBF, age)
# MAGIC - Assign replacement priority
# MAGIC - Track calibration status

# COMMAND ----------

@dlt.table(
    name="equipment_silver",
    comment="Cleaned equipment data with health metrics and replacement priority",
    table_properties={
        "quality": "silver",
        "pipelines.autoOptimize.managed": "true"
    }
)
@dlt.expect_or_drop("valid_equipment_id", "equipment_id IS NOT NULL")
@dlt.expect_or_drop("valid_work_center", "work_center_id IS NOT NULL")
@dlt.expect("valid_oee", "oee_rating BETWEEN 0 AND 1")
def equipment_silver():
    """
    Transform equipment_bronze to equipment_silver.
    
    Adds health scoring, calibration tracking, and replacement prioritization.
    """
    # Get work center lookup
    work_centers = spark.table(f"{SOURCE_SCHEMA}.work_centers_bronze").select(
        "work_center_id",
        F.col("work_center_name").alias("work_center_name_lookup"),
        F.col("work_center_type").alias("work_center_type_lookup")
    )
    
    return (
        spark.table(f"{SOURCE_SCHEMA}.equipment_bronze")
        # Standardize IDs
        .withColumn("equipment_id", F.upper(F.trim(F.col("equipment_id"))))
        .withColumn("work_center_id", F.upper(F.trim(F.col("work_center_id"))))
        .withColumn("plant_id", F.upper(F.trim(F.col("plant_id"))))
        
        # Maintenance and calibration tracking
        .withColumn("days_until_maintenance",
            F.datediff(F.col("next_maintenance_date"), F.current_date())
        )
        .withColumn("days_since_calibration",
            F.datediff(F.current_date(), F.col("last_calibration_date"))
        )
        
        # Calculate composite health score
        # Formula: 50% OEE + 30% MTBF (normalized) + 20% Age factor
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
        
        # Replacement priority based on age and OEE
        .withColumn("replacement_priority",
            F.when((F.col("age_years") > 15) & (F.col("oee_rating") < 0.75), "High")
            .when((F.col("age_years") > 10) & (F.col("oee_rating") < 0.80), "Medium")
            .otherwise("Low")
        )
        
        # Calibration status
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
# MAGIC ## 👷 Operators Silver
# MAGIC 
# MAGIC **Purpose**: Clean operator data and calculate skill metrics.
# MAGIC 
# MAGIC **Transformations**:
# MAGIC - Create full name from first/last
# MAGIC - Calculate tenure in years
# MAGIC - Convert skill level to numeric score

# COMMAND ----------

@dlt.table(
    name="operators_silver",
    comment="Cleaned operator data with skill assessments and tenure",
    table_properties={
        "quality": "silver",
        "pipelines.autoOptimize.managed": "true"
    }
)
@dlt.expect_or_drop("valid_operator_id", "operator_id IS NOT NULL")
@dlt.expect_or_drop("valid_plant_id", "plant_id IS NOT NULL")
@dlt.expect("valid_shift", "primary_shift IN (1, 2, 3)")
def operators_silver():
    """
    Transform operators_bronze to operators_silver.
    
    Adds full name, tenure calculation, and skill scoring.
    """
    plants = spark.table(f"{SOURCE_SCHEMA}.plants_bronze").select(
        "plant_id",
        F.col("plant_name").alias("plant_name_lookup"),
        F.col("city").alias("plant_city")
    )
    
    return (
        spark.table(f"{SOURCE_SCHEMA}.operators_bronze")
        .withColumn("operator_id", F.upper(F.trim(F.col("operator_id"))))
        .withColumn("plant_id", F.upper(F.trim(F.col("plant_id"))))
        
        # Create full name
        .withColumn("full_name", F.concat_ws(" ", F.col("first_name"), F.col("last_name")))
        
        # Calculate tenure
        .withColumn("tenure_years", 
            F.round(F.datediff(F.current_date(), F.col("hire_date")) / 365.25, 1)
        )
        
        # Convert skill level to numeric score (1-5)
        .withColumn("skill_score",
            F.when(F.col("skill_level") == "Expert", 5)
            .when(F.col("skill_level") == "Senior", 4)
            .when(F.col("skill_level") == "Intermediate", 3)
            .when(F.col("skill_level") == "Junior", 2)
            .otherwise(1)
        )
        
        # Friendly shift name
        .withColumn("shift_name",
            F.when(F.col("primary_shift") == 1, "Day Shift")
            .when(F.col("primary_shift") == 2, "Evening Shift")
            .otherwise("Night Shift")
        )
        
        .join(plants, "plant_id", "left")
        .drop("is_high_error_flag")  # Remove internal ML flag
        .withColumn("processed_at", F.current_timestamp())
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## 💰 Cost Centers Silver
# MAGIC 
# MAGIC **Purpose**: Clean cost center data and add budget categorization.

# COMMAND ----------

@dlt.table(
    name="cost_centers_silver",
    comment="Cleaned cost center master data with budget metrics",
    table_properties={
        "quality": "silver",
        "pipelines.autoOptimize.managed": "true"
    }
)
@dlt.expect_or_drop("valid_cost_center_id", "cost_center_id IS NOT NULL")
@dlt.expect("valid_budget", "budget_annual > 0")
def cost_centers_silver():
    """
    Transform cost_centers_bronze to cost_centers_silver.
    
    Adds monthly budget and budget category classification.
    """
    return (
        spark.table(f"{SOURCE_SCHEMA}.cost_centers_bronze")
        .withColumn("cost_center_id", F.upper(F.trim(F.col("cost_center_id"))))
        
        # Calculate monthly budget
        .withColumn("budget_monthly", F.round(F.col("budget_annual") / 12, 2))
        
        # Categorize budget size
        .withColumn("budget_category",
            F.when(F.col("budget_annual") >= 2000000, "High")
            .when(F.col("budget_annual") >= 500000, "Medium")
            .otherwise("Low")
        )
        
        .withColumn("processed_at", F.current_timestamp())
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## ⚠️ Defect Codes Silver
# MAGIC 
# MAGIC **Purpose**: Clean defect code hierarchy and add severity classification.

# COMMAND ----------

@dlt.table(
    name="defect_codes_silver",
    comment="Cleaned defect code hierarchy with severity classification",
    table_properties={
        "quality": "silver",
        "pipelines.autoOptimize.managed": "true"
    }
)
@dlt.expect_or_drop("valid_defect_code", "defect_code IS NOT NULL")
def defect_codes_silver():
    """
    Transform defect_codes_bronze to defect_codes_silver.
    
    Standardizes codes and adds severity categorization.
    """
    return (
        spark.table(f"{SOURCE_SCHEMA}.defect_codes_bronze")
        .withColumn("defect_code", F.upper(F.trim(F.col("defect_code"))))
        .withColumn("parent_code", F.upper(F.trim(F.col("parent_code"))))
        
        # Flag leaf codes (level 2 = most specific)
        .withColumn("is_leaf_code", F.col("hierarchy_level") == 2)
        
        # Categorize severity
        .withColumn("severity_category",
            F.when(F.col("severity_weight") >= 0.7, "High")
            .when(F.col("severity_weight") >= 0.4, "Medium")
            .otherwise("Low")
        )
        
        .withColumn("processed_at", F.current_timestamp())
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🔩 Parts Master Silver
# MAGIC 
# MAGIC **Purpose**: Clean parts data and add supplier enrichment.
# MAGIC 
# MAGIC **Transformations**:
# MAGIC - Join supplier information
# MAGIC - Classify by cost category
# MAGIC - Classify by lead time
# MAGIC - Calculate inventory criticality

# COMMAND ----------

@dlt.table(
    name="parts_master_silver",
    comment="Cleaned parts master with supplier enrichment and cost classification",
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
    """
    Transform parts_master_bronze to parts_master_silver.
    
    Adds supplier lookup, cost/lead time classification, and inventory criticality.
    """
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
        
        # Cost classification
        .withColumn("cost_category",
            F.when(F.col("unit_cost") >= 100, "High Value")
            .when(F.col("unit_cost") >= 25, "Medium Value")
            .otherwise("Low Value")
        )
        
        # Lead time classification
        .withColumn("lead_time_category",
            F.when(F.col("lead_time_days") >= 30, "Long Lead")
            .when(F.col("lead_time_days") >= 14, "Medium Lead")
            .otherwise("Short Lead")
        )
        
        # Inventory criticality (combination of critical flag and lead time)
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
# MAGIC ---
# MAGIC # 📊 Fact Tables
# MAGIC ---

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🗑️ Scrap Transactions Silver (Main Fact Table)
# MAGIC 
# MAGIC **Purpose**: Clean and enrich scrap transactions - the primary fact table.
# MAGIC 
# MAGIC **Transformations**:
# MAGIC - Standardize all IDs
# MAGIC - Add time dimension columns (year, month, quarter, week, day)
# MAGIC - Join all dimension lookups (parts, work centers, equipment, operators, plants, suppliers, defects)
# MAGIC - Add severity classifications
# MAGIC - Partition by year/month for performance
# MAGIC 
# MAGIC **Data Quality Expectations**:
# MAGIC - Must have transaction ID and date
# MAGIC - Quantity must be positive
# MAGIC - Cost must be non-negative
# MAGIC - Shift must be 1, 2, or 3

# COMMAND ----------

@dlt.table(
    name="scrap_transactions_silver",
    comment="Cleaned and enriched scrap transactions - main fact table for analysis",
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
    """
    Transform scrap_transactions_bronze to scrap_transactions_silver.
    
    This is the main fact table. Joins all dimension lookups and adds
    time dimensions and severity classifications.
    """
    # Load all dimension lookups
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
        # Standardize all IDs
        .withColumn("scrap_transaction_id", F.upper(F.trim(F.col("scrap_transaction_id"))))
        .withColumn("part_id", F.upper(F.trim(F.col("part_id"))))
        .withColumn("work_center_id", F.upper(F.trim(F.col("work_center_id"))))
        .withColumn("equipment_id", F.upper(F.trim(F.col("equipment_id"))))
        .withColumn("operator_id", F.upper(F.trim(F.col("operator_id"))))
        .withColumn("plant_id", F.upper(F.trim(F.col("plant_id"))))
        .withColumn("supplier_id", F.upper(F.trim(F.col("supplier_id"))))
        .withColumn("defect_code", F.upper(F.trim(F.col("defect_code"))))
        
        # Add time dimension columns for partitioning and analysis
        .withColumn("transaction_year", F.year(F.col("transaction_date")))
        .withColumn("transaction_month", F.month(F.col("transaction_date")))
        .withColumn("transaction_quarter", F.quarter(F.col("transaction_date")))
        .withColumn("transaction_week", F.weekofyear(F.col("transaction_date")))
        .withColumn("transaction_day_of_week", F.dayofweek(F.col("transaction_date")))
        .withColumn("day_name", F.date_format(F.col("transaction_date"), "EEEE"))
        .withColumn("is_weekend", F.dayofweek(F.col("transaction_date")).isin(1, 7))
        .withColumn("is_month_end", F.dayofmonth(F.col("transaction_date")) >= 25)
        
        # Add shift friendly name
        .withColumn("shift_name",
            F.when(F.col("shift") == 1, "Day Shift")
            .when(F.col("shift") == 2, "Evening Shift")
            .otherwise("Night Shift")
        )
        
        # Add severity classifications
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
        
        # Join all dimension lookups
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
# MAGIC ## 📋 8D Reports Silver
# MAGIC 
# MAGIC **Purpose**: Clean 8D quality reports and add effectiveness metrics.
# MAGIC 
# MAGIC **Transformations**:
# MAGIC - Calculate days open
# MAGIC - Calculate days until target close
# MAGIC - Flag overdue 8Ds
# MAGIC - Count words in text fields (for ML analysis)
# MAGIC - Add resolution time category

# COMMAND ----------

@dlt.table(
    name="eight_d_reports_silver",
    comment="Cleaned and enriched 8D quality reports with effectiveness metrics",
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
    """
    Transform eight_d_reports_bronze to eight_d_reports_silver.
    
    Adds time metrics, overdue flags, and text analysis preparation fields.
    """
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
        
        # Calculate days open
        .withColumn("days_open",
            F.when(F.col("date_closed").isNotNull(),
                F.datediff(F.col("date_closed"), F.col("date_opened"))
            ).otherwise(
                F.datediff(F.current_date(), F.col("date_opened"))
            )
        )
        
        # Days until target close date
        .withColumn("days_until_target",
            F.datediff(F.col("target_close_date"), F.current_date())
        )
        
        # Flag overdue 8Ds
        .withColumn("is_overdue",
            (F.col("status") != "Closed") & (F.col("days_until_target") < 0)
        )
        
        # Word counts for text analysis (useful for LLM prep)
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
# MAGIC ## 🔗 Scrap to 8D Mapping Silver
# MAGIC 
# MAGIC **Purpose**: Clean the mapping table linking scrap transactions to 8D reports.

# COMMAND ----------

@dlt.table(
    name="scrap_8d_mapping_silver",
    comment="Cleaned scrap transaction to 8D report mapping",
    table_properties={
        "quality": "silver",
        "pipelines.autoOptimize.managed": "true"
    }
)
@dlt.expect_or_drop("valid_scrap_id", "scrap_transaction_id IS NOT NULL")
@dlt.expect_or_drop("valid_8d_id", "eight_d_id IS NOT NULL")
def scrap_8d_mapping_silver():
    """
    Transform scrap_8d_mapping_bronze to scrap_8d_mapping_silver.
    
    Standardizes IDs for join integrity.
    """
    return (
        spark.table(f"{SOURCE_SCHEMA}.scrap_8d_mapping_bronze")
        .withColumn("scrap_transaction_id", F.upper(F.trim(F.col("scrap_transaction_id"))))
        .withColumn("eight_d_id", F.upper(F.trim(F.col("eight_d_id"))))
        .withColumn("processed_at", F.current_timestamp())
    )
