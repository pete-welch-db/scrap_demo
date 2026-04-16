# Databricks notebook source
# MAGIC %md
# MAGIC # Scrap Data Demo - Setup & Data Model
# MAGIC 
# MAGIC ## Overview
# MAGIC 
# MAGIC This demo generates synthetic scrap data for an automotive Tier 1/2 supplier to demonstrate:
# MAGIC 1. **Scrap Forecasting** - Time series ML to predict future scrap quantities/costs
# MAGIC 2. **Scrap Reduction Policies** - Using 8D reports and scrap data to identify reduction strategies
# MAGIC 
# MAGIC ## Medallion Architecture
# MAGIC 
# MAGIC ```
# MAGIC ┌─────────────────────────────────────────────────────────────────────────────────┐
# MAGIC │                           BRONZE LAYER (Raw Data)                               │
# MAGIC │  - suppliers_bronze          - equipment_bronze        - scrap_transactions_bronze  │
# MAGIC │  - plants_bronze             - operators_bronze        - eight_d_reports_bronze     │
# MAGIC │  - work_centers_bronze       - cost_centers_bronze     - scrap_8d_mapping_bronze    │
# MAGIC │  - parts_master_bronze       - defect_codes_bronze                                  │
# MAGIC └─────────────────────────────────────────────────────────────────────────────────┘
# MAGIC                                        │
# MAGIC                                        ▼ DLT Pipeline
# MAGIC ┌─────────────────────────────────────────────────────────────────────────────────┐
# MAGIC │                         SILVER LAYER (Cleaned & Enriched)                       │
# MAGIC │  - Data quality validation       - Standardized data types                      │
# MAGIC │  - Denormalized lookups          - Business rule enforcement                    │
# MAGIC │  - Derived columns               - FK/PK validation                             │
# MAGIC └─────────────────────────────────────────────────────────────────────────────────┘
# MAGIC                                        │
# MAGIC                                        ▼ DLT Pipeline
# MAGIC ┌─────────────────────────────────────────────────────────────────────────────────┐
# MAGIC │                          GOLD LAYER (Analytics Ready)                           │
# MAGIC │  Aggregations:                    ML Feature Tables:                            │
# MAGIC │  - scrap_daily_summary_gold       - ml_scrap_forecast_features_gold             │
# MAGIC │  - scrap_monthly_summary_gold     - ml_scrap_policy_features_gold               │
# MAGIC │  - scrap_by_plant_gold            - eight_d_text_analysis_gold                  │
# MAGIC │  - scrap_by_supplier_gold                                                       │
# MAGIC │  - scrap_by_defect_gold           Dashboards:                                   │
# MAGIC │  - scrap_by_equipment_gold        - kpi_dashboard_gold                          │
# MAGIC │  - eight_d_summary_gold           - supplier_scorecard_gold                     │
# MAGIC └─────────────────────────────────────────────────────────────────────────────────┘
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## Entity Relationship Diagram
# MAGIC 
# MAGIC ```
# MAGIC ┌──────────────────┐       ┌──────────────────┐       ┌──────────────────┐
# MAGIC │    SUPPLIERS     │       │     PLANTS       │       │  COST_CENTERS    │
# MAGIC ├──────────────────┤       ├──────────────────┤       ├──────────────────┤
# MAGIC │ PK supplier_id   │       │ PK plant_id      │       │ PK cost_center_id│
# MAGIC │    supplier_name │       │    plant_name    │       │    cc_name       │
# MAGIC │    supplier_tier │       │    city/country  │       │    department    │
# MAGIC │    quality_score │       │    capacity      │       │    budget        │
# MAGIC │    specialization│       │    year_opened   │       └──────────────────┘
# MAGIC └────────┬─────────┘       └────────┬─────────┘
# MAGIC          │                          │
# MAGIC          │                          │
# MAGIC          ▼                          ▼
# MAGIC ┌──────────────────┐       ┌──────────────────┐       ┌──────────────────┐
# MAGIC │   PARTS_MASTER   │       │   WORK_CENTERS   │       │    OPERATORS     │
# MAGIC ├──────────────────┤       ├──────────────────┤       ├──────────────────┤
# MAGIC │ PK part_id       │       │ PK work_center_id│       │ PK operator_id   │
# MAGIC │ FK supplier_id   │───┐   │ FK plant_id      │───┐   │ FK plant_id      │───┐
# MAGIC │    part_number   │   │   │    wc_type       │   │   │    skill_level   │   │
# MAGIC │    part_category │   │   │    capacity      │   │   │    shift         │   │
# MAGIC │    unit_cost     │   │   └────────┬─────────┘   │   │    experience    │   │
# MAGIC └────────┬─────────┘   │            │             │   └────────┬─────────┘   │
# MAGIC          │             │            │             │            │             │
# MAGIC          │             │            ▼             │            │             │
# MAGIC          │             │   ┌──────────────────┐   │            │             │
# MAGIC          │             │   │    EQUIPMENT     │   │            │             │
# MAGIC          │             │   ├──────────────────┤   │            │             │
# MAGIC          │             │   │ PK equipment_id  │   │            │             │
# MAGIC          │             │   │ FK work_center_id│───┤            │             │
# MAGIC          │             │   │ FK plant_id      │───┘            │             │
# MAGIC          │             │   │    age_years     │                │             │
# MAGIC          │             │   │    oee_rating    │                │             │
# MAGIC          │             │   └────────┬─────────┘                │             │
# MAGIC          │             │            │                          │             │
# MAGIC          │             │            │                          │             │
# MAGIC          ▼             ▼            ▼                          ▼             │
# MAGIC ┌──────────────────────────────────────────────────────────────────────────────┐
# MAGIC │                           SCRAP_TRANSACTIONS                                  │
# MAGIC ├──────────────────────────────────────────────────────────────────────────────┤
# MAGIC │ PK scrap_transaction_id                                                       │
# MAGIC │ FK part_id ─────────────────► parts_master                                    │
# MAGIC │ FK supplier_id ─────────────► suppliers                                       │
# MAGIC │ FK plant_id ────────────────► plants                                          │
# MAGIC │ FK work_center_id ──────────► work_centers                                    │
# MAGIC │ FK equipment_id ────────────► equipment                                       │
# MAGIC │ FK operator_id ─────────────► operators                                       │
# MAGIC │ FK defect_code ─────────────► defect_codes                                    │
# MAGIC │ FK cost_center_id ──────────► cost_centers                                    │
# MAGIC │    transaction_datetime, shift, quantity_scrapped, total_scrap_cost           │
# MAGIC │    disposition, root_cause_category, lot_number, work_order                   │
# MAGIC └──────────────────────────────────────────────────────────────────────────────┘
# MAGIC                                        │
# MAGIC                                        │ 1:N via mapping table
# MAGIC                                        ▼
# MAGIC ┌──────────────────────────────────────────────────────────────────────────────┐
# MAGIC │                              SCRAP_8D_MAPPING                                 │
# MAGIC ├──────────────────────────────────────────────────────────────────────────────┤
# MAGIC │ FK scrap_transaction_id ────► scrap_transactions                              │
# MAGIC │ FK eight_d_id ──────────────► eight_d_reports                                 │
# MAGIC └──────────────────────────────────────────────────────────────────────────────┘
# MAGIC                                        │
# MAGIC                                        │
# MAGIC                                        ▼
# MAGIC ┌──────────────────────────────────────────────────────────────────────────────┐
# MAGIC │                              EIGHT_D_REPORTS                                  │
# MAGIC ├──────────────────────────────────────────────────────────────────────────────┤
# MAGIC │ PK eight_d_id                                                                 │
# MAGIC │ FK plant_id ────────────────► plants                                          │
# MAGIC │ FK supplier_id ─────────────► suppliers                                       │
# MAGIC │ FK work_center_id ──────────► work_centers                                    │
# MAGIC │ FK primary_part_id ─────────► parts_master                                    │
# MAGIC │    problem_description (TEXT), root_cause_analysis (TEXT)                     │
# MAGIC │    corrective_actions (TEXT), preventive_actions (TEXT)                       │
# MAGIC │    status, severity, days_open, verification_result                           │
# MAGIC └──────────────────────────────────────────────────────────────────────────────┘
# MAGIC 
# MAGIC ┌──────────────────┐
# MAGIC │  DEFECT_CODES    │  (Hierarchical - 2 levels)
# MAGIC ├──────────────────┤
# MAGIC │ PK defect_code   │
# MAGIC │ FK parent_code   │──► self-reference
# MAGIC │    defect_name   │
# MAGIC │    category      │
# MAGIC │    severity      │
# MAGIC └──────────────────┘
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Volume Summary
# MAGIC 
# MAGIC | Table | Record Count | Notes |
# MAGIC |-------|-------------|-------|
# MAGIC | suppliers_bronze | 45 | ~15% flagged as problematic for ML patterns |
# MAGIC | plants_bronze | 6 | North America (USA, Canada, Mexico) |
# MAGIC | work_centers_bronze | 48 | 8 per plant |
# MAGIC | equipment_bronze | ~180 | 3-5 per work center |
# MAGIC | operators_bronze | 300 | 50 per plant |
# MAGIC | cost_centers_bronze | 20 | Standard manufacturing cost centers |
# MAGIC | defect_codes_bronze | ~45 | Hierarchical (8 categories, ~5 codes each) |
# MAGIC | parts_master_bronze | 500 | 7 part categories |
# MAGIC | scrap_transactions_bronze | 60,000 | 3 years of history |
# MAGIC | eight_d_reports_bronze | ~Variable | One 8D can cover multiple scrap records |
# MAGIC | scrap_8d_mapping_bronze | 60,000 | 1:1 with scrap transactions |

# COMMAND ----------

# MAGIC %md
# MAGIC ## Embedded ML Patterns
# MAGIC 
# MAGIC The synthetic data includes realistic patterns for ML models to discover:
# MAGIC 
# MAGIC ### 1. Seasonal Patterns
# MAGIC - **Summer spike (July-August)**: Model year changeover period - 40% increase in scrap
# MAGIC - **Winter spike (December-January)**: Holiday skeleton crews - 25% increase
# MAGIC - **Spring ramp-up (March-April)**: Production increases - 15% increase
# MAGIC 
# MAGIC ### 2. Day-of-Week Patterns
# MAGIC - **Monday effect**: Startup issues cause 20% higher scrap rates
# MAGIC - **Weekend effect**: Lower volume but similar patterns
# MAGIC 
# MAGIC ### 3. Shift Patterns
# MAGIC - **3rd shift (Night)**: 15% higher scrap rates due to fatigue/supervision
# MAGIC 
# MAGIC ### 4. Supplier Quality Correlation
# MAGIC - ~15% of suppliers flagged as "problematic" with lower quality scores
# MAGIC - Parts from problematic suppliers have 20-50% higher scrap rates
# MAGIC 
# MAGIC ### 5. Equipment Age Correlation
# MAGIC - Older equipment (>12 years) has lower OEE and higher scrap rates
# MAGIC - Equipment age correlates with plant age
# MAGIC 
# MAGIC ### 6. Operator Experience Correlation
# MAGIC - ~10% of operators flagged as "high error" for ML validation
# MAGIC - Higher scrap rates from inexperienced operators

# COMMAND ----------

# MAGIC %md
# MAGIC ## Running the Notebooks
# MAGIC 
# MAGIC ### Step 1: Generate Bronze Data
# MAGIC Run notebook `1_Generate_Bronze_Data.py`:
# MAGIC - Installs faker library
# MAGIC - Creates all bronze tables in `scrap_catalog.scrap_demo`
# MAGIC - Generates 60K scrap transactions with embedded ML patterns
# MAGIC - Creates 8D reports linked to scrap records
# MAGIC 
# MAGIC ### Step 2: Create DLT Pipelines
# MAGIC Create two DLT pipelines in Databricks:
# MAGIC 
# MAGIC **Pipeline 1: Silver Transformations**
# MAGIC ```
# MAGIC Name: scrap_demo_silver_pipeline
# MAGIC Source: 2_DLT_Silver_Pipeline.py
# MAGIC Target Schema: scrap_catalog.scrap_demo
# MAGIC ```
# MAGIC 
# MAGIC **Pipeline 2: Gold Aggregations**
# MAGIC ```
# MAGIC Name: scrap_demo_gold_pipeline
# MAGIC Source: 3_DLT_Gold_Pipeline.py
# MAGIC Target Schema: scrap_catalog.scrap_demo
# MAGIC Dependencies: scrap_demo_silver_pipeline (run after silver)
# MAGIC ```
# MAGIC 
# MAGIC ### Step 3: Verify Data
# MAGIC Run the validation queries below to confirm data generation.

# COMMAND ----------

# Configuration - Use widget parameters from job, with defaults for interactive use
dbutils.widgets.text("catalog", "scrap_catalog", "Catalog Name")
dbutils.widgets.text("schema", "scrap_demo", "Schema Name")

CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
FULL_SCHEMA = f"{CATALOG}.{SCHEMA}"

print(f"Using catalog: {CATALOG}, schema: {SCHEMA}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Validation Queries

# COMMAND ----------

# Check all tables exist and row counts
tables = [
    "suppliers_bronze", "plants_bronze", "work_centers_bronze", "equipment_bronze",
    "operators_bronze", "cost_centers_bronze", "defect_codes_bronze", "parts_master_bronze",
    "scrap_transactions_bronze", "eight_d_reports_bronze", "scrap_8d_mapping_bronze"
]

print(f"{'Table':<40} {'Row Count':>15}")
print("-" * 55)

for table in tables:
    try:
        count = spark.table(f"{FULL_SCHEMA}.{table}").count()
        print(f"{table:<40} {count:>15,}")
    except Exception as e:
        print(f"{table:<40} {'NOT FOUND':>15}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Sample Queries for Analysis

# COMMAND ----------

# Top 10 defect codes by scrap cost (uses job widget catalog/schema, not hardcoded catalog)
display(
    spark.sql(
        f"""
SELECT
    defect_code,
    defect_category,
    COUNT(*) as scrap_records,
    SUM(quantity_scrapped) as total_qty,
    ROUND(SUM(total_scrap_cost), 2) as total_cost
FROM {FULL_SCHEMA}.scrap_transactions_bronze
GROUP BY defect_code, defect_category
ORDER BY total_cost DESC
LIMIT 10
"""
    )
)

# COMMAND ----------

display(
    spark.sql(
        f"""
SELECT
    DATE_TRUNC('month', transaction_date) as month,
    SUM(quantity_scrapped) as total_qty,
    ROUND(SUM(total_scrap_cost), 2) as total_cost,
    COUNT(DISTINCT part_id) as unique_parts
FROM {FULL_SCHEMA}.scrap_transactions_bronze
GROUP BY DATE_TRUNC('month', transaction_date)
ORDER BY month
"""
    )
)

# COMMAND ----------

display(
    spark.sql(
        f"""
SELECT
    s.supplier_name,
    s.supplier_tier,
    s.quality_score,
    COUNT(*) as scrap_records,
    SUM(t.quantity_scrapped) as total_qty_scrapped,
    ROUND(SUM(t.total_scrap_cost), 2) as total_scrap_cost
FROM {FULL_SCHEMA}.scrap_transactions_bronze t
JOIN {FULL_SCHEMA}.suppliers_bronze s ON t.supplier_id = s.supplier_id
GROUP BY s.supplier_name, s.supplier_tier, s.quality_score
ORDER BY total_scrap_cost DESC
LIMIT 15
"""
    )
)

# COMMAND ----------

display(
    spark.sql(
        f"""
SELECT
    status,
    severity,
    COUNT(*) as num_reports,
    ROUND(AVG(total_scrap_cost), 2) as avg_scrap_cost,
    ROUND(AVG(num_parts_affected), 1) as avg_parts_affected
FROM {FULL_SCHEMA}.eight_d_reports_bronze
GROUP BY status, severity
ORDER BY status, severity
"""
    )
)

# COMMAND ----------

display(
    spark.sql(
        f"""
SELECT
    eight_d_id,
    title,
    defect_category,
    severity,
    LEFT(root_cause_analysis, 300) as root_cause_preview,
    LEFT(corrective_actions, 300) as corrective_action_preview
FROM {FULL_SCHEMA}.eight_d_reports_bronze
WHERE status = 'Closed'
LIMIT 5
"""
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next Steps
# MAGIC 
# MAGIC After data generation and DLT pipeline execution:
# MAGIC 
# MAGIC 1. **ML - Scrap Forecasting**
# MAGIC    - Use `ml_scrap_forecast_features_gold` table
# MAGIC    - Time series models: Prophet, ARIMA, or deep learning
# MAGIC    - Predict daily/weekly scrap quantities and costs
# MAGIC 
# MAGIC 2. **ML - Scrap Reduction Policies**
# MAGIC    - Use `ml_scrap_policy_features_gold` table
# MAGIC    - Classification/clustering to identify reduction opportunities
# MAGIC    - Combine with 8D text analysis for root cause patterns
# MAGIC 
# MAGIC 3. **LLM - 8D Text Analysis**
# MAGIC    - Use `eight_d_text_analysis_gold` table
# MAGIC    - Summarization of root cause and corrective actions
# MAGIC    - RAG system for querying historical 8D patterns
# MAGIC 
# MAGIC 4. **Dashboards**
# MAGIC    - Use `kpi_dashboard_gold` for executive metrics
# MAGIC    - Use `supplier_scorecard_gold` for vendor management
# MAGIC    - Time series visualizations from `scrap_monthly_summary_gold`

