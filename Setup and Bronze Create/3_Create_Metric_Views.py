# Databricks notebook source
# MAGIC %md
# MAGIC # Create UC Metric Views
# MAGIC Creates Unity Catalog Metric Views for the Scrap Reduction Platform.
# MAGIC These provide centralized, governed KPI definitions consumed by Lakeview dashboards, Genie spaces, and the Streamlit app.

# COMMAND ----------

# Parameters
dbutils.widgets.text("catalog", "scrap_catalog", "Catalog")
dbutils.widgets.text("schema", "scrap_demo", "Schema")

CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")

spark.sql(f"USE CATALOG {CATALOG}")
spark.sql(f"USE SCHEMA {SCHEMA}")

print(f"Creating metric views in {CATALOG}.{SCHEMA}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Scrap Operations Metric View
# MAGIC Core scrap KPIs: cost, volume, incidents — sliceable by plant, defect, supplier, shift, and time.

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE VIEW {CATALOG}.{SCHEMA}.scrap_operations_metrics
WITH METRICS
LANGUAGE YAML
AS $$
  version: 1.1
  comment: "Core scrap operations KPIs for manufacturing analytics. Sliceable by plant, defect category, supplier, shift, and time period."
  source: {CATALOG}.{SCHEMA}.scrap_transactions_silver
  filter: transaction_date IS NOT NULL
  dimensions:
    - name: Transaction Month
      expr: DATE_TRUNC('MONTH', transaction_date)
      comment: "Month of scrap transaction for trend analysis"
    - name: Transaction Year
      expr: YEAR(transaction_date)
      comment: "Year of scrap transaction"
    - name: Plant
      expr: plant_name_lookup
      comment: "Manufacturing plant where scrap occurred"
    - name: Defect Category
      expr: defect_category
      comment: "Category of defect that caused scrap"
    - name: Root Cause
      expr: root_cause_category
      comment: "Root cause classification"
    - name: Supplier
      expr: supplier_name_lookup
      comment: "Supplier of the scrapped part"
    - name: Shift
      expr: "CASE WHEN shift = 1 THEN 'Day (6am-2pm)' WHEN shift = 2 THEN 'Evening (2pm-10pm)' WHEN shift = 3 THEN 'Night (10pm-6am)' END"
      comment: "Production shift: Day, Evening, or Night"
    - name: Part Category
      expr: part_category_lookup
      comment: "Category of the scrapped part"
  measures:
    - name: Total Scrap Cost
      expr: SUM(total_scrap_cost)
      comment: "Total dollar cost of scrapped parts (USD)"
    - name: Total Units Scrapped
      expr: SUM(quantity_scrapped)
      comment: "Total number of units scrapped"
    - name: Scrap Incidents
      expr: COUNT(1)
      comment: "Number of scrap transaction events"
    - name: Avg Scrap Cost per Incident
      expr: SUM(total_scrap_cost) / COUNT(1)
      comment: "Average cost per scrap incident"
    - name: Avg Unit Cost
      expr: SUM(total_scrap_cost) / SUM(quantity_scrapped)
      comment: "Average cost per scrapped unit"
    - name: Unique Parts Affected
      expr: COUNT(DISTINCT part_id)
      comment: "Number of distinct parts involved in scrap"
    - name: Unique Suppliers Involved
      expr: COUNT(DISTINCT supplier_id)
      comment: "Number of distinct suppliers with scrapped parts"
    - name: YTD Scrap Cost
      expr: SUM(total_scrap_cost) FILTER (WHERE YEAR(transaction_date) = YEAR(CURRENT_DATE()))
      comment: "Year-to-date total scrap cost"
    - name: YTD Units Scrapped
      expr: SUM(quantity_scrapped) FILTER (WHERE YEAR(transaction_date) = YEAR(CURRENT_DATE()))
      comment: "Year-to-date total units scrapped"
$$
""")

print("Created: scrap_operations_metrics")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Supplier Quality Metric View
# MAGIC Supplier-level KPIs: quality scores, scrap rates, delivery performance.

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE VIEW {CATALOG}.{SCHEMA}.supplier_quality_metrics
WITH METRICS
LANGUAGE YAML
AS $$
  version: 1.1
  comment: "Supplier quality and performance KPIs. Measures quality scores, scrap rates, and delivery across supplier tiers and regions."
  source: {CATALOG}.{SCHEMA}.supplier_scenario_gold
  filter: parts_supplied > 0
  dimensions:
    - name: Supplier
      expr: supplier_name
      comment: "Supplier name"
    - name: Supplier Tier
      expr: supplier_tier
      comment: "Supplier tier: Tier 1 (direct), Tier 2, or Tier 3"
    - name: Specialization
      expr: specialization
      comment: "Supplier product specialization"
    - name: Region
      expr: region
      comment: "Supplier geographic region"
    - name: Is Preferred
      expr: "CASE WHEN is_preferred THEN 'Preferred' ELSE 'Standard' END"
      comment: "Whether supplier is a preferred vendor"
  measures:
    - name: Avg Quality Score
      expr: SUM(quality_score * parts_supplied) / SUM(parts_supplied)
      comment: "Weighted average quality score (0-1 scale, multiply by 100 for percentage)"
    - name: Avg On-Time Delivery
      expr: SUM(on_time_delivery_rate * parts_supplied) / SUM(parts_supplied)
      comment: "Weighted average on-time delivery rate"
    - name: Total Annual Scrap Cost
      expr: SUM(annual_scrap_cost)
      comment: "Total annual scrap cost across suppliers (USD)"
    - name: Total Annual Qty Scrapped
      expr: SUM(annual_qty_scrapped)
      comment: "Total annual quantity scrapped"
    - name: Avg Scrap Rate
      expr: SUM(scrap_rate * parts_supplied) / SUM(parts_supplied)
      comment: "Weighted average scrap rate"
    - name: Avg Risk Score
      expr: SUM(risk_score * parts_supplied) / SUM(parts_supplied)
      comment: "Weighted average supplier risk score"
    - name: Supplier Count
      expr: COUNT(DISTINCT supplier_id)
      comment: "Number of active suppliers"
    - name: Total Parts Supplied
      expr: SUM(parts_supplied)
      comment: "Total number of parts supplied"
    - name: Total Available Capacity
      expr: SUM(available_capacity)
      comment: "Aggregate available production capacity"
$$
""")

print("Created: supplier_quality_metrics")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8D Quality Reports Metric View
# MAGIC Root cause investigation KPIs: report volume, closure rates, effectiveness.

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE VIEW {CATALOG}.{SCHEMA}.eight_d_quality_metrics
WITH METRICS
LANGUAGE YAML
AS $$
  version: 1.1
  comment: "8D quality report KPIs for root cause tracking. Measures report volume, closure rates, and corrective action effectiveness."
  source: {CATALOG}.{SCHEMA}.eight_d_reports_silver
  dimensions:
    - name: Report Month
      expr: DATE_TRUNC('MONTH', date_opened)
      comment: "Month the 8D report was filed"
    - name: Status
      expr: status
      comment: "Report status: Open, In Progress, Closed"
    - name: Priority
      expr: priority
      comment: "Report priority level"
    - name: Defect Category
      expr: defect_category
      comment: "Type of defect investigated"
  measures:
    - name: Total Reports
      expr: COUNT(1)
      comment: "Total number of 8D reports"
    - name: Open Reports
      expr: COUNT(1) FILTER (WHERE status = 'Open')
      comment: "Number of currently open 8D reports"
    - name: Closed Reports
      expr: COUNT(1) FILTER (WHERE status = 'Closed')
      comment: "Number of closed 8D reports"
    - name: Closure Rate
      expr: COUNT(1) FILTER (WHERE status = 'Closed') * 1.0 / COUNT(1)
      comment: "Percentage of reports that have been closed"
    - name: YTD Open Reports
      expr: COUNT(1) FILTER (WHERE status != 'Closed' AND YEAR(date_opened) = YEAR(CURRENT_DATE()))
      comment: "Year-to-date open 8D reports"
$$
""")

print("Created: eight_d_quality_metrics")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Grant SP Access to Metric Views

# COMMAND ----------

# Grant the app service principal SELECT on the metric views
SP_CLIENT_ID = "cb92b8a6-f086-4365-9bce-17c57aedfa1e"

for mv in ["scrap_operations_metrics", "supplier_quality_metrics", "eight_d_quality_metrics"]:
    spark.sql(f"GRANT SELECT ON VIEW {CATALOG}.{SCHEMA}.{mv} TO `{SP_CLIENT_ID}`")
    print(f"Granted SELECT on {mv} to app SP")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Validate Metric Views

# COMMAND ----------

# Test each metric view
print("=== Scrap Operations Metrics ===")
display(spark.sql(f"""
SELECT
    `Plant`,
    MEASURE(`Total Scrap Cost`) as total_cost,
    MEASURE(`Total Units Scrapped`) as total_units,
    MEASURE(`Scrap Incidents`) as incidents
FROM {CATALOG}.{SCHEMA}.scrap_operations_metrics
WHERE `Transaction Year` = YEAR(CURRENT_DATE())
GROUP BY `Plant`
ORDER BY total_cost DESC
"""))

# COMMAND ----------

print("=== Supplier Quality Metrics ===")
display(spark.sql(f"""
SELECT
    `Supplier Tier`,
    MEASURE(`Supplier Count`) as suppliers,
    MEASURE(`Avg Quality Score`) as avg_quality,
    MEASURE(`Total Annual Scrap Cost`) as scrap_cost
FROM {CATALOG}.{SCHEMA}.supplier_quality_metrics
GROUP BY `Supplier Tier`
"""))

# COMMAND ----------

print("=== 8D Quality Metrics ===")
display(spark.sql(f"""
SELECT
    `Status`,
    MEASURE(`Total Reports`) as total,
    MEASURE(`Closure Rate`) as closure_rate
FROM {CATALOG}.{SCHEMA}.eight_d_quality_metrics
GROUP BY `Status`
"""))

# COMMAND ----------

print("All metric views created and validated successfully!")
print(f"""
Metric views available in {CATALOG}.{SCHEMA}:
  - scrap_operations_metrics  (core scrap KPIs by plant/defect/supplier/shift/time)
  - supplier_quality_metrics  (supplier quality/risk/delivery KPIs by tier/region)
  - eight_d_quality_metrics   (8D report KPIs by status/priority/defect)

These can be added to:
  - Lakeview dashboards as datasets (use MEASURE() function)
  - Genie spaces as table sources
  - SQL queries from any editor
""")
