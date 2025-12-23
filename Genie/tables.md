# Genie Space - Tables Configuration

Add these tables to your Genie Space for scrap analytics.

## Catalog & Schema
- **Catalog**: `scrap_catalog`
- **Schema**: `scrap_demo`

---

## Required Tables

### Fact Tables (Add These First)

| Table Name | Description | Key Columns |
|------------|-------------|-------------|
| `scrap_transactions_silver` | Main scrap events | transaction_id, part_id, plant_id, supplier_id, total_scrap_cost, quantity_scrapped, defect_category, shift, transaction_date |
| `eight_d_reports_silver` | Quality reports | report_number, title, severity, status, root_cause_category, problem_description, corrective_actions, days_open |

### Gold Aggregation Tables (For Performance)

| Table Name | Description | Key Columns |
|------------|-------------|-------------|
| `scrap_daily_summary_gold` | Daily rollups | transaction_date, total_scrap_cost, total_quantity_scrapped, unique_parts, unique_defects |
| `scrap_monthly_summary_gold` | Monthly rollups | year_month, transaction_year, transaction_month, total_scrap_cost |
| `scrap_by_plant_gold` | Plant summaries | plant_id, plant_name_lookup, total_scrap_cost, quantity_scrapped |
| `scrap_by_supplier_gold` | Supplier summaries | supplier_id, supplier_name_lookup, supplier_quality_score, total_scrap_cost |
| `scrap_by_defect_gold` | Defect pareto | defect_category, total_scrap_cost, quantity_scrapped |

### Scenario Tables (Optional)

| Table Name | Description | Key Columns |
|------------|-------------|-------------|
| `supplier_scenario_gold` | Supplier what-if data | supplier_id, supplier_name, quality_score, scrap_rate, annual_scrap_cost |
| `kpi_dashboard_gold` | KPI metrics | year, total_scrap_cost, total_qty_scrapped |

---

## Column Reference

### Cost Columns (Always Round)
```sql
-- Use this pattern for cost displays
ROUND(total_scrap_cost, 2) as cost
ROUND(SUM(total_scrap_cost), 2) as total_cost
```

### Lookup Columns (Denormalized)
The silver tables include pre-joined lookup columns with `_lookup` suffix:
- `plant_name_lookup` - Plant name from plants table
- `supplier_name_lookup` - Supplier name
- `supplier_tier_lookup` - Tier 1/2/3
- `supplier_quality_score` - Quality score (0-1)
- `part_name_lookup` - Part description
- `part_category_lookup` - Part category
- `defect_category` - Defect type
- `work_center_name_lookup` - Work center
- `equipment_name_lookup` - Equipment

### Date Columns
- `transaction_date` - Date of scrap event
- `transaction_datetime` - Full timestamp
- `transaction_year` - Year (INTEGER)
- `transaction_month` - Month (INTEGER)
- `transaction_quarter` - Quarter (INTEGER)
- `year_month` - YYYY-MM format (STRING)

### Quality Columns
- `supplier_quality_score` - 0.0 to 1.0 scale
- `equipment_oee` - OEE rating 0.0 to 1.0
- `severity` - 8D severity (Low/Medium/High/Critical)
- `verification_result` - 8D effectiveness

---

## Sample Queries for Genie

### Top Plants by Scrap Cost
```sql
SELECT 
    plant_name_lookup as Plant,
    CONCAT('$', FORMAT_NUMBER(ROUND(SUM(total_scrap_cost), 2), '#,##0.00')) as "Scrap Cost"
FROM scrap_catalog.scrap_demo.scrap_transactions_silver
WHERE transaction_year = YEAR(CURRENT_DATE())
GROUP BY plant_name_lookup
ORDER BY SUM(total_scrap_cost) DESC
```

### Supplier Quality Issues
```sql
SELECT 
    supplier_name_lookup as Supplier,
    supplier_tier_lookup as Tier,
    ROUND(AVG(supplier_quality_score) * 100, 1) as "Quality %",
    CONCAT('$', FORMAT_NUMBER(ROUND(SUM(total_scrap_cost), 2), '#,##0.00')) as "Scrap Cost"
FROM scrap_catalog.scrap_demo.scrap_transactions_silver
WHERE supplier_quality_score < 0.85
GROUP BY supplier_name_lookup, supplier_tier_lookup
ORDER BY SUM(total_scrap_cost) DESC
```

### Monthly Trend
```sql
SELECT 
    year_month as Month,
    CONCAT('$', FORMAT_NUMBER(ROUND(SUM(total_scrap_cost), 2), '#,##0.00')) as "Scrap Cost"
FROM scrap_catalog.scrap_demo.scrap_monthly_summary_gold
ORDER BY year_month DESC
LIMIT 12
```

### Open 8D Reports
```sql
SELECT 
    report_number as "8D #",
    title as Problem,
    severity as Severity,
    days_open as "Days Open",
    plant_name_lookup as Plant
FROM scrap_catalog.scrap_demo.eight_d_reports_silver
WHERE status = 'Open'
ORDER BY days_open DESC
```

---

## Entity Relationships

```
suppliers_silver ──┬──► parts_master_silver ──┐
                   │                          │
plants_silver ─────┼──► work_centers_silver ──┤
                   │         │                │
                   │    equipment_silver ─────┼───► scrap_transactions_silver
                   │                          │              │
operators_silver ──┼──────────────────────────┤              │
                   │                          │              ▼
defect_codes_silver┴──────────────────────────┘     eight_d_reports_silver
                                                            │
                                                            ▼
                                                   scrap_8d_mapping_silver
```

