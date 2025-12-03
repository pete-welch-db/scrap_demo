# Databricks Genie Configuration

This folder contains configuration files for the Databricks Genie Space.

## Files

| File | Purpose |
|------|---------|
| `instructions_short.txt` | **Use this** - Optimized instructions for Genie (avoids "too long" warning) |
| `instructions.md` | Full documentation of instructions |
| `tables.md` | Complete table reference with columns, relationships, and sample queries |

## Setup

### 1. Create Genie Space
- Go to Databricks > Machine Learning > Genie
- Create new space: "Scrap Analytics Assistant"

### 2. Add Tables
Add these tables from `welch.scrap_demo`:
- `scrap_transactions_silver` ✅ (required)
- `eight_d_reports_silver` ✅ (required)
- `scrap_by_plant_gold`
- `scrap_by_supplier_gold`
- `scrap_monthly_summary_gold`
- `supplier_scenario_gold`

### 3. Add Instructions
Copy contents of `instructions_short.txt` into:
**Settings > Instructions**

### 4. Test Queries
Try these:
- "What's our total scrap cost this year?"
- "Show top 5 plants by scrap cost"
- "Which suppliers have quality below 85%?"
- "How many 8D reports are open?"

## Space ID
Current Space ID: `01f0d05d42ed11eeae85802c1d5bcccd`

Update this in `.env` file:
```
GENIE_SPACE_ID=01f0d05d42ed11eeae85802c1d5bcccd
```

