# Databricks notebook source
# MAGIC %md
# MAGIC # Refresh Lakeview Dashboard
# MAGIC Publishes the Executive Scrap Dashboard to refresh cached data after pipeline runs.

# COMMAND ----------

import requests
import json

# Get workspace context
host = spark.conf.get("spark.databricks.workspaceUrl", "")
token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

# Dashboard and warehouse IDs (passed as parameters)
dashboard_id = dbutils.widgets.get("dashboard_id")
warehouse_id = dbutils.widgets.get("warehouse_id")

print(f"Refreshing dashboard: {dashboard_id}")
print(f"Using warehouse: {warehouse_id}")

# COMMAND ----------

# Publish the dashboard to force a data refresh
response = requests.post(
    f"https://{host}/api/2.0/lakeview/dashboards/{dashboard_id}/published",
    headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
    json={
        "embed_credentials": True,
        "warehouse_id": warehouse_id
    }
)

if response.status_code == 200:
    result = response.json()
    print(f"Dashboard published successfully: {result.get('display_name', 'OK')}")
else:
    print(f"Warning: Dashboard publish returned {response.status_code}: {response.text}")
    # Don't fail the job if dashboard refresh fails
    print("Continuing - dashboard can be manually refreshed")
