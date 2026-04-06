# 🚀 Databricks Asset Bundle Deployment Guide

This project uses **Databricks Asset Bundles (DAB)** for automated deployment of all resources to Databricks.

## ⚡ Serverless Architecture

This bundle is configured to use **serverless compute** throughout:

| Resource | Compute Type |
|----------|--------------|
| Jobs/Workflows | Serverless Job Compute |
| DLT Pipeline | Serverless DLT |
| SQL Warehouse | Shared Unity Catalog Serverless |
| Databricks App | Serverless (default) |

**Benefits:**
- ⚡ Near-instant startup (no cluster provisioning)
- 💰 Pay-per-use pricing
- 🔧 Zero infrastructure management
- 📈 Automatic scaling

## 📋 Prerequisites

1. **Databricks CLI v0.200+** installed and configured
   ```bash
   # Install Databricks CLI
   curl -fsSL https://raw.githubusercontent.com/databricks/setup-cli/main/install.sh | sh
   
   # Verify installation
   databricks --version
   ```

2. **Authentication** configured via one of:
   - `databricks configure` (interactive setup)
   - Environment variables (`DATABRICKS_HOST`, `DATABRICKS_TOKEN`)
   - OAuth with `databricks auth login`

3. **Workspace Requirements**:
   - Unity Catalog enabled
   - Serverless compute enabled for workspace
   - "Shared Unity Catalog Serverless" SQL Warehouse available
   - Appropriate permissions for creating jobs, pipelines, and apps
   - **Catalog must exist** before first deployment (the DLT pipeline references it)
   
   To create the catalog manually before first deployment:
   ```sql
   CREATE CATALOG IF NOT EXISTS scrap_catalog;
   CREATE SCHEMA IF NOT EXISTS scrap_catalog.scrap_demo;
   ```

## 📁 Bundle Structure

```
scrap_demo/
├── databricks.yml              # Main bundle configuration
├── resources/
│   ├── jobs.yml               # Serverless workflow definitions
│   ├── pipelines.yml          # Serverless DLT pipeline configuration
│   ├── apps.yml               # Streamlit app deployment
│   └── dashboards.yml         # Lakeview dashboard deployment
├── Setup and Bronze Create/    # Data setup notebooks
├── Scrap DLT Pipeline/        # DLT Silver/Gold notebooks
├── ML Models/                 # ML training notebooks
├── Dashboards/                # Dashboard JSON files
└── app/                       # Streamlit application
```

## 🎯 Deployment Targets

| Target | Catalog | Schema | Usage |
|--------|---------|--------|-------|
| `dev` | `scrap` | `scrap_demo` | Development/testing (default) |

> **Note**: Staging and prod targets are commented out in `databricks.yml`. Uncomment and configure them as needed.

## 🔧 Quick Start Commands

### One-Command Deploy (Recommended)
```bash
# Deploy and run full pipeline automatically
./deploy.sh dev
```

This script will:
1. Validate the bundle
2. Deploy all resources (job, DLT pipeline, app, dashboard)
3. Run the complete pipeline (Setup → DLT → ML Training)

### Manual Deployment

#### Validate Bundle Configuration
```bash
# Check for syntax errors and validate resources
databricks bundle validate
```

#### Deploy to Development
```bash
# Deploy all resources to dev environment
databricks bundle deploy --target dev

# Then run setup job manually
databricks bundle run scrap_setup_job --target dev
```

#### Deploy to Production
```bash
# Deploy to production (requires additional permissions)
databricks bundle deploy --target prod
```

### Run Jobs Manually
```bash
# Step 1: Run setup job (creates catalog, schema, bronze data)
databricks bundle run scrap_setup_job --target dev

# Step 2: Run pipeline job (DLT + ML) - requires setup to complete first
databricks bundle run scrap_pipeline_job --target dev
```

### Destroy Deployment
```bash
# Remove all deployed resources
databricks bundle destroy --target dev
```

## 📊 Deployed Resources

### Jobs/Workflows (Serverless)
| Job Name | Description | Run Order |
|----------|-------------|-----------|
| `scrap_setup_job` | Creates catalog, schema, bronze data | Run FIRST |
| `scrap_pipeline_job` | DLT pipeline + ML training | Run SECOND |

**Two-Stage Deployment:**
1. **Setup Job** - Creates catalog/schema and generates bronze data
2. **Pipeline Job** - DLT transformations (Silver/Gold) + ML model training

### DLT Pipeline (Serverless)
- **Name**: Scrap Reduction - DLT Pipeline
- **Layers**: Bronze → Silver → Gold
- **Features**: Serverless, Photon enabled, auto-scaling

### Databricks App
- **Name**: scrap-reduction-{target}
- **Type**: Streamlit application
- **SQL Warehouse**: Shared Unity Catalog Serverless

### Dashboard
- **Name**: Executive Scrap Dashboard
- **Type**: Lakeview (AI/BI Dashboard)

## ⚙️ Configuration Variables

Override variables per environment in `databricks.yml` or via CLI:

```bash
# Override catalog for deployment
databricks bundle deploy --target dev --var="catalog=my_catalog"
```

| Variable | Description | Default |
|----------|-------------|---------|
| `catalog` | Unity Catalog name | `scrap_catalog` |
| `schema` | Schema for tables | `scrap_demo` |
| `warehouse_name` | Serverless SQL Warehouse | `Shared Unity Catalog Serverless` |

## 🔐 Production Configuration

For production deployments, update `databricks.yml`:

1. **Service Principal**: Update `run_as.service_principal_name`
2. **Permissions**: Configure appropriate group access
3. **Notifications**: Add production email recipients

```yaml
targets:
  prod:
    run_as:
      service_principal_name: your-prod-sp
    permissions:
      - level: CAN_MANAGE
        group_name: scrap-platform-admins
```

## 🔄 CI/CD Integration

### GitHub Actions Example
```yaml
name: Deploy Scrap Platform
on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Install Databricks CLI
        run: curl -fsSL https://raw.githubusercontent.com/databricks/setup-cli/main/install.sh | sh
      
      - name: Deploy to Production
        env:
          DATABRICKS_HOST: ${{ secrets.DATABRICKS_HOST }}
          DATABRICKS_TOKEN: ${{ secrets.DATABRICKS_TOKEN }}
        run: |
          databricks bundle validate
          databricks bundle deploy --target prod
```

## 🐛 Troubleshooting

### Common Issues

1. **Authentication Failed**
   ```bash
   # Re-authenticate
   databricks auth login --host https://your-workspace.databricks.com
   ```

2. **Serverless Not Available**
   - Ensure serverless compute is enabled for your workspace
   - Check with your Databricks admin

3. **SQL Warehouse Not Found**
   - Verify "Shared Unity Catalog Serverless" warehouse exists
   - Or override with: `--var="warehouse_name=Your Warehouse Name"`

4. **Permission Denied**
   - Ensure your user/SP has workspace admin or appropriate job/pipeline permissions

5. **DLT Pipeline Fails**
   - Check catalog/schema permissions
   - Verify source tables exist (run setup job first)

### View Deployment State
```bash
# Check what's deployed
databricks bundle summary --target dev
```

## 📞 Support

For questions about this deployment, contact the Data Engineering team.
