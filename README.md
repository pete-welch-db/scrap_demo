# 🔧 Scrap Reduction Intelligence Platform

A comprehensive Databricks demo showcasing end-to-end scrap reduction analytics for automotive manufacturing, including data generation, ML models, dashboards, and a customer-facing Streamlit application.

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         DATA GENERATION & BRONZE                             │
│  📁 Setup and Bronze Create/                                                 │
│  - 0_Setup_and_Data_Model.py     → Schema creation, ERD documentation        │
│  - 1_Generate_Bronze_Data.py     → Synthetic data: 60K records, 3 years      │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         DLT PIPELINES                                        │
│  📁 Scrap DLT Pipeline/                                                      │
│  - 2_DLT_Silver_Pipeline.py      → Data quality, enrichment, denormalization │
│  - 3_DLT_Gold_Pipeline.py        → Aggregations, ML features, KPIs           │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         MACHINE LEARNING                                     │
│  📁 ML Models/                                                               │
│  - 1_Scrap_Forecasting.py        → Prophet time series, 3-month forecasts    │
│  - 2_Root_Cause_8D_Validation.py → NLP + classification, LLM analysis        │
│  - 3_Advanced_ML_Models.py       → Anomaly detection, clustering, risk       │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         VISUALIZATION & APPS                                 │
│  📁 Dashboards/                                                              │
│  - Enhanced_Executive_Dashboard.lvdash.json → 6-page AI/BI Dashboard         │
│                                                                              │
│  📁 app/                                                                     │
│  - Streamlit app with What-If scenarios, ML insights, Genie integration      │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 📊 Data Model

### Entity Relationship
```
SUPPLIERS ─────► PARTS_MASTER ─────┐
                                    │
PLANTS ────────► WORK_CENTERS ─────┤
                     │              │
                EQUIPMENT ─────────┼───► SCRAP_TRANSACTIONS
                                   │            │
OPERATORS ─────────────────────────┤            │
                                   │            ▼
DEFECT_CODES ──────────────────────┤     EIGHT_D_REPORTS
                                   │            │
COST_CENTERS ──────────────────────┘            │
                                                ▼
                                        SCRAP_8D_MAPPING
```

### Data Volume
| Entity | Records | Notes |
|--------|---------|-------|
| Parts Master | 500 | Automotive parts with categories |
| Scrap Transactions | 60,000 | 3 years of history |
| 8D Reports | ~1,200 | One 8D can cover multiple parts |
| Plants | 6 | North American facilities |
| Suppliers | 45 | Tier 1/2/3 distribution |
| Equipment | ~240 | 5 per work center |

## 🏆 Medallion Architecture

### Bronze Layer (Raw)
- Direct ingestion from simulated WMS/Quality systems
- Schema-enforced, no transformations
- Tables: `*_bronze`

### Silver Layer (Cleaned)
- Data quality validation via DLT expectations
- Denormalized with dimension lookups
- Business rule enforcement
- Tables: `*_silver`

### Gold Layer (Analytics)
- Pre-aggregated for dashboards
- ML feature tables
- KPI calculations
- Tables: `*_gold`

## 🤖 ML Models

### 1. Scrap Forecasting
- **Model**: Facebook Prophet
- **Levels**: Total, Plant, Supplier, Part
- **Horizon**: 90 days
- **Tracking**: MLflow experiment + Unity Catalog model registry

### 2. Root Cause Validation
- **Text Analysis**: TF-IDF + Classification
- **8D Similarity**: Cosine similarity for matching historical 8Ds
- **LLM Analysis**: Foundation Model (databricks-gpt-5-1) for 8D review
- **Output**: High-confidence disagreement flags

### 3. Advanced Models
- **Anomaly Detection**: Isolation Forest for unusual scrap patterns
- **Defect Clustering**: K-Means for hidden patterns
- **Supplier Risk Scoring**: Multi-factor risk assessment
- **Equipment Failure Prediction**: Age/OEE correlation
- **Supplier Reallocation Optimizer**: Category-based recommendations

## 📈 Dashboard Pages

1. **Executive Overview**: KPIs, trends, plant comparison
2. **Forecasting & Predictions**: ML forecasts with confidence intervals
3. **Supplier Intelligence**: Risk scoring, quality correlation
4. **Plant Operations**: Work center analysis, shift patterns
5. **8D & Quality**: Root cause breakdown, closure metrics
6. **ML Anomalies**: Real-time anomaly detection

### Features
- Global filters (Plant, Year, Supplier)
- Cross-filtering between visualizations
- Responsive layout

## 🖥️ Streamlit Application

### Pages
1. **Executive Dashboard**: Embedded Databricks dashboard + KPI cards
2. **What-If Scenarios**: Interactive scenario planning
3. **ML Insights**: Forecast charts, anomaly timeline, risk matrix
4. **AI Assistant**: Genie API integration

### Running Locally
```bash
cd app
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

### Environment Variables
Create `.env` file:
```
DATABRICKS_HOST=adb-984752964297111.11.azuredatabricks.net
DATABRICKS_HTTP_PATH=/sql/1.0/warehouses/148ccb90800933a1
DATABRICKS_TOKEN=<your-token>
GENIE_SPACE_ID=01f0d05d42ed11eeae85802c1d5bcccd
```

## 🚀 Quick Start

### 1. Setup Environment
```python
# Run in Databricks notebook
# Notebook: Setup and Bronze Create/0_Setup_and_Data_Model.py
# Creates catalog/schema: scrap_catalog.scrap_demo
```

### 2. Generate Bronze Data
```python
# Notebook: Setup and Bronze Create/1_Generate_Bronze_Data.py
# Generates 60K scrap records + master data
```

### 3. Run DLT Pipelines
```python
# Create DLT pipeline in Databricks pointing to:
# - Scrap DLT Pipeline/2_DLT_Silver_Pipeline.py
# - Scrap DLT Pipeline/3_DLT_Gold_Pipeline.py
```

### 4. Train ML Models
```python
# Run notebooks in order:
# 1. ML Models/1_Scrap_Forecasting.py
# 2. ML Models/2_Root_Cause_8D_Validation.py
# 3. ML Models/3_Advanced_ML_Models.py
```

### 5. Import Dashboard
```
# In Databricks:
# SQL > Dashboards > Import
# Select: Dashboards/Enhanced_Executive_Dashboard.lvdash.json
```

### 6. Configure Genie
```
# Create Genie Space with tables:
# - scrap_transactions_silver
# - eight_d_reports_silver
# - scrap_by_plant_gold
# - scrap_by_supplier_gold
# - supplier_scenario_gold
```

## 📋 Best Practices Implemented

### Databricks
- ✅ Unity Catalog for governance
- ✅ Delta Lake for all tables
- ✅ DLT with data quality expectations
- ✅ MLflow experiment tracking
- ✅ Model registry in Unity Catalog
- ✅ Parameterized notebooks

### Code Quality
- ✅ Comprehensive markdown documentation in notebooks
- ✅ Type hints in Python code
- ✅ Modular application structure
- ✅ Configuration management via environment variables
- ✅ Error handling with fallback to mock data

### ML Ops
- ✅ Feature engineering in Gold layer
- ✅ Cross-validation for model selection
- ✅ Model signatures for Unity Catalog
- ✅ Artifact logging (plots, metrics)

## 📁 Project Structure

```
scrap_demo/
├── README.md                      # This file
├── Setup and Bronze Create/       # Data generation
├── Scrap DLT Pipeline/           # DLT pipelines
├── ML Models/                     # ML notebooks
├── Dashboards/                    # AI/BI dashboard JSON
├── 8D PDF Ingestion/             # Document processing (optional)
└── app/                          # Streamlit application
    ├── app.py
    ├── config.py
    ├── requirements.txt
    ├── components/
    ├── views/
    └── data/
```

## 🔐 Security Notes

- Never commit `.env` files or tokens to git
- Use Databricks secrets for production
- Token in `.env` expires in ~90 days
- Dashboard embed URL is workspace-specific

## 📧 Contact

Internal demo - contact Pete Welch for questions.

