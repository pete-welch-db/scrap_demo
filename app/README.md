# Scrap Reduction Intelligence App

A modern, dark-mode Streamlit application for scrap data analysis, what-if scenarios, ML insights, and AI-powered analytics.

## 📁 Project Structure

```
app/
├── app.py                 # Main application entry point
├── config.py              # Configuration management
├── requirements.txt       # Python dependencies
├── app.yaml              # Databricks Apps deployment config
├── .env                  # Environment variables (not in git)
├── .env.example          # Example environment file
│
├── assets/               # Static assets
│   └── Databricks_Logo.png
│
├── components/           # Reusable UI components
│   ├── __init__.py
│   ├── styles.py         # Custom CSS styling
│   ├── sidebar.py        # Sidebar navigation
│   └── metrics.py        # Metric cards & chart themes
│
├── views/                # Application pages
│   ├── __init__.py
│   ├── dashboard.py      # Executive Dashboard
│   ├── scenarios.py      # What-If Scenarios
│   ├── ml_insights.py    # ML Model Results
│   └── assistant.py      # AI Assistant (Genie)
│
├── data/                 # Data layer
│   ├── __init__.py
│   ├── connection.py     # Database connection
│   ├── queries.py        # SQL queries
│   ├── genie_client.py   # Databricks Genie API client
│   └── mock_data.py      # Mock data fallback
│
└── .streamlit/
    └── config.toml       # Streamlit theme config
```

## Features

### 📊 Executive Dashboard
- Embedded Databricks AI/BI dashboard via iframe
- Real-time KPI cards with Databricks branding
- Adjustable dashboard height
- Direct link to full Databricks dashboard

### 🎯 What-If Scenarios
Interactive scenario planning for scrap reduction:
- **By Plant**: Simulate plant-level improvement initiatives
- **By Work Center**: Model equipment upgrades and OEE improvements  
- **By Part**: Analyze part-specific defect reduction strategies
- **By Supplier**: Evaluate supplier reallocation opportunities

### 🤖 ML Insights
Visualize results from Databricks ML models:
- **Forecasting**: 3-month scrap cost predictions (Prophet)
- **Root Cause Validation**: ML vs Quality team comparison
- **Anomaly Detection**: Unusual scrap patterns (Isolation Forest)
- **Supplier Risk**: Multi-factor risk scoring
- **Recommendations**: ML-powered supplier reallocation suggestions

### 💬 AI Assistant
Databricks Genie integration:
- Natural language queries over scrap data
- SQL query generation and execution
- Link to full Genie Space in Databricks

## Quick Start

### 1. Clone and Setup

```bash
cd scrap_demo/app

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
# Copy example env file
cp .env.example .env

# Edit .env with your credentials
# The token is pre-configured but expires in ~90 days
```

### 3. Run Locally

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DATABRICKS_HOST` | Workspace hostname | `fe-vm-industry-solutions-buildathon.cloud.databricks.com` |
| `DATABRICKS_HTTP_PATH` | SQL Warehouse path | `/sql/1.0/warehouses/288a7ec183eea397` |
| `DATABRICKS_TOKEN` | Access token | (from .env) |
| `DASHBOARD_EMBED_URL` | Dashboard iframe URL | (pre-configured) |
| `DATABRICKS_CATALOG` | Unity Catalog name | `scrap` |
| `DATABRICKS_SCHEMA` | Schema name | `scrap_demo` |
| `GENIE_SPACE_ID` | Genie Space ID | (optional) |

### Adding New Pages

1. Create a new file in `pages/` (e.g., `pages/reports.py`)
2. Implement a `render_*_page()` function
3. Add to `pages/__init__.py`
4. Add navigation option in `components/sidebar.py`
5. Add routing in `app.py`

### Customizing Styles

Edit `components/styles.py` to modify CSS variables:

```css
:root {
    --bg-primary: #0a0a0f;
    --accent-primary: #6366f1;
    --text-primary: #f1f5f9;
}
```

## Deploy to Databricks Apps

```bash
# Create app
databricks apps create --name scrap-intelligence \
  --source-code-path /Repos/your-repo/scrap_demo/app

# Deploy
databricks apps deploy scrap-intelligence
```

## Security Notes

⚠️ **Token Management**: 
- Never commit `.env` files to git
- Use `.env.example` as a template
- For production, use Databricks secrets management
- Current token expires in ~90 days

## License

Internal use only.
