# Scrap Reduction Intelligence App

A modern, dark-mode Streamlit application for scrap data analysis, what-if scenarios, and AI-powered insights.

## 📁 Project Structure

```
app/
├── app.py                 # Main application entry point
├── config.py              # Configuration management
├── requirements.txt       # Python dependencies
├── app.yaml              # Databricks Apps deployment config
├── .env                  # Environment variables (not in git)
├── .env.example          # Example environment file
├── .gitignore            # Git ignore rules
│
├── components/           # Reusable UI components
│   ├── __init__.py
│   ├── styles.py         # Custom CSS styling
│   ├── sidebar.py        # Sidebar navigation
│   └── metrics.py        # Metric cards & chart themes
│
├── pages/                # Application pages
│   ├── __init__.py
│   ├── dashboard.py      # Executive Dashboard
│   ├── scenarios.py      # What-If Scenarios
│   └── assistant.py      # AI Assistant
│
├── data/                 # Data layer
│   ├── __init__.py
│   ├── connection.py     # Database connection
│   ├── queries.py        # SQL queries
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

### 🎯 What-If Scenarios
Interactive scenario planning for scrap reduction:
- **By Plant**: Simulate plant-level improvement initiatives
- **By Work Center**: Model equipment upgrades and OEE improvements  
- **By Part**: Analyze part-specific defect reduction strategies

### 🤖 AI Assistant
Placeholder for AI integrations:
- **Databricks Genie**: Natural language queries over scrap data
- **Multi-Agent System**: Complex analysis orchestration
- **8D Knowledge Base**: RAG over historical 8D reports

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
| `DATABRICKS_HOST` | Workspace hostname | `adb-984752964297111.11.azuredatabricks.net` |
| `DATABRICKS_HTTP_PATH` | SQL Warehouse path | `/sql/1.0/warehouses/148ccb90800933a1` |
| `DATABRICKS_TOKEN` | Access token | (from .env) |
| `DASHBOARD_EMBED_URL` | Dashboard iframe URL | (pre-configured) |
| `DATABRICKS_CATALOG` | Unity Catalog name | `welch` |
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
