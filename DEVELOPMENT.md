# 🏗️ Building High-Quality Databricks Demo Apps: A Playbook

This document captures how the Scrap Intelligence Platform was built and serves as a repeatable guide for Databricks Solution Architects to create polished, production-quality demo applications.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Technology Stack](#technology-stack)
3. [Key Design Decisions](#key-design-decisions)
4. [Development Workflow](#development-workflow)
5. [Prompt Engineering Patterns](#prompt-engineering-patterns)
6. [Common Patterns & Solutions](#common-patterns--solutions)
7. [Bugs We Fixed (So You Don't Have To)](#bugs-we-fixed-so-you-dont-have-to)
8. [Files to Customize](#files-to-customize)
9. [Step-by-Step: Building a New Demo](#step-by-step-building-a-new-demo)
10. [Pro Tips](#pro-tips)
11. [Recreating This App From Scratch](#recreating-this-app-from-scratch)
12. [Complete Prompt History](#complete-prompt-history)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     Streamlit Frontend                          │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐           │
│  │ Landing  │ │Dashboard │ │Scenarios │ │Assistant │           │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘           │
│        │            │            │            │                 │
│  ┌─────┴────────────┴────────────┴────────────┴─────┐          │
│  │              Components Layer                     │          │
│  │   (styles.py, sidebar.py, metrics.py)            │          │
│  └───────────────────────────────────────────────────┘          │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────┴──────────────────────────────────────┐
│                      Data Layer                                  │
│  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐   │
│  │connection  │ │  queries   │ │genie_client│ │ mock_data  │   │
│  └────────────┘ └────────────┘ └────────────┘ └────────────┘   │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────┴──────────────────────────────────────┐
│                   Databricks Platform                            │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐             │
│  │ SQL Warehouse│ │ Unity Catalog│ │  Genie API   │             │
│  └──────────────┘ └──────────────┘ └──────────────┘             │
│  ┌──────────────┐ ┌──────────────┐                               │
│  │  ML Models   │ │AI/BI Dashboard│                              │
│  └──────────────┘ └──────────────┘                               │
└─────────────────────────────────────────────────────────────────┘
```

### File Structure

```
app/
├── app.py                 # Entry point + routing only
├── config.py              # Centralized config (env vars)
├── requirements.txt       # Python dependencies
├── app.yaml               # Databricks Apps deployment config
├── .env                   # Environment variables (not in git)
├── .env.example           # Example environment file
│
├── assets/                # Static assets (logos, images)
│
├── components/            # Reusable UI pieces
│   ├── __init__.py
│   ├── styles.py          # Custom CSS styling & header
│   ├── sidebar.py         # Sidebar navigation
│   └── metrics.py         # Metric cards & Plotly themes
│
├── views/                 # Page-level modules
│   ├── __init__.py
│   ├── landing.py         # Landing/pitch page
│   ├── dashboard.py       # Main analytics dashboard
│   ├── scenarios.py       # What-If Scenarios
│   └── assistant.py       # AI Assistant (Genie)
│
└── data/                  # All data access logic
    ├── __init__.py
    ├── connection.py      # Database connection
    ├── queries.py         # SQL queries
    ├── genie_client.py    # Databricks Genie API client
    └── mock_data.py       # Mock data fallback
```

---

## Technology Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| Frontend | Streamlit | Rapid UI development |
| Styling | Custom CSS | Databricks branding |
| Charts | Plotly | Interactive visualizations |
| Data | Databricks SQL | Query execution |
| AI Chat | Genie API | Natural language queries |
| Dashboards | AI/BI Dashboards | Embedded analytics |
| ML | MLflow models | Pre-built predictions |
| Deployment | Databricks Apps | Production hosting |

---

## Key Design Decisions

### 1. Graceful Degradation
Every database call has a try/except with mock data fallback. The demo works offline or when the warehouse is unavailable.

```python
def load_data():
    try:
        df = run_query("SELECT * FROM table")
        if df is not None and len(df) > 0:
            return df
    except Exception as e:
        st.warning(f"Database unavailable: {e}")
    
    # Fall back to mock data
    return generate_mock_data()
```

### 2. Databricks Branding
- **Theme**: Oat Medium (light mode)
- **Font**: DM Sans (Databricks brand font)
- **Colors**: 
  - Navy: `#1B3139`
  - Lava: `#FF3621`
  - Oat: `#F5F3EF`
  - Text: `#151513`

> 📘 **See [STYLING.md](STYLING.md) for the complete Databricks styling guide** with all colors, typography, component CSS, and Plotly themes.

### 3. Modular Architecture
- Each page is a separate module in `views/`
- Reusable components in `components/`
- All data access isolated in `data/`
- Configuration centralized in `config.py`

### 4. Settings Toggle
A sidebar "Settings" section allows toggling between real and mock data, useful for:
- Offline demos
- Testing without warehouse access
- Consistent demo data

---

## Development Workflow

We followed an iterative cycle throughout development:

```
1. Request Feature/Change
   ↓
2. Implement Change
   ↓
3. Test (run app locally)
   ↓
4. Report Issues (with screenshots/errors)
   ↓
5. Debug & Fix
   ↓
6. Refine Styling/UX
   ↓
(repeat)
```

### Example Iteration Chain
This is a real example from our development:

1. "Add supplier risk chart" → implemented scatter plot
2. "TypeError on supplier risk" → fixed Plotly legend conflict
3. "Query error: column 'tier' not found" → updated query to match schema
4. "Add axis labels to the chart" → added x/y labels
5. "Change to pie chart with drill-down" → refactored visualization

---

## Prompt Engineering Patterns

### Effective Prompts That Worked Well

#### Initial Setup
> "Update the app to use the Databricks Oat Medium theme with DM Sans font and add a branded header."

#### Feature Addition
> "Add anomaly detection and root cause validation as tabs in the executive dashboard. Keep the styling consistent with the existing tabs."

#### UX Refinement
> "The pink bar looks like an error even though it isn't. Fix the styling to look like an insight/tip instead."

#### Debugging (with context)
> "TypeError on the Forecasting tab. [paste error stack trace]. The issue seems to be with how create_plotly_theme() is being used."

#### Configuration Migration
> "We have a new Databricks workspace. Update all configurations: host is X, catalog/schema is Y, warehouse path is Z. Ask any clarifying questions."

### Anti-Patterns to Avoid

❌ Vague: "Fix the chart"
✅ Specific: "Add x and y axis labels to the supplier risk scatter plot"

❌ Multiple asks: "Fix the error and also change the color and add a new feature"
✅ One at a time: "Fix the TypeError on the Forecasting tab" → then → "Change the color" → then → "Add the new feature"

---

## Common Patterns & Solutions

### Pattern 1: Plotly Theme Application

```python
# ❌ DON'T: Spread theme dict with separate legend param
fig.update_layout(**create_plotly_theme(), legend=dict(...))  # Causes conflict!

# ✅ DO: Get theme first, apply properties separately
theme = create_plotly_theme()
fig.update_layout(
    paper_bgcolor=theme.get('paper_bgcolor'),
    plot_bgcolor=theme.get('plot_bgcolor'),
    font=theme.get('font'),
    legend=dict(...)  # Your custom legend
)
```

### Pattern 2: Database Query with Fallback

```python
@st.cache_data(ttl=300)
def load_data():
    query = f"SELECT * FROM {SCHEMA}.my_table"
    df = run_query(query)
    
    if df is None or len(df) == 0:
        return generate_mock_data()
    return df
```

### Pattern 3: Conditional Data Source

```python
if app_config.data_source == "mock":
    st.info("📊 Using demo data")
    df = get_mock_data()
else:
    df = load_real_data()
    if df is None:
        st.warning("⚠️ Database unavailable, using demo data")
        df = get_mock_data()
```

### Pattern 4: Chart with Proper Labels

```python
fig.update_layout(
    xaxis_title="Quality Score",
    yaxis_title="Annual Scrap Cost ($)",
    xaxis=dict(
        tickformat='.0%',
        title_font=dict(size=12, color='#1B3139')
    ),
    yaxis=dict(
        tickprefix='$',
        tickformat=',.0f',
        title_font=dict(size=12, color='#1B3139')
    ),
    margin=dict(l=80, r=20, t=30, b=80)  # Room for labels
)
```

### Pattern 5: Embedding Dashboards

```python
# Use /embed/ format for iframes
embed_url = "https://workspace.cloud.databricks.com/embed/dashboardsv3/..."

st.markdown(f'''
<iframe 
    src="{embed_url}" 
    width="100%" 
    height="700" 
    frameborder="0"
    style="border-radius: 8px;">
</iframe>
''', unsafe_allow_html=True)
```

---

## Bugs We Fixed (So You Don't Have To)

| Issue | Cause | Fix |
|-------|-------|-----|
| `TypeError: got multiple values for keyword argument 'legend'` | Spreading theme dict that includes `legend` key alongside a separate `legend` param | Apply theme properties individually, not with `**spread` |
| `ValueError` on Plotly marker size | Plotly can't use pandas Series directly for marker sizes | Use `.tolist()` on the Series |
| `use_container_width` deprecation warning | Streamlit 1.45+ deprecated this parameter | Use `width='stretch'` instead |
| `UNRESOLVED_COLUMN` SQL error | Query references columns that don't exist in the table | Verify column names match your actual schema |
| Dashboard iframe "refused to connect" | Using `/published` URL format | Use `/embed/` format for iframe embedding |
| `DATATYPE_MISMATCH` SQL error | Comparing BIGINT column to boolean `true` | Use `= 1` instead of `= true` for BIGINT columns |
| Plotly font `weight` error | Plotly fonts don't support `weight` property | Remove `weight`, use only `family`, `size`, `color` |

---

## Files to Customize

When adapting this app for a new demo:

| File | What to Change |
|------|----------------|
| `config.py` | Workspace URL, warehouse path, catalog/schema |
| `app.yaml` | Same as config.py, for Databricks Apps deployment |
| `.env.example` | Template for environment variables |
| `components/styles.py` | Colors, fonts, branding |
| `data/queries.py` | SQL queries for your specific tables |
| `views/dashboard.py` | Main dashboard content and tabs |
| `views/landing.py` | Pitch/story content |
| `data/mock_data.py` | Mock data matching your schema |

---

## Step-by-Step: Building a New Demo

### Phase 1: Foundation (Day 1)

1. **Copy this app structure** to your new repo
2. **Update `config.py`** with your workspace details:
   ```python
   DATABRICKS_HOST = "your-workspace.cloud.databricks.com"
   DATABRICKS_HTTP_PATH = "/sql/1.0/warehouses/your-warehouse-id"
   DATABRICKS_CATALOG = "your_catalog"
   DATABRICKS_SCHEMA = "your_schema"
   ```
3. **Create `.env`** from `.env.example`, add your PAT token
4. **Test connection**: Run `streamlit run app.py` and verify data loads

### Phase 2: Content (Day 2-3)

1. **Design your views** - Keep to 3-4 pages max:
   - Landing/pitch page
   - Main analytics dashboard
   - Interactive scenarios
   - AI assistant (optional)

2. **Update queries** in `data/queries.py` for your tables

3. **Create mock data** in `data/mock_data.py` matching your schema

4. **Add your charts/metrics** - Follow the Plotly patterns above

5. **Integrate ML outputs** if you have pre-trained models

6. **Add Genie** for natural language queries (update space ID)

### Phase 3: Polish (Day 4)

1. **Refine styling** - Consistent colors, proper spacing
2. **Add axis labels** to ALL charts
3. **Test on different screen sizes**
4. **Test with mock data** for offline demos
5. **Remove debug/development elements**

### Phase 4: Deploy

1. **Test locally** one final time
2. **Deploy to Databricks Apps**:
   ```bash
   databricks apps create --name your-demo-name
   databricks apps deploy your-demo-name
   ```
3. **Share the URL** with stakeholders

---

## Pro Tips

### Development
1. **Always provide error context** - Screenshots and stack traces speed up debugging
2. **Iterate in small steps** - One change at a time is easier to debug
3. **Use mock data first** - Get the UI right before connecting real data
4. **Comment out, don't delete** - Features you remove might come back
5. **Check column names** - Schema mismatches are the #1 query error

### Presentation
1. **Use mock data for demos** - More consistent, no warehouse latency
2. **Pre-load the app** - Avoid cold start delays during presentations
3. **Know your fallbacks** - If something breaks, mock data should work
4. **Keep it simple** - 3-4 polished pages beats 10 incomplete ones

### Debugging
1. **Check the Streamlit logs** - Run with `streamlit run app.py` in terminal
2. **Verify SQL in Databricks UI** - Test queries directly before adding to app
3. **Print DataFrame shapes** - `st.write(df.shape)` helps diagnose data issues
4. **Cache carefully** - `@st.cache_data` can hide updates; clear cache when debugging

---

## Recreating This App From Scratch

Want to build a similar demo from zero? Follow this phased approach with the exact prompts to use with an AI coding assistant.

### Prerequisites

Before starting, ensure you have:
1. A Databricks workspace with Unity Catalog
2. Your data tables created (bronze → silver → gold with ML outputs)
3. A Genie Space configured with your tables
4. An AI/BI Dashboard built
5. Python 3.10+ installed locally

### Phase 1: Foundation

Start with the structure and core infrastructure.

**Prompt 1 - Project Setup:**
> "Create a Streamlit app for a Databricks demo called '[Your Demo Name]'. Set up the folder structure with:
> - `app/` folder with `app.py` as entry point
> - `components/` for reusable UI (sidebar, styles, metrics)
> - `views/` for pages (dashboard, scenarios, assistant)
> - `data/` for database connection and queries
> - Use environment variables for configuration (host, warehouse path, catalog, schema)
> - Include mock data fallback for when database is unavailable"

**Prompt 2 - Database Connection:**
> "Create the database connection layer in `data/connection.py`. Use databricks-sql-connector to connect to Databricks SQL Warehouse. Configuration should come from environment variables. Include error handling and return None if connection fails."

**Prompt 3 - Configuration:**
> "Create `config.py` with centralized configuration: DATABRICKS_HOST, DATABRICKS_HTTP_PATH, DATABRICKS_CATALOG, DATABRICKS_SCHEMA, GENIE_SPACE_ID, DASHBOARD_EMBED_URL. All should load from environment variables with sensible defaults."

**Prompt 4 - Requirements:**
> "Create requirements.txt with: streamlit, plotly, pandas, databricks-sql-connector, python-dotenv, requests"

### Phase 2: Branding & Styling

Apply Databricks look and feel.

**Prompt 5 - Theme:**
> "Update the app to use Databricks Oat Medium theme:
> - Background: #F5F3EF (Oat)
> - Primary accent: #FF3621 (Lava)
> - Text: #151513 (Charcoal)
> - Secondary: #1B3139 (Navy)
> - Font: DM Sans from Google Fonts
> Add custom CSS in `components/styles.py`"

**Prompt 6 - Header:**
> "Add a branded header with the Databricks logo (SVG inline) and app title. Keep minimal whitespace - follow accessibility best practices for spacing."

**Prompt 7 - Sidebar:**
> "Create a sidebar with app title, navigation buttons for each page, and a Settings expander at the bottom with data source toggle. Use the same Databricks theme colors."

**Prompt 8 - Plotly Theme:**
> "Create a Plotly chart theme in `components/metrics.py` that matches our Databricks branding: transparent backgrounds, DM Sans font, color palette (Navy, Lava, teal, amber), minimal gridlines."

### Phase 3: Landing Page

Create the pitch/story page.

**Prompt 9 - Landing Structure:**
> "Create a landing page in `views/landing.py` with: hero section with headline and subheadline, key statistics, 'How it works' section, and CTA button to navigate to the dashboard."

**Prompt 10 - Landing Content:**
> "Update the landing page with content about [your use case]. Include the business problem, the opportunity size, and how Databricks enables the solution."

### Phase 4: Main Dashboard

Build the analytics hub with multiple tabs.

**Prompt 11 - Dashboard Structure:**
> "Create the main dashboard in `views/dashboard.py` with tabs: Overview (embedded AI/BI dashboard), [Tab 2 name], [Tab 3 name], [Tab 4 name]."

**Prompt 12 - Dashboard Embed:**
> "Embed the Databricks AI/BI dashboard in the Overview tab using an iframe. Use the /embed/ URL format."

**Prompt 13 - First Visualization Tab:**
> "Add a [visualization type] to the [Tab Name] tab. Query [table_name] and show [what to display]. Include proper axis labels."

**Prompt 14 - Additional Tabs:**
> Repeat for each tab, being specific about:
> - Which table to query
> - What columns to use
> - How to visualize (chart type, colors)
> - What interactivity to include

**Prompt 15 - Mock Data:**
> "Create mock data generators in `data/mock_data.py` for all tables. Mock data should match the real schema exactly so fallback works seamlessly."

### Phase 5: What-If Scenarios (Optional)

Add interactive scenario modeling.

**Prompt 16 - Scenarios Page:**
> "Create what-if scenarios page with sections for [Scenario 1], [Scenario 2], etc. Each should have sliders/inputs and show projected outcomes."

**Prompt 17 - Scenario Charts:**
> "For each scenario, add a chart showing current vs projected state with clear axis labels and legend."

### Phase 6: AI Assistant (Optional)

Integrate Genie for natural language.

**Prompt 18 - Genie Integration:**
> "Create `data/genie_client.py` to interact with Databricks Genie API: start conversation, send messages, poll for responses, handle follow-up questions."

**Prompt 19 - Chat Interface:**
> "Create chat interface in `views/assistant.py`: full-width chat display, input at bottom, quick question buttons (only show when no conversation), tips in collapsible expander."

### Phase 7: Polish

Refine the experience.

**Prompt 20 - Spacing:**
> "Reduce whitespace throughout: smaller header margins, less padding above content, sidebar title closer to top."

**Prompt 21 - Chart Labels:**
> "Add proper axis labels to ALL charts with appropriate formatting (currency, percentages, etc.)."

**Prompt 22 - Error Handling:**
> "Add graceful error handling to all data loads. If query fails, show warning and fall back to mock data. Never let the app crash."

**Prompt 23 - Final Review:**
> "Review all pages for consistent styling, all charts have axis labels, no errors in any tab, mock data fallback works."

### Phase 8: Deployment

Ship it!

**Prompt 24 - App.yaml:**
> "Create app.yaml for Databricks Apps deployment with all environment variables."

**Prompt 25 - Env Example:**
> "Create .env.example with all required variables (without actual secrets) as a template."

**Prompt 26 - Documentation:**
> "Create README.md with setup instructions, features overview, and deployment steps."

---

### Quick Reference: The Core 10 Prompts

For a minimal working demo, use these essential prompts:

| # | Prompt | Purpose |
|---|--------|---------|
| 1 | "Create Streamlit app with components/, views/, data/ structure" | Foundation |
| 2 | "Add Databricks SQL connection with env vars" | Data layer |
| 3 | "Apply Databricks Oat Medium theme with DM Sans font" | Branding |
| 4 | "Create navigation sidebar with page buttons" | Navigation |
| 5 | "Create dashboard with tabs for each analytics view" | Main content |
| 6 | "Add Plotly charts with proper axis labels" | Visualizations |
| 7 | "Create mock data fallback for offline demos" | Reliability |
| 8 | "Integrate Databricks Genie for AI chat" | AI features |
| 9 | "Reduce whitespace, fix deprecations, consistent styling" | Polish |
| 10 | "Create app.yaml for Databricks Apps deployment" | Deployment |

---

## Complete Prompt History

Here's the actual sequence of prompts used to build this app:

### Theming & Branding
1. "Update the app to use the Databricks Oat Medium theme with DM Sans font and add a branded header"
2. "Update the footer and Plotly theme to match the new branding"
3. "The pink bar looks like an error - fix it to look like an insight/tip"
4. "Change selected tab styling to orange instead of black, still in theme"

### Layout & UX
5. "Get this cleaned up to fit on one page without scrolling"
6. "Kill the quick stats, it's redundant"
7. "Less whitespace at the top of the sidebar, move settings to the bottom"
8. "Reduce whitespace at the top margin"
9. "Move the chat input above the footer"
10. "Fix the black box at the bottom of the chat"

### Feature Organization
11. "Comment out anomaly detection, recommendations, and root cause validation"
12. "Move supplier risk and forecasting to exec dashboard as tabs"
13. "Comment out ML Insights section"
14. "Add back anomaly detection and root cause validation as tabs in exec dashboard"
15. "Update title from Executive Dashboard to Scrap Insights"
16. "Completely comment out ML Insights tab, no longer needed"

### Chart Improvements
17. Multiple `TypeError` fixes for Plotly legend conflicts
18. "Add labels to x,y axis on forecasting chart"
19. "Add labels to x,y axis on supplier risk chart"
20. "Change supplier risk to a pie chart with drill-down"
21. "Add labels to x,y axis on all What-If Scenario charts"
22. "Update color on recommended reallocation so I can see it"

### Data & Configuration
23. "New Databricks workspace: [URL], catalog: scrap_catalog.scrap_demo"
24. "Ensure Genie space ID is correct"
25. "Update SQL warehouse path"
26. "Update .env.example with correct values"
27. "New dashboard URL - update everywhere"
28. "Iframe refused to connect - why?"
29. "Remove banner about only working in Databricks Apps"

### Query Fixes
30. "Anomalies tab using demo data - fix to call real data"
31. "Query error: combined_anomaly datatype mismatch"
32. "Query error: daily_cost cannot be resolved"
33. "Anomaly chart using wrong date - use transaction_date"

### Content
34. "Create a landing page based on the pitch deck"
35. "Root cause tab should filter on total disagreements, not just high-confidence"

---

## License

Internal Databricks use only.
