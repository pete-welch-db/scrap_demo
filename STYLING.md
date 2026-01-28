# 🎨 Databricks Styling Guide

This document provides the complete styling reference for building Databricks-branded Streamlit applications. Follow these standards for consistent, professional demos.

---

## Table of Contents

1. [Brand Colors](#brand-colors)
2. [Typography](#typography)
3. [Component Styling](#component-styling)
4. [Plotly Chart Theme](#plotly-chart-theme)
5. [Header Component](#header-component)
6. [CSS Variables Reference](#css-variables-reference)
7. [Key Styling Rules](#key-styling-rules)
8. [Complete styles.py Template](#complete-stytespy-template)

---

## Brand Colors

| Color Name | Hex Code | CSS Variable | Usage |
|------------|----------|--------------|-------|
| **Lava 600** | `#FF3621` | `--lava-600` | Primary accent, buttons, active tabs, CTAs |
| **Lava 500** | `#FF5F46` | `--lava-500` | Hover states, secondary accent |
| **Navy 800** | `#1B3139` | `--navy-800` | Headers, dark text, secondary buttons |
| **Navy 900** | `#0B2026` | `--navy-900` | Deep contrast elements |
| **Oat Medium** | `#EEEDE9` | `--bg-primary` | Page background |
| **Oat Light** | `#F9F7F4` | `--bg-secondary` | Sidebar, alternate sections |
| **White** | `#FFFFFF` | `--bg-card` | Cards, modals, inputs |
| **Charcoal** | `#151513` | `--text-primary` | Primary text, headings |
| **Gray** | `#4a4a48` | `--text-secondary` | Body text, labels |
| **Muted** | `#6b6b69` | `--text-muted` | Captions, helper text |
| **Border** | `#e0dfdc` | `--border-color` | Borders, dividers |
| **Success** | `#10b981` | `--success` | Positive indicators |
| **Warning** | `#f59e0b` | `--warning` | Caution indicators |
| **Danger** | `#ef4444` | `--danger` | Error states, negative deltas |

### Visual Reference

```
┌─────────────────────────────────────────────────────────────┐
│  Lava 600        Lava 500        Navy 800        Navy 900   │
│  #FF3621         #FF5F46         #1B3139         #0B2026    │
│  ████████        ████████        ████████        ████████   │
├─────────────────────────────────────────────────────────────┤
│  Oat Medium      Oat Light       White           Charcoal   │
│  #EEEDE9         #F9F7F4         #FFFFFF         #151513    │
│  ░░░░░░░░        ░░░░░░░░        ▒▒▒▒▒▒▒▒        ████████   │
├─────────────────────────────────────────────────────────────┤
│  Success         Warning         Danger          Border     │
│  #10b981         #f59e0b         #ef4444         #e0dfdc    │
│  ████████        ████████        ████████        ░░░░░░░░   │
└─────────────────────────────────────────────────────────────┘
```

---

## Typography

### Font Family

**DM Sans** - The official Databricks brand font

```css
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,100..1000;1,9..40,100..1000&display=swap');

font-family: "DM Sans", sans-serif;
font-variation-settings: "opsz" 14;
```

### Font Weights

| Element | Weight | Usage |
|---------|--------|-------|
| H1, Page Titles | 700 | Main page headers |
| H2, H3, Section Headers | 600 | Section headings |
| Buttons, Labels | 600 | Interactive elements |
| Navigation, Tabs | 500 | Navigation items |
| Body Text | 400 | Regular content |

### Font Sizes

| Size | Rem | Pixels | Usage |
|------|-----|--------|-------|
| XL | 2.5rem | 40px | Hero headlines |
| H1 | 2rem | 32px | Page titles |
| H2 | 1.5rem | 24px | Section headers |
| H3 | 1.25rem | 20px | Subsections |
| Body | 1rem | 16px | Regular text |
| Small | 0.875rem | 14px | Labels, captions |
| XSmall | 0.75rem | 12px | Badges, metadata |

---

## Component Styling

### Cards

Standard card for metrics, content blocks, and containers:

```css
.metric-card {
    background: #FFFFFF;
    border: 1px solid #e0dfdc;
    border-radius: 12px;
    padding: 24px;
    margin: 8px 0;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
    transition: all 0.2s ease;
}

.metric-card:hover {
    border-color: #FF5F46;
    box-shadow: 0 4px 12px rgba(255, 54, 33, 0.1);
}
```

### Metric Values

```css
.metric-value {
    font-size: 2.5rem;
    font-weight: 700;
    color: #151513;
    font-family: "DM Sans", sans-serif;
}

.metric-label {
    font-size: 0.875rem;
    color: #6b6b69;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-top: 8px;
}

.metric-delta-positive {
    color: #10b981;
    font-size: 0.875rem;
}

.metric-delta-negative {
    color: #ef4444;
    font-size: 0.875rem;
}
```

### Buttons

**Primary Button (Lava):**

```css
.stButton > button {
    background: #FF3621;
    color: white;
    border: none;
    border-radius: 8px;
    padding: 12px 24px;
    font-weight: 600;
    font-family: "DM Sans", sans-serif;
    transition: all 0.2s ease;
}

.stButton > button:hover {
    background: #FF5F46;
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(255, 54, 33, 0.25);
}
```

**Secondary Button (Outline):**

```css
.stButton > button[kind="secondary"] {
    background: #FFFFFF;
    color: #151513;
    border: 1px solid #e0dfdc;
}

.stButton > button[kind="secondary"]:hover {
    background: #F9F7F4;
    border-color: #FF3621;
}
```

### Tabs

```css
/* Tab container */
.stTabs [data-baseweb="tab-list"] {
    background: #FFFFFF;
    border-radius: 8px;
    padding: 4px;
    gap: 4px;
    border: 1px solid #e0dfdc;
}

/* Individual tab */
.stTabs [data-baseweb="tab"] {
    background: transparent;
    border-radius: 6px;
    color: #4a4a48;
    font-weight: 500;
    font-family: "DM Sans", sans-serif;
    padding: 8px 16px;
}

/* Active/Selected tab */
.stTabs [aria-selected="true"] {
    background: #FF3621 !important;
    color: white !important;
}

/* Hover state */
.stTabs [data-baseweb="tab"]:hover {
    background: rgba(255, 54, 33, 0.1);
    color: #FF3621;
}
```

### Form Inputs

```css
.stSelectbox > div > div,
.stMultiSelect > div > div {
    background: #FFFFFF;
    border: 1px solid #e0dfdc;
    border-radius: 8px;
    color: #151513;
}

.stTextInput > div > div > input,
.stNumberInput > div > div > input {
    background: #FFFFFF;
    border: 1px solid #e0dfdc;
    border-radius: 8px;
    color: #151513;
    font-family: "DM Sans", sans-serif;
}

/* Focus state */
.stTextInput > div > div > input:focus,
.stNumberInput > div > div > input:focus {
    border-color: #FF3621;
    box-shadow: 0 0 0 2px rgba(255, 54, 33, 0.15);
}
```

### Status Badges

```css
.status-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 6px 12px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 500;
    font-family: "DM Sans", sans-serif;
}

.status-badge.connected {
    background: rgba(16, 185, 129, 0.1);
    color: #059669;
    border: 1px solid rgba(16, 185, 129, 0.2);
}

.status-badge.disconnected {
    background: rgba(239, 68, 68, 0.1);
    color: #dc2626;
    border: 1px solid rgba(239, 68, 68, 0.2);
}

.status-badge.warning {
    background: rgba(245, 158, 11, 0.1);
    color: #d97706;
    border: 1px solid rgba(245, 158, 11, 0.2);
}
```

### Expanders

```css
[data-testid="stExpander"] {
    background: #FFFFFF;
    border: 1px solid #e0dfdc;
    border-radius: 8px;
}

[data-testid="stExpander"] summary {
    padding: 12px 16px;
    background: #FFFFFF !important;
    border-radius: 8px;
    color: #151513 !important;
}

[data-testid="stExpander"] summary:hover {
    background: #F9F7F4 !important;
}
```

### DataFrames

```css
.stDataFrame {
    background: #FFFFFF;
    border-radius: 8px;
    border: 1px solid #e0dfdc;
}
```

### Chat Components

```css
.stChatInput > div {
    background: #FFFFFF !important;
    border: 1px solid #e0dfdc;
    border-radius: 8px;
}

.stChatMessage {
    background: #FFFFFF;
    border-radius: 8px;
    border: 1px solid #e0dfdc;
}

/* Position chat input above footer */
[data-testid="stChatInput"] {
    bottom: 60px !important;
    background: #EEEDE9 !important;
    padding: 12px 0 !important;
}
```

---

## Plotly Chart Theme

### Theme Configuration

Use this function for consistent Plotly styling:

```python
def create_plotly_theme() -> dict:
    """Databricks Oat Medium theme for Plotly charts."""
    return {
        'paper_bgcolor': 'rgba(255,255,255,0)',
        'plot_bgcolor': 'rgba(255,255,255,0)',
        'font': {
            'color': '#4a4a48', 
            'family': 'DM Sans, sans-serif'
        },
        'title': {
            'font': {
                'color': '#151513',
                'family': 'DM Sans, sans-serif',
                'size': 16
            }
        },
        'xaxis': {
            'gridcolor': 'rgba(224, 223, 220, 0.7)',
            'linecolor': '#e0dfdc',
            'tickfont': {'color': '#4a4a48'}
        },
        'yaxis': {
            'gridcolor': 'rgba(224, 223, 220, 0.7)',
            'linecolor': '#e0dfdc',
            'tickfont': {'color': '#4a4a48'}
        },
        'colorway': [
            '#FF3621',  # Lava 600 (primary)
            '#1B3139',  # Navy 800
            '#FF5F46',  # Lava 500
            '#0B2026',  # Navy 900
            '#10b981',  # Success green
            '#f59e0b',  # Warning orange
        ],
        'legend': {
            'font': {'color': '#4a4a48'},
            'bgcolor': 'rgba(255,255,255,0.8)'
        }
    }
```

### Applying the Theme

**✅ Correct approach - apply properties individually:**

```python
theme = create_plotly_theme()

fig.update_layout(
    paper_bgcolor=theme.get('paper_bgcolor'),
    plot_bgcolor=theme.get('plot_bgcolor'),
    font=theme.get('font'),
    xaxis=theme.get('xaxis'),
    yaxis=theme.get('yaxis'),
    xaxis_title="Your X Label",
    yaxis_title="Your Y Label",
    legend=dict(orientation='h', y=1.1)  # Your custom legend
)
```

**❌ Don't spread if you have conflicting keys:**

```python
# This causes TypeError: got multiple values for keyword argument 'legend'
fig.update_layout(**create_plotly_theme(), legend=dict(...))
```

### Chart Best Practices

1. **Always add axis labels:**
```python
fig.update_layout(
    xaxis_title="Date",
    yaxis_title="Daily Scrap Cost ($)"
)
```

2. **Format currency properly:**
```python
fig.update_layout(
    yaxis=dict(
        tickprefix='$',
        tickformat=',.0f'
    )
)
```

3. **Format percentages:**
```python
fig.update_layout(
    xaxis=dict(
        tickformat='.0%'
    )
)
```

4. **Leave room for labels:**
```python
fig.update_layout(
    margin=dict(l=80, r=20, t=30, b=80)
)
```

---

## Header Component

The branded header with Databricks logo:

```python
def render_header():
    """Render the Databricks branded header."""
    # Databricks logo SVG (official)
    databricks_logo_svg = '''<svg width="132" height="22" viewBox="0 0 132 22" fill="none" xmlns="http://www.w3.org/2000/svg">...</svg>'''
    
    st.markdown(f"""
    <header style="
        display: flex;
        align-items: center;
        justify-content: space-between;
        white-space: nowrap;
        border-bottom: 1px solid #e0dfdc;
        padding: 8px 24px;
        background: white;
        margin: -1rem -1rem 0.25rem -1rem;
        width: calc(100% + 2rem);
        min-height: 48px;
    ">
        <div style="display: flex; align-items: center; gap: 16px; color: #151513;">
            <a href="https://dbricks.co/manufacturing" target="_blank" 
               style="display: flex; align-items: center; text-decoration: none;">
                {databricks_logo_svg}
            </a>
            <h2 style="
                color: #151513 !important; 
                font-size: 1.25rem; 
                font-weight: 500; 
                margin: 0;
                font-family: 'DM Sans', sans-serif;
            ">[Category]</h2>
        </div>
        <div style="display: flex; align-items: center; gap: 24px;">
            <span style="
                color: #151513; 
                font-size: 1rem;
                font-family: 'DM Sans', sans-serif;
            ">[App Name]</span>
        </div>
    </header>
    """, unsafe_allow_html=True)
```

---

## CSS Variables Reference

Add these root variables to your `styles.py`:

```css
:root {
    /* Databricks Primary Palette */
    --lava-600: #FF3621;
    --lava-500: #FF5F46;
    --navy-800: #1B3139;
    --navy-900: #0B2026;
    --oat-light: #F9F7F4;
    --oat-medium: #EEEDE9;
    
    /* Semantic colors */
    --bg-primary: #EEEDE9;
    --bg-secondary: #F9F7F4;
    --bg-card: #FFFFFF;
    --text-primary: #151513;
    --text-secondary: #4a4a48;
    --text-muted: #6b6b69;
    --border-color: #e0dfdc;
    --accent-primary: #FF3621;
    --accent-secondary: #1B3139;
    --success: #10b981;
    --warning: #f59e0b;
    --danger: #ef4444;
}
```

---

## Key Styling Rules

### Do's ✅

1. **Backgrounds**: Use Oat Medium (`#EEEDE9`) for page, White for cards
2. **Borders**: Always `#e0dfdc`, 1px solid, 8-12px radius
3. **Shadows**: Subtle only: `0 1px 3px rgba(0, 0, 0, 0.05)`
4. **Transitions**: Use `all 0.2s ease` for interactive elements
5. **Spacing**: 8px base unit (8, 16, 24, 32px)
6. **Primary CTA**: Lava 600 (`#FF3621`)
7. **Secondary elements**: Navy 800 (`#1B3139`)
8. **Font**: Always DM Sans for consistency

### Don'ts ❌

1. **Don't use pure black** (`#000000`) - use Charcoal (`#151513`)
2. **Don't use harsh shadows** - keep them subtle
3. **Don't mix font families** - stick to DM Sans
4. **Don't forget axis labels** on charts
5. **Don't use default Streamlit colors** - override with theme
6. **Don't use font weight property in Plotly** - it's not supported

### Hide Default Streamlit Elements

```css
/* Hide default header */
header[data-testid="stHeader"] {
    display: none;
}

/* Hide hamburger menu */
#MainMenu {
    visibility: hidden;
}

/* Hide default footer */
footer {
    visibility: hidden;
}
```

### Reduce Default Padding

```css
/* Reduce top padding in main content area */
.stMainBlockContainer {
    padding-top: 0 !important;
}

[data-testid="stAppViewBlockContainer"] {
    padding-top: 0 !important;
}

.block-container {
    padding-top: 0.5rem !important;
}
```

---

## Complete styles.py Template

For the full implementation, see:
- `app/components/styles.py` - Complete CSS styling
- `app/components/metrics.py` - Plotly theme and metric cards

Copy these files to your new project and customize as needed.

---

## Quick Copy-Paste Snippets

### Insight/Tip Callout

```python
st.markdown("""
<div style="
    background: #1B3139;
    color: white;
    padding: 16px 20px;
    border-radius: 8px;
    margin: 16px 0;
    display: flex;
    align-items: center;
    gap: 12px;
">
    <span style="font-size: 1.25rem;">💡</span>
    <span style="font-family: 'DM Sans', sans-serif;">
        Your insight text here
    </span>
</div>
""", unsafe_allow_html=True)
```

### Section Divider

```python
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
```

### Custom Metric Card

```python
def render_metric(value, label, delta=None):
    delta_html = ""
    if delta:
        color = "#10b981" if delta > 0 else "#ef4444"
        symbol = "↑" if delta > 0 else "↓"
        delta_html = f'<div style="color: {color}; font-size: 0.875rem;">{symbol} {abs(delta):.1f}%</div>'
    
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{value}</div>
        <div class="metric-label">{label}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)
```

---

## License

Internal Databricks use only.
