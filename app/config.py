"""
Configuration Management
========================
Centralized configuration for the Scrap Reduction Intelligence app.
Loads settings from environment variables with sensible defaults.

INTEGRATION NOTE:
- Base: scrap_demo-main (lines 1-127)
- ML Branch: Scrap (lines 1-115) 
- Changes: Added environment detection (is_databricks/is_local) from main
- Theme: Using main's light theme (will be overridden in styles.py)
"""

import os
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


@dataclass
class DatabricksConfig:
    """Databricks connection configuration."""
    host: str
    http_path: str
    token: str
    catalog: str
    schema: str
    dashboard_embed_url: str
    
    @classmethod
    def from_env(cls) -> "DatabricksConfig":
        """Create configuration from environment variables."""
        return cls(
            host=os.getenv("DATABRICKS_HOST", "adb-984752964297111.11.azuredatabricks.net"),
            http_path=os.getenv("DATABRICKS_HTTP_PATH", "/sql/1.0/warehouses/148ccb90800933a1"),
            token=os.getenv("DATABRICKS_TOKEN", ""),
            catalog=os.getenv("DATABRICKS_CATALOG", "scrap_catalog"),
            schema=os.getenv("DATABRICKS_SCHEMA", "scrap_demo"),
            dashboard_embed_url=os.getenv(
                "DASHBOARD_EMBED_URL",
                "https://adb-984752964297111.11.azuredatabricks.net/embed/dashboardsv3/01f0cfcf667f1b8ba6b3f8158b94f453?o=984752964297111"
            )
        )
    
    @property
    def full_schema(self) -> str:
        """Return fully qualified schema name."""
        return f"{self.catalog}.{self.schema}"
    
    def is_configured(self) -> bool:
        """Check if essential configuration is present."""
        return bool(self.host and self.token)


@dataclass
class GenieConfig:
    """Databricks Genie configuration."""
    space_id: Optional[str]
    space_name: str = "Scrap Analytics Assistant"
    
    @classmethod
    def from_env(cls) -> "GenieConfig":
        return cls(
            space_id=os.getenv("GENIE_SPACE_ID", "01f0d05d42ed11eeae85802c1d5bcccd"),
            space_name="Scrap Analytics Assistant"
        )
    
    def is_configured(self) -> bool:
        return bool(self.space_id)


@dataclass
class AppConfig:
    """Main application configuration."""
    title: str = "Scrap Reduction Intelligence"
    icon: str = "🔧"
    layout: str = "wide"

    # Environment: "LOCAL" (local dev) or "DBX" (Databricks Apps/DAB)
    # SOURCE: scrap_demo-main (enhancement for production deployment)
    environment: str = "LOCAL"

    # Data refresh interval (seconds)
    cache_ttl: int = 300

    # UI settings
    default_dashboard_height: int = 800

    # Data source: "auto" (try live, fallback to mock) or "mock" (always mock)
    data_source: str = "auto"

    @classmethod
    def from_env(cls) -> "AppConfig":
        return cls(
            environment=os.getenv("ENVIRONMENT", "LOCAL"),
            data_source=os.getenv("DATA_SOURCE", "auto")
        )

    def is_databricks(self) -> bool:
        """Check if running in Databricks environment.
        SOURCE: scrap_demo-main (new method)"""
        return self.environment.upper() == "DBX"

    def is_local(self) -> bool:
        """Check if running in local environment.
        SOURCE: scrap_demo-main (new method)"""
        return self.environment.upper() == "LOCAL"
    

# Global configuration instances
databricks_config = DatabricksConfig.from_env()
genie_config = GenieConfig.from_env()
app_config = AppConfig.from_env()


# Theme colors (matching CSS variables)
# SOURCE: Will be overridden with Databricks Oat Medium light theme in styles.py
# This dark theme kept for reference but not used
THEME = {
    "bg_primary": "#EEEDE9",  # Oat Medium light background
    "bg_secondary": "white",
    "bg_tertiary": "#f8f8f6",
    "bg_card": "white",
    "accent_primary": "#FF3621",  # Databricks Lava Red
    "accent_secondary": "#1B3139",  # Databricks Navy
    "accent_tertiary": "#a855f7",
    "text_primary": "#1B3139",  # Dark text for light background
    "text_secondary": "#4a4a48",
    "text_muted": "#6b6b69",
    "border_color": "#e0dfdc",
    "success": "#10b981",
    "warning": "#f59e0b",
    "danger": "#ef4444",
}
