"""
Configuration Management
========================
Centralized configuration for the Scrap Reduction Intelligence app.
Loads settings from environment variables with sensible defaults.
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
    
    # Data refresh interval (seconds)
    cache_ttl: int = 300
    
    # UI settings
    default_dashboard_height: int = 800
    
    # Data source: "auto" (try live, fallback to mock) or "mock" (always mock)
    data_source: str = "auto"
    
    @classmethod
    def from_env(cls) -> "AppConfig":
        return cls(
            data_source=os.getenv("DATA_SOURCE", "auto")
        )
    

# Global configuration instances
databricks_config = DatabricksConfig.from_env()
genie_config = GenieConfig.from_env()
app_config = AppConfig.from_env()


# Theme colors (matching CSS variables)
THEME = {
    "bg_primary": "#0a0a0f",
    "bg_secondary": "#12121a",
    "bg_tertiary": "#1a1a2e",
    "bg_card": "#16162a",
    "accent_primary": "#6366f1",
    "accent_secondary": "#8b5cf6",
    "accent_tertiary": "#a855f7",
    "text_primary": "#f1f5f9",
    "text_secondary": "#94a3b8",
    "text_muted": "#64748b",
    "border_color": "#2d2d44",
    "success": "#10b981",
    "warning": "#f59e0b",
    "danger": "#ef4444",
}

