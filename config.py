"""
AeroTwin Edge - Configuration Management
Production-grade settings using pydantic-settings for environment variable loading.
Supports both local development and cloud deployment modes.
"""

from functools import lru_cache
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    
    Priority order:
    1. Environment variables
    2. .env file
    3. Default values
    """
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # Database
    database_url: str = Field(
        default="sqlite:///./aerotwin_prod.db",
        description="Database connection URL"
    )
    
    # API Security
    api_key: str = Field(
        default="aerotwin_secret_2026",
        description="API key for authenticating edge nodes"
    )
    
    # ML Configuration
    ml_anomaly_interval: int = Field(
        default=10,
        description="Seconds between anomaly detection cycles"
    )
    ml_min_samples: int = Field(
        default=10,
        description="Minimum samples required before ML model fitting"
    )
    ml_contamination: float = Field(
        default=0.1,
        ge=0.0,
        le=0.5,
        description="Expected proportion of anomalies in the dataset"
    )
    
    # Server Configuration
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8000, description="Server port")
    debug: bool = Field(default=False, description="Enable debug mode")
    
    # Forecasting Configuration
    forecast_intervals: int = Field(default=3, description="Number of intervals to forecast")
    forecast_history_size: int = Field(default=20, description="Historical data points for forecasting")
    
    # Cloud Deployment Configuration
    cloud_mode: bool = Field(
        default=True,
        description="Enable cloud deployment mode with bundled simulator"
    )
    simulator_enabled: bool = Field(
        default=True,
        description="Enable built-in edge simulator"
    )
    simulator_interval: float = Field(
        default=5.0,
        description="Seconds between simulator telemetry cycles"
    )
    simulator_zone: str = Field(
        default="Building-A-Floor-1",
        description="Default zone ID for simulator"
    )
    demo_spike_interval: int = Field(
        default=15,
        description="Cycles between demo spike events"
    )
    
    # Frontend Configuration
    api_url: str = Field(
        default="http://localhost:8000",
        description="Backend API URL for dashboard"
    )


@lru_cache
def get_settings() -> Settings:
    """
    Get cached settings instance.
    Uses lru_cache for performance - settings are loaded once.
    """
    return Settings()


settings = get_settings()
