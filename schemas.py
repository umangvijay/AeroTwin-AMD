"""
AeroTwin Edge - Pydantic Schemas
Request/Response validation models for the FastAPI endpoints
"""

from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field, field_validator


class TelemetryCreate(BaseModel):
    """Schema for creating new telemetry records."""
    zone: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Campus zone identifier",
        examples=["Building-A-Floor-1"]
    )
    occupancy: int = Field(
        ...,
        ge=0,
        le=10000,
        description="Number of people in the zone"
    )
    ai_mode: str = Field(
        ...,
        min_length=1,
        max_length=50,
        description="Current AI operation mode",
        examples=["auto", "manual", "eco"]
    )
    hardware_cpu: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="CPU utilization percentage"
    )

    @field_validator("ai_mode")
    @classmethod
    def validate_ai_mode(cls, v: str) -> str:
        allowed_modes = {"auto", "manual", "eco", "performance", "standby"}
        if v.lower() not in allowed_modes:
            raise ValueError(f"ai_mode must be one of: {', '.join(allowed_modes)}")
        return v.lower()

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "zone": "Building-A-Floor-1",
                    "occupancy": 45,
                    "ai_mode": "auto",
                    "hardware_cpu": 67.5
                }
            ]
        }
    }


class TelemetryResponse(BaseModel):
    """Schema for telemetry response."""
    id: int
    timestamp: datetime
    zone: str
    occupancy: int
    ai_mode: str
    hardware_cpu: float

    model_config = {"from_attributes": True}


class AnomalyResponse(BaseModel):
    """Schema for anomaly response."""
    id: int
    timestamp: datetime
    severity: str
    description: str

    model_config = {"from_attributes": True}


class ForecastPoint(BaseModel):
    """Schema for a single forecast point."""
    interval: int = Field(..., description="Forecast interval (1, 2, or 3)")
    predicted_occupancy: float = Field(..., description="Predicted occupancy value")
    confidence_lower: float = Field(..., description="Lower confidence bound")
    confidence_upper: float = Field(..., description="Upper confidence bound")


class MLDashboardResponse(BaseModel):
    """Schema for ML Dashboard endpoint response."""
    latest_telemetry: Optional[TelemetryResponse] = Field(
        None,
        description="Most recent telemetry reading"
    )
    recent_anomalies: List[AnomalyResponse] = Field(
        default_factory=list,
        description="List of recent anomalies (last 10)"
    )
    forecast: List[ForecastPoint] = Field(
        default_factory=list,
        description="Occupancy forecast for next 3 intervals"
    )
    model_status: str = Field(
        default="active",
        description="Current ML model status"
    )
    total_telemetry_count: int = Field(
        default=0,
        description="Total number of telemetry records"
    )
    total_anomaly_count: int = Field(
        default=0,
        description="Total number of detected anomalies"
    )


class TelemetryIngestResponse(BaseModel):
    """Schema for telemetry ingestion response."""
    success: bool
    message: str
    telemetry_id: int
    timestamp: datetime
