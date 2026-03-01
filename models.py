"""
AeroTwin Edge - SQLAlchemy Database Models
Telemetry and Anomaly tables for Smart Campus AI platform
"""

from datetime import datetime
from sqlalchemy import Column, Integer, Float, String, DateTime
from database import Base


class Telemetry(Base):
    """
    Telemetry data model for storing sensor readings from campus zones.
    
    Attributes:
        id: Primary key
        timestamp: When the reading was recorded
        zone: Campus zone identifier (e.g., "Building-A-Floor-1")
        occupancy: Number of people detected in the zone
        ai_mode: Current AI operation mode (e.g., "auto", "manual", "eco")
        hardware_cpu: CPU utilization percentage of edge hardware
    """
    __tablename__ = "telemetry"

    id: int = Column(Integer, primary_key=True, index=True, autoincrement=True)
    timestamp: datetime = Column(DateTime, default=datetime.utcnow, index=True)
    zone: str = Column(String(100), nullable=False, index=True)
    occupancy: int = Column(Integer, nullable=False)
    ai_mode: str = Column(String(50), nullable=False)
    hardware_cpu: float = Column(Float, nullable=False)

    def __repr__(self) -> str:
        return f"<Telemetry(id={self.id}, zone={self.zone}, occupancy={self.occupancy})>"


class Anomaly(Base):
    """
    Anomaly records detected by the ML pipeline.
    
    Attributes:
        id: Primary key
        timestamp: When the anomaly was detected
        severity: Severity level (e.g., "low", "medium", "high", "critical")
        description: Human-readable description of the anomaly
    """
    __tablename__ = "anomalies"

    id: int = Column(Integer, primary_key=True, index=True, autoincrement=True)
    timestamp: datetime = Column(DateTime, default=datetime.utcnow, index=True)
    severity: str = Column(String(20), nullable=False, index=True)
    description: str = Column(String(500), nullable=False)

    def __repr__(self) -> str:
        return f"<Anomaly(id={self.id}, severity={self.severity})>"
