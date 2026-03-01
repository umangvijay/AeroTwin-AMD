"""
AeroTwin Edge - Database Configuration
Production-grade SQLite/SQLAlchemy setup with environment-based configuration.
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.pool import StaticPool

from config import settings

# Connection pooling configuration for robust database operations
# SQLite: Use StaticPool for thread-safe single connection with background tasks
# PostgreSQL/MySQL: Use default QueuePool with connection pooling
if "sqlite" in settings.database_url:
    engine = create_engine(
        settings.database_url,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        echo=False
    )
else:
    # Production database with connection pooling
    engine = create_engine(
        settings.database_url,
        pool_size=10,
        max_overflow=20,
        pool_pre_ping=True,
        pool_recycle=3600,
        echo=False
    )

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def get_db():
    """Dependency for FastAPI to get database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Initialize database tables."""
    from models import Telemetry, Anomaly
    Base.metadata.create_all(bind=engine)
