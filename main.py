"""
AeroTwin Edge - Smart Campus AI Platform
Cloud-Ready FastAPI Backend with Bundled Edge Simulator

Features:
- ML-powered Anomaly Detection and Forecasting
- API key authentication for telemetry ingestion
- Environment-based configuration via pydantic-settings
- Bundled AMD Ryzen AI edge simulator for cloud deployment
- Cloud-safe hardware mocking (no psutil dependency on cloud)
"""

import asyncio
import logging
import sys
import random
import time
from contextlib import asynccontextmanager
from datetime import datetime
from typing import AsyncGenerator, Optional, Literal

from fastapi import FastAPI, Depends, HTTPException, status, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from sqlalchemy.orm import Session
from sqlalchemy import text

from config import settings
from database import get_db, init_db, SessionLocal
from models import Telemetry, Anomaly
from schemas import (
    TelemetryCreate,
    TelemetryResponse,
    TelemetryIngestResponse,
    MLDashboardResponse,
    AnomalyResponse,
)
from ml_services import anomaly_service, forecasting_service


class AeroTwinFormatter(logging.Formatter):
    """Custom formatter with color-coded log levels for enterprise logging."""
    
    COLORS = {
        'DEBUG': '\033[36m',
        'INFO': '\033[32m',
        'WARNING': '\033[33m',
        'ERROR': '\033[31m',
        'CRITICAL': '\033[35m',
    }
    RESET = '\033[0m'
    BOLD = '\033[1m'
    
    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, self.RESET)
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        formatted = (
            f"{self.BOLD}[{record.levelname:^8}]{self.RESET} "
            f"{color}{timestamp}{self.RESET} - "
            f"{self.BOLD}{record.name}{self.RESET}: {record.getMessage()}"
        )
        
        return formatted


def setup_logging() -> logging.Logger:
    """Configure enterprise-grade logging with custom formatting."""
    logger = logging.getLogger("AeroTwin")
    logger.setLevel(logging.INFO)
    
    if logger.handlers:
        logger.handlers.clear()
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(AeroTwinFormatter())
    
    logger.addHandler(console_handler)
    
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
    
    return logger


logger = setup_logging()

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


AIMode = Literal["ECO_INT8", "PERFORMANCE_FP32", "BALANCED_FP16", "STANDBY"]


class CloudEdgeSimulator:
    """
    Cloud-safe AMD Ryzen AI edge simulator.
    
    Generates realistic telemetry data with mocked hardware metrics
    perfect for cloud deployment where actual AMD NPU is not available.
    
    Features:
    - Realistic CPU/RAM simulation based on AI mode
    - Demo spike feature for guaranteed anomaly triggers
    - No psutil dependency - fully simulated metrics
    """
    
    MODE_CPU_PROFILES = {
        "ECO_INT8": (15, 30),
        "BALANCED_FP16": (35, 55),
        "PERFORMANCE_FP32": (60, 85),
        "STANDBY": (5, 15)
    }
    
    MODE_LATENCY_PROFILES = {
        "ECO_INT8": (8, 15),
        "BALANCED_FP16": (15, 25),
        "PERFORMANCE_FP32": (25, 45),
        "STANDBY": (50, 100)
    }
    
    def __init__(
        self,
        zone_id: str = None,
        demo_spike_interval: int = None
    ):
        self.zone_id = zone_id or settings.simulator_zone
        self.demo_spike_interval = demo_spike_interval or settings.demo_spike_interval
        self._cycle_count = 0
        self._base_cpu = random.uniform(20, 35)
        self._cpu_drift = 0.0
        
    def generate_occupancy(self) -> int:
            """
            Generate dynamic occupancy count (0-50) based on real-time datetime.

            Uses time-of-day logic for realistic patterns:
            - Business hours (9-17): High occupancy with natural variance
            - Transition hours (6-9, 17-20): Moderate occupancy
            - Night hours: Low occupancy

            Every Nth iteration forces a DEMO SPIKE (occupancy > 45)
            to guarantee PERFORMANCE_FP32 mode and anomaly detection triggers.
            """
            self._cycle_count += 1

            # Demo spike functionality - preserved for guaranteed anomaly triggers
            if self._cycle_count > 0 and self._cycle_count % self.demo_spike_interval == 0:
                occupancy = random.randint(46, 50)
                logger.warning(
                    f"Simulator: DEMO SPIKE triggered! Forcing high occupancy: {occupancy}"
                )
                return occupancy

            # Dynamic time-of-day based occupancy generation
            hour = datetime.now().hour

            # Business hours: high occupancy with natural variance
            if 9 <= hour <= 17:
                base = random.gauss(28, 10)
            # Transition hours: moderate occupancy
            elif 6 <= hour <= 9 or 17 <= hour <= 20:
                base = random.gauss(18, 7)
            # Night hours: low occupancy
            else:
                base = random.gauss(8, 4)

            # Clamp to valid range [0, 50]
            return max(0, min(50, int(base)))
    
    def determine_mode(self, occupancy: int) -> tuple[AIMode, str]:
        """Determine AI mode based on occupancy levels."""
        if occupancy < 15:
            mode: AIMode = "ECO_INT8"
            log = f"[ECO] Low occupancy ({occupancy}). INT8 quantization for power efficiency."
        elif occupancy <= 35:
            mode = "BALANCED_FP16"
            log = f"[BALANCED] Moderate activity ({occupancy}). FP16 mixed-precision mode."
        else:
            mode = "PERFORMANCE_FP32"
            log = f"[PERFORMANCE] High occupancy ({occupancy}). FP32 full-precision engaged."
        
        return mode, log
    
    def generate_cpu_metrics(self, mode: AIMode, occupancy: int) -> float:
        """
        Generate dynamic CPU metrics that correlate with AI mode and occupancy.

        CPU usage is influenced by:
        1. AI mode base range (ECO: 15-30%, BALANCED: 35-55%, PERFORMANCE: 60-85%)
        2. Occupancy level (higher occupancy = more processing = higher CPU)
        3. Realistic drift simulation for natural fluctuations
        4. Random variance for authentic system behavior

        Args:
            mode: Current AI mode determining base CPU profile
            occupancy: Current occupancy count (0-50) affecting CPU load

        Returns:
            CPU percentage (5-95%)
        """
        cpu_range = self.MODE_CPU_PROFILES[mode]

        # Update CPU drift with random walk for realistic fluctuations
        self._cpu_drift += random.uniform(-2, 2)
        self._cpu_drift = max(-5, min(5, self._cpu_drift))

        # Base CPU from mode profile
        base_cpu = random.uniform(*cpu_range)

        # Occupancy correlation: higher occupancy increases CPU load
        # Normalized occupancy (0-1 range)
        occupancy_normalized = min(occupancy / 50.0, 1.0)

        # Occupancy adds 0-15% CPU depending on load
        # Uses exponential curve for realistic resource scaling
        occupancy_impact = 15.0 * (occupancy_normalized ** 1.3)

        # Combine base CPU, occupancy impact, and drift
        cpu = base_cpu + occupancy_impact + self._cpu_drift

        # Add small random variance for natural system fluctuations
        cpu += random.uniform(-3, 3)

        # Clamp to realistic bounds [5%, 95%]
        return round(max(5, min(95, cpu)), 1)
    
    def generate_latency(self, mode: AIMode, cpu_percent: float) -> float:
        """
        Generate dynamic inference latency based on AI mode and CPU load.

        Latency is influenced by:
        1. AI mode base range (ECO: 8-15ms, BALANCED: 15-25ms, PERFORMANCE: 25-45ms)
        2. CPU load (higher CPU = higher latency due to resource contention)
        3. Random variance for realistic system behavior

        Args:
            mode: Current AI mode determining base latency profile
            cpu_percent: Current CPU usage percentage (0-100)

        Returns:
            Inference latency in milliseconds
        """
        latency_range = self.MODE_LATENCY_PROFILES[mode]
        
        # Base latency from mode profile
        base_latency = random.uniform(*latency_range)
        
        # CPU load impact: higher CPU increases latency due to resource contention
        # Normalized CPU (0-1 range)
        cpu_normalized = min(cpu_percent / 100.0, 1.0)
        
        # CPU adds 0-100% of base latency depending on load
        # Uses exponential curve for realistic resource contention
        cpu_impact_factor = cpu_normalized ** 1.5
        cpu_impact = base_latency * cpu_impact_factor
        
        # Combine base latency and CPU impact
        latency = base_latency + cpu_impact
        
        # Add random variance (±5%) for natural system fluctuations
        variance = latency * random.uniform(-0.05, 0.05)
        latency += variance
        
        # Ensure positive latency
        return round(max(1.0, latency), 2)
    
    def generate_telemetry(self) -> dict:
        """Generate a complete telemetry payload for the backend."""
        occupancy = self.generate_occupancy()
        mode, system_log = self.determine_mode(occupancy)
        cpu = self.generate_cpu_metrics(mode, occupancy)
        latency = self.generate_latency(mode, cpu)

        mode_mapping = {
            "ECO_INT8": "eco",
            "BALANCED_FP16": "auto",
            "PERFORMANCE_FP32": "performance",
            "STANDBY": "standby"
        }

        return {
            "zone": self.zone_id,
            "occupancy": occupancy,
            "ai_mode": mode_mapping[mode],
            "hardware_cpu": cpu,
            "inference_latency": latency,
            "system_log": system_log,
            "cycle": self._cycle_count
        }


cloud_simulator = CloudEdgeSimulator()


async def retry_db_operation(operation_name: str, max_retries: int = 3, base_delay: float = 1.0):
    """
    Decorator-like helper for database operations with exponential backoff.
    
    Args:
        operation_name: Name of the operation for logging
        max_retries: Maximum number of retry attempts (default: 3)
        base_delay: Initial delay in seconds for exponential backoff (default: 1.0)
    
    Returns:
        Tuple of (success: bool, retry_count: int, error: Optional[Exception])
    """
    retry_count = 0
    last_error = None
    
    while retry_count < max_retries:
        try:
            return True, retry_count, None
        except Exception as e:
            last_error = e
            retry_count += 1
            
            if retry_count < max_retries:
                delay = base_delay * (2 ** (retry_count - 1))  # Exponential backoff
                logger.warning(
                    f"Database: {operation_name} failed (attempt {retry_count}/{max_retries}) - "
                    f"Retrying in {delay:.1f}s - Error: {str(e)}"
                )
                await asyncio.sleep(delay)
            else:
                logger.error(
                    f"Database: {operation_name} failed after {max_retries} attempts - "
                    f"Final error: {str(e)}"
                )
    
    return False, retry_count, last_error


async def execute_with_retry(db_operation, operation_name: str, max_retries: int = 3):
    """
    Execute a database operation with retry logic and exponential backoff.
    
    Args:
        db_operation: Callable that performs the database operation
        operation_name: Name of the operation for logging
        max_retries: Maximum number of retry attempts
    
    Returns:
        Result of the operation or None if all retries failed
    """
    base_delay = 1.0
    
    for attempt in range(max_retries):
        try:
            return await db_operation()
        except Exception as e:
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)  # Exponential backoff: 1s, 2s, 4s
                logger.warning(
                    f"Database: {operation_name} failed (attempt {attempt + 1}/{max_retries}) - "
                    f"Retrying in {delay:.1f}s - Error: {str(e)}"
                )
                await asyncio.sleep(delay)
            else:
                logger.error(
                    f"Database: {operation_name} failed after {max_retries} attempts - "
                    f"Final error: {str(e)}"
                )
                raise


async def run_anomaly_detection_task() -> None:
    """
    Background task that runs anomaly detection at configured intervals.
    Uses Isolation Forest to detect anomalies in telemetry data.
    
    Implements robust session lifecycle management:
    - Creates new session per cycle
    - Automatic cleanup with context manager pattern
    - Proper error handling and rollback
    - Retry logic with exponential backoff for transient errors
    - No session leaks
    """
    logger.info("AI Engine: Anomaly detection pipeline initialized")
    logger.info(f"AI Engine: Detection interval set to {settings.ml_anomaly_interval}s")
    
    consecutive_failures = 0
    max_consecutive_failures = 5
    
    while True:
        try:
            async def perform_detection():
                """Inner function to perform detection with retry support."""
                db = SessionLocal()
                try:
                    data_count = db.query(Telemetry).count()
                    
                    if data_count < settings.ml_min_samples:
                        logger.info(
                            f"AI Engine: Warming up... ({data_count}/{settings.ml_min_samples} samples collected)"
                        )
                    else:
                        anomalies_detected = anomaly_service.detect_and_save_anomalies(db)
                        
                        if anomalies_detected > 0:
                            latest = db.query(Telemetry).order_by(Telemetry.timestamp.desc()).first()
                            zone = latest.zone if latest else "Unknown"
                            logger.warning(
                                f"AI Engine: Anomaly detected in {zone} - "
                                f"Isolation Forest flagged unusual pattern"
                            )
                        else:
                            logger.info("AI Engine: Scan complete - No anomalies detected")
                    
                    return True
                            
                except Exception as e:
                    logger.error(f"AI Engine: Detection cycle failed - {e}")
                    db.rollback()
                    raise
                finally:
                    db.close()
            
            # Execute with retry logic
            try:
                await execute_with_retry(perform_detection, "Anomaly Detection", max_retries=3)
                consecutive_failures = 0  # Reset on success
            except Exception as e:
                consecutive_failures += 1
                logger.error(
                    f"AI Engine: Detection cycle failed after retries "
                    f"(consecutive failures: {consecutive_failures}/{max_consecutive_failures})"
                )
                
                if consecutive_failures >= max_consecutive_failures:
                    logger.critical(
                        f"AI Engine: {max_consecutive_failures} consecutive failures detected - "
                        f"System may require manual intervention"
                    )
                    # Continue running but log critical status
                
        except Exception as e:
            logger.error(f"AI Engine: Unexpected error in detection task - {e}")
            consecutive_failures += 1
        
        await asyncio.sleep(settings.ml_anomaly_interval)



async def run_cloud_simulator_task() -> None:
    """
    Background task that generates simulated edge telemetry.
    
    Runs alongside the API server for cloud deployment where
    the external edge_simulator.py cannot be used.
    
    Features:
    - Cloud-safe AMD hardware mocking
    - No psutil dependency
    - Realistic CPU patterns per AI mode
    - Demo spike for guaranteed anomaly triggers
    - Robust session lifecycle management with retry logic
    """
    logger.info("=" * 60)
    logger.info("[System] Cloud Deployment Detected. Simulating AMD Ryzen AI edge telemetry.")
    logger.info("=" * 60)
    logger.info(f"Simulator: Zone = {cloud_simulator.zone_id}")
    logger.info(f"Simulator: Interval = {settings.simulator_interval}s")
    logger.info(f"Simulator: Demo Spike = Every {settings.demo_spike_interval} cycles")
    
    await asyncio.sleep(2.0)
    
    consecutive_failures = 0
    max_consecutive_failures = 5
    
    while True:
        try:
            async def generate_and_save_telemetry():
                """Inner function to generate and save telemetry with retry support."""
                db = SessionLocal()
                try:
                    telemetry = cloud_simulator.generate_telemetry()
                    
                    telemetry_record = Telemetry(
                        timestamp=datetime.utcnow(),
                        zone=telemetry["zone"],
                        occupancy=telemetry["occupancy"],
                        ai_mode=telemetry["ai_mode"],
                        hardware_cpu=telemetry["hardware_cpu"]
                    )
                    
                    db.add(telemetry_record)
                    db.commit()
                    db.refresh(telemetry_record)
                    
                    logger.info(
                        f"Simulator: Cycle {telemetry['cycle']} | "
                        f"Zone={telemetry['zone']} | "
                        f"Occ={telemetry['occupancy']} | "
                        f"Mode={telemetry['ai_mode'].upper()} | "
                        f"CPU={telemetry['hardware_cpu']}% | "
                        f"Latency={telemetry['inference_latency']}ms"
                    )
                    
                    return True
                        
                except Exception as e:
                    logger.error(f"Simulator: Telemetry generation failed - {e}")
                    db.rollback()
                    raise
                finally:
                    db.close()
            
            # Execute with retry logic
            try:
                await execute_with_retry(generate_and_save_telemetry, "Telemetry Generation", max_retries=3)
                consecutive_failures = 0  # Reset on success
            except Exception as e:
                consecutive_failures += 1
                logger.error(
                    f"Simulator: Telemetry generation failed after retries "
                    f"(consecutive failures: {consecutive_failures}/{max_consecutive_failures})"
                )
                
                if consecutive_failures >= max_consecutive_failures:
                    logger.critical(
                        f"Simulator: {max_consecutive_failures} consecutive failures detected - "
                        f"System may require manual intervention"
                    )
                    # Continue running but log critical status
                
        except Exception as e:
            logger.error(f"Simulator: Unexpected error in simulator task - {e}")
            consecutive_failures += 1
        
        await asyncio.sleep(settings.simulator_interval)



async def verify_api_key(api_key: Optional[str] = Security(api_key_header)) -> str:
    """
    Validate API key from request header.
    Raises 401 Unauthorized if key is missing or invalid.
    """
    if api_key is None:
        logger.warning("Security: API key missing from request")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required. Include 'X-API-Key' header.",
            headers={"WWW-Authenticate": "ApiKey"}
        )
    
    if api_key != settings.api_key:
        logger.warning("Security: Invalid API key attempted")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "ApiKey"}
        )
    
    return api_key


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan handler.
    Initializes database and starts background tasks on startup.
    """
    logger.info("=" * 60)
    logger.info("AeroTwin Edge Platform - Cloud-Ready Production Mode")
    logger.info("=" * 60)
    logger.info(f"Config: Database URL = {settings.database_url}")
    logger.info(f"Config: API Key configured = {'Yes' if settings.api_key else 'No'}")
    logger.info(f"Config: ML Interval = {settings.ml_anomaly_interval}s")
    logger.info(f"Config: Cloud Mode = {settings.cloud_mode}")
    logger.info(f"Config: Simulator Enabled = {settings.simulator_enabled}")
    
    init_db()
    logger.info("Database: Connection established")
    logger.info("Database: Schema validation complete")
    
    background_tasks = []
    
    anomaly_task = asyncio.create_task(run_anomaly_detection_task())
    background_tasks.append(anomaly_task)
    logger.info("ML Pipeline: Anomaly detection task started")
    
    if settings.simulator_enabled:
        simulator_task = asyncio.create_task(run_cloud_simulator_task())
        background_tasks.append(simulator_task)
        logger.info("Simulator: Cloud edge simulator task started")
    
    logger.info("System: AeroTwin Edge is now ONLINE")
    logger.info("=" * 60)
    
    yield
    
    logger.info("System: Initiating graceful shutdown...")
    
    for task in background_tasks:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
    
    logger.info("ML Pipeline: Background tasks terminated")
    logger.info("System: AeroTwin Edge shutdown complete")


app = FastAPI(
    title="AeroTwin Edge",
    description="Cloud-Ready Smart Campus AI Platform with ML-powered analytics and bundled edge simulator",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["Health"])
async def root() -> dict:
    """Root endpoint with service information."""
    return {
        "service": "AeroTwin Edge",
        "status": "operational",
        "version": "1.0.0",
        "model": "AeroTwin Edge AI",
        "deployment": "cloud" if settings.cloud_mode else "local",
        "simulator": "active" if settings.simulator_enabled else "disabled",
        "timestamp": datetime.utcnow().isoformat()
    }


@app.get("/api/health", tags=["Health"])
async def health_check(db: Session = Depends(get_db)) -> dict:
    """
    Comprehensive health check endpoint.
    Returns service status, database connectivity, and ML model state.
    """
    db_status = "disconnected"
    data_count = 0
    ml_status = "unknown"
    
    try:
        db.execute(text("SELECT 1"))
        db_status = "connected"
        
        data_count = db.query(Telemetry).count()
        
        if data_count < settings.ml_min_samples:
            ml_status = f"warming_up ({data_count}/{settings.ml_min_samples} samples)"
        elif anomaly_service._is_fitted:
            ml_status = "active"
        else:
            ml_status = "fitting_model"
            
    except Exception as e:
        error_msg = str(e)
        
        # Classify error severity for health check failures
        if "connection" in error_msg.lower() or "timeout" in error_msg.lower():
            logger.critical(f"Health Check: Database connection failure - {error_msg}")
        elif "operational" in error_msg.lower():
            logger.error(f"Health Check: Database operational error - {error_msg}")
        else:
            logger.error(f"Health Check: Database query failed - {error_msg}")
        
        db_status = "error"
        ml_status = "unavailable"
    
    return {
        "status": "online",
        "model": "AeroTwin Edge AI",
        "database": db_status,
        "ml_engine": ml_status,
        "simulator": "active" if settings.simulator_enabled else "disabled",
        "telemetry_count": data_count,
        "cloud_mode": settings.cloud_mode,
        "uptime": "operational",
        "timestamp": datetime.utcnow().isoformat()
    }


@app.get("/api/simulator/status", tags=["Simulator"])
async def simulator_status() -> dict:
    """Get current simulator status and statistics."""
    return {
        "enabled": settings.simulator_enabled,
        "zone": cloud_simulator.zone_id,
        "cycle_count": cloud_simulator._cycle_count,
        "demo_spike_interval": cloud_simulator.demo_spike_interval,
        "next_spike_in": cloud_simulator.demo_spike_interval - (cloud_simulator._cycle_count % cloud_simulator.demo_spike_interval),
        "interval_seconds": settings.simulator_interval
    }


@app.post(
    "/api/telemetry",
    response_model=TelemetryIngestResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Telemetry"]
)
async def ingest_telemetry(
    telemetry_data: TelemetryCreate,
    db: Session = Depends(get_db),
    api_key: str = Depends(verify_api_key)
) -> TelemetryIngestResponse:
    """
    Ingest telemetry data from campus sensors.
    
    **Requires API Key**: Include `X-API-Key` header with valid key.
    
    Accepts occupancy, AI mode, and hardware metrics from edge devices.
    Data is validated and stored for ML pipeline processing.
    """
    try:
        telemetry_record = Telemetry(
            timestamp=datetime.utcnow(),
            zone=telemetry_data.zone,
            occupancy=telemetry_data.occupancy,
            ai_mode=telemetry_data.ai_mode,
            hardware_cpu=telemetry_data.hardware_cpu
        )
        
        db.add(telemetry_record)
        db.commit()
        db.refresh(telemetry_record)
        
        logger.info(
            f"Telemetry: Ingested from {telemetry_data.zone} | "
            f"Occupancy={telemetry_data.occupancy} | "
            f"Mode={telemetry_data.ai_mode.upper()}"
        )
        
        return TelemetryIngestResponse(
            success=True,
            message="Telemetry data ingested successfully",
            telemetry_id=telemetry_record.id,
            timestamp=telemetry_record.timestamp
        )
    
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Telemetry: Ingestion failed - {error_msg}")
        
        try:
            db.rollback()
            logger.info("Database: Transaction rolled back successfully")
        except Exception as rollback_error:
            logger.critical(f"Database: Rollback failed - {rollback_error}")
        
        # Classify error severity
        if "connection" in error_msg.lower() or "timeout" in error_msg.lower():
            logger.critical(f"Database: Connection error during telemetry ingestion - {error_msg}")
        elif "constraint" in error_msg.lower() or "integrity" in error_msg.lower():
            logger.warning(f"Database: Data integrity error during telemetry ingestion - {error_msg}")
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to ingest telemetry data: {error_msg}"
        )


@app.get(
    "/api/ml_dashboard",
    response_model=MLDashboardResponse,
    tags=["ML Dashboard"]
)
async def get_ml_dashboard(
    db: Session = Depends(get_db)
) -> MLDashboardResponse:
    """
    Get ML dashboard data including latest telemetry, anomalies, and forecasts.
    
    Returns:
        - Latest telemetry reading
        - Recent anomalies (last 10)
        - Occupancy forecast for next 3 intervals (Holt-Winters)
        - Model status and statistics
    """
    try:
        latest_telemetry = (
            db.query(Telemetry)
            .order_by(Telemetry.timestamp.desc())
            .first()
        )
        
        recent_anomalies = (
            db.query(Anomaly)
            .order_by(Anomaly.timestamp.desc())
            .limit(10)
            .all()
        )
        
        total_telemetry = db.query(Telemetry).count()
        
        forecast = []
        if total_telemetry >= 5:
            forecast = forecasting_service.forecast_occupancy(db)
        
        total_anomalies = db.query(Anomaly).count()
        
        latest_telemetry_response = None
        if latest_telemetry:
            latest_telemetry_response = TelemetryResponse(
                id=latest_telemetry.id,
                timestamp=latest_telemetry.timestamp,
                zone=latest_telemetry.zone,
                occupancy=latest_telemetry.occupancy,
                ai_mode=latest_telemetry.ai_mode,
                hardware_cpu=latest_telemetry.hardware_cpu
            )
        
        anomaly_responses = [
            AnomalyResponse(
                id=a.id,
                timestamp=a.timestamp,
                severity=a.severity,
                description=a.description
            )
            for a in recent_anomalies
        ]
        
        if total_telemetry < settings.ml_min_samples:
            model_status = "warming_up"
        elif anomaly_service._is_fitted:
            model_status = "active"
        else:
            model_status = "fitting"
        
        return MLDashboardResponse(
            latest_telemetry=latest_telemetry_response,
            recent_anomalies=anomaly_responses,
            forecast=forecast,
            model_status=model_status,
            total_telemetry_count=total_telemetry,
            total_anomaly_count=total_anomalies
        )
    
    except Exception as e:
        logger.error(f"Dashboard: Failed to fetch data - {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch dashboard data: {str(e)}"
        )


@app.get(
    "/api/telemetry/recent",
    response_model=list[TelemetryResponse],
    tags=["Telemetry"]
)
async def get_recent_telemetry(
    limit: int = 50,
    db: Session = Depends(get_db)
) -> list[TelemetryResponse]:
    """Get recent telemetry records."""
    records = (
        db.query(Telemetry)
        .order_by(Telemetry.timestamp.desc())
        .limit(min(limit, 100))
        .all()
    )
    
    return [
        TelemetryResponse(
            id=r.id,
            timestamp=r.timestamp,
            zone=r.zone,
            occupancy=r.occupancy,
            ai_mode=r.ai_mode,
            hardware_cpu=r.hardware_cpu
        )
        for r in records
    ]


@app.get(
    "/api/anomalies",
    response_model=list[AnomalyResponse],
    tags=["Anomalies"]
)
async def get_anomalies(
    limit: int = 50,
    severity: str | None = None,
    db: Session = Depends(get_db)
) -> list[AnomalyResponse]:
    """Get anomaly records with optional severity filter."""
    query = db.query(Anomaly)
    
    if severity:
        query = query.filter(Anomaly.severity == severity.lower())
    
    records = (
        query
        .order_by(Anomaly.timestamp.desc())
        .limit(min(limit, 100))
        .all()
    )
    
    return [
        AnomalyResponse(
            id=r.id,
            timestamp=r.timestamp,
            severity=r.severity,
            description=r.description
        )
        for r in records
    ]


if __name__ == "__main__":
    import uvicorn
    
    logger.info("Starting AeroTwin Edge server...")
    
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level="warning"
    )
