"""
AeroTwin Edge - Machine Learning Services
Production-grade Anomaly Detection and Time-Series Forecasting pipelines.

Uses:
- scikit-learn IsolationForest for anomaly detection
- statsmodels ExponentialSmoothing (Holt-Winters) for time-series forecasting
"""

import logging
from datetime import datetime
from typing import List, Tuple, Optional
import warnings

import numpy as np
from sklearn.ensemble import IsolationForest
from sqlalchemy.orm import Session

from models import Telemetry, Anomaly
from schemas import ForecastPoint
from config import settings

warnings.filterwarnings("ignore", category=UserWarning)

logger = logging.getLogger("AeroTwin")


class AnomalyDetectionService:
    """
    Isolation Forest-based anomaly detection service.
    Continuously monitors telemetry data and detects anomalies
    in occupancy and CPU load patterns.
    """

    def __init__(
        self,
        contamination: float = None,
        n_estimators: int = 100
    ):
        self.contamination = contamination or settings.ml_contamination
        self.n_estimators = n_estimators
        self.model: Optional[IsolationForest] = None
        self._is_fitted = False
        self._min_samples = settings.ml_min_samples

    def fit_and_predict(
        self,
        db: Session,
        window_size: int = 50
    ) -> List[Tuple[Telemetry, bool]]:
        """
        Fetch recent telemetry, fit the model, and predict anomalies.
        """
        telemetry_records = (
            db.query(Telemetry)
            .order_by(Telemetry.timestamp.desc())
            .limit(window_size)
            .all()
        )

        if len(telemetry_records) < self._min_samples:
            logger.info(
                f"Insufficient data for anomaly detection. "
                f"Need {self._min_samples}, have {len(telemetry_records)}"
            )
            return []

        features = np.array([
            [record.occupancy, record.hardware_cpu]
            for record in telemetry_records
        ])

        self.model = IsolationForest(
            contamination=self.contamination,
            n_estimators=self.n_estimators,
            random_state=42,
            n_jobs=-1
        )

        predictions = self.model.fit_predict(features)
        self._is_fitted = True

        results = []
        for record, prediction in zip(telemetry_records, predictions):
            is_anomaly = prediction == -1
            results.append((record, is_anomaly))

        return results

    def detect_and_save_anomalies(self, db: Session) -> int:
        """
        Run anomaly detection and save detected anomalies to database.
        """
        results = self.fit_and_predict(db)

        if not results:
            return 0

        latest_record, is_anomaly = results[0]

        if not is_anomaly:
            return 0

        severity = self._calculate_severity(
            latest_record.occupancy,
            latest_record.hardware_cpu,
            results
        )

        description = self._generate_description(
            latest_record,
            severity
        )

        existing = (
            db.query(Anomaly)
            .filter(
                Anomaly.timestamp >= latest_record.timestamp,
                Anomaly.description.contains(latest_record.zone)
            )
            .first()
        )

        if existing:
            return 0

        anomaly = Anomaly(
            timestamp=datetime.utcnow(),
            severity=severity,
            description=description
        )
        db.add(anomaly)
        db.commit()

        logger.warning(f"Anomaly detected: {description}")
        return 1

    def _calculate_severity(
        self,
        occupancy: int,
        cpu: float,
        results: List[Tuple[Telemetry, bool]]
    ) -> str:
        """Calculate severity based on deviation from normal patterns."""
        normal_records = [r for r, is_anom in results if not is_anom]

        if not normal_records:
            return "medium"

        avg_occupancy = np.mean([r.occupancy for r in normal_records])
        avg_cpu = np.mean([r.hardware_cpu for r in normal_records])

        occupancy_deviation = abs(occupancy - avg_occupancy) / max(avg_occupancy, 1)
        cpu_deviation = abs(cpu - avg_cpu) / max(avg_cpu, 1)

        combined_deviation = (occupancy_deviation + cpu_deviation) / 2

        if combined_deviation > 1.0:
            return "critical"
        elif combined_deviation > 0.5:
            return "high"
        elif combined_deviation > 0.25:
            return "medium"
        else:
            return "low"

    def _generate_description(
        self,
        record: Telemetry,
        severity: str
    ) -> str:
        """Generate human-readable anomaly description."""
        return (
            f"Anomaly detected in zone '{record.zone}': "
            f"Occupancy={record.occupancy}, CPU={record.hardware_cpu:.1f}%. "
            f"Pattern deviates significantly from normal behavior. "
            f"Severity: {severity.upper()}"
        )


class ForecastingService:
    """
    Production-grade time-series forecasting service using statsmodels
    Holt-Winters Exponential Smoothing for mathematically accurate predictions.
    
    Falls back to Double Exponential Smoothing if statsmodels is unavailable.
    """

    def __init__(self):
        self._statsmodels_available = False
        try:
            from statsmodels.tsa.holtwinters import ExponentialSmoothing
            self._statsmodels_available = True
            self._ExponentialSmoothing = ExponentialSmoothing
            logger.info("ForecastingService: Using statsmodels Holt-Winters")
        except ImportError:
            logger.warning(
                "ForecastingService: statsmodels not available, using fallback method"
            )

    def forecast_occupancy(
        self,
        db: Session,
        history_size: int = None,
        forecast_intervals: int = None
    ) -> List[ForecastPoint]:
        """
        Predict occupancy for next intervals using Holt-Winters Exponential Smoothing.
        
        This provides mathematically sound forecasting with:
        - Level component (baseline)
        - Trend component (direction)
        - Optional seasonal component
        """
        history_size = history_size or settings.forecast_history_size
        forecast_intervals = forecast_intervals or settings.forecast_intervals
        
        telemetry_records = (
            db.query(Telemetry)
            .order_by(Telemetry.timestamp.desc())
            .limit(history_size)
            .all()
        )

        if len(telemetry_records) < 5:
            logger.info("Insufficient data for forecasting")
            return []

        occupancy_series = np.array([
            record.occupancy for record in reversed(telemetry_records)
        ], dtype=float)

        if self._statsmodels_available:
            return self._holt_winters_forecast(
                occupancy_series, forecast_intervals
            )
        else:
            return self._fallback_forecast(
                occupancy_series, forecast_intervals
            )

    def _holt_winters_forecast(
        self,
        series: np.ndarray,
        forecast_periods: int
    ) -> List[ForecastPoint]:
        """
        Production Holt-Winters Exponential Smoothing forecast.
        
        Uses additive trend for occupancy data which typically shows
        linear growth patterns during peak hours.
        """
        try:
            series_clean = np.maximum(series, 0.1)
            
            if len(series_clean) < 10:
                model = self._ExponentialSmoothing(
                    series_clean,
                    trend='add',
                    seasonal=None,
                    damped_trend=True,
                    initialization_method='estimated'
                )
            else:
                model = self._ExponentialSmoothing(
                    series_clean,
                    trend='add',
                    seasonal=None,
                    damped_trend=True,
                    initialization_method='estimated'
                )
            
            fitted_model = model.fit(optimized=True, use_brute=False)
            
            forecasts = fitted_model.forecast(forecast_periods)
            
            residuals = series_clean - fitted_model.fittedvalues
            std_residuals = np.std(residuals)
            
            result = []
            for i in range(forecast_periods):
                h = i + 1
                margin = 1.96 * std_residuals * np.sqrt(h)
                
                forecast_value = max(0, forecasts.iloc[i] if hasattr(forecasts, 'iloc') else forecasts[i])
                
                forecast_point = ForecastPoint(
                    interval=h,
                    predicted_occupancy=round(forecast_value, 2),
                    confidence_lower=round(max(0, forecast_value - margin), 2),
                    confidence_upper=round(forecast_value + margin, 2)
                )
                result.append(forecast_point)
            
            return result
            
        except Exception as e:
            logger.warning(f"Holt-Winters forecast failed: {e}, using fallback")
            return self._fallback_forecast(series, forecast_periods)

    def _fallback_forecast(
        self,
        series: np.ndarray,
        forecast_periods: int
    ) -> List[ForecastPoint]:
        """
        Fallback Double Exponential Smoothing implementation.
        Used when statsmodels is not available.
        """
        alpha = 0.3
        beta = 0.1
        
        n = len(series)
        level = series[0]
        trend = np.mean(np.diff(series[:min(5, n)]))

        levels = np.zeros(n)
        trends = np.zeros(n)
        fitted = np.zeros(n)

        for t in range(n):
            if t == 0:
                levels[t] = level
                trends[t] = trend
                fitted[t] = level
            else:
                levels[t] = alpha * series[t] + (1 - alpha) * (levels[t-1] + trends[t-1])
                trends[t] = beta * (levels[t] - levels[t-1]) + (1 - beta) * trends[t-1]
                fitted[t] = levels[t-1] + trends[t-1]

        residuals = series - fitted
        std_residuals = np.std(residuals)

        result = []
        for h in range(1, forecast_periods + 1):
            forecast = levels[-1] + h * trends[-1]
            forecast = max(0, forecast)

            margin = 1.96 * std_residuals * np.sqrt(h)
            
            forecast_point = ForecastPoint(
                interval=h,
                predicted_occupancy=round(forecast, 2),
                confidence_lower=round(max(0, forecast - margin), 2),
                confidence_upper=round(forecast + margin, 2)
            )
            result.append(forecast_point)

        return result


anomaly_service = AnomalyDetectionService()
forecasting_service = ForecastingService()
