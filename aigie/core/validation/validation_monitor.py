"""
Validation Performance Monitor - Real-time monitoring and optimization.
"""

import asyncio
import time
import logging
import psutil
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import deque, defaultdict
import threading
import json

from ..types.validation_types import ExecutionStep, ValidationResult, ValidationStatus
from .runtime_validator import RuntimeValidator as AdvancedRuntimeValidator
from .validation_pipeline import ValidationPipeline


@dataclass
class PerformanceAlert:
    """Performance alert configuration."""
    
    metric_name: str
    threshold: float
    operator: str  # "gt", "lt", "eq", "gte", "lte"
    severity: str  # "low", "medium", "high", "critical"
    enabled: bool = True
    cooldown_seconds: int = 60
    last_triggered: Optional[datetime] = None


@dataclass
class ValidationMetrics:
    """Comprehensive validation metrics."""
    
    # Basic metrics
    total_validations: int = 0
    successful_validations: int = 0
    failed_validations: int = 0
    
    # Timing metrics
    avg_validation_time: float = 0.0
    min_validation_time: float = float('inf')
    max_validation_time: float = 0.0
    p95_validation_time: float = 0.0
    p99_validation_time: float = 0.0
    
    # Quality metrics
    avg_confidence: float = 0.0
    high_confidence_rate: float = 0.0  # Percentage with confidence > 0.8
    low_confidence_rate: float = 0.0   # Percentage with confidence < 0.5
    
    # Performance metrics
    cache_hit_rate: float = 0.0
    parallel_utilization: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    
    # Error metrics
    error_rate: float = 0.0
    timeout_rate: float = 0.0
    retry_rate: float = 0.0
    
    # Pattern learning metrics
    pattern_matches: int = 0
    pattern_accuracy: float = 0.0
    learning_rate: float = 0.0
    
    def update(self, validation_time: float, confidence: float, success: bool, 
               cache_hit: bool, parallel_used: bool, error: bool = False, 
               timeout: bool = False, retry: bool = False, pattern_match: bool = False):
        """Update metrics with new validation data."""
        self.total_validations += 1
        
        if success:
            self.successful_validations += 1
        else:
            self.failed_validations += 1
        
        # Update timing metrics
        self.avg_validation_time = (
            (self.avg_validation_time * (self.total_validations - 1) + validation_time) 
            / self.total_validations
        )
        self.min_validation_time = min(self.min_validation_time, validation_time)
        self.max_validation_time = max(self.max_validation_time, validation_time)
        
        # Update quality metrics
        self.avg_confidence = (
            (self.avg_confidence * (self.total_validations - 1) + confidence) 
            / self.total_validations
        )
        
        if confidence > 0.8:
            self.high_confidence_rate = (
                (self.high_confidence_rate * (self.total_validations - 1) + 1) 
                / self.total_validations
            )
        
        if confidence < 0.5:
            self.low_confidence_rate = (
                (self.low_confidence_rate * (self.total_validations - 1) + 1) 
                / self.total_validations
            )
        
        # Update performance metrics
        if cache_hit:
            self.cache_hit_rate = (
                (self.cache_hit_rate * (self.total_validations - 1) + 1) 
                / self.total_validations
            )
        
        if parallel_used:
            self.parallel_utilization = (
                (self.parallel_utilization * (self.total_validations - 1) + 1) 
                / self.total_validations
            )
        
        # Update error metrics
        if error:
            self.error_rate = (
                (self.error_rate * (self.total_validations - 1) + 1) 
                / self.total_validations
            )
        
        if timeout:
            self.timeout_rate = (
                (self.timeout_rate * (self.total_validations - 1) + 1) 
                / self.total_validations
            )
        
        if retry:
            self.retry_rate = (
                (self.retry_rate * (self.total_validations - 1) + 1) 
                / self.total_validations
            )
        
        if pattern_match:
            self.pattern_matches += 1


@dataclass
class ValidationTrend:
    """Validation trend analysis."""
    
    metric_name: str
    current_value: float
    trend_direction: str  # "increasing", "decreasing", "stable"
    trend_strength: float  # 0.0 to 1.0
    prediction: Optional[float] = None
    confidence: float = 0.0


class ValidationMonitor:
    """Real-time validation performance monitor."""
    
    def __init__(self, validator: Optional[AdvancedRuntimeValidator] = None, 
                 pipeline: Optional[ValidationPipeline] = None):
        self.validator = validator
        self.pipeline = pipeline
        self.metrics = ValidationMetrics()
        
        # Performance tracking
        self.validation_times = deque(maxlen=1000)  # Keep last 1000 validation times
        self.confidence_scores = deque(maxlen=1000)
        self.error_counts = defaultdict(int)
        self.performance_history = deque(maxlen=100)  # Keep last 100 performance snapshots
        
        # Alerting system
        self.alerts: List[PerformanceAlert] = []
        self.alert_handlers: List[Callable] = []
        self.alert_cooldowns = {}
        
        # Monitoring thread
        self.monitoring_active = False
        self.monitoring_thread = None
        self.monitoring_interval = 5.0  # seconds
        
        # Performance optimization
        self.optimization_enabled = True
        self.auto_tuning_enabled = True
        
        logging.info("ValidationMonitor initialized")
    
    def start_monitoring(self):
        """Start the monitoring thread."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        logging.info("Validation monitoring started")
    
    def stop_monitoring(self):
        """Stop the monitoring thread."""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
        logging.info("Validation monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                # Update system metrics
                self._update_system_metrics()
                
                # Check for alerts
                self._check_alerts()
                
                # Perform optimization if enabled
                if self.optimization_enabled:
                    self._perform_optimization()
                
                # Record performance snapshot
                self._record_performance_snapshot()
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                logging.error(f"Monitoring loop error: {e}")
                time.sleep(self.monitoring_interval)
    
    def _update_system_metrics(self):
        """Update system-level metrics."""
        try:
            # Memory usage
            process = psutil.Process()
            memory_info = process.memory_info()
            self.metrics.memory_usage_mb = memory_info.rss / 1024 / 1024
            
            # CPU usage
            self.metrics.cpu_usage_percent = process.cpu_percent()
            
        except Exception as e:
            logging.warning(f"Failed to update system metrics: {e}")
    
    def _check_alerts(self):
        """Check for performance alerts."""
        current_time = datetime.now()
        
        for alert in self.alerts:
            if not alert.enabled:
                continue
            
            # Check cooldown
            if (alert.last_triggered and 
                (current_time - alert.last_triggered).total_seconds() < alert.cooldown_seconds):
                continue
            
            # Get current metric value
            metric_value = self._get_metric_value(alert.metric_name)
            if metric_value is None:
                continue
            
            # Check threshold
            should_alert = self._evaluate_alert_condition(
                metric_value, alert.threshold, alert.operator
            )
            
            if should_alert:
                self._trigger_alert(alert, metric_value)
                alert.last_triggered = current_time
    
    def _get_metric_value(self, metric_name: str) -> Optional[float]:
        """Get current value of a metric."""
        metric_map = {
            "avg_validation_time": self.metrics.avg_validation_time,
            "error_rate": self.metrics.error_rate,
            "cache_hit_rate": self.metrics.cache_hit_rate,
            "memory_usage_mb": self.metrics.memory_usage_mb,
            "cpu_usage_percent": self.metrics.cpu_usage_percent,
            "success_rate": self.successful_validations / max(1, self.total_validations),
            "avg_confidence": self.metrics.avg_confidence
        }
        return metric_map.get(metric_name)
    
    def _evaluate_alert_condition(self, value: float, threshold: float, operator: str) -> bool:
        """Evaluate alert condition."""
        if operator == "gt":
            return value > threshold
        elif operator == "lt":
            return value < threshold
        elif operator == "eq":
            return abs(value - threshold) < 0.001
        elif operator == "gte":
            return value >= threshold
        elif operator == "lte":
            return value <= threshold
        else:
            return False
    
    def _trigger_alert(self, alert: PerformanceAlert, current_value: float):
        """Trigger a performance alert."""
        alert_data = {
            "metric_name": alert.metric_name,
            "current_value": current_value,
            "threshold": alert.threshold,
            "severity": alert.severity,
            "timestamp": datetime.now().isoformat()
        }
        
        logging.warning(f"Performance alert: {alert_data}")
        
        # Notify alert handlers
        for handler in self.alert_handlers:
            try:
                handler(alert_data)
            except Exception as e:
                logging.error(f"Alert handler failed: {e}")
    
    def _perform_optimization(self):
        """Perform automatic optimization based on metrics."""
        if not self.auto_tuning_enabled:
            return
        
        try:
            # Optimize cache settings
            if self.metrics.cache_hit_rate < 0.3:
                self._optimize_cache_settings()
            
            # Optimize parallel processing
            if self.metrics.parallel_utilization < 0.5:
                self._optimize_parallel_settings()
            
            # Optimize validation strategies
            if self.metrics.avg_validation_time > 5.0:
                self._optimize_validation_strategies()
            
        except Exception as e:
            logging.warning(f"Optimization failed: {e}")
    
    def _optimize_cache_settings(self):
        """Optimize cache settings based on performance."""
        if not self.validator:
            return
        
        # Increase cache TTL if hit rate is low
        if self.metrics.cache_hit_rate < 0.3:
            self.validator.config.cache_ttl_seconds = min(
                self.validator.config.cache_ttl_seconds * 1.5, 600
            )
            logging.info(f"Increased cache TTL to {self.validator.config.cache_ttl_seconds}s")
    
    def _optimize_parallel_settings(self):
        """Optimize parallel processing settings."""
        if not self.validator:
            return
        
        # Increase parallel workers if utilization is low
        if self.metrics.parallel_utilization < 0.5:
            self.validator.config.max_concurrent_validations = min(
                self.validator.config.max_concurrent_validations + 2, 20
            )
            logging.info(f"Increased parallel workers to {self.validator.config.max_concurrent_validations}")
    
    def _optimize_validation_strategies(self):
        """Optimize validation strategies based on performance."""
        if not self.validator:
            return
        
        # Reduce strategies if validation is too slow
        if self.metrics.avg_validation_time > 5.0:
            # Keep only essential strategies
            essential_strategies = [
                ValidationStrategy.GOAL_ALIGNMENT,
                ValidationStrategy.SAFETY_COMPLIANCE
            ]
            self.validator.config.enabled_strategies = essential_strategies
            logging.info("Reduced validation strategies for performance")
    
    def _record_performance_snapshot(self):
        """Record a performance snapshot."""
        snapshot = {
            "timestamp": datetime.now(),
            "metrics": {
                "total_validations": self.metrics.total_validations,
                "avg_validation_time": self.metrics.avg_validation_time,
                "error_rate": self.metrics.error_rate,
                "cache_hit_rate": self.metrics.cache_hit_rate,
                "memory_usage_mb": self.metrics.memory_usage_mb,
                "cpu_usage_percent": self.metrics.cpu_usage_percent
            }
        }
        self.performance_history.append(snapshot)
    
    def record_validation(self, step: ExecutionStep, result: ValidationResult, 
                         validation_time: float, cache_hit: bool = False, 
                         parallel_used: bool = False, error: bool = False,
                         timeout: bool = False, retry: bool = False,
                         pattern_match: bool = False):
        """Record a validation event."""
        # Update metrics
        self.metrics.update(
            validation_time=validation_time,
            confidence=result.confidence,
            success=result.is_valid,
            cache_hit=cache_hit,
            parallel_used=parallel_used,
            error=error,
            timeout=timeout,
            retry=retry,
            pattern_match=pattern_match
        )
        
        # Record in history
        self.validation_times.append(validation_time)
        self.confidence_scores.append(result.confidence)
        
        # Track errors
        if error:
            error_type = type(result).__name__
            self.error_counts[error_type] += 1
    
    def add_alert(self, alert: PerformanceAlert):
        """Add a performance alert."""
        self.alerts.append(alert)
        logging.info(f"Added alert for {alert.metric_name}")
    
    def remove_alert(self, metric_name: str):
        """Remove alerts for a specific metric."""
        self.alerts = [alert for alert in self.alerts if alert.metric_name != metric_name]
        logging.info(f"Removed alerts for {metric_name}")
    
    def add_alert_handler(self, handler: Callable):
        """Add an alert handler."""
        self.alert_handlers.append(handler)
    
    def remove_alert_handler(self, handler: Callable):
        """Remove an alert handler."""
        if handler in self.alert_handlers:
            self.alert_handlers.remove(handler)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics."""
        return {
            "basic_metrics": {
                "total_validations": self.metrics.total_validations,
                "successful_validations": self.metrics.successful_validations,
                "failed_validations": self.metrics.failed_validations,
                "success_rate": self.metrics.successful_validations / max(1, self.metrics.total_validations)
            },
            "timing_metrics": {
                "avg_validation_time": self.metrics.avg_validation_time,
                "min_validation_time": self.metrics.min_validation_time if self.metrics.min_validation_time != float('inf') else 0,
                "max_validation_time": self.metrics.max_validation_time,
                "p95_validation_time": self._calculate_percentile(list(self.validation_times), 95),
                "p99_validation_time": self._calculate_percentile(list(self.validation_times), 99)
            },
            "quality_metrics": {
                "avg_confidence": self.metrics.avg_confidence,
                "high_confidence_rate": self.metrics.high_confidence_rate,
                "low_confidence_rate": self.metrics.low_confidence_rate
            },
            "performance_metrics": {
                "cache_hit_rate": self.metrics.cache_hit_rate,
                "parallel_utilization": self.metrics.parallel_utilization,
                "memory_usage_mb": self.metrics.memory_usage_mb,
                "cpu_usage_percent": self.metrics.cpu_usage_percent
            },
            "error_metrics": {
                "error_rate": self.metrics.error_rate,
                "timeout_rate": self.metrics.timeout_rate,
                "retry_rate": self.metrics.retry_rate,
                "error_counts": dict(self.error_counts)
            },
            "pattern_metrics": {
                "pattern_matches": self.metrics.pattern_matches,
                "pattern_accuracy": self.metrics.pattern_accuracy,
                "learning_rate": self.metrics.learning_rate
            }
        }
    
    def get_trends(self) -> List[ValidationTrend]:
        """Get trend analysis for key metrics."""
        trends = []
        
        # Analyze validation time trend
        if len(self.validation_times) > 10:
            time_trend = self._analyze_trend(list(self.validation_times))
            trends.append(ValidationTrend(
                metric_name="validation_time",
                current_value=self.metrics.avg_validation_time,
                trend_direction=time_trend["direction"],
                trend_strength=time_trend["strength"],
                prediction=time_trend.get("prediction"),
                confidence=time_trend.get("confidence", 0.0)
            ))
        
        # Analyze confidence trend
        if len(self.confidence_scores) > 10:
            confidence_trend = self._analyze_trend(list(self.confidence_scores))
            trends.append(ValidationTrend(
                metric_name="confidence",
                current_value=self.metrics.avg_confidence,
                trend_direction=confidence_trend["direction"],
                trend_strength=confidence_trend["strength"],
                prediction=confidence_trend.get("prediction"),
                confidence=confidence_trend.get("confidence", 0.0)
            ))
        
        return trends
    
    def _analyze_trend(self, values: List[float]) -> Dict[str, Any]:
        """Analyze trend in a series of values."""
        if len(values) < 3:
            return {"direction": "stable", "strength": 0.0}
        
        # Simple linear regression
        n = len(values)
        x = list(range(n))
        y = values
        
        # Calculate slope
        x_mean = sum(x) / n
        y_mean = sum(y) / n
        
        numerator = sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))
        
        if denominator == 0:
            return {"direction": "stable", "strength": 0.0}
        
        slope = numerator / denominator
        
        # Determine direction and strength
        if abs(slope) < 0.01:
            direction = "stable"
            strength = 0.0
        elif slope > 0:
            direction = "increasing"
            strength = min(abs(slope) * 100, 1.0)
        else:
            direction = "decreasing"
            strength = min(abs(slope) * 100, 1.0)
        
        # Simple prediction (extend trend)
        prediction = y[-1] + slope if len(values) > 1 else None
        
        return {
            "direction": direction,
            "strength": strength,
            "prediction": prediction,
            "confidence": min(strength * 2, 1.0)
        }
    
    def _calculate_percentile(self, values: List[float], percentile: int) -> float:
        """Calculate percentile of values."""
        if not values:
            return 0.0
        
        sorted_values = sorted(values)
        index = int((percentile / 100) * len(sorted_values))
        index = min(index, len(sorted_values) - 1)
        return sorted_values[index]
    
    def export_metrics(self, filepath: str):
        """Export metrics to JSON file."""
        try:
            metrics_data = {
                "timestamp": datetime.now().isoformat(),
                "metrics": self.get_metrics(),
                "trends": [trend.__dict__ for trend in self.get_trends()],
                "performance_history": [
                    {
                        "timestamp": snapshot["timestamp"].isoformat(),
                        "metrics": snapshot["metrics"]
                    }
                    for snapshot in self.performance_history
                ]
            }
            
            with open(filepath, 'w') as f:
                json.dump(metrics_data, f, indent=2)
            
            logging.info(f"Metrics exported to {filepath}")
            
        except Exception as e:
            logging.error(f"Failed to export metrics: {e}")
    
    def reset_metrics(self):
        """Reset all metrics."""
        self.metrics = ValidationMetrics()
        self.validation_times.clear()
        self.confidence_scores.clear()
        self.error_counts.clear()
        self.performance_history.clear()
        logging.info("Metrics reset")
    
    def shutdown(self):
        """Shutdown the monitor."""
        self.stop_monitoring()
        logging.info("ValidationMonitor shutdown complete")
