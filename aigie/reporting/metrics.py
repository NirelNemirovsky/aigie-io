"""
Metrics collection and analysis for Aigie.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass


@dataclass
class MetricPoint:
    """A single metric data point."""
    
    timestamp: datetime
    value: float
    labels: Dict[str, str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "value": self.value,
            "labels": self.labels
        }


class MetricsCollector:
    """Collects and manages metrics for Aigie monitoring."""
    
    def __init__(self):
        self.metrics: Dict[str, List[MetricPoint]] = {}
        self.max_metric_points = 1000
    
    def record_metric(self, metric_name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Record a metric value."""
        if metric_name not in self.metrics:
            self.metrics[metric_name] = []
        
        metric_point = MetricPoint(
            timestamp=datetime.now(),
            value=value,
            labels=labels or {}
        )
        
        self.metrics[metric_name].append(metric_point)
        
        # Keep only the most recent points
        if len(self.metrics[metric_name]) > self.max_metric_points:
            self.metrics[metric_name] = self.metrics[metric_name][-self.max_metric_points:]
    
    def get_metric(self, metric_name: str, window_minutes: int = 60) -> List[MetricPoint]:
        """Get metric points within a time window."""
        if metric_name not in self.metrics:
            return []
        
        cutoff_time = datetime.now() - timedelta(minutes=window_minutes)
        return [
            point for point in self.metrics[metric_name]
            if point.timestamp >= cutoff_time
        ]
    
    def get_metric_summary(self, metric_name: str, window_minutes: int = 60) -> Dict[str, Any]:
        """Get summary statistics for a metric."""
        points = self.get_metric(metric_name, window_minutes)
        
        if not points:
            return {
                "metric_name": metric_name,
                "window_minutes": window_minutes,
                "count": 0,
                "min": None,
                "max": None,
                "avg": None,
                "latest": None
            }
        
        values = [point.value for point in points]
        
        return {
            "metric_name": metric_name,
            "window_minutes": window_minutes,
            "count": len(points),
            "min": min(values),
            "max": max(values),
            "avg": sum(values) / len(values),
            "latest": points[-1].value if points else None,
            "latest_timestamp": points[-1].timestamp.isoformat() if points else None
        }
    
    def get_all_metrics_summary(self, window_minutes: int = 60) -> Dict[str, Any]:
        """Get summary for all metrics."""
        summary = {
            "window_minutes": window_minutes,
            "total_metrics": len(self.metrics),
            "metrics": {}
        }
        
        for metric_name in self.metrics:
            summary["metrics"][metric_name] = self.get_metric_summary(metric_name, window_minutes)
        
        return summary
    
    def clear_metrics(self, metric_name: Optional[str] = None):
        """Clear metrics, optionally for a specific metric."""
        if metric_name is None:
            self.metrics.clear()
        elif metric_name in self.metrics:
            self.metrics[metric_name].clear()
    
    def export_metrics(self, format: str = "json") -> str:
        """Export metrics in the specified format."""
        if format.lower() == "json":
            import json
            return json.dumps(self.get_all_metrics_summary(), indent=2)
        else:
            raise ValueError(f"Unsupported export format: {format}")


# Global metrics collector instance
_metrics_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector instance."""
    global _metrics_collector
    
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    
    return _metrics_collector


def record_metric(metric_name: str, value: float, labels: Optional[Dict[str, str]] = None):
    """Record a metric using the global collector."""
    collector = get_metrics_collector()
    collector.record_metric(metric_name, value, labels)


def get_metric_summary(metric_name: str, window_minutes: int = 60) -> Dict[str, Any]:
    """Get metric summary using the global collector."""
    collector = get_metrics_collector()
    return collector.get_metric_summary(metric_name, window_minutes)
