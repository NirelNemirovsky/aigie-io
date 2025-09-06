"""
Performance monitoring and resource tracking for Aigie.
"""

import time
import psutil
import threading
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from contextlib import contextmanager


@dataclass
class PerformanceMetrics:
    """Performance metrics for a single execution."""
    
    start_time: datetime
    end_time: Optional[datetime] = None
    execution_time: Optional[float] = None
    memory_start: Optional[float] = None
    memory_end: Optional[float] = None
    memory_delta: Optional[float] = None
    cpu_start: Optional[float] = None
    cpu_end: Optional[float] = None
    cpu_delta: Optional[float] = None
    
    def finalize(self):
        """Calculate final metrics."""
        if self.end_time is None:
            self.end_time = datetime.now()
        
        if self.start_time and self.end_time:
            self.execution_time = (self.end_time - self.start_time).total_seconds()
        
        if self.memory_start is not None and self.memory_end is not None:
            self.memory_delta = self.memory_end - self.memory_start
        
        if self.cpu_start is not None and self.cpu_end is not None:
            self.cpu_delta = self.cpu_end - self.cpu_start
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "execution_time": self.execution_time,
            "memory_start_mb": self.memory_start,
            "memory_end_mb": self.memory_end,
            "memory_delta_mb": self.memory_delta,
            "cpu_start_percent": self.cpu_start,
            "cpu_end_percent": self.cpu_end,
            "cpu_delta_percent": self.cpu_delta,
        }


class PerformanceMonitor:
    """Monitors performance metrics for AI agent executions."""
    
    def __init__(self, enable_memory_monitoring: bool = True, enable_cpu_monitoring: bool = True):
        self.enable_memory_monitoring = enable_memory_monitoring
        self.enable_cpu_monitoring = enable_cpu_monitoring
        self.metrics_history: list[PerformanceMetrics] = []
        self.current_metrics: Optional[PerformanceMetrics] = None
        self._lock = threading.Lock()
        
        # Thresholds for performance warnings
        self.execution_time_threshold = 30.0  # seconds
        self.memory_threshold = 1024.0  # MB
        self.cpu_threshold = 80.0  # percent
    
    def start_monitoring(self, component: str, method: str) -> PerformanceMetrics:
        """Start monitoring a new execution."""
        with self._lock:
            metrics = PerformanceMetrics(
                start_time=datetime.now(),
                memory_start=self._get_memory_usage() if self.enable_memory_monitoring else None,
                cpu_start=self._get_cpu_usage() if self.enable_cpu_monitoring else None
            )
            self.current_metrics = metrics
            return metrics
    
    def stop_monitoring(self, metrics: PerformanceMetrics):
        """Stop monitoring and finalize metrics."""
        with self._lock:
            if self.enable_memory_monitoring:
                metrics.memory_end = self._get_memory_usage()
            
            if self.enable_cpu_monitoring:
                metrics.cpu_end = self._get_cpu_usage()
            
            metrics.finalize()
            self.metrics_history.append(metrics)
            self.current_metrics = None
    
    @contextmanager
    def monitor_execution(self, component: str, method: str):
        """Context manager for monitoring execution."""
        metrics = self.start_monitoring(component, method)
        try:
            yield metrics
        finally:
            self.stop_monitoring(metrics)
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            return memory_info.rss / 1024 / 1024  # Convert to MB
        except Exception:
            return 0.0
    
    def _get_cpu_usage(self) -> float:
        """Get current CPU usage percentage."""
        try:
            return psutil.cpu_percent(interval=0.1)
        except Exception:
            return 0.0
    
    def get_performance_summary(self, window_minutes: int = 60) -> Dict[str, Any]:
        """Get performance summary for the last N minutes."""
        cutoff_time = datetime.now() - timedelta(minutes=window_minutes)
        
        recent_metrics = [
            m for m in self.metrics_history 
            if m.start_time >= cutoff_time
        ]
        
        if not recent_metrics:
            return {}
        
        execution_times = [m.execution_time for m in recent_metrics if m.execution_time is not None]
        memory_deltas = [m.memory_delta for m in recent_metrics if m.memory_delta is not None]
        cpu_deltas = [m.cpu_delta for m in recent_metrics if m.cpu_delta is not None]
        
        summary = {
            "total_executions": len(recent_metrics),
            "window_minutes": window_minutes,
            "avg_execution_time": sum(execution_times) / len(execution_times) if execution_times else 0,
            "max_execution_time": max(execution_times) if execution_times else 0,
            "min_execution_time": min(execution_times) if execution_times else 0,
        }
        
        if memory_deltas:
            summary.update({
                "avg_memory_delta": sum(memory_deltas) / len(memory_deltas),
                "max_memory_delta": max(memory_deltas),
                "min_memory_delta": min(memory_deltas),
            })
        
        if cpu_deltas:
            summary.update({
                "avg_cpu_delta": sum(cpu_deltas) / len(cpu_deltas),
                "max_cpu_delta": max(cpu_deltas),
                "min_cpu_delta": min(cpu_deltas),
            })
        
        return summary
    
    def check_performance_issues(self, metrics: PerformanceMetrics) -> list[str]:
        """Check for performance issues and return warnings."""
        warnings = []
        
        if metrics.execution_time and metrics.execution_time > self.execution_time_threshold:
            warnings.append(f"Slow execution: {metrics.execution_time:.2f}s (threshold: {self.execution_time_threshold}s)")
        
        if metrics.memory_delta and abs(metrics.memory_delta) > self.memory_threshold:
            warnings.append(f"High memory usage: {metrics.memory_delta:.2f}MB (threshold: {self.memory_threshold}MB)")
        
        if metrics.cpu_delta and abs(metrics.cpu_delta) > self.cpu_threshold:
            warnings.append(f"High CPU usage: {metrics.cpu_delta:.2f}% (threshold: {self.cpu_threshold}%)")
        
        return warnings
    
    def clear_history(self):
        """Clear metrics history."""
        with self._lock:
            self.metrics_history.clear()


class ResourceMonitor:
    """Monitors system resources and detects potential issues."""
    
    def __init__(self):
        self.memory_threshold = 0.9  # 90% of available memory
        self.cpu_threshold = 0.8     # 80% of available CPU
        self.disk_threshold = 0.9    # 90% of disk usage
    
    def check_system_health(self) -> Dict[str, Any]:
        """Check overall system health."""
        health_status = {
            "timestamp": datetime.now().isoformat(),
            "memory": self._check_memory_health(),
            "cpu": self._check_cpu_health(),
            "disk": self._check_disk_health(),
            "overall_status": "healthy"
        }
        
        # Determine overall status
        if any(status["status"] == "critical" for status in [health_status["memory"], health_status["cpu"], health_status["disk"]]):
            health_status["overall_status"] = "critical"
        elif any(status["status"] == "warning" for status in [health_status["memory"], health_status["cpu"], health_status["disk"]]):
            health_status["overall_status"] = "warning"
        
        return health_status
    
    def _check_memory_health(self) -> Dict[str, Any]:
        """Check memory health."""
        try:
            memory = psutil.virtual_memory()
            usage_percent = memory.percent / 100
            
            if usage_percent > self.memory_threshold:
                status = "critical"
            elif usage_percent > self.memory_threshold * 0.8:
                status = "warning"
            else:
                status = "healthy"
            
            return {
                "status": status,
                "usage_percent": memory.percent,
                "available_mb": memory.available / 1024 / 1024,
                "total_mb": memory.total / 1024 / 1024,
                "threshold": self.memory_threshold * 100
            }
        except Exception as e:
            return {
                "status": "unknown",
                "error": str(e)
            }
    
    def _check_cpu_health(self) -> Dict[str, Any]:
        """Check CPU health."""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            usage_percent = cpu_percent / 100
            
            if usage_percent > self.cpu_threshold:
                status = "critical"
            elif usage_percent > self.cpu_threshold * 0.8:
                status = "warning"
            else:
                status = "healthy"
            
            return {
                "status": status,
                "usage_percent": cpu_percent,
                "threshold": self.cpu_threshold * 100
            }
        except Exception as e:
            return {
                "status": "unknown",
                "error": str(e)
            }
    
    def _check_disk_health(self) -> Dict[str, Any]:
        """Check disk health."""
        try:
            disk = psutil.disk_usage('/')
            usage_percent = disk.percent / 100
            
            if usage_percent > self.disk_threshold:
                status = "critical"
            elif usage_percent > self.disk_threshold * 0.8:
                status = "warning"
            else:
                status = "healthy"
            
            return {
                "status": status,
                "usage_percent": disk.percent,
                "free_gb": disk.free / 1024 / 1024 / 1024,
                "total_gb": disk.total / 1024 / 1024 / 1024,
                "threshold": self.disk_threshold * 100
            }
        except Exception as e:
            return {
                "status": "unknown",
                "error": str(e)
            }
