"""
Main error detection engine for Aigie.
"""

import traceback
import asyncio
from typing import Dict, Any, Optional, Callable, List, Union
from datetime import datetime, timedelta
from contextlib import contextmanager

from .error_types import (
    ErrorType, ErrorSeverity, ErrorContext, DetectedError,
    classify_error, determine_severity
)
from .monitoring import PerformanceMonitor, ResourceMonitor


class ErrorDetector:
    """Main error detection engine for AI agent applications."""
    
    def __init__(self, enable_performance_monitoring: bool = True, enable_resource_monitoring: bool = True):
        self.performance_monitor = PerformanceMonitor() if enable_performance_monitoring else None
        self.resource_monitor = ResourceMonitor() if enable_resource_monitoring else None
        self.error_handlers: List[Callable[[DetectedError], None]] = []
        self.error_history: List[DetectedError] = []
        self.is_monitoring = False
        
        # Error detection settings
        self.enable_timeout_detection = True
        self.timeout_threshold = 300.0  # 5 minutes
        self.enable_memory_leak_detection = True
        self.memory_leak_threshold = 100.0  # MB increase over time
        
    def add_error_handler(self, handler: Callable[[DetectedError], None]):
        """Add a custom error handler."""
        self.error_handlers.append(handler)
    
    def start_monitoring(self):
        """Start the error detection system."""
        self.is_monitoring = True
        
    def stop_monitoring(self):
        """Stop the error detection system."""
        self.is_monitoring = False
    
    @contextmanager
    def monitor_execution(self, framework: str, component: str, method: str, **kwargs):
        """Context manager for monitoring execution and detecting errors."""
        if not self.is_monitoring:
            yield
            return
        
        # Start performance monitoring
        perf_metrics = None
        if self.performance_monitor:
            perf_metrics = self.performance_monitor.start_monitoring(component, method)
        
        # Create error context
        context = ErrorContext(
            timestamp=datetime.now(),
            framework=framework,
            component=component,
            method=method,
            input_data=kwargs.get('input_data'),
            state=kwargs.get('state')
        )
        
        start_time = datetime.now()
        
        try:
            yield
            
            # Check for performance issues
            if perf_metrics:
                self.performance_monitor.stop_monitoring(perf_metrics)
                performance_warnings = self.performance_monitor.check_performance_issues(perf_metrics)
                
                for warning in performance_warnings:
                    self._detect_performance_issue(warning, context, perf_metrics)
                    
        except Exception as e:
            # Detect and handle the error
            self._detect_error(e, context, perf_metrics)
            raise
        finally:
            # Check for timeout
            if self.enable_timeout_detection:
                execution_time = (datetime.now() - start_time).total_seconds()
                if execution_time > self.timeout_threshold:
                    self._detect_timeout(execution_time, context)
    
    def _detect_error(self, exception: Exception, context: ErrorContext, perf_metrics: Optional[Any] = None):
        """Detect and process an error."""
        # Classify the error
        error_type = classify_error(exception, context)
        severity = determine_severity(error_type, context)
        
        # Update context with performance metrics
        if perf_metrics:
            context.execution_time = perf_metrics.execution_time
            context.memory_usage = perf_metrics.memory_delta
            context.cpu_usage = perf_metrics.cpu_delta
            context.stack_trace = traceback.format_exc()
        
        # Create detected error
        detected_error = DetectedError(
            error_type=error_type,
            severity=severity,
            message=str(exception),
            exception=exception,
            context=context,
            suggestions=self._generate_suggestions(error_type, exception, context)
        )
        
        # Store error
        self.error_history.append(detected_error)
        
        # Notify handlers
        self._notify_handlers(detected_error)
        
        return detected_error
    
    def _detect_performance_issue(self, warning: str, context: ErrorContext, perf_metrics: Any):
        """Detect performance-related issues."""
        if "slow execution" in warning.lower():
            error_type = ErrorType.SLOW_EXECUTION
        elif "memory" in warning.lower():
            error_type = ErrorType.HIGH_MEMORY_USAGE
        elif "cpu" in warning.lower():
            error_type = ErrorType.HIGH_CPU_USAGE
        else:
            error_type = ErrorType.UNKNOWN_ERROR
        
        detected_error = DetectedError(
            error_type=error_type,
            severity=ErrorSeverity.MEDIUM,
            message=warning,
            context=context,
            suggestions=self._generate_suggestions(error_type, None, context)
        )
        
        self.error_history.append(detected_error)
        self._notify_handlers(detected_error)
    
    def _detect_timeout(self, execution_time: float, context: ErrorContext):
        """Detect timeout issues."""
        detected_error = DetectedError(
            error_type=ErrorType.TIMEOUT,
            severity=ErrorSeverity.HIGH,
            message=f"Execution timed out after {execution_time:.2f} seconds",
            context=context,
            suggestions=[
                "Increase timeout configuration",
                "Optimize the execution logic",
                "Check for blocking operations",
                "Consider asynchronous execution"
            ]
        )
        
        self.error_history.append(detected_error)
        self._notify_handlers(detected_error)
    
    def _detect_memory_leak(self, memory_increase: float, context: ErrorContext):
        """Detect potential memory leaks."""
        if memory_increase > self.memory_leak_threshold:
            detected_error = DetectedError(
                error_type=ErrorType.MEMORY_LEAK,
                severity=ErrorSeverity.HIGH,
                message=f"Potential memory leak detected: {memory_increase:.2f}MB increase",
                context=context,
                suggestions=[
                    "Check for unclosed resources (files, connections, etc.)",
                    "Review memory management in loops",
                    "Consider using context managers",
                    "Monitor object lifecycle and cleanup"
                ]
            )
            
            self.error_history.append(detected_error)
            self._notify_handlers(detected_error)
    
    def _generate_suggestions(self, error_type: ErrorType, exception: Optional[Exception], context: ErrorContext) -> List[str]:
        """Generate suggestions for fixing the error."""
        suggestions = []
        
        if error_type == ErrorType.TIMEOUT:
            suggestions.extend([
                "Increase timeout configuration",
                "Optimize the execution logic",
                "Check for blocking operations",
                "Consider asynchronous execution"
            ])
        
        elif error_type == ErrorType.API_ERROR:
            suggestions.extend([
                "Check API endpoint configuration",
                "Verify authentication credentials",
                "Review rate limiting settings",
                "Check network connectivity"
            ])
        
        elif error_type == ErrorType.MEMORY_ERROR:
            suggestions.extend([
                "Check for memory leaks in loops",
                "Review large data structures",
                "Consider streaming for large datasets",
                "Monitor memory usage patterns"
            ])
        
        elif error_type == ErrorType.STATE_ERROR:
            suggestions.extend([
                "Validate state transitions",
                "Check data type consistency",
                "Review state initialization",
                "Add state validation checks"
            ])
        
        elif error_type == ErrorType.SLOW_EXECUTION:
            suggestions.extend([
                "Profile the execution path",
                "Optimize database queries",
                "Consider caching strategies",
                "Review algorithm complexity"
            ])
        
        # Framework-specific suggestions
        if context.framework == "langchain":
            suggestions.extend([
                "Check chain configuration",
                "Review tool implementations",
                "Verify memory setup",
                "Check agent reasoning logic"
            ])
        
        elif context.framework == "langgraph":
            suggestions.extend([
                "Review node implementations",
                "Check state graph configuration",
                "Verify checkpoint settings",
                "Review transition logic"
            ])
        
        return suggestions
    
    def _notify_handlers(self, detected_error: DetectedError):
        """Notify all registered error handlers."""
        for handler in self.error_handlers:
            try:
                handler(detected_error)
            except Exception as e:
                # Don't let handler errors break the system
                print(f"Error in error handler: {e}")
    
    def get_error_summary(self, window_minutes: int = 60) -> Dict[str, Any]:
        """Get summary of errors in the last N minutes."""
        cutoff_time = datetime.now() - timedelta(minutes=window_minutes)
        
        recent_errors = [
            e for e in self.error_history 
            if e.context and e.context.timestamp >= cutoff_time
        ]
        
        if not recent_errors:
            return {"total_errors": 0, "window_minutes": window_minutes}
        
        # Count by severity
        severity_counts = {}
        for error in recent_errors:
            severity = error.severity.value
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        # Count by type
        type_counts = {}
        for error in recent_errors:
            error_type = error.error_type.value
            type_counts[error_type] = type_counts.get(error_type, 0) + 1
        
        # Count by component
        component_counts = {}
        for error in recent_errors:
            if error.context:
                component = error.context.component
                component_counts[component] = component_counts.get(component, 0) + 1
        
        return {
            "total_errors": len(recent_errors),
            "window_minutes": window_minutes,
            "severity_distribution": severity_counts,
            "type_distribution": type_counts,
            "component_distribution": component_counts,
            "most_recent_error": recent_errors[-1].to_dict() if recent_errors else None
        }
    
    def clear_history(self):
        """Clear error history."""
        self.error_history.clear()
        if self.performance_monitor:
            self.performance_monitor.clear_history()
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health status."""
        health_status = {
            "timestamp": datetime.now().isoformat(),
            "is_monitoring": self.is_monitoring,
            "total_errors": len(self.error_history),
            "recent_errors": len([e for e in self.error_history if (datetime.now() - e.context.timestamp).seconds < 300])  # Last 5 minutes
        }
        
        if self.resource_monitor:
            health_status["system_health"] = self.resource_monitor.check_system_health()
        
        if self.performance_monitor:
            health_status["performance_summary"] = self.performance_monitor.get_performance_summary(window_minutes=60)
        
        return health_status


class AsyncErrorDetector(ErrorDetector):
    """Asynchronous version of the error detector for async operations."""
    
    async def monitor_execution_async(self, framework: str, component: str, method: str, **kwargs):
        """Async context manager for monitoring execution."""
        if not self.is_monitoring:
            yield
            return
        
        # Start performance monitoring
        perf_metrics = None
        if self.performance_monitor:
            perf_metrics = self.performance_monitor.start_monitoring(component, method)
        
        # Create error context
        context = ErrorContext(
            timestamp=datetime.now(),
            framework=framework,
            component=component,
            method=method,
            input_data=kwargs.get('input_data'),
            state=kwargs.get('state')
        )
        
        start_time = datetime.now()
        
        try:
            yield
            
            # Check for performance issues
            if perf_metrics:
                self.performance_monitor.stop_monitoring(perf_metrics)
                performance_warnings = self.performance_monitor.check_performance_issues(perf_metrics)
                
                for warning in performance_warnings:
                    self._detect_performance_issue(warning, context, perf_metrics)
                    
        except Exception as e:
            # Detect and handle the error
            self._detect_error(e, context, perf_metrics)
            raise
        finally:
            # Check for timeout
            if self.enable_timeout_detection:
                execution_time = (datetime.now() - start_time).total_seconds()
                if execution_time > self.timeout_threshold:
                    self._detect_timeout(execution_time, context)
    
    async def _notify_handlers_async(self, detected_error: DetectedError):
        """Notify all registered error handlers asynchronously."""
        for handler in self.error_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(detected_error)
                else:
                    handler(detected_error)
            except Exception as e:
                print(f"Error in async error handler: {e}")
