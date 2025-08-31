"""
Real-time logging and error reporting for Aigie.
"""

import json
import sys
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.syntax import Syntax
import structlog

from ..core.error_types import DetectedError, ErrorSeverity


class AigieLogger:
    """Real-time logger for Aigie error detection and monitoring."""
    
    def __init__(self, 
                 log_level: str = "INFO",
                 enable_console: bool = True,
                 enable_file: bool = False,
                 log_file_path: Optional[str] = None,
                 enable_rich_formatting: bool = True):
        
        self.enable_console = enable_console
        self.enable_file = enable_file
        self.log_file_path = log_file_path or "aigie.log"
        self.enable_rich_formatting = enable_rich_formatting
        
        # Initialize rich console
        if enable_rich_formatting:
            self.console = Console()
        else:
            self.console = None
        
        # Initialize structured logging
        self._setup_structured_logging(log_level)
        
        # Error statistics
        self.error_counts = {
            "total": 0,
            "critical": 0,
            "high": 0,
            "medium": 0,
            "low": 0
        }
        
        # Recent errors for display
        self.recent_errors: List[DetectedError] = []
        self.max_recent_errors = 10
    
    def _setup_structured_logging(self, log_level: str):
        """Setup structured logging with structlog."""
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                structlog.processors.JSONRenderer()
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )
        
        self.logger = structlog.get_logger()
        # Set log level using standard logging levels
        import logging
        self.logger.setLevel(getattr(logging, log_level.upper()))
    
    def log_error(self, error: DetectedError):
        """Log a detected error with rich formatting."""
        # Update statistics
        self.error_counts["total"] += 1
        severity = error.severity.value
        if severity in self.error_counts:
            self.error_counts[severity] += 1
        
        # Add to recent errors
        self.recent_errors.append(error)
        if len(self.recent_errors) > self.max_recent_errors:
            self.recent_errors.pop(0)
        
        # Log to structured logger
        self.logger.error(
            "Error detected",
            error_type=error.error_type.value,
            severity=error.severity.value,
            message=error.message,
            component=error.context.component if error.context else "unknown",
            method=error.context.method if error.context else "unknown",
            framework=error.context.framework if error.context else "unknown",
            execution_time=error.context.execution_time if error.context else None,
            suggestions=error.suggestions
        )
        
        # Rich console output
        if self.enable_rich_formatting and self.console:
            self._display_error_rich(error)
        
        # File logging
        if self.enable_file:
            self._log_to_file(error)
    
    def _display_error_rich(self, error: DetectedError):
        """Display error with rich formatting in console."""
        # Create error panel
        severity_color = self._get_severity_color(error.severity)
        
        # Error header
        header = Text(f"🚨 {error.error_type.value.upper()}", style=severity_color)
        if error.context:
            header.append(f" in {error.context.component}.{error.context.method}", style="dim")
        
        # Error details
        details = []
        if error.context and error.context.execution_time:
            details.append(f"⏱️  Execution time: {error.context.execution_time:.2f}s")
        
        if error.context and error.context.memory_usage:
            details.append(f"💾 Memory usage: {error.context.memory_usage:.2f}MB")
        
        if error.context and error.context.cpu_usage:
            details.append(f"🖥️  CPU usage: {error.context.cpu_usage:.2f}%")
        
        # Suggestions
        if error.suggestions:
            suggestions_text = "\n".join([f"• {s}" for s in error.suggestions])
            details.append(f"💡 Suggestions:\n{suggestions_text}")
        
        # Create panel
        panel = Panel(
            f"{error.message}\n\n" + "\n".join(details),
            title=header,
            border_style=severity_color,
            padding=(1, 2)
        )
        
        self.console.print(panel)
    
    def _get_severity_color(self, severity: ErrorSeverity) -> str:
        """Get color for severity level."""
        colors = {
            ErrorSeverity.LOW: "green",
            ErrorSeverity.MEDIUM: "yellow",
            ErrorSeverity.HIGH: "red",
            ErrorSeverity.CRITICAL: "bold red"
        }
        return colors.get(severity, "white")
    
    def _log_to_file(self, error: DetectedError):
        """Log error to file."""
        try:
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "error": error.to_dict()
            }
            
            with open(self.log_file_path, "a") as f:
                f.write(json.dumps(log_entry) + "\n")
        except Exception as e:
            # Fallback to stderr if file logging fails
            print(f"Failed to log to file: {e}", file=sys.stderr)
    
    def display_error_summary(self):
        """Display summary of all errors."""
        if not self.console:
            return
        
        # Create summary table
        table = Table(title="🚨 Aigie Error Summary")
        table.add_column("Severity", style="bold")
        table.add_column("Count", justify="right")
        table.add_column("Percentage", justify="right")
        
        total = self.error_counts["total"]
        if total > 0:
            for severity in ["critical", "high", "medium", "low"]:
                count = self.error_counts[severity]
                percentage = (count / total) * 100
                color = self._get_severity_color(ErrorSeverity(severity))
                table.add_row(
                    severity.upper(),
                    str(count),
                    f"{percentage:.1f}%",
                    style=color
                )
        
        self.console.print(table)
        
        # Display recent errors
        if self.recent_errors:
            self.console.print("\n📋 Recent Errors:")
            for error in self.recent_errors[-5:]:  # Show last 5
                self._display_error_rich(error)
    
    def display_system_health(self, health_data: Dict[str, Any]):
        """Display system health information."""
        if not self.console:
            return
        
        # System health panel
        health_panel = Panel(
            f"🔄 Monitoring: {'Active' if health_data.get('is_monitoring') else 'Inactive'}\n"
            f"📊 Total Errors: {health_data.get('total_errors', 0)}\n"
            f"⚡ Recent Errors (5min): {health_data.get('recent_errors', 0)}",
            title="🏥 System Health",
            border_style="blue"
        )
        
        self.console.print(health_panel)
        
        # Performance summary
        if "performance_summary" in health_data:
            perf = health_data["performance_summary"]
            if perf:
                perf_panel = Panel(
                    f"📈 Total Executions: {perf.get('total_executions', 0)}\n"
                    f"⏱️  Avg Execution Time: {perf.get('avg_execution_time', 0):.2f}s\n"
                    f"🚀 Max Execution Time: {perf.get('max_execution_time', 0):.2f}s",
                    title="📊 Performance Metrics",
                    border_style="green"
                )
                self.console.print(perf_panel)
        
        # System health
        if "system_health" in health_data:
            sys_health = health_data["system_health"]
            if sys_health:
                status = sys_health.get("overall_status", "unknown")
                status_color = {
                    "healthy": "green",
                    "warning": "yellow",
                    "critical": "red"
                }.get(status, "white")
                
                sys_panel = Panel(
                    f"💾 Memory: {sys_health.get('memory', {}).get('status', 'unknown')}\n"
                    f"🖥️  CPU: {sys_health.get('cpu', {}).get('status', 'unknown')}\n"
                    f"💿 Disk: {sys_health.get('disk', {}).get('status', 'unknown')}",
                    title="🖥️  System Resources",
                    border_style=status_color
                )
                self.console.print(sys_panel)
    
    def log_performance_issue(self, warning: str, context: Dict[str, Any]):
        """Log performance-related warnings."""
        self.logger.warning(
            "Performance issue detected",
            extra={"warning": warning, **context}
        )
        
        if self.enable_rich_formatting and self.console:
            self.console.print(f"⚠️  {warning}", style="yellow")
    
    def log_system_event(self, event: str, details: Dict[str, Any] = None):
        """Log system events."""
        self.logger.info(
            "System event",
            extra={"event": event, "details": details or {}}
        )
        
        if self.enable_rich_formatting and self.console:
            self.console.print(f"ℹ️  {event}", style="blue")
    
    def clear_logs(self):
        """Clear log files."""
        if self.enable_file and Path(self.log_file_path).exists():
            Path(self.log_file_path).unlink()
        
        # Clear recent errors
        self.recent_errors.clear()
        
        # Reset error counts
        for key in self.error_counts:
            self.error_counts[key] = 0
    
    def get_log_stats(self) -> Dict[str, Any]:
        """Get logging statistics."""
        return {
            "total_errors_logged": self.error_counts["total"],
            "severity_distribution": self.error_counts.copy(),
            "recent_errors_count": len(self.recent_errors),
            "log_file_enabled": self.enable_file,
            "log_file_path": self.log_file_path if self.enable_file else None,
            "rich_formatting_enabled": self.enable_rich_formatting
        }


class ConsoleLogger(AigieLogger):
    """Console-only logger for development and testing."""
    
    def __init__(self, log_level: str = "INFO"):
        super().__init__(
            log_level=log_level,
            enable_console=True,
            enable_file=False,
            enable_rich_formatting=True
        )


class FileLogger(AigieLogger):
    """File-only logger for production environments."""
    
    def __init__(self, log_file_path: str, log_level: str = "INFO"):
        super().__init__(
            log_level=log_level,
            enable_console=False,
            enable_file=True,
            log_file_path=log_file_path,
            enable_rich_formatting=False
        )


class CloudLogger(AigieLogger):
    """Cloud-ready logger with structured output for cloud monitoring."""
    
    def __init__(self, log_level: str = "INFO", enable_console: bool = True):
        super().__init__(
            log_level=log_level,
            enable_console=enable_console,
            enable_file=False,
            enable_rich_formatting=False
        )
        
        # Configure for cloud logging
        self._setup_cloud_logging()
    
    def _setup_cloud_logging(self):
        """Setup logging optimized for cloud environments."""
        # Add cloud-specific processors
        processors = [
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            # Add cloud-specific fields
            self._add_cloud_metadata,
            structlog.processors.JSONRenderer()
        ]
        
        structlog.configure(
            processors=processors,
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )
        
        self.logger = structlog.get_logger()
    
    def _add_cloud_metadata(self, logger, method_name, event_dict):
        """Add cloud-specific metadata to log entries."""
        # Add environment information
        event_dict["environment"] = "production"
        event_dict["service"] = "aigie"
        event_dict["version"] = "0.1.0"
        
        # Add timestamp in cloud-friendly format
        event_dict["timestamp"] = datetime.now().isoformat()
        
        return event_dict
