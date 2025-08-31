"""
Aigie - AI Agent Runtime Error Detection

A real-time error detection and monitoring system for LangChain and LangGraph applications.
"""

__version__ = "0.1.0"
__author__ = "Aigie Team"
__email__ = "team@aigie.io"

from .core.error_detector import ErrorDetector
from .core.monitoring import PerformanceMonitor
from .interceptors.langchain import LangChainInterceptor
from .interceptors.langgraph import LangGraphInterceptor
from .reporting.logger import AigieLogger
from .utils.config import AigieConfig

# Main entry point for automatic integration
from .auto_integration import (
    auto_integrate, get_integrator, stop_integration, 
    get_status, get_analysis, enable_monitoring, 
    disable_monitoring, show_status, show_analysis
)

__all__ = [
    "ErrorDetector",
    "PerformanceMonitor", 
    "LangChainInterceptor",
    "LangGraphInterceptor",
    "AigieLogger",
    "AigieConfig",
    "auto_integrate",
    "get_integrator",
    "stop_integration",
    "get_status",
    "get_analysis",
    "enable_monitoring",
    "disable_monitoring",
    "show_status",
    "show_analysis",
]
