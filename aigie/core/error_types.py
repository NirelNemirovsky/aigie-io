"""
Error type definitions and classification for Aigie.
"""

from enum import Enum
from typing import Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime


class ErrorType(Enum):
    """Types of errors that can occur in AI agent applications."""
    
    # Execution errors
    RUNTIME_EXCEPTION = "runtime_exception"
    TIMEOUT = "timeout"
    INFINITE_LOOP = "infinite_loop"
    STACK_OVERFLOW = "stack_overflow"
    
    # API and external service errors
    API_ERROR = "api_error"
    RATE_LIMIT = "rate_limit"
    AUTHENTICATION_ERROR = "authentication_error"
    NETWORK_ERROR = "network_error"
    SERVICE_UNAVAILABLE = "service_unavailable"
    
    # State and data errors
    STATE_ERROR = "state_error"
    DATA_CORRUPTION = "data_corruption"
    TYPE_MISMATCH = "type_mismatch"
    VALIDATION_ERROR = "validation_error"
    
    # Memory and resource errors
    MEMORY_ERROR = "memory_error"
    MEMORY_OVERFLOW = "memory_overflow"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    DISK_SPACE_ERROR = "disk_space_error"
    
    # Performance issues
    SLOW_EXECUTION = "slow_execution"
    HIGH_MEMORY_USAGE = "high_memory_usage"
    HIGH_CPU_USAGE = "high_cpu_usage"
    MEMORY_LEAK = "memory_leak"
    
    # LangChain specific errors
    LANGCHAIN_CHAIN_ERROR = "langchain_chain_error"
    LANGCHAIN_TOOL_ERROR = "langchain_tool_error"
    LANGCHAIN_MEMORY_ERROR = "langchain_memory_error"
    LANGCHAIN_AGENT_ERROR = "langchain_agent_error"
    
    # LangGraph specific errors
    LANGGRAPH_NODE_ERROR = "langgraph_node_error"
    LANGGRAPH_STATE_ERROR = "langgraph_state_error"
    LANGGRAPH_TRANSITION_ERROR = "langgraph_transition_error"
    LANGGRAPH_CHECKPOINT_ERROR = "langgraph_checkpoint_error"
    
    # Unknown or uncategorized errors
    UNKNOWN_ERROR = "unknown_error"


class ErrorSeverity(Enum):
    """Severity levels for errors."""
    
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ErrorContext:
    """Context information for an error."""
    
    timestamp: datetime
    framework: str  # "langchain" or "langgraph"
    component: str  # e.g., "LLMChain", "StateGraph", "Tool"
    method: str     # e.g., "run", "invoke", "call"
    input_data: Optional[Dict[str, Any]] = None
    state: Optional[Dict[str, Any]] = None
    stack_trace: Optional[str] = None
    execution_time: Optional[float] = None
    memory_usage: Optional[float] = None
    cpu_usage: Optional[float] = None


@dataclass
class DetectedError:
    """A detected error with full context."""
    
    error_type: ErrorType
    severity: ErrorSeverity
    message: str
    exception: Optional[Exception] = None
    context: Optional[ErrorContext] = None
    suggestions: Optional[list[str]] = None
    
    def __post_init__(self):
        if self.context is None:
            self.context = ErrorContext(
                timestamp=datetime.now(),
                framework="unknown",
                component="unknown",
                method="unknown"
            )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for logging/transmission."""
        return {
            "error_type": self.error_type.value,
            "severity": self.severity.value,
            "message": self.message,
            "exception_type": type(self.exception).__name__ if self.exception else None,
            "exception_message": str(self.exception) if self.exception else None,
            "timestamp": self.context.timestamp.isoformat() if self.context else None,
            "framework": self.context.framework if self.context else None,
            "component": self.context.component if self.context else None,
            "method": self.context.method if self.context else None,
            "execution_time": self.context.execution_time if self.context else None,
            "memory_usage": self.context.memory_usage if self.context else None,
            "cpu_usage": self.context.cpu_usage if self.context else None,
            "suggestions": self.suggestions or [],
        }


def classify_error(exception: Exception, context: ErrorContext) -> ErrorType:
    """Classify an exception into an error type based on context."""
    
    exception_type = type(exception).__name__
    exception_message = str(exception).lower()
    
    # Timeout errors
    if any(keyword in exception_message for keyword in ["timeout", "timed out", "deadline exceeded"]):
        return ErrorType.TIMEOUT
    
    # API errors
    if any(keyword in exception_message for keyword in ["api", "http", "status code", "rate limit"]):
        if "rate limit" in exception_message:
            return ErrorType.RATE_LIMIT
        elif "401" in exception_message or "unauthorized" in exception_message:
            return ErrorType.AUTHENTICATION_ERROR
        elif "503" in exception_message or "unavailable" in exception_message:
            return ErrorType.SERVICE_UNAVAILABLE
        else:
            return ErrorType.API_ERROR
    
    # Network errors
    if any(keyword in exception_message for keyword in ["connection", "network", "dns", "ssl"]):
        return ErrorType.NETWORK_ERROR
    
    # Memory errors
    if any(keyword in exception_message for keyword in ["memory", "out of memory", "memory error"]):
        return ErrorType.MEMORY_ERROR
    
    # State errors
    if any(keyword in exception_message for keyword in ["state", "invalid state", "state transition"]):
        return ErrorType.STATE_ERROR
    
    # LangChain specific
    if context.framework == "langchain":
        if "chain" in context.component.lower():
            return ErrorType.LANGCHAIN_CHAIN_ERROR
        elif "tool" in context.component.lower():
            return ErrorType.LANGCHAIN_TOOL_ERROR
        elif "memory" in context.component.lower():
            return ErrorType.LANGCHAIN_MEMORY_ERROR
        elif "agent" in context.component.lower():
            return ErrorType.LANGCHAIN_AGENT_ERROR
    
    # LangGraph specific
    if context.framework == "langgraph":
        if "node" in context.component.lower():
            return ErrorType.LANGGRAPH_NODE_ERROR
        elif "state" in context.component.lower():
            return ErrorType.LANGGRAPH_STATE_ERROR
        elif "transition" in context.component.lower():
            return ErrorType.LANGGRAPH_TRANSITION_ERROR
        elif "checkpoint" in context.component.lower():
            return ErrorType.LANGGRAPH_CHECKPOINT_ERROR
    
    # Default to runtime exception
    return ErrorType.RUNTIME_EXCEPTION


def determine_severity(error_type: ErrorType, context: ErrorContext) -> ErrorSeverity:
    """Determine the severity of an error based on type and context."""
    
    # Critical errors
    if error_type in [
        ErrorType.MEMORY_OVERFLOW,
        ErrorType.STACK_OVERFLOW,
        ErrorType.RESOURCE_EXHAUSTION,
        ErrorType.DATA_CORRUPTION
    ]:
        return ErrorSeverity.CRITICAL
    
    # High severity errors
    if error_type in [
        ErrorType.API_ERROR,
        ErrorType.SERVICE_UNAVAILABLE,
        ErrorType.STATE_ERROR,
        ErrorType.MEMORY_ERROR,
        ErrorType.TIMEOUT
    ]:
        return ErrorSeverity.HIGH
    
    # Medium severity errors
    if error_type in [
        ErrorType.RATE_LIMIT,
        ErrorType.NETWORK_ERROR,
        ErrorType.VALIDATION_ERROR,
        ErrorType.SLOW_EXECUTION
    ]:
        return ErrorSeverity.MEDIUM
    
    # Low severity errors
    if error_type in [
        ErrorType.TYPE_MISMATCH,
        ErrorType.HIGH_MEMORY_USAGE,
        ErrorType.HIGH_CPU_USAGE
    ]:
        return ErrorSeverity.LOW
    
    # Default to medium
    return ErrorSeverity.MEDIUM
