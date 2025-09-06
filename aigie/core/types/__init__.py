"""
Type definitions and data structures for Aigie.
"""

from .validation_types import (
    ExecutionStep, ValidationResult, ValidationStatus, ValidationStrategy,
    RiskLevel, CorrectionStrategy, CorrectionResult, ProcessedStep,
    ValidationMetrics, ValidationReport
)
from .error_types import (
    ErrorType, ErrorSeverity, ErrorContext, DetectedError,
    classify_error, determine_severity
)

__all__ = [
    # Validation types
    "ExecutionStep", "ValidationResult", "ValidationStatus", "ValidationStrategy",
    "RiskLevel", "CorrectionStrategy", "CorrectionResult", "ProcessedStep",
    "ValidationMetrics", "ValidationReport",
    
    # Error types
    "ErrorType", "ErrorSeverity", "ErrorContext", "DetectedError",
    "classify_error", "determine_severity"
]
