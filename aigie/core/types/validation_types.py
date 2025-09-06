"""
Validation types and data structures for Runtime Validation System.
"""

from enum import Enum
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field
from datetime import datetime
import uuid


class ValidationStatus(Enum):
    """Status of validation for an execution step."""
    
    PENDING = "pending"
    VALID = "valid"
    INVALID = "invalid"
    CORRECTED = "corrected"
    SKIPPED = "skipped"
    ERROR = "error"


class ValidationStrategy(Enum):
    """Different validation strategies."""
    
    GOAL_ALIGNMENT = "goal_alignment"
    LOGICAL_CONSISTENCY = "logical_consistency"
    OUTPUT_QUALITY = "output_quality"
    STATE_COHERENCE = "state_coherence"
    SAFETY_COMPLIANCE = "safety_compliance"
    PERFORMANCE_OPTIMALITY = "performance_optimality"


class RiskLevel(Enum):
    """Risk levels for validation results."""
    
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class CorrectionStrategy(Enum):
    """Strategies for correcting invalid steps."""
    
    PARAMETER_ADJUSTMENT = "parameter_adjustment"
    PROMPT_REFINEMENT = "prompt_refinement"
    STATE_RESTORATION = "state_restoration"
    TOOL_SUBSTITUTION = "tool_substitution"
    LOGIC_REPAIR = "logic_repair"
    GOAL_REALIGNMENT = "goal_realignment"


@dataclass
class ExecutionStep:
    """Rich context for each agent execution step."""
    
    # Core identification
    step_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Framework and component info
    framework: str = "unknown"  # "langchain" | "langgraph" | "custom"
    component: str = "unknown"  # e.g., "LLMChain", "Tool", "StateGraph"
    operation: str = "unknown"  # e.g., "invoke", "run", "call"
    
    # Execution context
    input_data: Dict[str, Any] = field(default_factory=dict)
    output_data: Optional[Dict[str, Any]] = None
    intermediate_state: Optional[Dict[str, Any]] = None
    
    # Agent context (auto-extracted, no manual input required)
    agent_goal: Optional[str] = None
    conversation_history: Optional[List[Dict[str, Any]]] = None
    step_reasoning: Optional[str] = None
    parent_step_id: Optional[str] = None
    
    # Auto-extracted context (inferred automatically)
    inferred_goal: Optional[str] = None
    context_clues: Optional[List[str]] = None
    operation_pattern: Optional[str] = None
    auto_confidence: Optional[float] = None
    
    # Performance metrics
    execution_time: Optional[float] = None
    memory_usage: Optional[float] = None
    token_usage: Optional[Dict[str, int]] = None
    
    # Validation results
    validation_status: ValidationStatus = ValidationStatus.PENDING
    validation_score: Optional[float] = None
    validation_reasoning: Optional[str] = None
    validation_timestamp: Optional[datetime] = None
    
    def copy(self) -> 'ExecutionStep':
        """Create a copy of this execution step."""
        return ExecutionStep(
            step_id=self.step_id,
            timestamp=self.timestamp,
            framework=self.framework,
            component=self.component,
            operation=self.operation,
            input_data=self.input_data.copy() if self.input_data else {},
            output_data=self.output_data.copy() if self.output_data else None,
            intermediate_state=self.intermediate_state.copy() if self.intermediate_state else None,
            agent_goal=self.agent_goal,
            conversation_history=self.conversation_history.copy() if self.conversation_history else None,
            step_reasoning=self.step_reasoning,
            parent_step_id=self.parent_step_id,
            inferred_goal=self.inferred_goal,
            context_clues=self.context_clues.copy() if self.context_clues else None,
            operation_pattern=self.operation_pattern,
            auto_confidence=self.auto_confidence,
            execution_time=self.execution_time,
            memory_usage=self.memory_usage,
            token_usage=self.token_usage.copy() if self.token_usage else None,
            validation_status=self.validation_status,
            validation_score=self.validation_score,
            validation_reasoning=self.validation_reasoning,
            validation_timestamp=self.validation_timestamp
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "step_id": self.step_id,
            "timestamp": self.timestamp.isoformat(),
            "framework": self.framework,
            "component": self.component,
            "operation": self.operation,
            "input_data": self.input_data,
            "output_data": self.output_data,
            "intermediate_state": self.intermediate_state,
            "agent_goal": self.agent_goal,
            "conversation_history": self.conversation_history,
            "step_reasoning": self.step_reasoning,
            "parent_step_id": self.parent_step_id,
            "execution_time": self.execution_time,
            "memory_usage": self.memory_usage,
            "token_usage": self.token_usage,
            "validation_status": self.validation_status.value,
            "validation_score": self.validation_score,
            "validation_reasoning": self.validation_reasoning,
            "validation_timestamp": self.validation_timestamp.isoformat() if self.validation_timestamp else None,
        }


@dataclass
class ValidationResult:
    """Result of validation for an execution step."""
    
    is_valid: bool
    confidence: float  # 0.0 to 1.0
    reasoning: str
    issues: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    risk_level: RiskLevel = RiskLevel.LOW
    
    # Strategy-specific results
    strategy_results: Dict[ValidationStrategy, Dict[str, Any]] = field(default_factory=dict)
    
    # Metadata
    validation_timestamp: datetime = field(default_factory=datetime.now)
    validator_version: str = "1.0.0"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "is_valid": self.is_valid,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "issues": self.issues,
            "suggestions": self.suggestions,
            "risk_level": self.risk_level.value,
            "strategy_results": {
                strategy.value: result 
                for strategy, result in self.strategy_results.items()
            },
            "validation_timestamp": self.validation_timestamp.isoformat(),
            "validator_version": self.validator_version,
        }


@dataclass
class CorrectionResult:
    """Result of step correction attempt."""
    
    original_step: ExecutionStep
    corrected_step: Optional[ExecutionStep] = None
    correction_strategy: Optional[CorrectionStrategy] = None
    validation_result: Optional[ValidationResult] = None
    success: bool = False
    correction_reasoning: Optional[str] = None
    
    # Metadata
    correction_timestamp: datetime = field(default_factory=datetime.now)
    correction_attempts: int = 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "original_step": self.original_step.to_dict(),
            "corrected_step": self.corrected_step.to_dict() if self.corrected_step else None,
            "correction_strategy": self.correction_strategy.value if self.correction_strategy else None,
            "validation_result": self.validation_result.to_dict() if self.validation_result else None,
            "success": self.success,
            "correction_reasoning": self.correction_reasoning,
            "correction_timestamp": self.correction_timestamp.isoformat(),
            "correction_attempts": self.correction_attempts,
        }


@dataclass
class ProcessedStep:
    """Final result of processing an execution step."""
    
    step: ExecutionStep
    validation_result: ValidationResult
    correction_result: Optional[CorrectionResult] = None
    
    @property
    def is_successful(self) -> bool:
        """Check if the step was successfully processed."""
        return (
            self.validation_result.is_valid or 
            (self.correction_result and self.correction_result.success)
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "step": self.step.to_dict(),
            "validation_result": self.validation_result.to_dict(),
            "correction_result": self.correction_result.to_dict() if self.correction_result else None,
            "is_successful": self.is_successful,
        }


@dataclass
class ValidationMetrics:
    """Metrics for validation performance."""
    
    total_validations: int = 0
    successful_validations: int = 0
    failed_validations: int = 0
    corrections_attempted: int = 0
    successful_corrections: int = 0
    
    # Timing metrics
    avg_validation_time: float = 0.0
    avg_correction_time: float = 0.0
    
    # Quality metrics
    avg_confidence: float = 0.0
    high_confidence_rate: float = 0.0  # Percentage of validations with confidence > 0.8
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "total_validations": self.total_validations,
            "successful_validations": self.successful_validations,
            "failed_validations": self.failed_validations,
            "corrections_attempted": self.corrections_attempted,
            "successful_corrections": self.successful_corrections,
            "avg_validation_time": self.avg_validation_time,
            "avg_correction_time": self.avg_correction_time,
            "avg_confidence": self.avg_confidence,
            "high_confidence_rate": self.high_confidence_rate,
        }


@dataclass
class ValidationReport:
    """Comprehensive validation report."""
    
    report_timestamp: datetime = field(default_factory=datetime.now)
    time_span_minutes: int = 60
    
    # Core metrics
    metrics: ValidationMetrics = field(default_factory=ValidationMetrics)
    
    # Pattern analysis
    common_issues: List[Dict[str, Any]] = field(default_factory=list)
    common_suggestions: List[Dict[str, Any]] = field(default_factory=list)
    
    # Performance analysis
    performance_impact: Dict[str, float] = field(default_factory=dict)
    
    # Recommendations
    recommendations: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "report_timestamp": self.report_timestamp.isoformat(),
            "time_span_minutes": self.time_span_minutes,
            "metrics": self.metrics.to_dict(),
            "common_issues": self.common_issues,
            "common_suggestions": self.common_suggestions,
            "performance_impact": self.performance_impact,
            "recommendations": self.recommendations,
        }
