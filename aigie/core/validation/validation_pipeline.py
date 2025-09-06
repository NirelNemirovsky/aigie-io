"""
Advanced Validation Pipeline - Multi-stage validation with performance optimization.
"""

import asyncio
import time
import logging
from typing import Dict, Any, Optional, List, Callable, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed
import weakref

from ..types.validation_types import ExecutionStep, ValidationResult, ValidationStatus, RiskLevel, ValidationStrategy
from .runtime_validator import RuntimeValidator as AdvancedRuntimeValidator, ValidationConfig


class ValidationStage(Enum):
    """Different stages of validation pipeline."""
    
    PRE_VALIDATION = "pre_validation"
    FAST_VALIDATION = "fast_validation"
    DEEP_VALIDATION = "deep_validation"
    POST_VALIDATION = "post_validation"


@dataclass
class ValidationStageConfig:
    """Configuration for a validation stage."""
    
    stage: ValidationStage
    enabled: bool = True
    timeout_seconds: float = 5.0
    priority: int = 1  # Lower number = higher priority
    required_confidence: float = 0.0
    strategies: List[str] = field(default_factory=list)
    parallel_execution: bool = True
    cache_results: bool = True


@dataclass
class PipelineMetrics:
    """Metrics for the validation pipeline."""
    
    total_steps_processed: int = 0
    stage_completion_rates: Dict[ValidationStage, float] = field(default_factory=dict)
    avg_pipeline_time: float = 0.0
    avg_stage_times: Dict[ValidationStage, float] = field(default_factory=dict)
    cache_hit_rates: Dict[ValidationStage, float] = field(default_factory=dict)
    parallel_utilization: float = 0.0
    error_rates: Dict[ValidationStage, float] = field(default_factory=dict)
    
    def update_stage_metrics(self, stage: ValidationStage, completion_time: float, 
                           cache_hit: bool, error: bool = False):
        """Update metrics for a specific stage."""
        if stage not in self.stage_completion_rates:
            self.stage_completion_rates[stage] = 0.0
            self.avg_stage_times[stage] = 0.0
            self.cache_hit_rates[stage] = 0.0
            self.error_rates[stage] = 0.0
        
        # Update completion rate
        self.stage_completion_rates[stage] = (
            (self.stage_completion_rates[stage] * (self.total_steps_processed - 1) + 1) 
            / self.total_steps_processed
        )
        
        # Update average stage time
        self.avg_stage_times[stage] = (
            (self.avg_stage_times[stage] * (self.total_steps_processed - 1) + completion_time) 
            / self.total_steps_processed
        )
        
        # Update cache hit rate
        if cache_hit:
            self.cache_hit_rates[stage] = (
                (self.cache_hit_rates[stage] * (self.total_steps_processed - 1) + 1) 
                / self.total_steps_processed
            )
        
        # Update error rate
        if error:
            self.error_rates[stage] = (
                (self.error_rates[stage] * (self.total_steps_processed - 1) + 1) 
                / self.total_steps_processed
            )


class ValidationPipeline:
    """Multi-stage validation pipeline with performance optimization."""
    
    def __init__(self, validator: AdvancedRuntimeValidator, config: Optional[Dict[ValidationStage, ValidationStageConfig]] = None):
        self.validator = validator
        self.config = config or self._get_default_config()
        self.metrics = PipelineMetrics()
        
        # Pipeline stages
        self.stages = self._initialize_stages()
        
        # Performance optimization
        self.executor = ThreadPoolExecutor(max_workers=10)
        self.stage_cache = {}
        self.pipeline_cache = {}
        
        # Event handlers
        self.stage_handlers: Dict[ValidationStage, List[Callable]] = {
            stage: [] for stage in ValidationStage
        }
        
        logging.info("ValidationPipeline initialized with multi-stage processing")
    
    def _get_default_config(self) -> Dict[ValidationStage, ValidationStageConfig]:
        """Get default configuration for validation stages."""
        return {
            ValidationStage.PRE_VALIDATION: ValidationStageConfig(
                stage=ValidationStage.PRE_VALIDATION,
                enabled=True,
                timeout_seconds=1.0,
                priority=1,
                required_confidence=0.0,
                strategies=["basic_checks"],
                parallel_execution=False,
                cache_results=True
            ),
            ValidationStage.FAST_VALIDATION: ValidationStageConfig(
                stage=ValidationStage.FAST_VALIDATION,
                enabled=True,
                timeout_seconds=3.0,
                priority=2,
                required_confidence=0.6,
                strategies=["goal_alignment", "safety_compliance"],
                parallel_execution=True,
                cache_results=True
            ),
            ValidationStage.DEEP_VALIDATION: ValidationStageConfig(
                stage=ValidationStage.DEEP_VALIDATION,
                enabled=True,
                timeout_seconds=10.0,
                priority=3,
                required_confidence=0.8,
                strategies=["all"],
                parallel_execution=True,
                cache_results=True
            ),
            ValidationStage.POST_VALIDATION: ValidationStageConfig(
                stage=ValidationStage.POST_VALIDATION,
                enabled=True,
                timeout_seconds=2.0,
                priority=4,
                required_confidence=0.0,
                strategies=["result_analysis"],
                parallel_execution=False,
                cache_results=False
            )
        }
    
    def _initialize_stages(self) -> Dict[ValidationStage, Callable]:
        """Initialize validation stage handlers."""
        return {
            ValidationStage.PRE_VALIDATION: self._pre_validation_stage,
            ValidationStage.FAST_VALIDATION: self._fast_validation_stage,
            ValidationStage.DEEP_VALIDATION: self._deep_validation_stage,
            ValidationStage.POST_VALIDATION: self._post_validation_stage
        }
    
    async def process_step(self, step: ExecutionStep) -> ValidationResult:
        """Process a step through the validation pipeline."""
        start_time = time.time()
        self.metrics.total_steps_processed += 1
        
        try:
            # Check pipeline cache first
            pipeline_key = self._get_pipeline_cache_key(step)
            if pipeline_key in self.pipeline_cache:
                cached_result = self.pipeline_cache[pipeline_key]
                if self._is_pipeline_cache_valid(cached_result):
                    logging.debug(f"Using cached pipeline result for step {step.step_id}")
                    return cached_result['result']
            
            # Execute validation stages
            result = await self._execute_pipeline(step)
            
            # Cache the result
            self.pipeline_cache[pipeline_key] = {
                'result': result,
                'timestamp': datetime.now()
            }
            
            # Update overall pipeline metrics
            pipeline_time = time.time() - start_time
            self.metrics.avg_pipeline_time = (
                (self.metrics.avg_pipeline_time * (self.metrics.total_steps_processed - 1) + pipeline_time) 
                / self.metrics.total_steps_processed
            )
            
            logging.info(f"Pipeline completed for step {step.step_id} in {pipeline_time:.3f}s")
            return result
            
        except Exception as e:
            logging.error(f"Pipeline processing failed for step {step.step_id}: {e}")
            return ValidationResult(
                is_valid=False,
                confidence=0.0,
                reasoning=f"Pipeline error: {str(e)}",
                issues=[f"Pipeline processing failed: {str(e)}"],
                suggestions=["Check pipeline configuration"],
                risk_level=RiskLevel.HIGH
            )
    
    async def _execute_pipeline(self, step: ExecutionStep) -> ValidationResult:
        """Execute the validation pipeline stages."""
        # Sort stages by priority
        sorted_stages = sorted(
            [stage for stage in ValidationStage if self.config[stage].enabled],
            key=lambda s: self.config[s].priority
        )
        
        current_result = None
        
        for stage in sorted_stages:
            stage_config = self.config[stage]
            stage_start_time = time.time()
            
            try:
                # Execute stage with timeout
                stage_result = await asyncio.wait_for(
                    self.stages[stage](step, current_result),
                    timeout=stage_config.timeout_seconds
                )
                
                # Update stage metrics
                stage_time = time.time() - stage_start_time
                self.metrics.update_stage_metrics(stage, stage_time, False, False)
                
                # Check if we can short-circuit
                if self._can_short_circuit(stage_result, stage_config):
                    logging.debug(f"Short-circuiting pipeline at stage {stage.value}")
                    return stage_result
                
                current_result = stage_result
                
                # Notify stage handlers
                await self._notify_stage_handlers(stage, step, stage_result)
                
            except asyncio.TimeoutError:
                logging.warning(f"Stage {stage.value} timed out after {stage_config.timeout_seconds}s")
                self.metrics.update_stage_metrics(stage, stage_config.timeout_seconds, False, True)
                
                # Continue with next stage or return current result
                if current_result is None:
                    current_result = ValidationResult(
                        is_valid=False,
                        confidence=0.0,
                        reasoning=f"Stage {stage.value} timed out",
                        issues=[f"Stage timeout: {stage.value}"],
                        suggestions=["Increase timeout or optimize stage"],
                        risk_level=RiskLevel.MEDIUM
                    )
            
            except Exception as e:
                logging.error(f"Stage {stage.value} failed: {e}")
                self.metrics.update_stage_metrics(stage, time.time() - stage_start_time, False, True)
                
                # Continue with next stage or return current result
                if current_result is None:
                    current_result = ValidationResult(
                        is_valid=False,
                        confidence=0.0,
                        reasoning=f"Stage {stage.value} failed: {str(e)}",
                        issues=[f"Stage error: {stage.value}"],
                        suggestions=["Check stage implementation"],
                        risk_level=RiskLevel.HIGH
                    )
        
        return current_result or ValidationResult(
            is_valid=True,
            confidence=0.5,
            reasoning="Pipeline completed without result",
            issues=[],
            suggestions=[],
            risk_level=RiskLevel.LOW
        )
    
    async def _pre_validation_stage(self, step: ExecutionStep, previous_result: Optional[ValidationResult]) -> ValidationResult:
        """Pre-validation stage: Basic checks and filtering."""
        stage_config = self.config[ValidationStage.PRE_VALIDATION]
        
        # Basic validation checks
        issues = []
        suggestions = []
        
        # Check if step has required fields
        if not step.component:
            issues.append("Missing component information")
            suggestions.append("Ensure component is specified")
        
        if not step.operation:
            issues.append("Missing operation information")
            suggestions.append("Ensure operation is specified")
        
        # Check input data
        if not step.input_data:
            issues.append("No input data provided")
            suggestions.append("Provide input data for validation")
        
        # Determine validity based on basic checks
        is_valid = len(issues) == 0
        confidence = 1.0 if is_valid else 0.3
        
        return ValidationResult(
            is_valid=is_valid,
            confidence=confidence,
            reasoning="Pre-validation basic checks",
            issues=issues,
            suggestions=suggestions,
            risk_level=RiskLevel.LOW if is_valid else RiskLevel.MEDIUM
        )
    
    async def _fast_validation_stage(self, step: ExecutionStep, previous_result: Optional[ValidationResult]) -> ValidationResult:
        """Fast validation stage: Quick validation using essential strategies."""
        stage_config = self.config[ValidationStage.FAST_VALIDATION]
        
        # Use only essential strategies for speed
        essential_strategies = [
            ValidationStrategy.GOAL_ALIGNMENT,
            ValidationStrategy.SAFETY_COMPLIANCE
        ]
        
        # Create validation config for fast validation
        fast_config = ValidationConfig(
            enabled_strategies=essential_strategies,
            enable_parallel_strategies=True,
            cache_ttl_seconds=60,  # Shorter cache for fast validation
            max_concurrent_validations=5
        )
        
        # Temporarily update validator config
        original_config = self.validator.config
        self.validator.config = fast_config
        
        try:
            result = await self.validator.validate_step(step)
            return result
        finally:
            # Restore original config
            self.validator.config = original_config
    
    async def _deep_validation_stage(self, step: ExecutionStep, previous_result: Optional[ValidationResult]) -> ValidationResult:
        """Deep validation stage: Comprehensive validation using all strategies."""
        stage_config = self.config[ValidationStage.DEEP_VALIDATION]
        
        # Use all available strategies
        all_strategies = list(ValidationStrategy)
        
        # Create validation config for deep validation
        deep_config = ValidationConfig(
            enabled_strategies=all_strategies,
            enable_parallel_strategies=True,
            enable_adaptive_validation=True,
            enable_pattern_learning=True,
            cache_ttl_seconds=300
        )
        
        # Temporarily update validator config
        original_config = self.validator.config
        self.validator.config = deep_config
        
        try:
            result = await self.validator.validate_step(step)
            return result
        finally:
            # Restore original config
            self.validator.config = original_config
    
    async def _post_validation_stage(self, step: ExecutionStep, previous_result: Optional[ValidationResult]) -> ValidationResult:
        """Post-validation stage: Result analysis and final processing."""
        if not previous_result:
            return ValidationResult(
                is_valid=False,
                confidence=0.0,
                reasoning="No previous validation result",
                issues=["Missing validation result"],
                suggestions=["Check validation pipeline"],
                risk_level=RiskLevel.HIGH
            )
        
        # Analyze the validation result
        analysis_issues = []
        analysis_suggestions = []
        
        # Check confidence levels
        if previous_result.confidence < 0.5:
            analysis_issues.append("Low confidence validation result")
            analysis_suggestions.append("Consider additional validation strategies")
        
        # Check for critical issues
        if previous_result.risk_level == RiskLevel.CRITICAL:
            analysis_issues.append("Critical risk level detected")
            analysis_suggestions.append("Immediate attention required")
        
        # Combine with previous result
        combined_issues = previous_result.issues + analysis_issues
        combined_suggestions = previous_result.suggestions + analysis_suggestions
        
        # Determine final validity
        final_valid = previous_result.is_valid and len(analysis_issues) == 0
        
        return ValidationResult(
            is_valid=final_valid,
            confidence=previous_result.confidence,
            reasoning=f"Post-validation analysis: {previous_result.reasoning}",
            issues=combined_issues,
            suggestions=combined_suggestions,
            risk_level=previous_result.risk_level,
            strategy_results=previous_result.strategy_results
        )
    
    def _can_short_circuit(self, result: ValidationResult, stage_config: ValidationStageConfig) -> bool:
        """Check if pipeline can short-circuit based on current result."""
        # Short-circuit if confidence is high enough and no critical issues
        if (result.confidence >= stage_config.required_confidence and 
            result.risk_level != RiskLevel.CRITICAL and
            len(result.issues) == 0):
            return True
        
        # Short-circuit if validation failed and we're in early stages
        if not result.is_valid and stage_config.priority <= 2:
            return True
        
        return False
    
    def _get_pipeline_cache_key(self, step: ExecutionStep) -> str:
        """Generate cache key for pipeline results."""
        key_data = {
            "framework": step.framework,
            "component": step.component,
            "operation": step.operation,
            "input_hash": hash(str(sorted(step.input_data.items()))),
            "goal": step.agent_goal,
            "pipeline_version": "1.0"
        }
        return f"pipeline_{hash(str(key_data))}"
    
    def _is_pipeline_cache_valid(self, cached_result: Dict[str, Any]) -> bool:
        """Check if cached pipeline result is still valid."""
        cache_time = cached_result.get("timestamp", datetime.min)
        age = (datetime.now() - cache_time).total_seconds()
        return age < 300  # 5 minutes cache TTL
    
    async def _notify_stage_handlers(self, stage: ValidationStage, step: ExecutionStep, result: ValidationResult):
        """Notify registered handlers for a stage."""
        handlers = self.stage_handlers.get(stage, [])
        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(step, result)
                else:
                    handler(step, result)
            except Exception as e:
                logging.warning(f"Stage handler failed for {stage.value}: {e}")
    
    def add_stage_handler(self, stage: ValidationStage, handler: Callable):
        """Add a handler for a specific stage."""
        if stage not in self.stage_handlers:
            self.stage_handlers[stage] = []
        self.stage_handlers[stage].append(handler)
    
    def remove_stage_handler(self, stage: ValidationStage, handler: Callable):
        """Remove a handler for a specific stage."""
        if stage in self.stage_handlers:
            try:
                self.stage_handlers[stage].remove(handler)
            except ValueError:
                pass
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get pipeline metrics."""
        return {
            "total_steps_processed": self.metrics.total_steps_processed,
            "avg_pipeline_time": self.metrics.avg_pipeline_time,
            "stage_completion_rates": {
                stage.value: rate for stage, rate in self.metrics.stage_completion_rates.items()
            },
            "avg_stage_times": {
                stage.value: time for stage, time in self.metrics.avg_stage_times.items()
            },
            "cache_hit_rates": {
                stage.value: rate for stage, rate in self.metrics.cache_hit_rates.items()
            },
            "error_rates": {
                stage.value: rate for stage, rate in self.metrics.error_rates.items()
            },
            "pipeline_cache_size": len(self.pipeline_cache)
        }
    
    def clear_caches(self):
        """Clear all pipeline caches."""
        self.pipeline_cache.clear()
        self.stage_cache.clear()
        logging.info("Pipeline caches cleared")
    
    def shutdown(self):
        """Shutdown the pipeline and cleanup resources."""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)
        logging.info("ValidationPipeline shutdown complete")
