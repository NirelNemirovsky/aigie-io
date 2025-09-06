"""
Validation Engine - Orchestrates the entire runtime validation process.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass

from .runtime_validator import RuntimeValidator
from .step_corrector import StepCorrector
from ..types.validation_types import (
    ExecutionStep, ProcessedStep, ValidationResult, 
    ValidationMetrics, ValidationReport, ValidationStatus
)


@dataclass
class ValidationPolicies:
    """Policies for when to validate steps."""
    
    enable_validation: bool = True
    validate_all_steps: bool = False
    skip_validation_for: List[str] = None  # Component names to skip
    min_confidence_threshold: float = 0.5
    max_validation_frequency: int = 10  # Max validations per minute
    
    def __post_init__(self):
        if self.skip_validation_for is None:
            self.skip_validation_for = []
    
    def should_validate(self, step: ExecutionStep) -> bool:
        """Determine if a step should be validated."""
        if not self.enable_validation:
            return False
        
        if step.component in self.skip_validation_for:
            return False
        
        if self.validate_all_steps:
            return True
        
        # Default: validate steps with goals and reasoning
        return bool(step.agent_goal and step.step_reasoning)


@dataclass
class ValidationOptimizer:
    """Optimizes validation performance."""
    
    enable_caching: bool = True
    enable_batching: bool = False
    max_concurrent_validations: int = 5
    validation_timeout: float = 30.0
    
    def __init__(self):
        self.active_validations = 0
        self.validation_queue = []
        self.performance_metrics = {
            "total_validations": 0,
            "avg_validation_time": 0.0,
            "cache_hit_rate": 0.0,
            "concurrent_validations": 0
        }


class ValidationEngine:
    """Orchestrates the entire runtime validation process."""
    
    def __init__(self, validator: RuntimeValidator, corrector: StepCorrector):
        self.validator = validator
        self.corrector = corrector
        self.validation_policies = ValidationPolicies()
        self.performance_optimizer = ValidationOptimizer()
        
        # Metrics and reporting
        self.metrics = ValidationMetrics()
        self.validation_history = []
        self.correction_history = []
        
        # Event handlers
        self.validation_handlers: List[Callable[[ProcessedStep], None]] = []
        self.correction_handlers: List[Callable[[ProcessedStep], None]] = []
        
        logging.info("ValidationEngine initialized with full orchestration capabilities")
    
    async def process_step(self, step: ExecutionStep) -> ProcessedStep:
        """Main processing pipeline for each execution step."""
        
        start_time = datetime.now()
        
        try:
            # 1. Pre-validation filtering
            if not self.validation_policies.should_validate(step):
                logging.debug(f"Skipping validation for step {step.step_id}")
                return ProcessedStep(
                    step=step,
                    validation_result=ValidationResult(
                        is_valid=True,
                        confidence=1.0,
                        reasoning="Skipped validation per policy",
                        issues=[],
                        suggestions=[]
                    )
                )
            
            # 2. Perform validation
            logging.debug(f"Validating step {step.step_id}")
            validation_result = await self.validator.validate_step(step)
            
            # 3. Handle validation outcome
            if validation_result.is_valid:
                processed_step = ProcessedStep(step=step, validation_result=validation_result)
                self._update_metrics(validation_result, None, start_time)
                self._notify_handlers(processed_step, self.validation_handlers)
                return processed_step
            
            # 4. Attempt automatic correction
            logging.info(f"Step {step.step_id} failed validation, attempting correction")
            correction_result = await self.corrector.correct_step(step, validation_result)
            
            # 5. Create final result
            processed_step = ProcessedStep(
                step=step,
                validation_result=validation_result,
                correction_result=correction_result
            )
            
            # 6. Update metrics and notify handlers
            self._update_metrics(validation_result, correction_result, start_time)
            self._notify_handlers(processed_step, self.validation_handlers)
            
            if correction_result.success:
                self._notify_handlers(processed_step, self.correction_handlers)
            
            return processed_step
            
        except Exception as e:
            logging.error(f"Step processing failed for {step.step_id}: {e}")
            
            # Create error result
            error_validation = ValidationResult(
                is_valid=False,
                confidence=0.0,
                reasoning=f"Processing error: {str(e)}",
                issues=[f"System error: {str(e)}"],
                suggestions=["Check system configuration", "Retry the operation"]
            )
            
            processed_step = ProcessedStep(step=step, validation_result=error_validation)
            self._update_metrics(error_validation, None, start_time)
            
            return processed_step
    
    async def process_steps_batch(self, steps: List[ExecutionStep]) -> List[ProcessedStep]:
        """Process multiple steps in batch for efficiency."""
        
        if not self.performance_optimizer.enable_batching:
            # Process sequentially
            results = []
            for step in steps:
                result = await self.process_step(step)
                results.append(result)
            return results
        
        # Process in parallel with concurrency limit
        semaphore = asyncio.Semaphore(self.performance_optimizer.max_concurrent_validations)
        
        async def process_with_semaphore(step):
            async with semaphore:
                return await self.process_step(step)
        
        tasks = [process_with_semaphore(step) for step in steps]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logging.error(f"Batch processing failed for step {steps[i].step_id}: {result}")
                error_validation = ValidationResult(
                    is_valid=False,
                    confidence=0.0,
                    reasoning=f"Batch processing error: {str(result)}",
                    issues=[f"Processing error: {str(result)}"],
                    suggestions=["Check system configuration"]
                )
                processed_results.append(ProcessedStep(step=steps[i], validation_result=error_validation))
            else:
                processed_results.append(result)
        
        return processed_results
    
    def add_validation_handler(self, handler: Callable[[ProcessedStep], None]):
        """Add a handler for validation events."""
        self.validation_handlers.append(handler)
        logging.info(f"Added validation handler: {handler.__name__}")
    
    def add_correction_handler(self, handler: Callable[[ProcessedStep], None]):
        """Add a handler for correction events."""
        self.correction_handlers.append(handler)
        logging.info(f"Added correction handler: {handler.__name__}")
    
    def remove_validation_handler(self, handler: Callable[[ProcessedStep], None]):
        """Remove a validation handler."""
        if handler in self.validation_handlers:
            self.validation_handlers.remove(handler)
            logging.info(f"Removed validation handler: {handler.__name__}")
    
    def remove_correction_handler(self, handler: Callable[[ProcessedStep], None]):
        """Remove a correction handler."""
        if handler in self.correction_handlers:
            self.correction_handlers.remove(handler)
            logging.info(f"Removed correction handler: {handler.__name__}")
    
    def configure_policies(self, **kwargs):
        """Configure validation policies."""
        for key, value in kwargs.items():
            if hasattr(self.validation_policies, key):
                setattr(self.validation_policies, key, value)
                logging.info(f"Updated policy {key}: {value}")
    
    def configure_optimizer(self, **kwargs):
        """Configure performance optimizer."""
        for key, value in kwargs.items():
            if hasattr(self.performance_optimizer, key):
                setattr(self.performance_optimizer, key, value)
                logging.info(f"Updated optimizer {key}: {value}")
    
    def get_metrics(self) -> ValidationMetrics:
        """Get current validation metrics."""
        return self.metrics
    
    def get_validation_report(self, time_span_minutes: int = 60) -> ValidationReport:
        """Generate comprehensive validation report."""
        
        cutoff_time = datetime.now() - timedelta(minutes=time_span_minutes)
        
        # Filter recent validations
        recent_validations = [
            v for v in self.validation_history 
            if v.get('timestamp', datetime.min) >= cutoff_time
        ]
        
        recent_corrections = [
            c for c in self.correction_history 
            if c.get('timestamp', datetime.min) >= cutoff_time
        ]
        
        # Calculate metrics
        total_validations = len(recent_validations)
        successful_validations = len([v for v in recent_validations if v.get('is_valid', False)])
        failed_validations = total_validations - successful_validations
        
        corrections_attempted = len(recent_corrections)
        successful_corrections = len([c for c in recent_corrections if c.get('success', False)])
        
        # Calculate averages
        validation_times = [v.get('validation_time', 0) for v in recent_validations]
        correction_times = [c.get('correction_time', 0) for c in recent_corrections]
        
        avg_validation_time = sum(validation_times) / len(validation_times) if validation_times else 0.0
        avg_correction_time = sum(correction_times) / len(correction_times) if correction_times else 0.0
        
        # Calculate confidence metrics
        confidences = [v.get('confidence', 0) for v in recent_validations]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        high_confidence_rate = len([c for c in confidences if c > 0.8]) / len(confidences) if confidences else 0.0
        
        # Update metrics
        self.metrics.total_validations = total_validations
        self.metrics.successful_validations = successful_validations
        self.metrics.failed_validations = failed_validations
        self.metrics.corrections_attempted = corrections_attempted
        self.metrics.successful_corrections = successful_corrections
        self.metrics.avg_validation_time = avg_validation_time
        self.metrics.avg_correction_time = avg_correction_time
        self.metrics.avg_confidence = avg_confidence
        self.metrics.high_confidence_rate = high_confidence_rate
        
        # Generate recommendations
        recommendations = self._generate_recommendations(recent_validations, recent_corrections)
        
        return ValidationReport(
            report_timestamp=datetime.now(),
            time_span_minutes=time_span_minutes,
            metrics=self.metrics,
            common_issues=self._analyze_common_issues(recent_validations),
            common_suggestions=self._analyze_common_suggestions(recent_validations),
            performance_impact=self._analyze_performance_impact(recent_validations),
            recommendations=recommendations
        )
    
    def _update_metrics(self, validation_result: ValidationResult, 
                       correction_result: Optional[Any], start_time: datetime):
        """Update internal metrics."""
        
        validation_time = (datetime.now() - start_time).total_seconds()
        
        # Store validation record
        validation_record = {
            'timestamp': datetime.now(),
            'is_valid': validation_result.is_valid,
            'confidence': validation_result.confidence,
            'validation_time': validation_time,
            'issues': validation_result.issues,
            'suggestions': validation_result.suggestions
        }
        self.validation_history.append(validation_record)
        
        # Store correction record if applicable
        if correction_result:
            correction_record = {
                'timestamp': datetime.now(),
                'success': correction_result.success,
                'strategy': correction_result.correction_strategy.value if correction_result.correction_strategy else None,
                'correction_time': validation_time,
                'attempts': correction_result.correction_attempts
            }
            self.correction_history.append(correction_record)
        
        # Keep only recent history (last 1000 records)
        if len(self.validation_history) > 1000:
            self.validation_history = self.validation_history[-1000:]
        if len(self.correction_history) > 1000:
            self.correction_history = self.correction_history[-1000:]
    
    def _notify_handlers(self, processed_step: ProcessedStep, handlers: List[Callable]):
        """Notify event handlers."""
        for handler in handlers:
            try:
                handler(processed_step)
            except Exception as e:
                logging.warning(f"Handler {handler.__name__} failed: {e}")
    
    def _analyze_common_issues(self, validations: List[Dict]) -> List[Dict[str, Any]]:
        """Analyze common issues from recent validations."""
        issue_counts = {}
        
        for validation in validations:
            for issue in validation.get('issues', []):
                issue_counts[issue] = issue_counts.get(issue, 0) + 1
        
        # Sort by frequency
        common_issues = [
            {"issue": issue, "count": count, "frequency": count / len(validations) if validations else 0}
            for issue, count in sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)
        ]
        
        return common_issues[:10]  # Top 10 issues
    
    def _analyze_common_suggestions(self, validations: List[Dict]) -> List[Dict[str, Any]]:
        """Analyze common suggestions from recent validations."""
        suggestion_counts = {}
        
        for validation in validations:
            for suggestion in validation.get('suggestions', []):
                suggestion_counts[suggestion] = suggestion_counts.get(suggestion, 0) + 1
        
        # Sort by frequency
        common_suggestions = [
            {"suggestion": suggestion, "count": count, "frequency": count / len(validations) if validations else 0}
            for suggestion, count in sorted(suggestion_counts.items(), key=lambda x: x[1], reverse=True)
        ]
        
        return common_suggestions[:10]  # Top 10 suggestions
    
    def _analyze_performance_impact(self, validations: List[Dict]) -> Dict[str, float]:
        """Analyze performance impact of validation."""
        if not validations:
            return {}
        
        total_time = sum(v.get('validation_time', 0) for v in validations)
        avg_time = total_time / len(validations)
        max_time = max(v.get('validation_time', 0) for v in validations)
        min_time = min(v.get('validation_time', 0) for v in validations)
        
        return {
            "total_validation_time": total_time,
            "avg_validation_time": avg_time,
            "max_validation_time": max_time,
            "min_validation_time": min_time,
            "validations_per_minute": len(validations) / (total_time / 60) if total_time > 0 else 0
        }
    
    def _generate_recommendations(self, validations: List[Dict], corrections: List[Dict]) -> List[str]:
        """Generate recommendations based on validation patterns."""
        recommendations = []
        
        if not validations:
            return ["No validation data available for recommendations"]
        
        # Analyze failure rate
        failure_rate = len([v for v in validations if not v.get('is_valid', True)]) / len(validations)
        if failure_rate > 0.3:
            recommendations.append("High failure rate detected - consider reviewing agent logic and goals")
        
        # Analyze correction success rate
        if corrections:
            correction_success_rate = len([c for c in corrections if c.get('success', False)]) / len(corrections)
            if correction_success_rate < 0.5:
                recommendations.append("Low correction success rate - consider improving correction strategies")
        
        # Analyze confidence levels
        avg_confidence = sum(v.get('confidence', 0) for v in validations) / len(validations)
        if avg_confidence < 0.6:
            recommendations.append("Low average confidence - consider improving validation prompts")
        
        # Analyze performance
        avg_time = sum(v.get('validation_time', 0) for v in validations) / len(validations)
        if avg_time > 5.0:
            recommendations.append("Slow validation performance - consider optimizing validation process")
        
        if not recommendations:
            recommendations.append("Validation system is performing well - no specific recommendations")
        
        return recommendations
