"""
Enhanced interceptor that integrates runtime validation with existing interceptors.
"""

import functools
import inspect
import asyncio
from typing import Any, Callable, Dict, Optional, Union
from datetime import datetime

from ..core.error_handling.error_detector import ErrorDetector
from ..core.types.error_types import ErrorContext
from ..core.validation.validation_engine import ValidationEngine
from ..core.types.validation_types import ExecutionStep, ProcessedStep
from ..core.validation.context_extractor import ContextExtractor
from ..reporting.logger import AigieLogger


class ValidationInterceptor:
    """Enhanced interceptor that adds runtime validation to existing interceptors."""
    
    def __init__(self, error_detector: ErrorDetector, validation_engine: ValidationEngine, logger: AigieLogger):
        self.error_detector = error_detector
        self.validation_engine = validation_engine
        self.logger = logger
        # Get GeminiAnalyzer from validation_engine for context extraction
        gemini_analyzer = getattr(validation_engine.validator, 'gemini_analyzer', None)
        self.context_extractor = ContextExtractor(gemini_analyzer)
        
        # Track validation state (auto-managed, no manual input required)
        self.validation_enabled = True
        self.current_agent_goal = None
        self.conversation_history = []
        
        # Performance tracking
        self.validation_stats = {
            "total_steps": 0,
            "validated_steps": 0,
            "corrected_steps": 0,
            "failed_validations": 0
        }
        
        self.logger.log_system_event("ValidationInterceptor initialized with runtime validation")
    
    def set_agent_goal(self, goal: str):
        """Set the current agent goal for validation context (optional override)."""
        self.current_agent_goal = goal
        self.logger.log_system_event(f"Agent goal manually set: {goal}")
    
    def add_conversation_message(self, role: str, content: str):
        """Add a message to the conversation history (optional manual addition)."""
        self.conversation_history.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now()
        })
        
        # Keep only recent messages (last 10)
        if len(self.conversation_history) > 10:
            self.conversation_history = self.conversation_history[-10:]
    
    def clear_conversation_history(self):
        """Clear the conversation history."""
        self.conversation_history = []
        self.logger.log_system_event("Conversation history cleared")
    
    def enable_validation(self, enabled: bool = True):
        """Enable or disable runtime validation."""
        self.validation_enabled = enabled
        self.logger.log_system_event(f"Runtime validation {'enabled' if enabled else 'disabled'}")
    
    def create_enhanced_patched_method(self, original_method: Callable, class_name: str, method_name: str):
        """Create an enhanced patched method with validation."""
        
        if inspect.iscoroutinefunction(original_method):
            return self._create_async_enhanced_method(original_method, class_name, method_name)
        else:
            return self._create_sync_enhanced_method(original_method, class_name, method_name)
    
    def _create_sync_enhanced_method(self, original_method: Callable, class_name: str, method_name: str):
        """Create a synchronous enhanced method with validation."""
        
        @functools.wraps(original_method)
        def enhanced_method(self_instance, *args, **kwargs):
            # Create execution step
            step = self._create_execution_step(
                framework="langchain",
                component=class_name,
                operation=method_name,
                input_data=self._extract_input_data(args, kwargs, method_name),
                instance=self_instance
            )
            
            # Process with validation engine
            if self.validation_enabled:
                try:
                    # Check if we're already in an async context
                    try:
                        loop = asyncio.get_running_loop()
                        # We're in an async context, need to handle differently
                        # For now, skip validation in sync methods when in async context
                        self.logger.log_system_event("Skipping validation in sync method due to async context")
                    except RuntimeError:
                        # No running loop, safe to create new one
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        try:
                            processed_step = loop.run_until_complete(
                                self.validation_engine.process_step(step)
                            )
                            
                            # Update stats
                            self._update_validation_stats(processed_step)
                            
                            # Log validation result
                            self._log_validation_result(processed_step)
                            
                            # If validation failed and correction failed, log warning
                            if not processed_step.is_successful:
                                self.logger.log_system_event(
                                    f"Step validation failed: {processed_step.validation_result.reasoning}",
                                    {
                                        "step_id": step.step_id,
                                        "component": class_name,
                                        "operation": method_name,
                                        "issues": processed_step.validation_result.issues
                                    }
                                )
                        finally:
                            loop.close()
                    
                except Exception as e:
                    self.logger.log_system_event(f"Validation failed: {e}")
                    # Continue with original execution even if validation fails
            
            # Execute original method with enhanced monitoring
            with self.error_detector.monitor_execution(
                framework="langchain",
                component=class_name,
                method=method_name,
                input_data=step.input_data
            ):
                try:
                    result = original_method(self_instance, *args, **kwargs)
                    
                    # Update step with output data
                    if self.validation_enabled:
                        step.output_data = self._extract_output_data(result)
                        step.execution_time = (datetime.now() - step.timestamp).total_seconds()
                    
                    return result
                except Exception as e:
                    # Error will be detected by the context manager
                    raise
        
        return enhanced_method
    
    def _create_async_enhanced_method(self, original_method: Callable, class_name: str, method_name: str):
        """Create an asynchronous enhanced method with validation."""
        
        @functools.wraps(original_method)
        async def enhanced_method(self_instance, *args, **kwargs):
            # Create execution step
            step = self._create_execution_step(
                framework="langchain",
                component=class_name,
                operation=method_name,
                input_data=self._extract_input_data(args, kwargs, method_name),
                instance=self_instance
            )
            
            # Process with validation engine
            if self.validation_enabled:
                try:
                    processed_step = await self.validation_engine.process_step(step)
                    
                    # Update stats
                    self._update_validation_stats(processed_step)
                    
                    # Log validation result
                    self._log_validation_result(processed_step)
                    
                    # If validation failed and correction failed, log warning
                    if not processed_step.is_successful:
                        self.logger.log_system_event(
                            f"Step validation failed: {processed_step.validation_result.reasoning}",
                            {
                                "step_id": step.step_id,
                                "component": class_name,
                                "operation": method_name,
                                "issues": processed_step.validation_result.issues
                            }
                        )
                    
                except Exception as e:
                    self.logger.log_system_event(f"Validation failed: {e}")
                    # Continue with original execution even if validation fails
            
            # Execute original method with enhanced monitoring
            async with self.error_detector.monitor_execution_async(
                framework="langchain",
                component=class_name,
                method=method_name,
                input_data=step.input_data
            ):
                try:
                    result = await original_method(self_instance, *args, **kwargs)
                    
                    # Update step with output data
                    if self.validation_enabled:
                        step.output_data = self._extract_output_data(result)
                        step.execution_time = (datetime.now() - step.timestamp).total_seconds()
                    
                    return result
                except Exception as e:
                    # Error will be detected by the context manager
                    raise
        
        return enhanced_method
    
    def _create_execution_step(self, framework: str, component: str, operation: str, 
                              input_data: Dict[str, Any], instance: Any = None) -> ExecutionStep:
        """Create an ExecutionStep with automatically extracted context."""
        
        # Create basic step
        step = ExecutionStep(
            framework=framework,
            component=component,
            operation=operation,
            input_data=input_data,
            agent_goal=self.current_agent_goal,  # May be None, will be auto-inferred
            conversation_history=self.conversation_history.copy(),
            step_reasoning=self._generate_step_reasoning(component, operation, input_data),
            intermediate_state=self._extract_intermediate_state(instance) if instance else None
        )
        
        # Automatically extract context and infer goals
        step = self.context_extractor.extract_context(step)
        
        # Update conversation history with new context
        self._update_conversation_history(step)
        
        return step
    
    def _update_conversation_history(self, step: ExecutionStep):
        """Update conversation history with new step context."""
        try:
            # Extract messages from input data
            if step.input_data and 'messages' in step.input_data:
                messages = step.input_data['messages']
                if isinstance(messages, list):
                    for message in messages:
                        if isinstance(message, dict) and 'content' in message:
                            self.conversation_history.append({
                                'role': message.get('role', 'user'),
                                'content': str(message['content'])[:200],  # Truncate
                                'timestamp': datetime.now().isoformat()
                            })
            
            # Update current agent goal if we have a better inferred one
            if step.inferred_goal and step.auto_confidence and step.auto_confidence > 0.7:
                if not self.current_agent_goal or step.auto_confidence > 0.8:
                    self.current_agent_goal = step.inferred_goal
                    self.logger.log_system_event(f"Agent goal auto-updated: {step.inferred_goal}")
            
            # Keep conversation history manageable (last 10 messages)
            if len(self.conversation_history) > 10:
                self.conversation_history = self.conversation_history[-10:]
                
        except Exception as e:
            self.logger.warning(f"Failed to update conversation history: {e}")
    
    def _extract_input_data(self, args: tuple, kwargs: dict, method_name: str) -> Dict[str, Any]:
        """Extract relevant input data for monitoring."""
        input_data = {}
        
        # Extract common input parameters
        if args:
            if method_name in ['run', '__call__', 'acall', 'arun', 'invoke', 'ainvoke']:
                if args:
                    input_data['input'] = str(args[0])[:200]  # Truncate long inputs
        
        # Extract keyword arguments
        for key, value in kwargs.items():
            if key in ['input', 'inputs', 'query', 'text', 'prompt', 'messages']:
                input_data[key] = str(value)[:200]  # Truncate long inputs
            elif key in ['memory', 'tools', 'callbacks', 'config']:
                input_data[key] = type(value).__name__  # Just the type
            elif key in ['temperature', 'max_tokens', 'top_p']:
                input_data[key] = value  # Keep numeric values
        
        return input_data
    
    def _extract_output_data(self, result: Any) -> Dict[str, Any]:
        """Extract output data from method result."""
        if result is None:
            return {}
        
        output_data = {
            "result_type": type(result).__name__,
            "timestamp": datetime.now().isoformat()
        }
        
        # Extract specific output data based on result type
        if hasattr(result, 'content'):
            output_data['content'] = str(result.content)[:200]
        elif hasattr(result, 'text'):
            output_data['text'] = str(result.text)[:200]
        elif isinstance(result, str):
            output_data['text'] = result[:200]
        elif isinstance(result, dict):
            # Extract key fields from dict result
            for key in ['output', 'result', 'response', 'answer']:
                if key in result:
                    output_data[key] = str(result[key])[:200]
                    break
        
        return output_data
    
    def _extract_intermediate_state(self, instance: Any) -> Optional[Dict[str, Any]]:
        """Extract intermediate state from instance."""
        if not instance:
            return None
        
        state = {}
        
        try:
            # Extract common state attributes
            if hasattr(instance, 'memory') and instance.memory:
                state['memory_type'] = type(instance.memory).__name__
            
            if hasattr(instance, 'tools') and instance.tools:
                state['tool_count'] = len(instance.tools)
                state['tool_types'] = [type(tool).__name__ for tool in instance.tools[:3]]
            
            if hasattr(instance, 'llm') and instance.llm:
                state['llm_type'] = type(instance.llm).__name__
            
            if hasattr(instance, 'prompt') and instance.prompt:
                state['prompt_type'] = type(instance.prompt).__name__
            
        except Exception:
            # Ignore errors in state extraction
            pass
        
        return state if state else None
    
    def _generate_step_reasoning(self, component: str, operation: str, input_data: Dict[str, Any]) -> str:
        """Generate reasoning for the step based on context."""
        
        reasoning_parts = []
        
        # Component-specific reasoning
        if "ChatOpenAI" in component or "LLM" in component:
            reasoning_parts.append(f"Using {component} to generate text response")
        elif "Tool" in component:
            reasoning_parts.append(f"Executing {component} to perform specific task")
        elif "Chain" in component:
            reasoning_parts.append(f"Running {component} to process input through chain")
        else:
            reasoning_parts.append(f"Executing {operation} on {component}")
        
        # Input-based reasoning
        if 'messages' in input_data:
            reasoning_parts.append("processing user messages")
        elif 'query' in input_data:
            reasoning_parts.append("executing query")
        elif 'input' in input_data:
            reasoning_parts.append("processing input data")
        
        # Goal-based reasoning
        if self.current_agent_goal:
            reasoning_parts.append(f"to achieve goal: {self.current_agent_goal}")
        
        return " ".join(reasoning_parts)
    
    def _update_validation_stats(self, processed_step: ProcessedStep):
        """Update validation statistics."""
        self.validation_stats["total_steps"] += 1
        
        if processed_step.validation_result.is_valid:
            self.validation_stats["validated_steps"] += 1
        else:
            self.validation_stats["failed_validations"] += 1
        
        if processed_step.correction_result and processed_step.correction_result.success:
            self.validation_stats["corrected_steps"] += 1
    
    def _log_validation_result(self, processed_step: ProcessedStep):
        """Log validation result."""
        step = processed_step.step
        validation = processed_step.validation_result
        
        log_data = {
            "step_id": step.step_id,
            "component": step.component,
            "operation": step.operation,
            "is_valid": validation.is_valid,
            "confidence": validation.confidence,
            "risk_level": validation.risk_level.value,
            "execution_time": step.execution_time
        }
        
        if validation.issues:
            log_data["issues"] = validation.issues
        
        if validation.suggestions:
            log_data["suggestions"] = validation.suggestions
        
        if processed_step.correction_result:
            log_data["correction_attempted"] = True
            log_data["correction_success"] = processed_step.correction_result.success
        
        self.logger.log_system_event(
            f"Step validation: {'PASSED' if validation.is_valid else 'FAILED'}",
            log_data
        )
    
    def get_validation_stats(self) -> Dict[str, Any]:
        """Get current validation statistics."""
        stats = self.validation_stats.copy()
        
        if stats["total_steps"] > 0:
            stats["validation_rate"] = stats["validated_steps"] / stats["total_steps"]
            stats["correction_rate"] = stats["corrected_steps"] / stats["total_steps"]
            stats["failure_rate"] = stats["failed_validations"] / stats["total_steps"]
        else:
            stats["validation_rate"] = 0.0
            stats["correction_rate"] = 0.0
            stats["failure_rate"] = 0.0
        
        return stats
    
    def get_validation_report(self) -> Dict[str, Any]:
        """Get comprehensive validation report."""
        try:
            # Get metrics from validation engine
            metrics = self.validation_engine.get_metrics()
            
            # Get validation report
            report = self.validation_engine.get_validation_report(time_span_minutes=60)
            
            return {
                "validation_enabled": self.validation_enabled,
                "current_goal": self.current_agent_goal,
                "conversation_length": len(self.conversation_history),
                "stats": self.get_validation_stats(),
                "metrics": metrics.to_dict(),
                "report": report.to_dict()
            }
        except Exception as e:
            self.logger.log_system_event(f"Failed to generate validation report: {e}")
            return {
                "validation_enabled": self.validation_enabled,
                "current_goal": self.current_agent_goal,
                "conversation_length": len(self.conversation_history),
                "stats": self.get_validation_stats(),
                "error": str(e)
            }
