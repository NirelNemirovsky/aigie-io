"""
Main error detection engine for Aigie.
"""

import traceback
import asyncio
import logging
from typing import Dict, Any, Optional, Callable, List, Union
from datetime import datetime, timedelta
from contextlib import contextmanager

from .error_types import (
    ErrorType, ErrorSeverity, ErrorContext, DetectedError,
    classify_error, determine_severity
)
from .monitoring import PerformanceMonitor, ResourceMonitor
from .gemini_analyzer import GeminiAnalyzer
from .intelligent_retry import IntelligentRetry


class ErrorDetector:
    """Main error detection engine for AI agent applications with real-time remediation."""
    
    def __init__(self, enable_performance_monitoring: bool = True, 
                 enable_resource_monitoring: bool = True,
                 enable_gemini_analysis: bool = True,
                 gemini_project_id: Optional[str] = None,
                 gemini_location: str = "us-central1"):
        self.performance_monitor = PerformanceMonitor() if enable_performance_monitoring else None
        self.resource_monitor = ResourceMonitor() if enable_resource_monitoring else None
        self.error_handlers: List[Callable[[DetectedError], None]] = []
        self.error_history: List[DetectedError] = []
        self.is_monitoring = False
        
        # Gemini integration
        self.gemini_analyzer = None
        self.intelligent_retry = None
        if enable_gemini_analysis:
            try:
                self.gemini_analyzer = GeminiAnalyzer(gemini_project_id, gemini_location)
                if self.gemini_analyzer.is_available():
                    self.intelligent_retry = IntelligentRetry(self.gemini_analyzer)
                    logging.info("Gemini-powered error analysis and retry enabled")
                else:
                    logging.info("Gemini not available - using fallback error analysis")
            except Exception as e:
                logging.warning(f"Failed to initialize Gemini: {e}")
        
        # Error detection settings
        self.enable_timeout_detection = True
        self.timeout_threshold = 300.0  # 5 minutes
        self.enable_memory_leak_detection = True
        self.memory_leak_threshold = 100.0  # MB increase over time
        
        # Retry settings
        self.enable_automatic_retry = True
        self.max_retries = 3
        self.retry_delay = 1.0
        
        # Operation storage for retry
        self.operation_store: Dict[str, Dict[str, Any]] = {}
        
        # Enhanced remediation capabilities
        self.enable_real_time_remediation = True
        self.enable_prompt_injection = True
        self.operation_context_stack: List[Dict[str, Any]] = []
        self.active_operations: Dict[str, Dict[str, Any]] = {}
        self.remediation_cache: Dict[str, Dict[str, Any]] = {}
        
    def add_error_handler(self, handler: Callable[[DetectedError], None]):
        """Add a custom error handler."""
        self.error_handlers.append(handler)
    
    def store_operation_for_retry(self, operation_id: str, operation: Callable, 
                                 args: tuple, kwargs: dict, context: ErrorContext):
        """Store an operation for potential retry with enhanced context."""
        self.operation_store[operation_id] = {
            'operation': operation,
            'args': args,
            'kwargs': kwargs,
            'context': context,
            'timestamp': datetime.now(),
            'retry_count': 0,
            'original_prompt': self._extract_prompt_from_args(args, kwargs),
            'operation_type': self._determine_operation_type(operation, context),
            'execution_stack': self.operation_context_stack.copy()
        }
        logging.info(f"Stored operation {operation_id} for potential retry")
    
    def register_active_operation(self, operation_id: str, operation_info: Dict[str, Any]):
        """Register an active operation for real-time monitoring."""
        self.active_operations[operation_id] = {
            **operation_info,
            'start_time': datetime.now(),
            'status': 'running',
            'context_stack': self.operation_context_stack.copy()
        }
    
    def complete_active_operation(self, operation_id: str, result: Any = None, error: Exception = None):
        """Mark an active operation as completed."""
        if operation_id in self.active_operations:
            self.active_operations[operation_id].update({
                'end_time': datetime.now(),
                'status': 'completed' if error is None else 'failed',
                'result': result,
                'error': error
            })
    
    def _extract_prompt_from_args(self, args: tuple, kwargs: dict) -> Optional[str]:
        """Extract prompt/input text from operation arguments."""
        # Common prompt parameter names
        prompt_keys = ['prompt', 'input', 'query', 'text', 'message', 'instruction', 'content']
        
        # Check kwargs first
        for key in prompt_keys:
            if key in kwargs and isinstance(kwargs[key], str):
                return kwargs[key]
        
        # Check positional args
        for arg in args:
            if isinstance(arg, str) and len(arg) > 10:  # Likely a prompt
                return arg
            elif isinstance(arg, dict):
                for key in prompt_keys:
                    if key in arg and isinstance(arg[key], str):
                        return arg[key]
        
        return None
    
    def _determine_operation_type(self, operation: Callable, context: ErrorContext) -> str:
        """Determine the type of operation being performed."""
        op_name = getattr(operation, '__name__', str(operation))
        component = context.component.lower()
        method = context.method.lower()
        
        if 'llm' in component or 'generate' in method:
            return 'llm_call'
        elif 'agent' in component or 'run' in method:
            return 'agent_execution'
        elif 'tool' in component or 'call' in method:
            return 'tool_call'
        elif 'chain' in component:
            return 'chain_execution'
        else:
            return 'unknown'
    
    def get_stored_operation(self, operation_id: str) -> Optional[Dict[str, Any]]:
        """Get a stored operation for retry."""
        return self.operation_store.get(operation_id)
    
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
            detected_error = self._detect_error(e, context, perf_metrics)
            
            # Try automatic retry if enabled and Gemini is available
            if self.enable_automatic_retry and self.intelligent_retry:
                try:
                    self._attempt_automatic_retry(e, context, detected_error)
                except Exception as retry_error:
                    logging.warning(f"Automatic retry failed: {retry_error}")
            
            raise
        finally:
            # Check for timeout
            if self.enable_timeout_detection:
                execution_time = (datetime.now() - start_time).total_seconds()
                if execution_time > self.timeout_threshold:
                    self._detect_timeout(execution_time, context)
    
    def _detect_error(self, exception: Exception, context: ErrorContext, perf_metrics: Optional[Any] = None):
        """Detect and process an error."""
        # Use Gemini for enhanced error analysis if available
        if self.gemini_analyzer and self.gemini_analyzer.is_available():
            try:
                gemini_analysis = self.gemini_analyzer.analyze_error(exception, context)
                
                # Use Gemini's error classification
                error_type = ErrorType(gemini_analysis.get("error_type", "RUNTIME_EXCEPTION"))
                severity = ErrorSeverity(gemini_analysis.get("severity", "MEDIUM"))
                suggestions = gemini_analysis.get("suggestions", [])
                
                # Store Gemini analysis for potential retry
                context.gemini_analysis = gemini_analysis
                
            except Exception as e:
                logging.warning(f"Gemini analysis failed, using fallback: {e}")
                # Fall back to basic classification
                error_type = classify_error(exception, context)
                severity = determine_severity(error_type, context)
                suggestions = self._generate_suggestions(error_type, exception, context)
        else:
            # Use basic error classification
            error_type = classify_error(exception, context)
            severity = determine_severity(error_type, context)
            suggestions = self._generate_suggestions(error_type, exception, context)
        
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
            suggestions=suggestions
        )
        
        # Store error
        self.error_history.append(detected_error)
        
        # Notify handlers
        self._notify_handlers(detected_error)
        
        return detected_error
    
    def _attempt_automatic_retry(self, exception: Exception, context: ErrorContext, 
                                detected_error: DetectedError):
        """Attempt automatic retry using Gemini-enhanced context with real-time remediation."""
        if not self.intelligent_retry or not self.enable_real_time_remediation:
            return
        
        try:
            # Check if error is retryable based on Gemini analysis
            if hasattr(context, 'gemini_analysis'):
                is_retryable = context.gemini_analysis.get("is_retryable", True)
                if not is_retryable:
                    logging.info("Error marked as non-retryable by Gemini")
                    return
            
            # Generate remediation strategy with prompt injection capabilities
            if self.gemini_analyzer and self.gemini_analyzer.is_available():
                # Get operation details for context-aware remediation
                operation_id = f"{context.framework}_{context.component}_{context.method}"
                stored_operation = self.get_stored_operation(operation_id)
                
                if not stored_operation:
                    logging.warning(f"No stored operation found for {operation_id}")
                    return
                
                # Generate specific remediation with prompt injection
                remediation = self.gemini_analyzer.generate_prompt_injection_remediation(
                    exception, context, stored_operation, detected_error
                )
                
                # Store remediation for potential use
                context.remediation_strategy = remediation
                
                logging.info(f"Generated prompt injection remediation with confidence: {remediation.get('confidence', 0)}")
                
                # Check if we should attempt retry based on confidence
                confidence = remediation.get('confidence', 0)
                if confidence < 0.7:  # Only retry if confidence is high enough
                    logging.info(f"Confidence too low for automatic retry: {confidence}")
                    return
                
                # Attempt real-time remediation with prompt injection
                return self._execute_real_time_remediation(exception, context, remediation, stored_operation)
                
        except Exception as e:
            logging.error(f"Failed to attempt automatic retry: {e}")
    
    def _execute_real_time_remediation(self, exception: Exception, context: ErrorContext, 
                                     remediation: Dict[str, Any], stored_operation: Dict[str, Any]) -> Any:
        """Execute real-time remediation with prompt injection."""
        try:
            operation_type = stored_operation.get('operation_type', 'unknown')
            original_prompt = stored_operation.get('original_prompt')
            
            logging.info(f"🔄 REAL-TIME REMEDIATION: Executing {operation_type} with prompt injection")
            
            # Generate enhanced prompt with specific error context
            enhanced_prompt = self._generate_enhanced_prompt(
                original_prompt, exception, context, remediation, operation_type
            )
            
            # Apply remediation to operation arguments
            enhanced_args, enhanced_kwargs = self._apply_prompt_injection(
                stored_operation['args'], stored_operation['kwargs'], 
                enhanced_prompt, remediation
            )
            
            # Execute the operation with enhanced context
            operation = stored_operation['operation']
            
            logging.info(f"🚀 EXECUTING REMEDIATED OPERATION: {operation_type}")
            logging.info(f"📝 ORIGINAL PROMPT: {original_prompt[:100]}...")
            logging.info(f"✨ ENHANCED PROMPT: {enhanced_prompt[:100]}...")
            
            # Use intelligent retry for actual execution
            result = self.intelligent_retry.retry_with_enhanced_context(
                operation, *enhanced_args, error_context=context, **enhanced_kwargs
            )
            
            # Log successful remediation
            logging.info(f"✅ REMEDIATION SUCCESS: {operation_type} completed successfully")
            
            # Cache successful remediation
            remediation_key = f"{operation_type}_{str(exception)[:50]}"
            self.remediation_cache[remediation_key] = {
                'remediation': remediation,
                'enhanced_prompt': enhanced_prompt,
                'success': True,
                'timestamp': datetime.now()
            }
            
            return result
            
        except Exception as e:
            logging.error(f"❌ REMEDIATION FAILED: {e}")
            # Cache failed remediation
            remediation_key = f"{operation_type}_{str(exception)[:50]}"
            self.remediation_cache[remediation_key] = {
                'remediation': remediation,
                'enhanced_prompt': enhanced_prompt if 'enhanced_prompt' in locals() else None,
                'success': False,
                'error': str(e),
                'timestamp': datetime.now()
            }
            raise
    
    def _generate_enhanced_prompt(self, original_prompt: Optional[str], exception: Exception, 
                                context: ErrorContext, remediation: Dict[str, Any], 
                                operation_type: str) -> str:
        """Generate enhanced prompt with specific error context and remediation hints."""
        if not original_prompt:
            original_prompt = "Please complete the requested task."
        
        error_context = f"""
ERROR CONTEXT:
- Error Type: {type(exception).__name__}
- Error Message: {str(exception)}
- Operation: {operation_type}
- Component: {context.component}
- Method: {context.method}
- Timestamp: {context.timestamp}
"""
        
        # Get specific remediation hints from Gemini analysis
        remediation_hints = remediation.get('prompt_injection_hints', [])
        if remediation_hints:
            hints_text = "\n".join([f"- {hint}" for hint in remediation_hints])
            remediation_context = f"""
REMEDIATION GUIDANCE:
{hints_text}
"""
        else:
            remediation_context = ""
        
        # Generate operation-specific guidance
        operation_guidance = self._get_operation_specific_guidance(operation_type, exception)
        
        enhanced_prompt = f"""You are an AI agent that needs to complete a task. A previous attempt failed, but you have been provided with specific guidance to succeed.

{error_context}
{remediation_context}
{operation_guidance}

ORIGINAL TASK:
{original_prompt}

IMPORTANT: 
1. Learn from the error context above
2. Apply the remediation guidance
3. Be more careful and specific in your approach
4. If you encounter similar issues, try alternative approaches
5. Provide detailed reasoning for your actions

Please complete the task now with this enhanced context:"""
        
        return enhanced_prompt
    
    def _get_operation_specific_guidance(self, operation_type: str, exception: Exception) -> str:
        """Get specific guidance based on operation type and error."""
        error_msg = str(exception).lower()
        
        if operation_type == 'llm_call':
            if 'timeout' in error_msg:
                return "\nLLM GUIDANCE: The previous call timed out. Try to be more concise and specific in your request."
            elif 'rate limit' in error_msg:
                return "\nLLM GUIDANCE: Rate limit was exceeded. The system will handle retry timing automatically."
            elif 'validation' in error_msg or 'format' in error_msg:
                return "\nLLM GUIDANCE: There was a format issue. Ensure your response follows the expected structure."
            else:
                return "\nLLM GUIDANCE: Focus on providing a clear, well-structured response."
        
        elif operation_type == 'tool_call':
            if 'timeout' in error_msg:
                return "\nTOOL GUIDANCE: The tool call timed out. Try with simpler parameters or break down the request."
            elif 'validation' in error_msg:
                return "\nTOOL GUIDANCE: Parameter validation failed. Check input format and required parameters."
            else:
                return "\nTOOL GUIDANCE: Ensure you're using the tool correctly with proper parameters."
        
        elif operation_type == 'agent_execution':
            if 'loop' in error_msg or 'recursion' in error_msg:
                return "\nAGENT GUIDANCE: Avoid infinite loops. Be decisive and move toward task completion."
            elif 'planning' in error_msg or 'reasoning' in error_msg:
                return "\nAGENT GUIDANCE: Improve your reasoning process. Break down the problem step by step."
            else:
                return "\nAGENT GUIDANCE: Focus on systematic problem-solving and clear decision-making."
        
        else:
            return "\nGENERAL GUIDANCE: Learn from the error and try a different approach."
    
    def _apply_prompt_injection(self, args: tuple, kwargs: dict, enhanced_prompt: str, 
                              remediation: Dict[str, Any]) -> tuple:
        """Apply prompt injection and other remediation parameters to operation arguments."""
        enhanced_args = list(args)
        enhanced_kwargs = kwargs.copy()
        
        # Apply prompt injection
        prompt_keys = ['prompt', 'input', 'query', 'text', 'message', 'instruction', 'content']
        
        # First try to replace in kwargs
        prompt_injected = False
        for key in prompt_keys:
            if key in enhanced_kwargs and isinstance(enhanced_kwargs[key], str):
                enhanced_kwargs[key] = enhanced_prompt
                prompt_injected = True
                logging.info(f"💉 PROMPT INJECTION: Applied to kwargs['{key}']")
                break
        
        # If not found in kwargs, try positional args
        if not prompt_injected:
            for i, arg in enumerate(enhanced_args):
                if isinstance(arg, str) and len(arg) > 10:  # Likely a prompt
                    enhanced_args[i] = enhanced_prompt
                    prompt_injected = True
                    logging.info(f"💉 PROMPT INJECTION: Applied to args[{i}]")
                    break
                elif isinstance(arg, dict):
                    for key in prompt_keys:
                        if key in arg and isinstance(arg[key], str):
                            arg[key] = enhanced_prompt
                            prompt_injected = True
                            logging.info(f"💉 PROMPT INJECTION: Applied to args[{i}]['{key}']")
                            break
                    if prompt_injected:
                        break
        
        # If still not injected, add as new parameter
        if not prompt_injected:
            enhanced_kwargs['enhanced_prompt'] = enhanced_prompt
            logging.info("💉 PROMPT INJECTION: Added as 'enhanced_prompt' parameter")
        
        # Apply other remediation parameters
        remediation_params = remediation.get('parameter_modifications', {})
        for param, value in remediation_params.items():
            enhanced_kwargs[param] = value
            logging.info(f"🔧 PARAMETER MODIFICATION: {param} = {value}")
        
        return tuple(enhanced_args), enhanced_kwargs
    
    def _execute_enhanced_retry(self, exception: Exception, context: ErrorContext, 
                               remediation: Dict[str, Any]):
        """Execute enhanced retry with Gemini-generated context."""
        try:
            # Extract retry strategy
            retry_strategy = remediation.get('retry_strategy', {})
            max_retries = retry_strategy.get('max_retries', self.max_retries)
            backoff_delay = retry_strategy.get('backoff_delay', self.retry_delay)
            
            # Look for stored operation to retry
            operation_id = f"{context.framework}_{context.component}_{context.method}"
            stored_operation = self.get_stored_operation(operation_id)
            
            if stored_operation and self.intelligent_retry:
                logging.info(f"Attempting enhanced retry with Gemini context")
                
                # Apply enhanced parameters from remediation
                enhanced_kwargs = stored_operation['kwargs'].copy()
                modified_params = remediation.get('modified_parameters', {})
                enhanced_kwargs.update(modified_params)
                
                # Apply enhanced prompt if available
                if remediation.get('enhanced_prompt'):
                    enhanced_kwargs = self._apply_enhanced_prompt(enhanced_kwargs, remediation['enhanced_prompt'])
                
                # Execute retry with enhanced context
                try:
                    result = stored_operation['operation'](*stored_operation['args'], **enhanced_kwargs)
                    
                    # Log successful retry
                    logging.info(f"Enhanced retry successful with Gemini context")
                    
                    # Store retry success in context
                    context.retry_attempts = context.retry_attempts or []
                    context.retry_attempts.append({
                        'timestamp': datetime.now(),
                        'enhanced_context': {
                            'enhanced_prompt': remediation.get('enhanced_prompt'),
                            'modified_parameters': modified_params,
                            'retry_attempt': 1
                        },
                        'remediation': remediation,
                        'success': True,
                        'result': str(result)[:200] if result else None
                    })
                    
                    return result
                    
                except Exception as retry_error:
                    logging.warning(f"Enhanced retry failed: {retry_error}")
                    
                    # Store retry failure in context
                    context.retry_attempts = context.retry_attempts or []
                    context.retry_attempts.append({
                        'timestamp': datetime.now(),
                        'enhanced_context': {
                            'enhanced_prompt': remediation.get('enhanced_prompt'),
                            'modified_parameters': modified_params,
                            'retry_attempt': 1
                        },
                        'remediation': remediation,
                        'success': False,
                        'error': str(retry_error)
                    })
                    
                    # Fall back to original operation
                    return stored_operation['operation'](*stored_operation['args'], **stored_operation['kwargs'])
            
            else:
                logging.info("No stored operation found for retry or intelligent retry not available")
                
        except Exception as e:
            logging.error(f"Failed to execute enhanced retry: {e}")
    
    def _apply_enhanced_prompt(self, kwargs: Dict[str, Any], enhanced_prompt: str) -> Dict[str, Any]:
        """Apply enhanced prompt to operation parameters."""
        enhanced_kwargs = kwargs.copy()
        
        # Common prompt parameter names
        prompt_params = ['prompt', 'input', 'query', 'text', 'message', 'instruction', 'template']
        
        for param in prompt_params:
            if param in enhanced_kwargs:
                # Enhance existing prompt
                original_prompt = enhanced_kwargs[param]
                enhanced_kwargs[param] = f"{enhanced_prompt}\n\nOriginal: {original_prompt}"
                break
        else:
            # No existing prompt parameter, add enhanced prompt
            enhanced_kwargs['enhanced_context'] = enhanced_prompt
        
        return enhanced_kwargs
    
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
        
        # Gemini analysis stats
        gemini_analyzed = len([e for e in recent_errors if hasattr(e.context, 'gemini_analysis')])
        retry_attempts = len([e for e in recent_errors if hasattr(e.context, 'retry_attempts')])
        
        return {
            "total_errors": len(recent_errors),
            "window_minutes": window_minutes,
            "severity_distribution": severity_counts,
            "type_distribution": type_counts,
            "component_distribution": component_counts,
            "gemini_analyzed": gemini_analyzed,
            "retry_attempts": retry_attempts,
            "most_recent_error": recent_errors[-1].to_dict() if recent_errors else None
        }
    
    def clear_history(self):
        """Clear error history."""
        self.error_history.clear()
        if self.performance_monitor:
            self.performance_monitor.clear_history()
        if self.intelligent_retry:
            self.intelligent_retry.clear_history()
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health status."""
        health_status = {
            "timestamp": datetime.now().isoformat(),
            "is_monitoring": self.is_monitoring,
            "total_errors": len(self.error_history),
            "recent_errors": len([e for e in self.error_history if (datetime.now() - e.context.timestamp).seconds < 300])  # Last 5 minutes
        }
        
        # Gemini status
        if self.gemini_analyzer:
            health_status["gemini_status"] = self.gemini_analyzer.get_status()
        
        # Retry stats
        if self.intelligent_retry:
            health_status["retry_stats"] = self.intelligent_retry.get_retry_stats()
        
        if self.resource_monitor:
            health_status["system_health"] = self.resource_monitor.check_system_health()
        
        if self.performance_monitor:
            health_status["performance_summary"] = self.performance_monitor.get_performance_summary(window_minutes=60)
        
        return health_status
    
    def enable_gemini_analysis(self, project_id: Optional[str] = None, location: str = "us-central1"):
        """Enable Gemini-powered error analysis."""
        try:
            self.gemini_analyzer = GeminiAnalyzer(project_id, location)
            if self.gemini_analyzer.is_available():
                self.intelligent_retry = IntelligentRetry(self.gemini_analyzer)
                logging.info("Gemini-powered error analysis enabled")
            else:
                logging.warning("Gemini not available - check project ID and authentication")
        except Exception as e:
            logging.error(f"Failed to enable Gemini analysis: {e}")
    
    def get_gemini_status(self) -> Dict[str, Any]:
        """Get Gemini integration status."""
        if self.gemini_analyzer:
            return self.gemini_analyzer.get_status()
        return {"enabled": False, "reason": "Gemini not initialized"}


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
            detected_error = self._detect_error(e, context, perf_metrics)
            
            # Try automatic retry if enabled and Gemini is available
            if self.enable_automatic_retry and self.intelligent_retry:
                try:
                    self._attempt_automatic_retry(e, context, detected_error)
                except Exception as retry_error:
                    logging.warning(f"Automatic retry failed: {retry_error}")
            
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
