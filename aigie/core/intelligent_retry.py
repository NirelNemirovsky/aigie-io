"""
Intelligent retry system for Aigie using Gemini-enhanced prompts and context.
"""

import time
import logging
from typing import Dict, Any, Optional, Callable, List, Union
from datetime import datetime
from functools import wraps

from .gemini_analyzer import GeminiAnalyzer
from .error_types import ErrorContext, DetectedError


class IntelligentRetry:
    """Intelligent retry system that uses Gemini to enhance retry attempts with real-time remediation."""
    
    def __init__(self, gemini_analyzer: GeminiAnalyzer, max_retries: int = 3, 
                 base_delay: float = 1.0, max_delay: float = 60.0):
        self.gemini_analyzer = gemini_analyzer
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.retry_history = []
        
        # Enhanced retry capabilities
        self.enable_prompt_injection = True
        self.enable_context_learning = True
        self.operation_memory: Dict[str, Dict[str, Any]] = {}
        self.success_patterns: Dict[str, List[Dict[str, Any]]] = {}
        
    def retry_with_gemini_context(self, operation: Callable, *args, 
                                 error_context: Optional[ErrorContext] = None,
                                 **kwargs) -> Any:
        """Retry an operation with Gemini-enhanced context and real-time remediation."""
        operation_signature = self._get_operation_signature(operation, error_context)
        last_error = None
        retry_attempts = []
        original_kwargs = kwargs.copy()  # Preserve original parameters
        
        # Check for previous successful patterns
        if operation_signature in self.success_patterns:
            logging.info(f"🧠 PATTERN MATCH: Found {len(self.success_patterns[operation_signature])} successful patterns for {operation_signature}")
        
        for attempt in range(self.max_retries + 1):  # +1 for initial attempt
            try:
                start_time = time.time()
                
                if attempt == 0:
                    # Initial attempt
                    result = operation(*args, **kwargs)
                    execution_time = time.time() - start_time
                    
                    # Store successful pattern
                    self._store_success_pattern(operation_signature, args, kwargs, result, execution_time)
                    
                    # Log successful execution
                    self._log_retry_attempt(attempt, True, None, execution_time, error_context)
                    return result
                    
                else:
                    # Retry attempt with enhanced context and prompt injection
                    enhanced_result = self._retry_with_prompt_injection(
                        operation, *args, attempt=attempt, 
                        last_error=last_error, error_context=error_context, 
                        original_kwargs=original_kwargs, retry_attempts=retry_attempts, **kwargs
                    )
                    
                    execution_time = time.time() - start_time
                    
                    # Store successful retry pattern
                    self._store_success_pattern(operation_signature, args, kwargs, enhanced_result, execution_time, retry_attempt=attempt)
                    
                    self._log_retry_attempt(attempt, True, None, execution_time, error_context)
                    return enhanced_result
                    
            except Exception as e:
                execution_time = time.time() - start_time
                last_error = e
                
                # Log failed attempt
                self._log_retry_attempt(attempt, False, e, execution_time, error_context)
                
                if attempt < self.max_retries:
                    # Generate enhanced context for retry with prompt injection
                    enhanced_context = self._generate_prompt_injection_context(
                        e, error_context, attempt, operation_signature, retry_attempts
                    )
                    retry_attempts.append({
                        'attempt': attempt,
                        'error': e,
                        'enhanced_context': enhanced_context,
                        'timestamp': datetime.now()
                    })
                    
                    # Wait before retry
                    delay = self._calculate_delay(attempt)
                    logging.info(f"🔄 RETRY ATTEMPT {attempt + 1} in {delay:.2f}s with prompt injection...")
                    time.sleep(delay)
                    
                else:
                    # Max retries exceeded
                    logging.error(f"❌ MAX RETRIES EXCEEDED: {self.max_retries} attempts failed. Final error: {e}")
                    
                    # Store failure pattern for learning
                    self._store_failure_pattern(operation_signature, args, kwargs, last_error, retry_attempts)
                    
                    raise e
        
        # This should never be reached
        raise RuntimeError("Unexpected retry loop termination")
    
    def retry_with_enhanced_context(self, operation: Callable, *args, 
                                  error_context: Optional[ErrorContext] = None,
                                  **kwargs) -> Any:
        """Enhanced retry with context learning and prompt injection (alias for backward compatibility)."""
        return self.retry_with_gemini_context(operation, *args, error_context=error_context, **kwargs)
    
    def _retry_with_enhanced_context(self, operation: Callable, *args, 
                                    attempt: int, last_error: Exception,
                                    error_context: Optional[ErrorContext], 
                                    original_kwargs: Dict[str, Any] = None, **kwargs) -> Any:
        """Execute retry with enhanced context from Gemini."""
        try:
            # Get enhanced context from Gemini
            enhanced_context = self._generate_enhanced_context(last_error, error_context, attempt)
            
            # Use original kwargs if available, otherwise fall back to current kwargs
            base_kwargs = original_kwargs if original_kwargs is not None else kwargs
            
            # Apply actual remediation strategies
            modified_kwargs = self._apply_remediation_strategies(base_kwargs, enhanced_context, last_error)
            
            # Log what we're about to execute
            logging.info(f"🔄 RETRY ATTEMPT {attempt}: Executing with remediated parameters: {modified_kwargs}")
            
            # Execute with enhanced context
            if enhanced_context.get('enhanced_prompt'):
                # For operations that accept prompts, inject the enhanced prompt
                enhanced_kwargs = self._inject_enhanced_prompt(modified_kwargs, enhanced_context['enhanced_prompt'])
                # Check if function accepts **kwargs (any keyword arguments)
                import inspect
                sig = inspect.signature(operation)
                logging.info(f"🔍 FUNCTION SIGNATURE: {sig.parameters}")
                
                # If function has **kwargs, accept all parameters; otherwise filter by signature
                has_kwargs = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in sig.parameters.values())
                if has_kwargs:
                    valid_kwargs = enhanced_kwargs
                    logging.info(f"🔧 EXECUTING: Function accepts **kwargs, using all remediated kwargs: {valid_kwargs}")
                else:
                    valid_kwargs = {k: v for k, v in enhanced_kwargs.items() if k in sig.parameters}
                    logging.info(f"🔧 EXECUTING: Function has strict signature, filtered kwargs: {valid_kwargs}")
                
                return operation(*args, **valid_kwargs)
            else:
                # Execute with modified parameters
                # Check if function accepts **kwargs (any keyword arguments)
                import inspect
                sig = inspect.signature(operation)
                logging.info(f"🔍 FUNCTION SIGNATURE: {sig.parameters}")
                
                # If function has **kwargs, accept all parameters; otherwise filter by signature
                has_kwargs = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in sig.parameters.values())
                if has_kwargs:
                    valid_kwargs = modified_kwargs
                    logging.info(f"🔧 EXECUTING: Function accepts **kwargs, using all remediated kwargs: {valid_kwargs}")
                else:
                    valid_kwargs = {k: v for k, v in modified_kwargs.items() if k in sig.parameters}
                    logging.info(f"🔧 EXECUTING: Function has strict signature, filtered kwargs: {valid_kwargs}")
                
                return operation(*args, **valid_kwargs)
                
        except Exception as e:
            logging.warning(f"Enhanced retry attempt {attempt} failed: {e}")
            # Fall back to original operation
            # Filter out any unexpected arguments that the function doesn't accept
            import inspect
            sig = inspect.signature(operation)
            valid_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
            return operation(*args, **valid_kwargs)
    
    def _generate_enhanced_context(self, error: Exception, 
                                  error_context: Optional[ErrorContext], 
                                  attempt: int) -> Dict[str, Any]:
        """Generate enhanced context using Gemini."""
        if not self.gemini_analyzer.is_available():
            return self._fallback_enhanced_context(error, attempt)
        
        try:
            # Create a basic context if none provided
            if not error_context:
                error_context = ErrorContext(
                    timestamp=datetime.now(),
                    framework="unknown",
                    component="unknown",
                    method="unknown"
                )
            
            # Analyze the error
            error_analysis = self.gemini_analyzer.analyze_error(error, error_context)
            
            # Generate remediation strategy
            remediation = self.gemini_analyzer.generate_remediation_strategy(
                error, error_context, error_analysis
            )
            
            # Enhance with retry attempt context
            enhanced_context = {
                **remediation,
                'retry_attempt': attempt,
                'original_error': str(error),
                'error_analysis': error_analysis,
                'timestamp': datetime.now().isoformat()
            }
            
            return enhanced_context
            
        except Exception as e:
            logging.error(f"Failed to generate enhanced context: {e}")
            return self._fallback_enhanced_context(error, attempt)
    
    def _retry_with_prompt_injection(self, operation: Callable, *args, 
                                   attempt: int, last_error: Exception,
                                   error_context: Optional[ErrorContext], 
                                   original_kwargs: Dict[str, Any],
                                   retry_attempts: List[Dict[str, Any]], **kwargs) -> Any:
        """Execute retry with advanced prompt injection and context learning."""
        try:
            operation_signature = self._get_operation_signature(operation, error_context)
            
            # Generate prompt injection context using Gemini
            injection_context = self._generate_prompt_injection_context(
                last_error, error_context, attempt, operation_signature, retry_attempts
            )
            
            # Apply learned patterns if available
            enhanced_kwargs = self._apply_learned_patterns(
                original_kwargs, operation_signature, last_error, injection_context
            )
            
            # Apply prompt injection with specific error guidance
            final_kwargs = self._apply_advanced_prompt_injection(
                enhanced_kwargs, injection_context, last_error, attempt
            )
            
            logging.info(f"🚀 PROMPT INJECTION RETRY {attempt}: Executing with enhanced context")
            logging.info(f"📊 APPLIED PATTERNS: {len(self.success_patterns.get(operation_signature, []))} success patterns available")
            
            # Execute with function signature validation
            result = self._execute_with_signature_validation(operation, args, final_kwargs)
            
            logging.info(f"✅ PROMPT INJECTION SUCCESS: Retry {attempt} completed successfully")
            return result
            
        except Exception as e:
            logging.warning(f"❌ PROMPT INJECTION FAILED: Retry {attempt} failed: {e}")
            # Fall back to basic retry
            return self._retry_with_enhanced_context(
                operation, *args, attempt=attempt, last_error=last_error, 
                error_context=error_context, original_kwargs=original_kwargs, **kwargs
            )
    
    def _get_operation_signature(self, operation: Callable, error_context: Optional[ErrorContext]) -> str:
        """Generate a unique signature for the operation for pattern matching."""
        op_name = getattr(operation, '__name__', str(operation))
        if error_context:
            return f"{error_context.framework}_{error_context.component}_{error_context.method}_{op_name}"
        else:
            return f"unknown_unknown_unknown_{op_name}"
    
    def _generate_prompt_injection_context(self, error: Exception, error_context: Optional[ErrorContext], 
                                         attempt: int, operation_signature: str, 
                                         retry_attempts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate advanced prompt injection context using Gemini and learned patterns."""
        if not self.gemini_analyzer.is_available():
            return self._fallback_prompt_injection_context(error, attempt, operation_signature)
        
        try:
            # Create mock stored operation for Gemini analysis
            stored_operation = {
                'operation_type': self._infer_operation_type(operation_signature),
                'original_prompt': self._extract_original_prompt_from_attempts(retry_attempts),
                'execution_stack': retry_attempts
            }
            
            # Create mock detected error
            from .error_types import DetectedError, ErrorSeverity, ErrorType
            detected_error = DetectedError(
                error_type=ErrorType.RUNTIME_EXCEPTION,
                severity=ErrorSeverity.MEDIUM,
                message=str(error),
                exception=error,
                context=error_context,
                suggestions=[]
            )
            
            # Get Gemini's prompt injection remediation
            remediation = self.gemini_analyzer.generate_prompt_injection_remediation(
                error, error_context, stored_operation, detected_error
            )
            
            # Enhance with learned patterns
            learned_context = self._get_learned_context(operation_signature, error)
            
            return {
                **remediation,
                'learned_patterns': learned_context,
                'retry_attempt': attempt,
                'operation_signature': operation_signature,
                'previous_attempts': len(retry_attempts),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logging.error(f"Failed to generate prompt injection context: {e}")
            return self._fallback_prompt_injection_context(error, attempt, operation_signature)
    
    def _apply_learned_patterns(self, kwargs: Dict[str, Any], operation_signature: str, 
                               error: Exception, injection_context: Dict[str, Any]) -> Dict[str, Any]:
        """Apply learned successful patterns to the operation parameters."""
        enhanced_kwargs = kwargs.copy()
        
        if operation_signature in self.success_patterns:
            patterns = self.success_patterns[operation_signature]
            
            # Find patterns that might apply to this error type
            applicable_patterns = []
            error_type = type(error).__name__
            
            for pattern in patterns:
                # Check if this pattern has dealt with similar errors
                if 'error_context' in pattern:
                    if error_type in pattern['error_context'].get('handled_errors', []):
                        applicable_patterns.append(pattern)
                elif pattern.get('retry_attempt', 0) > 0:  # This was a successful retry
                    applicable_patterns.append(pattern)
            
            if applicable_patterns:
                # Use the most recent successful pattern
                best_pattern = max(applicable_patterns, key=lambda p: p.get('timestamp', datetime.min))
                
                logging.info(f"🧠 APPLYING LEARNED PATTERN: Using successful pattern from {best_pattern.get('timestamp', 'unknown time')}")
                
                # Apply successful parameters
                if 'successful_params' in best_pattern:
                    for param, value in best_pattern['successful_params'].items():
                        if param not in enhanced_kwargs:  # Don't override existing params
                            enhanced_kwargs[param] = value
                            logging.info(f"📚 LEARNED PARAM: Applied {param} = {value}")
        
        return enhanced_kwargs
    
    def _apply_advanced_prompt_injection(self, kwargs: Dict[str, Any], injection_context: Dict[str, Any], 
                                       error: Exception, attempt: int) -> Dict[str, Any]:
        """Apply advanced prompt injection with specific error context."""
        final_kwargs = kwargs.copy()
        
        # Get prompt injection hints from Gemini
        hints = injection_context.get('prompt_injection_hints', [])
        if not hints:
            hints = [f"Previous attempt {attempt-1} failed with {type(error).__name__}: {str(error)}"]
        
        # Create comprehensive enhanced prompt
        error_guidance = f"""
🔄 RETRY CONTEXT (Attempt {attempt}):
Previous Error: {type(error).__name__}: {str(error)}

🎯 SPECIFIC GUIDANCE:
{chr(10).join([f"• {hint}" for hint in hints])}

📋 OPERATION GUIDANCE:
{injection_context.get('operation_specific_guidance', {}).get('primary_approach', 'Be more careful and systematic')}

⚠️ ERROR PREVENTION:
{chr(10).join([f"• {step}" for step in injection_context.get('operation_specific_guidance', {}).get('error_prevention', ['Double-check your work'])])}

🔧 ALTERNATIVE APPROACHES:
{chr(10).join([f"• {alt}" for alt in injection_context.get('operation_specific_guidance', {}).get('fallback_approaches', ['Try a different method if the first fails'])])}

IMPORTANT: Learn from the above context and execute the task successfully this time.
"""
        
        # Apply prompt injection
        prompt_keys = ['prompt', 'input', 'query', 'text', 'message', 'instruction', 'content']
        
        prompt_injected = False
        for key in prompt_keys:
            if key in final_kwargs and isinstance(final_kwargs[key], str):
                original_prompt = final_kwargs[key]
                final_kwargs[key] = f"{error_guidance}\n\nORIGINAL TASK:\n{original_prompt}"
                prompt_injected = True
                logging.info(f"💉 ADVANCED PROMPT INJECTION: Applied to '{key}' parameter")
                break
        
        if not prompt_injected:
            # Try positional arguments or add as new parameter
            final_kwargs['enhanced_context'] = error_guidance
            logging.info("💉 ADVANCED PROMPT INJECTION: Added as 'enhanced_context' parameter")
        
        # Apply parameter modifications from Gemini
        param_mods = injection_context.get('parameter_modifications', {})
        for param, value in param_mods.items():
            final_kwargs[param] = value
            logging.info(f"🔧 GEMINI PARAM MOD: {param} = {value}")
        
        return final_kwargs
    
    def _execute_with_signature_validation(self, operation: Callable, args: tuple, kwargs: Dict[str, Any]) -> Any:
        """Execute operation with signature validation to avoid parameter errors."""
        import inspect
        
        try:
            sig = inspect.signature(operation)
            
            # Filter kwargs to only include parameters the function accepts
            has_var_keyword = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in sig.parameters.values())
            
            if has_var_keyword:
                # Function accepts **kwargs, use all parameters
                valid_kwargs = kwargs
            else:
                # Function has strict signature, filter parameters
                valid_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
                
                # Log filtered parameters
                filtered = set(kwargs.keys()) - set(valid_kwargs.keys())
                if filtered:
                    logging.info(f"🔍 SIGNATURE FILTER: Removed parameters: {filtered}")
            
            return operation(*args, **valid_kwargs)
            
        except Exception as e:
            logging.error(f"Signature validation failed: {e}")
            # Fall back to original call
            return operation(*args, **kwargs)
    
    def _store_success_pattern(self, operation_signature: str, args: tuple, kwargs: Dict[str, Any], 
                             result: Any, execution_time: float, retry_attempt: int = 0):
        """Store successful operation pattern for future learning."""
        if operation_signature not in self.success_patterns:
            self.success_patterns[operation_signature] = []
        
        pattern = {
            'timestamp': datetime.now(),
            'args': args,
            'successful_params': kwargs.copy(),
            'result_summary': str(result)[:100] if result else None,
            'execution_time': execution_time,
            'retry_attempt': retry_attempt,
            'success': True
        }
        
        self.success_patterns[operation_signature].append(pattern)
        
        # Keep only recent patterns (last 10)
        if len(self.success_patterns[operation_signature]) > 10:
            self.success_patterns[operation_signature] = self.success_patterns[operation_signature][-10:]
        
        logging.info(f"📚 PATTERN STORED: Success pattern for {operation_signature} (attempt {retry_attempt})")
    
    def _store_failure_pattern(self, operation_signature: str, args: tuple, kwargs: Dict[str, Any], 
                             error: Exception, retry_attempts: List[Dict[str, Any]]):
        """Store failure pattern for learning what doesn't work."""
        if operation_signature not in self.operation_memory:
            self.operation_memory[operation_signature] = {'failures': [], 'successes': []}
        
        failure_pattern = {
            'timestamp': datetime.now(),
            'args': args,
            'failed_params': kwargs.copy(),
            'error': str(error),
            'error_type': type(error).__name__,
            'retry_attempts': len(retry_attempts),
            'attempts_details': retry_attempts
        }
        
        self.operation_memory[operation_signature]['failures'].append(failure_pattern)
        
        # Keep only recent failures (last 5)
        if len(self.operation_memory[operation_signature]['failures']) > 5:
            self.operation_memory[operation_signature]['failures'] = self.operation_memory[operation_signature]['failures'][-5:]
        
        logging.info(f"📝 FAILURE LOGGED: Pattern for {operation_signature} - {type(error).__name__}")
    
    def _get_learned_context(self, operation_signature: str, error: Exception) -> Dict[str, Any]:
        """Get learned context from previous operations."""
        if operation_signature not in self.operation_memory:
            return {}
        
        memory = self.operation_memory[operation_signature]
        error_type = type(error).__name__
        
        # Find similar errors
        similar_failures = [f for f in memory.get('failures', []) if f['error_type'] == error_type]
        
        return {
            'similar_failures': len(similar_failures),
            'total_failures': len(memory.get('failures', [])),
            'success_patterns_available': len(self.success_patterns.get(operation_signature, [])),
            'last_similar_error': similar_failures[-1] if similar_failures else None
        }
    
    def _infer_operation_type(self, operation_signature: str) -> str:
        """Infer operation type from signature."""
        sig_lower = operation_signature.lower()
        if 'llm' in sig_lower or 'generate' in sig_lower:
            return 'llm_call'
        elif 'agent' in sig_lower:
            return 'agent_execution'
        elif 'tool' in sig_lower:
            return 'tool_call'
        elif 'chain' in sig_lower:
            return 'chain_execution'
        else:
            return 'unknown'
    
    def _extract_original_prompt_from_attempts(self, retry_attempts: List[Dict[str, Any]]) -> str:
        """Extract original prompt from retry attempts."""
        if not retry_attempts:
            return "N/A"
        
        for attempt in retry_attempts:
            if 'enhanced_context' in attempt and 'original_prompt' in attempt['enhanced_context']:
                return attempt['enhanced_context']['original_prompt']
        
        return "N/A"
    
    def _fallback_prompt_injection_context(self, error: Exception, attempt: int, operation_signature: str) -> Dict[str, Any]:
        """Fallback prompt injection context when Gemini is not available."""
        return {
            'prompt_injection_hints': [
                f"Previous attempt {attempt-1} failed with {type(error).__name__}",
                "Try a different approach this time",
                "Be more careful and systematic",
                "Double-check your work before proceeding"
            ],
            'operation_specific_guidance': {
                'primary_approach': 'Retry with more care',
                'fallback_approaches': ['Break down the task', 'Use simpler approach'],
                'error_prevention': ['Validate inputs', 'Check for common issues']
            },
            'parameter_modifications': {},
            'confidence': 0.6,
            'reasoning': f"Fallback guidance for {operation_signature}",
            'retry_attempt': attempt,
            'operation_signature': operation_signature
        }
    
    def _fallback_enhanced_context(self, error: Exception, attempt: int) -> Dict[str, Any]:
        """Fallback enhanced context when Gemini is not available."""
        return {
            'retry_strategy': {
                'approach': 'Retry with exponential backoff',
                'max_retries': self.max_retries,
                'backoff_delay': self.base_delay
            },
            'enhanced_prompt': f"Retry operation (attempt {attempt + 1}). Previous error: {str(error)}",
            'modified_parameters': {},
            'implementation_steps': [
                f"Retry attempt {attempt + 1}",
                "Use exponential backoff",
                "Log retry attempts for debugging"
            ],
            'confidence': 0.5,
            'retry_attempt': attempt,
            'original_error': str(error),
            'timestamp': datetime.now().isoformat()
        }
    
    def _inject_enhanced_prompt(self, kwargs: Dict[str, Any], enhanced_prompt: str) -> Dict[str, Any]:
        """Inject enhanced prompt into operation parameters."""
        enhanced_kwargs = kwargs.copy()
        
        # Common prompt parameter names
        prompt_params = ['prompt', 'input', 'query', 'text', 'message', 'instruction']
        
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
    
    def _apply_remediation_strategies(self, kwargs: Dict[str, Any], 
                                     enhanced_context: Dict[str, Any], 
                                     error: Exception) -> Dict[str, Any]:
        """Apply actual remediation strategies based on Gemini's analysis."""
        modified_kwargs = kwargs.copy()
        
        # Log what remediation strategies we're applying
        if enhanced_context.get('implementation_steps'):
            logging.info(f"🔧 Applying remediation strategies: {enhanced_context['implementation_steps']}")
        
        # Apply timeout fixes
        if "timeout" in str(error).lower() and enhanced_context.get('timeout_fixes'):
            timeout_fixes = enhanced_context['timeout_fixes']
            if 'increase_timeout' in timeout_fixes:
                # Increase timeout parameters
                for param in ['timeout', 'timeout_seconds', 'max_wait']:
                    if param in modified_kwargs:
                        original_timeout = modified_kwargs[param]
                        modified_timeout = original_timeout * 2  # Double the timeout
                        modified_kwargs[param] = modified_timeout
                        logging.info(f"⏱️  TIMEOUT FIX: Increased {param} from {original_timeout} to {modified_timeout}")
                    else:
                        # Add timeout parameter if it doesn't exist
                        modified_kwargs[param] = 10  # Default 10 seconds
                        logging.info(f"⏱️  TIMEOUT FIX: Added {param}={modified_kwargs[param]}s")
        
        # Apply retry strategy fixes
        if enhanced_context.get('retry_strategy'):
            retry_strategy = enhanced_context['retry_strategy']
            if 'approach' in retry_strategy:
                logging.info(f"🔄 RETRY STRATEGY: Using approach: {retry_strategy['approach']}")
        
        # Apply parameter modifications from Gemini
        if enhanced_context.get('parameter_modifications'):
            for param, value in enhanced_context['parameter_modifications'].items():
                if param in modified_kwargs:
                    original_value = modified_kwargs[param]
                    modified_kwargs[param] = value
                    logging.info(f"🔧 PARAMETER FIX: Modified {param}: {original_value} → {value}")
                else:
                    modified_kwargs[param] = value
                    logging.info(f"➕ PARAMETER FIX: Added parameter {param}: {value}")
        
        # Also check for legacy modified_parameters for backward compatibility
        if enhanced_context.get('modified_parameters'):
            for param, value in enhanced_context['modified_parameters'].items():
                if param in modified_kwargs:
                    original_value = modified_kwargs[param]
                    modified_kwargs[param] = value
                    logging.info(f"🔧 PARAMETER FIX (legacy): Modified {param}: {original_value} → {value}")
                else:
                    modified_kwargs[param] = value
                    logging.info(f"➕ PARAMETER FIX (legacy): Added parameter {param}: {value}")
        
        # Apply error handling improvements
        if enhanced_context.get('error_handling'):
            error_handling = enhanced_context['error_handling']
            if 'add_validation' in error_handling:
                # Add input validation
                if 'input_data' in modified_kwargs:
                    # Validate and clean input data
                    input_data = modified_kwargs['input_data']
                    if isinstance(input_data, str):
                        # Clean string input
                        original_input = input_data
                        cleaned_input = input_data.strip()
                        modified_kwargs['input_data'] = cleaned_input
                        if original_input != cleaned_input:
                            logging.info(f"🧹 VALIDATION FIX: Cleaned input: '{original_input}' → '{cleaned_input}'")
                
                # Enable validation flags
                modified_kwargs['validate_input'] = True
                modified_kwargs['clean_input'] = True
                logging.info(f"✅ VALIDATION FIX: Enabled input validation and cleaning")
        
        # Apply circuit breaker pattern for API errors
        if "api" in str(error).lower() or "connection" in str(error).lower():
            if enhanced_context.get('circuit_breaker'):
                modified_kwargs['circuit_breaker_enabled'] = True
                modified_kwargs['circuit_breaker_threshold'] = enhanced_context.get('circuit_breaker_threshold', 3)
                logging.info(f"🔄 CIRCUIT BREAKER FIX: Enabled with threshold {modified_kwargs['circuit_breaker_threshold']}")
            
            # Add retry and connection improvements
            modified_kwargs['retry_on_failure'] = True
            modified_kwargs['connection_pool_size'] = 10
            logging.info(f"🌐 API FIX: Enabled retry_on_failure and connection pooling")
        
        # Apply input sanitization for validation errors
        if "validation" in str(error).lower() or "format" in str(error).lower():
            if enhanced_context.get('input_sanitization'):
                # Sanitize string inputs
                for key, value in modified_kwargs.items():
                    if isinstance(value, str):
                        original_value = value
                        # Remove extra whitespace and normalize
                        sanitized_value = ' '.join(value.split())
                        modified_kwargs[key] = sanitized_value
                        if original_value != sanitized_value:
                            logging.info(f"🧹 SANITIZATION FIX: {key}: '{original_value}' → '{sanitized_value}'")
                
                # Enable sanitization flags
                modified_kwargs['sanitize_input'] = True
                logging.info(f"🧹 SANITIZATION FIX: Enabled input sanitization")
        
        # Apply memory optimization for memory errors
        if "memory" in str(error).lower():
            if enhanced_context.get('memory_optimization'):
                modified_kwargs['batch_size'] = enhanced_context.get('optimized_batch_size', 1)
                modified_kwargs['streaming'] = True
                logging.info(f"💾 MEMORY FIX: Applied optimization: batch_size={modified_kwargs['batch_size']}, streaming=True")
            else:
                # Default memory optimization
                modified_kwargs['batch_size'] = 1
                modified_kwargs['streaming'] = True
                logging.info(f"💾 MEMORY FIX: Applied default optimization: batch_size=1, streaming=True")
        
        # Apply state recovery for state errors
        if "state" in str(error).lower():
            if enhanced_context.get('state_recovery'):
                modified_kwargs['reset_state'] = True
                modified_kwargs['state_validation'] = True
                logging.info(f"🔄 STATE FIX: Applied recovery: reset_state=True, state_validation=True")
            else:
                # Default state recovery
                modified_kwargs['reset_state'] = True
                modified_kwargs['state_validation'] = True
                logging.info(f"🔄 STATE FIX: Applied default recovery: reset_state=True, state_validation=True")
        
        # Apply rate limiting for rate limit errors
        if "rate limit" in str(error).lower():
            if enhanced_context.get('rate_limit_handling'):
                modified_kwargs['rate_limit_delay'] = enhanced_context.get('rate_limit_delay', 5.0)
                modified_kwargs['exponential_backoff'] = True
                logging.info(f"⏱️  RATE LIMIT FIX: Applied handling: delay={modified_kwargs['rate_limit_delay']}s, exponential_backoff=True")
            else:
                # Default rate limit handling
                modified_kwargs['rate_limit_delay'] = 5.0
                modified_kwargs['exponential_backoff'] = True
                logging.info(f"⏱️  RATE LIMIT FIX: Applied default handling: delay=5.0s, exponential_backoff=True")
        
        # Log summary of all applied fixes
        applied_fixes = []
        for key, value in modified_kwargs.items():
            if key not in kwargs:
                applied_fixes.append(f"{key}={value}")
        
        if applied_fixes:
            logging.info(f"🎯 REMEDIATION SUMMARY: Applied fixes: {', '.join(applied_fixes)}")
        
        return modified_kwargs
    
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay for retry attempt using exponential backoff."""
        delay = self.base_delay * (2 ** attempt)
        return min(delay, self.max_delay)
    
    def _log_retry_attempt(self, attempt: int, success: bool, error: Optional[Exception], 
                           execution_time: float, error_context: Optional[ErrorContext]):
        """Log retry attempt details."""
        log_entry = {
            'attempt': attempt,
            'success': success,
            'error': str(error) if error else None,
            'execution_time': execution_time,
            'timestamp': datetime.now(),
            'context': {
                'timestamp': error_context.timestamp.isoformat() if error_context else None,
                'framework': error_context.framework if error_context else None,
                'component': error_context.component if error_context else None,
                'method': error_context.method if error_context else None
            } if error_context else None
        }
        
        self.retry_history.append(log_entry)
        
        if success:
            logging.info(f"Operation {'succeeded' if attempt == 0 else f'retry {attempt} succeeded'} "
                        f"in {execution_time:.3f}s")
        else:
            logging.warning(f"Operation attempt {attempt} failed: {error}")
    
    def get_retry_stats(self) -> Dict[str, Any]:
        """Get statistics about retry attempts."""
        if not self.retry_history:
            return {"total_attempts": 0, "success_rate": 0.0}
        
        total_attempts = len(self.retry_history)
        successful_attempts = len([r for r in self.retry_history if r['success']])
        retry_attempts = len([r for r in self.retry_history if r['attempt'] > 0])
        successful_retries = len([r for r in self.retry_history if r['success'] and r['attempt'] > 0])
        
        # Calculate success rates
        overall_success_rate = successful_attempts / total_attempts if total_attempts > 0 else 0.0
        retry_success_rate = successful_retries / retry_attempts if retry_attempts > 0 else 0.0
        
        # Average execution times
        avg_execution_time = sum(r['execution_time'] for r in self.retry_history) / total_attempts
        
        return {
            "total_attempts": total_attempts,
            "successful_attempts": successful_attempts,
            "retry_attempts": retry_attempts,
            "successful_retries": successful_retries,
            "overall_success_rate": overall_success_rate,
            "retry_success_rate": retry_success_rate,
            "avg_execution_time": avg_execution_time,
            "retry_history": self.retry_history[-10:]  # Last 10 attempts
        }
    
    def clear_history(self):
        """Clear retry history."""
        self.retry_history.clear()


def intelligent_retry(max_retries: int = 3, base_delay: float = 1.0, 
                     gemini_analyzer: Optional[GeminiAnalyzer] = None):
    """Decorator for intelligent retry with Gemini context."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create retry instance if not provided
            if gemini_analyzer is None:
                # Create a basic analyzer without Gemini
                from .gemini_analyzer import GeminiAnalyzer
                analyzer = GeminiAnalyzer()
            else:
                analyzer = gemini_analyzer
            
            retry_system = IntelligentRetry(analyzer, max_retries, base_delay)
            
            # Create basic error context
            error_context = ErrorContext(
                timestamp=datetime.now(),
                framework="decorated_function",
                component=func.__module__,
                method=func.__name__
            )
            
            return retry_system.retry_with_gemini_context(
                func, *args, error_context=error_context, **kwargs
            )
        
        return wrapper
    return decorator


class LangChainRetryWrapper:
    """Wrapper for LangChain operations with intelligent retry."""
    
    def __init__(self, gemini_analyzer: GeminiAnalyzer, max_retries: int = 3):
        self.gemini_analyzer = gemini_analyzer
        self.max_retries = max_retries
        self.retry_system = IntelligentRetry(gemini_analyzer, max_retries)
    
    def wrap_chain(self, chain):
        """Wrap a LangChain chain with intelligent retry."""
        original_invoke = chain.invoke
        
        @wraps(original_invoke)
        def enhanced_invoke(*args, **kwargs):
            # Create error context
            error_context = ErrorContext(
                timestamp=datetime.now(),
                framework="langchain",
                component=chain.__class__.__name__,
                method="invoke",
                input_data=kwargs
            )
            
            return self.retry_system.retry_with_gemini_context(
                original_invoke, *args, error_context=error_context, **kwargs
            )
        
        # Replace the invoke method
        chain.invoke = enhanced_invoke
        return chain
    
    def wrap_llm(self, llm):
        """Wrap a LangChain LLM with intelligent retry."""
        original_invoke = llm.invoke
        
        @wraps(original_invoke)
        def enhanced_invoke(*args, **kwargs):
            # Create error context
            error_context = ErrorContext(
                timestamp=datetime.now(),
                framework="langchain",
                component=llm.__class__.__name__,
                method="invoke",
                input_data=kwargs
            )
            
            return self.retry_system.retry_with_gemini_context(
                original_invoke, *args, error_context=error_context, **kwargs
            )
        
        # Replace the invoke method
        llm.invoke = enhanced_invoke
        return llm


class LangGraphRetryWrapper:
    """Wrapper for LangGraph operations with intelligent retry."""
    
    def __init__(self, gemini_analyzer: GeminiAnalyzer, max_retries: int = 3):
        self.gemini_analyzer = gemini_analyzer
        self.max_retries = max_retries
        self.retry_system = IntelligentRetry(gemini_analyzer, max_retries)
    
    def wrap_graph(self, graph):
        """Wrap a LangGraph with intelligent retry."""
        original_invoke = graph.invoke
        
        @wraps(original_invoke)
        def enhanced_invoke(*args, **kwargs):
            # Create error context
            error_context = ErrorContext(
                timestamp=datetime.now(),
                framework="langgraph",
                component=graph.__class__.__name__,
                method="invoke",
                input_data=kwargs
            )
            
            return self.retry_system.retry_with_gemini_context(
                original_invoke, *args, error_context=error_context, **kwargs
            )
        
        # Replace the invoke method
        graph.invoke = enhanced_invoke
        return graph
