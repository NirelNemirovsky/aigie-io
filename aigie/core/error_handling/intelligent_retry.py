"""
Intelligent retry system for Aigie using Gemini-enhanced prompts and context.
"""

import time
import logging
from typing import Dict, Any, Optional, Callable, List, Union
from datetime import datetime
from functools import wraps

from ..ai.gemini_analyzer import GeminiAnalyzer
from ..types.error_types import ErrorContext, DetectedError


class IntelligentRetry:
    """Intelligent retry system that uses Gemini to enhance retry attempts with real-time remediation."""
    
    def __init__(self, gemini_analyzer: Optional[GeminiAnalyzer], max_retries: int = 3, 
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
        if not self.gemini_analyzer or not self.gemini_analyzer.is_available():
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
        logging.info(f"🔍 PROMPT INJECTION DEBUG: Starting retry attempt {attempt}")
        logging.info(f"🔍 PROMPT INJECTION DEBUG: Operation: {operation}")
        logging.info(f"🔍 PROMPT INJECTION DEBUG: Args: {args}")
        logging.info(f"🔍 PROMPT INJECTION DEBUG: Last error: {type(last_error).__name__}: {str(last_error)}")
        logging.info(f"🔍 PROMPT INJECTION DEBUG: Original kwargs keys: {list(original_kwargs.keys())}")
        logging.info(f"🔍 PROMPT INJECTION DEBUG: Current kwargs keys: {list(kwargs.keys())}")
        logging.info(f"🔍 PROMPT INJECTION DEBUG: Retry attempts count: {len(retry_attempts)}")
        
        try:
            logging.info(f"🔍 PROMPT INJECTION DEBUG: Step 1 - Getting operation signature")
            operation_signature = self._get_operation_signature(operation, error_context)
            logging.info(f"🔍 PROMPT INJECTION DEBUG: Operation signature: {operation_signature}")
            
            logging.info(f"🔍 PROMPT INJECTION DEBUG: Step 2 - Generating prompt injection context")
            # Generate prompt injection context using Gemini
            injection_context = self._generate_prompt_injection_context(
                last_error, error_context, attempt, operation_signature, retry_attempts
            )
            logging.info(f"🔍 PROMPT INJECTION DEBUG: Injection context keys: {list(injection_context.keys())}")
            logging.info(f"🔍 PROMPT INJECTION DEBUG: Injection context: {injection_context}")
            
            # Note: Pending remediation prompts are handled by the interceptor directly
            # The injection context is passed to the retry mechanism
            
            logging.info(f"🔍 PROMPT INJECTION DEBUG: Step 3 - Applying learned patterns")
            # Apply learned patterns if available
            enhanced_kwargs = self._apply_learned_patterns(
                original_kwargs, operation_signature, last_error, injection_context
            )
            logging.info(f"🔍 PROMPT INJECTION DEBUG: Enhanced kwargs keys: {list(enhanced_kwargs.keys())}")
            logging.info(f"🔍 PROMPT INJECTION DEBUG: Enhanced kwargs: {enhanced_kwargs}")
            
            logging.info(f"🔍 PROMPT INJECTION DEBUG: Step 4 - Applying advanced prompt injection")
            # Apply prompt injection with specific error guidance
            final_kwargs = self._apply_advanced_prompt_injection(
                enhanced_kwargs, injection_context, last_error, attempt, operation
            )
            logging.info(f"🔍 PROMPT INJECTION DEBUG: Final kwargs keys: {list(final_kwargs.keys())}")
            logging.info(f"🔍 PROMPT INJECTION DEBUG: Final kwargs: {final_kwargs}")
            
            logging.info(f"🚀 PROMPT INJECTION RETRY {attempt}: Executing with enhanced context")
            logging.info(f"📊 APPLIED PATTERNS: {len(self.success_patterns.get(operation_signature, []))} success patterns available")
            
            logging.info(f"🔍 PROMPT INJECTION DEBUG: Step 5 - Executing with signature validation")
            # Execute with function signature validation
            result = self._execute_with_signature_validation(operation, args, final_kwargs)
            
            logging.info(f"✅ PROMPT INJECTION SUCCESS: Retry {attempt} completed successfully")
            logging.info(f"🔍 PROMPT INJECTION DEBUG: Result type: {type(result)}")
            logging.info(f"🔍 PROMPT INJECTION DEBUG: Result: {result}")
            return result
            
        except Exception as e:
            logging.error(f"❌ PROMPT INJECTION FAILED: Retry {attempt} failed: {e}")
            logging.error(f"🔍 PROMPT INJECTION DEBUG: Exception type: {type(e).__name__}")
            logging.error(f"🔍 PROMPT INJECTION DEBUG: Exception details: {str(e)}")
            import traceback
            logging.error(f"🔍 PROMPT INJECTION DEBUG: Traceback: {traceback.format_exc()}")
            
            logging.info(f"🔍 PROMPT INJECTION DEBUG: Falling back to basic retry")
            # Fall back to basic retry
            return self._retry_with_enhanced_context(
                operation, *args, attempt=attempt, last_error=last_error, 
                error_context=error_context, original_kwargs=original_kwargs, **kwargs
            )
    
    def _get_operation_signature(self, operation: Callable, error_context: Optional[ErrorContext]) -> str:
        """Generate a unique signature for the operation for pattern matching."""
        if error_context:
            # Use the same format as the interceptor: framework_component_method
            return f"{error_context.framework}_{error_context.component}_{error_context.method}"
        else:
            op_name = getattr(operation, '__name__', str(operation))
            return f"unknown_unknown_unknown_{op_name}"
    
    def _set_pending_remediation_prompts(self, injection_context: Dict[str, Any], error: Exception, attempt: int) -> None:
        """Set pending remediation prompts for the interceptor to use."""
        logging.info("🔍 SETTING PENDING REMEDIATION PROMPTS: Creating prompts for interceptor")
        
        # Create remediation prompt from injection context
        remediation_prompt = self._create_remediation_prompt_from_context(injection_context, error, attempt)
        
        # Set the pending remediation prompts in the error detector
        if hasattr(self.error_detector, 'pending_remediation_prompts'):
            self.error_detector.pending_remediation_prompts = [remediation_prompt]
            logging.info(f"💉 PENDING PROMPTS SET: Added remediation prompt for interceptor")
            logging.info(f"🔍 PENDING PROMPTS DEBUG: Prompt length: {len(remediation_prompt)} characters")
        else:
            logging.warning("⚠️ PENDING PROMPTS: Error detector doesn't have pending_remediation_prompts attribute")
    
    def _create_remediation_prompt_from_context(self, injection_context: Dict[str, Any], error: Exception, attempt: int) -> str:
        """Create a remediation prompt from the injection context."""
        # Extract key information from injection context
        enhanced_prompt = injection_context.get('enhanced_prompt', '')
        prompt_injection_hints = injection_context.get('prompt_injection_hints', [])
        operation_specific_guidance = injection_context.get('operation_specific_guidance', {})
        
        # Create comprehensive remediation prompt
        remediation_prompt = f"""
🔄 RETRY CONTEXT (Attempt {attempt}):
Previous Error: {type(error).__name__}: {str(error)}

🎯 SPECIFIC GUIDANCE:
{chr(10).join([f"• {hint}" for hint in prompt_injection_hints]) if prompt_injection_hints else "• Learn from the previous error and execute successfully"}

📋 OPERATION GUIDANCE:
{operation_specific_guidance.get('primary_approach', 'Be more careful and systematic')}

⚠️ ERROR PREVENTION:
{chr(10).join([f"• {step}" for step in operation_specific_guidance.get('error_prevention', ['Double-check your work'])])}

🔧 ALTERNATIVE APPROACHES:
{chr(10).join([f"• {alt}" for alt in operation_specific_guidance.get('fallback_approaches', ['Try a different method if the first fails'])])}

{enhanced_prompt if enhanced_prompt else ""}

IMPORTANT: Learn from the above context and execute the task successfully this time.
"""
        
        return remediation_prompt.strip()
    
    def _generate_prompt_injection_context(self, error: Exception, error_context: Optional[ErrorContext], 
                                         attempt: int, operation_signature: str, 
                                         retry_attempts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate advanced prompt injection context using Gemini and learned patterns."""
        logging.info(f"🔍 GENERATE CONTEXT DEBUG: Starting context generation for attempt {attempt}")
        logging.info(f"🔍 GENERATE CONTEXT DEBUG: Error: {type(error).__name__}: {str(error)}")
        logging.info(f"🔍 GENERATE CONTEXT DEBUG: Operation signature: {operation_signature}")
        logging.info(f"🔍 GENERATE CONTEXT DEBUG: Retry attempts: {len(retry_attempts)}")
        
        if not self.gemini_analyzer or not self.gemini_analyzer.is_available():
            logging.info(f"🔍 GENERATE CONTEXT DEBUG: Gemini analyzer not available, using fallback")
            return self._fallback_prompt_injection_context(error, attempt, operation_signature)
        
        try:
            # Get the actual stored operation from the error detector if available
            stored_operation = None
            if hasattr(self, 'error_detector') and self.error_detector:
                stored_operation = self.error_detector.get_stored_operation(operation_signature)
                logging.info(f"🔍 GENERATE CONTEXT DEBUG: Found stored operation: {stored_operation is not None}")
                if stored_operation:
                    logging.info(f"🔍 GENERATE CONTEXT DEBUG: Stored operation kwargs keys: {list(stored_operation.get('kwargs', {}).keys())}")
                    if 'agent_scratchpad' in stored_operation.get('kwargs', {}):
                        logging.info(f"🔍 GENERATE CONTEXT DEBUG: agent_scratchpad type: {type(stored_operation['kwargs']['agent_scratchpad'])}")
            
            if not stored_operation:
                logging.info(f"🔍 GENERATE CONTEXT DEBUG: No stored operation found, creating minimal context")
                # Create minimal context without mock data
                stored_operation = {
                    'operation_type': self._infer_operation_type(operation_signature),
                    'original_prompt': self._extract_original_prompt_from_attempts(retry_attempts),
                    'kwargs': {}  # Empty kwargs to avoid corruption
                }
            
            logging.info(f"🔍 GENERATE CONTEXT DEBUG: Creating detected error")
            # Create detected error
            from ..types.error_types import DetectedError, ErrorSeverity, ErrorType
            detected_error = DetectedError(
                error_type=ErrorType.RUNTIME_EXCEPTION,
                severity=ErrorSeverity.MEDIUM,
                message=str(error),
                exception=error,
                context=error_context,
                suggestions=[]
            )
            
            logging.info(f"🔍 GENERATE CONTEXT DEBUG: Getting Gemini's prompt injection remediation")
            # Get Gemini's prompt injection remediation
            remediation = self.gemini_analyzer.generate_prompt_injection_remediation(
                error, error_context, stored_operation, detected_error
            )
            logging.info(f"🔍 GENERATE CONTEXT DEBUG: Gemini remediation keys: {list(remediation.keys())}")
            
            logging.info(f"🔍 GENERATE CONTEXT DEBUG: Getting learned context")
            # Enhance with learned patterns
            learned_context = self._get_learned_context(operation_signature, error)
            logging.info(f"🔍 GENERATE CONTEXT DEBUG: Learned context: {learned_context}")
            
            result = {
                **remediation,
                'learned_patterns': learned_context,
                'retry_attempt': attempt,
                'operation_signature': operation_signature,
                'previous_attempts': len(retry_attempts),
                'timestamp': datetime.now().isoformat()
            }
            logging.info(f"🔍 GENERATE CONTEXT DEBUG: Final context result keys: {list(result.keys())}")
            return result
            
        except Exception as e:
            logging.error(f"Failed to generate prompt injection context: {e}")
            logging.error(f"🔍 GENERATE CONTEXT DEBUG: Exception type: {type(e).__name__}")
            logging.error(f"🔍 GENERATE CONTEXT DEBUG: Exception details: {str(e)}")
            import traceback
            logging.error(f"🔍 GENERATE CONTEXT DEBUG: Traceback: {traceback.format_exc()}")
            return self._fallback_prompt_injection_context(error, attempt, operation_signature)
    
    def _apply_learned_patterns(self, kwargs: Dict[str, Any], operation_signature: str, 
                               error: Exception, injection_context: Dict[str, Any]) -> Dict[str, Any]:
        """Apply learned successful patterns to the operation parameters."""
        logging.info(f"🔍 APPLY PATTERNS DEBUG: Starting pattern application")
        logging.info(f"🔍 APPLY PATTERNS DEBUG: Operation signature: {operation_signature}")
        logging.info(f"🔍 APPLY PATTERNS DEBUG: Error type: {type(error).__name__}")
        logging.info(f"🔍 APPLY PATTERNS DEBUG: Input kwargs keys: {list(kwargs.keys())}")
        
        enhanced_kwargs = kwargs.copy()
        
        # Apply learned patterns intelligently
        # Note: agent_scratchpad is a LangChain internal parameter that should not be modified
        # but we can still apply other parameter modifications
        
        if operation_signature in self.success_patterns:
            patterns = self.success_patterns[operation_signature]
            logging.info(f"🔍 APPLY PATTERNS DEBUG: Found {len(patterns)} patterns for {operation_signature}")
            
            # Find patterns that might apply to this error type
            applicable_patterns = []
            error_type = type(error).__name__
            
            for i, pattern in enumerate(patterns):
                logging.info(f"🔍 APPLY PATTERNS DEBUG: Pattern {i}: {pattern}")
                # Check if this pattern has dealt with similar errors
                if 'error_context' in pattern:
                    if error_type in pattern['error_context'].get('handled_errors', []):
                        applicable_patterns.append(pattern)
                        logging.info(f"🔍 APPLY PATTERNS DEBUG: Pattern {i} applicable - error type match")
                elif pattern.get('retry_attempt', 0) > 0:  # This was a successful retry
                    applicable_patterns.append(pattern)
                    logging.info(f"🔍 APPLY PATTERNS DEBUG: Pattern {i} applicable - successful retry")
            
            logging.info(f"🔍 APPLY PATTERNS DEBUG: Found {len(applicable_patterns)} applicable patterns")
            
            if applicable_patterns:
                # Use the most recent successful pattern
                best_pattern = max(applicable_patterns, key=lambda p: p.get('timestamp', datetime.min))
                
                logging.info(f"🧠 APPLYING LEARNED PATTERN: Using successful pattern from {best_pattern.get('timestamp', 'unknown time')}")
                logging.info(f"🔍 APPLY PATTERNS DEBUG: Best pattern: {best_pattern}")
                
                # Apply successful parameters
                if 'successful_params' in best_pattern:
                    # Skip LangChain specific parameters that shouldn't be modified
                    langchain_special_params = ['agent_scratchpad', 'intermediate_steps', 'messages', 'input_variables']
                    
                    for param, value in best_pattern['successful_params'].items():
                        if (param not in enhanced_kwargs and  # Don't override existing params
                            param not in langchain_special_params):  # Don't modify LangChain special params
                            enhanced_kwargs[param] = value
                            logging.info(f"📚 LEARNED PARAM: Applied {param} = {value}")
                        elif param in langchain_special_params:
                            logging.info(f"🚫 LEARNED PARAM: Skipped LangChain special parameter {param}")
                        else:
                            logging.info(f"🔍 APPLY PATTERNS DEBUG: Skipped {param} - already exists in kwargs")
                else:
                    logging.info(f"🔍 APPLY PATTERNS DEBUG: No successful_params in best pattern")
            else:
                logging.info(f"🔍 APPLY PATTERNS DEBUG: No applicable patterns found")
        else:
            logging.info(f"🔍 APPLY PATTERNS DEBUG: No patterns found for {operation_signature}")
        
        logging.info(f"🔍 APPLY PATTERNS DEBUG: Enhanced kwargs keys: {list(enhanced_kwargs.keys())}")
        logging.info(f"🔍 APPLY PATTERNS DEBUG: Enhanced kwargs: {enhanced_kwargs}")
        return enhanced_kwargs
    
    def _apply_advanced_prompt_injection(self, kwargs: Dict[str, Any], injection_context: Dict[str, Any], 
                                       error: Exception, attempt: int, operation: Optional[Callable] = None) -> Dict[str, Any]:
        """Apply advanced prompt injection with specific error context."""
        logging.info(f"🔍 ADVANCED INJECTION DEBUG: Starting advanced prompt injection")
        logging.info(f"🔍 ADVANCED INJECTION DEBUG: Attempt: {attempt}")
        logging.info(f"🔍 ADVANCED INJECTION DEBUG: Error: {type(error).__name__}: {str(error)}")
        logging.info(f"🔍 ADVANCED INJECTION DEBUG: Operation: {operation}")
        logging.info(f"🔍 ADVANCED INJECTION DEBUG: Input kwargs keys: {list(kwargs.keys())}")
        logging.info(f"🔍 ADVANCED INJECTION DEBUG: Injection context keys: {list(injection_context.keys())}")
        
        final_kwargs = kwargs.copy()
        
        # Apply advanced prompt injection intelligently
        # Note: agent_scratchpad is a LangChain internal parameter that should not be modified
        # but we can still inject prompts via other parameters
        
        # Check if this is a high-level LangChain operation that shouldn't be modified
        logging.info(f"🔍 ADVANCED INJECTION DEBUG: Checking if high-level LangChain operation")
        is_high_level = self._is_high_level_langchain_operation(operation, kwargs)
        logging.info(f"🔍 ADVANCED INJECTION DEBUG: Is high-level operation: {is_high_level}")
        
        if is_high_level:
            logging.info("🚫 ADVANCED PROMPT INJECTION: Skipping high-level LangChain operation to avoid corruption")
            return final_kwargs
        
        # Debug: Log what we're working with
        logging.info(f"🔍 ADVANCED DEBUG: Original kwargs keys: {list(kwargs.keys())}")
        if 'agent_scratchpad' in kwargs:
            logging.info(f"🔍 ADVANCED DEBUG: agent_scratchpad type: {type(kwargs['agent_scratchpad'])}, value: {kwargs['agent_scratchpad']}")
        
        # Get prompt injection hints from Gemini
        logging.info(f"🔍 ADVANCED INJECTION DEBUG: Getting prompt injection hints")
        hints = injection_context.get('prompt_injection_hints', [])
        logging.info(f"🔍 ADVANCED INJECTION DEBUG: Original hints: {hints}")
        if not hints:
            hints = [f"Previous attempt {attempt-1} failed with {type(error).__name__}: {str(error)}"]
            logging.info(f"🔍 ADVANCED INJECTION DEBUG: Using fallback hints: {hints}")
        
        # Create comprehensive enhanced prompt
        logging.info(f"🔍 ADVANCED INJECTION DEBUG: Creating error guidance")
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
        logging.info(f"🔍 ADVANCED INJECTION DEBUG: Error guidance created: {len(error_guidance)} characters")
        
        # Apply prompt injection - be careful with LangChain specific parameters
        prompt_keys = ['prompt', 'input', 'query', 'text', 'message', 'instruction', 'content']
        logging.info(f"🔍 ADVANCED INJECTION DEBUG: Looking for prompt keys: {prompt_keys}")
        
        # Skip LangChain specific parameters that shouldn't be modified
        langchain_special_params = ['agent_scratchpad', 'intermediate_steps', 'messages', 'input_variables']
        logging.info(f"🔍 ADVANCED INJECTION DEBUG: LangChain special params to skip: {langchain_special_params}")
        
        prompt_injected = False
        for key in prompt_keys:
            logging.info(f"🔍 ADVANCED INJECTION DEBUG: Checking key '{key}'")
            if (key in final_kwargs and 
                isinstance(final_kwargs[key], str) and 
                key not in langchain_special_params):
                original_prompt = final_kwargs[key]
                final_kwargs[key] = f"{error_guidance}\n\nORIGINAL TASK:\n{original_prompt}"
                prompt_injected = True
                logging.info(f"💉 ADVANCED PROMPT INJECTION: Applied to '{key}' parameter")
                logging.info(f"🔍 ADVANCED INJECTION DEBUG: Original prompt length: {len(original_prompt)}")
                logging.info(f"🔍 ADVANCED INJECTION DEBUG: Enhanced prompt length: {len(final_kwargs[key])}")
                break
            else:
                logging.info(f"🔍 ADVANCED INJECTION DEBUG: Skipped key '{key}' - not suitable for injection")
        
        # Special handling for LangChain agent_scratchpad - DON'T modify input for high-level operations
        if 'agent_scratchpad' in final_kwargs:
            logging.info("💉 ADVANCED PROMPT INJECTION: Detected agent_scratchpad - skipping modification to avoid corruption")
            # Don't modify the input parameter for high-level LangChain operations
            # This prevents corruption of agent_scratchpad
        
        if not prompt_injected and operation is not None:
            logging.info(f"🔍 ADVANCED INJECTION DEBUG: No prompt injected, checking if operation accepts enhanced_context")
            # Only add enhanced_context if the function accepts it
            import inspect
            try:
                sig = inspect.signature(operation)
                logging.info(f"🔍 ADVANCED INJECTION DEBUG: Operation signature: {sig.parameters}")
                if 'enhanced_context' in sig.parameters:
                    final_kwargs['enhanced_context'] = error_guidance
                    logging.info("💉 ADVANCED PROMPT INJECTION: Added as 'enhanced_context' parameter")
                else:
                    logging.info("💉 ADVANCED PROMPT INJECTION: Function doesn't accept 'enhanced_context', skipping")
            except Exception as e:
                logging.warning(f"Could not validate function signature: {e}")
                # Don't add enhanced_context if we can't validate the signature
        
        # Apply parameter modifications from Gemini
        logging.info(f"🔍 ADVANCED INJECTION DEBUG: Applying parameter modifications from Gemini")
        param_mods = injection_context.get('parameter_modifications', {})
        logging.info(f"🔍 ADVANCED INJECTION DEBUG: Parameter modifications: {param_mods}")
        for param, value in param_mods.items():
            if param not in langchain_special_params:
                final_kwargs[param] = value
                logging.info(f"🔧 GEMINI PARAM MOD: {param} = {value}")
            else:
                logging.info(f"🚫 GEMINI PARAM MOD: Skipped LangChain special parameter {param}")
        
        # Debug: Log final result
        logging.info(f"🔍 ADVANCED DEBUG: Final kwargs keys: {list(final_kwargs.keys())}")
        if 'agent_scratchpad' in final_kwargs:
            logging.info(f"🔍 ADVANCED DEBUG: Final agent_scratchpad type: {type(final_kwargs['agent_scratchpad'])}, value: {final_kwargs['agent_scratchpad']}")
        
        logging.info(f"🔍 ADVANCED INJECTION DEBUG: Prompt injection completed. Prompt injected: {prompt_injected}")
        return final_kwargs
    
    def _is_high_level_langchain_operation(self, operation: Optional[Callable], kwargs: Dict[str, Any]) -> bool:
        """Check if this is a high-level LangChain operation that shouldn't be modified."""
        if not operation:
            return False
        
        # Get the operation name/class
        operation_name = getattr(operation, '__name__', str(operation))
        operation_class = getattr(operation, '__self__', None)
        
        # Check for high-level LangChain operations
        high_level_operations = [
            'AgentExecutor',
            'Agent',
            'LLMChain',
            'ConversationChain',
            'RetrievalQA',
            'VectorStoreRetriever'
        ]
        
        # Check if the operation is from a high-level LangChain class
        if operation_class:
            class_name = getattr(operation_class, '__class__', {}).get('__name__', '')
            if any(op in class_name for op in high_level_operations):
                return True
        
        # Check if the operation name suggests it's a high-level operation
        if any(op.lower() in operation_name.lower() for op in high_level_operations):
            return True
        
        # Check for specific method names that are high-level
        high_level_methods = ['invoke', 'ainvoke', 'run', '__call__']
        if operation_name in high_level_methods:
            # Additional check: if it has agent_scratchpad or similar LangChain-specific params
            langchain_params = ['agent_scratchpad', 'intermediate_steps', 'messages', 'input_variables']
            if any(param in kwargs for param in langchain_params):
                return True
        
        return False
    
    def _execute_with_signature_validation(self, operation: Callable, args: tuple, kwargs: Dict[str, Any]) -> Any:
        """Execute operation with signature validation to avoid parameter errors."""
        logging.info(f"🔍 SIGNATURE VALIDATION DEBUG: Starting signature validation")
        logging.info(f"🔍 SIGNATURE VALIDATION DEBUG: Operation: {operation}")
        logging.info(f"🔍 SIGNATURE VALIDATION DEBUG: Args: {args}")
        logging.info(f"🔍 SIGNATURE VALIDATION DEBUG: Kwargs keys: {list(kwargs.keys())}")
        
        import inspect
        
        # Debug: Check agent_scratchpad before execution
        if 'agent_scratchpad' in kwargs:
            logging.info(f"🔍 SIGNATURE VALIDATION: agent_scratchpad type: {type(kwargs['agent_scratchpad'])}, value: {kwargs['agent_scratchpad']}")
        
        try:
            logging.info(f"🔍 SIGNATURE VALIDATION DEBUG: Getting operation signature")
            sig = inspect.signature(operation)
            logging.info(f"🔍 SIGNATURE VALIDATION DEBUG: Signature: {sig}")
            logging.info(f"🔍 SIGNATURE VALIDATION DEBUG: Parameters: {sig.parameters}")
            
            # Filter kwargs to only include parameters the function accepts
            has_var_keyword = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in sig.parameters.values())
            logging.info(f"🔍 SIGNATURE VALIDATION DEBUG: Has **kwargs: {has_var_keyword}")
            
            if has_var_keyword:
                # Function accepts **kwargs, use all parameters
                valid_kwargs = kwargs
                logging.info(f"🔍 SIGNATURE VALIDATION DEBUG: Using all kwargs (function accepts **kwargs)")
            else:
                # Function has strict signature, filter parameters
                valid_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
                logging.info(f"🔍 SIGNATURE VALIDATION DEBUG: Filtered kwargs: {list(valid_kwargs.keys())}")
                
                # Log filtered parameters
                filtered = set(kwargs.keys()) - set(valid_kwargs.keys())
                if filtered:
                    logging.info(f"🔍 SIGNATURE FILTER: Removed parameters: {filtered}")
                else:
                    logging.info(f"🔍 SIGNATURE VALIDATION DEBUG: No parameters filtered")
            
            # Debug: Check agent_scratchpad before calling operation
            if 'agent_scratchpad' in valid_kwargs:
                logging.info(f"🔍 BEFORE OPERATION: agent_scratchpad type: {type(valid_kwargs['agent_scratchpad'])}, value: {valid_kwargs['agent_scratchpad']}")
            
            logging.info(f"🔍 SIGNATURE VALIDATION DEBUG: About to execute operation with {len(valid_kwargs)} kwargs")
            result = operation(*args, **valid_kwargs)
            logging.info(f"🔍 SIGNATURE VALIDATION DEBUG: Operation executed successfully")
            logging.info(f"🔍 SIGNATURE VALIDATION DEBUG: Result type: {type(result)}")
            return result
            
        except Exception as e:
            logging.error(f"Signature validation failed: {e}")
            logging.error(f"🔍 SIGNATURE VALIDATION DEBUG: Exception type: {type(e).__name__}")
            logging.error(f"🔍 SIGNATURE VALIDATION DEBUG: Exception details: {str(e)}")
            import traceback
            logging.error(f"🔍 SIGNATURE VALIDATION DEBUG: Traceback: {traceback.format_exc()}")
            
            logging.info(f"🔍 SIGNATURE VALIDATION DEBUG: Falling back to original call")
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
        
        # Skip LangChain specific parameters that shouldn't be modified
        langchain_special_params = ['agent_scratchpad', 'intermediate_steps', 'messages', 'input_variables']
        
        for param in prompt_params:
            if (param in enhanced_kwargs and 
                param not in langchain_special_params and
                isinstance(enhanced_kwargs[param], str)):
                # Enhance existing prompt
                original_prompt = enhanced_kwargs[param]
                enhanced_kwargs[param] = f"{enhanced_prompt}\n\nOriginal: {original_prompt}"
                logging.info(f"💉 INJECT ENHANCED PROMPT: Applied to '{param}' parameter")
                break
        else:
            # No existing prompt parameter, add enhanced prompt
            enhanced_kwargs['enhanced_context'] = enhanced_prompt
            logging.info("💉 INJECT ENHANCED PROMPT: Added as 'enhanced_context' parameter")
        
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
        
        # Skip LangChain specific parameters that shouldn't be modified
        langchain_special_params = ['agent_scratchpad', 'intermediate_steps', 'messages', 'input_variables']
        
        # Apply parameter modifications from Gemini
        if enhanced_context.get('parameter_modifications'):
            for param, value in enhanced_context['parameter_modifications'].items():
                # CRITICAL: Never modify agent_scratchpad or other LangChain internal parameters
                if param not in langchain_special_params and param != 'agent_scratchpad':
                    if param in modified_kwargs:
                        original_value = modified_kwargs[param]
                        modified_kwargs[param] = value
                        logging.info(f"🔧 PARAMETER FIX: Modified {param}: {original_value} → {value}")
                    else:
                        modified_kwargs[param] = value
                        logging.info(f"➕ PARAMETER FIX: Added parameter {param}: {value}")
                else:
                    logging.info(f"🚫 PARAMETER FIX: Skipped LangChain special parameter {param}")
        
        # Also check for legacy modified_parameters for backward compatibility
        if enhanced_context.get('modified_parameters'):
            for param, value in enhanced_context['modified_parameters'].items():
                # CRITICAL: Never modify agent_scratchpad or other LangChain internal parameters
                if param not in langchain_special_params and param != 'agent_scratchpad':
                    if param in modified_kwargs:
                        original_value = modified_kwargs[param]
                        modified_kwargs[param] = value
                        logging.info(f"🔧 PARAMETER FIX (legacy): Modified {param}: {original_value} → {value}")
                    else:
                        modified_kwargs[param] = value
                        logging.info(f"➕ PARAMETER FIX (legacy): Added parameter {param}: {value}")
                else:
                    logging.info(f"🚫 PARAMETER FIX (legacy): Skipped LangChain special parameter {param}")
        
        # Apply dynamic remediation based on Gemini's suggestions
        # This replaces hardcoded error type detection with AI-powered remediation
        self._apply_dynamic_remediation(enhanced_context, modified_kwargs, error)
        
        # Log summary of all applied fixes
        applied_fixes = []
        for key, value in modified_kwargs.items():
            if key not in kwargs:
                applied_fixes.append(f"{key}={value}")
        
        if applied_fixes:
            logging.info(f"🎯 REMEDIATION SUMMARY: Applied fixes: {', '.join(applied_fixes)}")
        
        return modified_kwargs
    
    def _apply_dynamic_remediation(self, enhanced_context: Dict[str, Any], modified_kwargs: Dict[str, Any], error: Exception) -> None:
        """
        Apply dynamic remediation based on Gemini's AI-powered suggestions.
        This replaces hardcoded error type detection with intelligent, context-aware fixes.
        """
        logging.info("🤖 DYNAMIC REMEDIATION: Applying AI-powered fixes based on Gemini analysis")
        
        # Apply remediation strategies suggested by Gemini
        remediation_strategies = enhanced_context.get('remediation_strategies', [])
        if not remediation_strategies:
            # Fallback: extract strategies from other Gemini response fields
            remediation_strategies = self._extract_remediation_strategies(enhanced_context)
        
        for strategy in remediation_strategies:
            self._apply_remediation_strategy(strategy, modified_kwargs, error)
        
        # Apply specific fixes suggested by Gemini
        specific_fixes = enhanced_context.get('specific_fixes', {})
        for fix_type, fix_details in specific_fixes.items():
            self._apply_specific_fix(fix_type, fix_details, modified_kwargs, error)
        
        # Apply configuration changes suggested by Gemini
        config_changes = enhanced_context.get('configuration_changes', {})
        for config_key, config_value in config_changes.items():
            if config_key not in ['agent_scratchpad', 'intermediate_steps', 'messages', 'input_variables']:
                original_value = modified_kwargs.get(config_key)
                modified_kwargs[config_key] = config_value
                logging.info(f"🔧 CONFIG FIX: {config_key}: {original_value} → {config_value}")
    
    def _extract_remediation_strategies(self, enhanced_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract remediation strategies from various Gemini response fields."""
        strategies = []
        
        # Extract from error_handling field
        if enhanced_context.get('error_handling'):
            error_handling = enhanced_context['error_handling']
            if isinstance(error_handling, dict):
                for key, value in error_handling.items():
                    strategies.append({
                        'type': 'error_handling',
                        'action': key,
                        'details': value
                    })
            elif isinstance(error_handling, list):
                strategies.extend([{'type': 'error_handling', 'action': item} for item in error_handling])
        
        # Extract from suggestions field
        if enhanced_context.get('suggestions'):
            suggestions = enhanced_context['suggestions']
            if isinstance(suggestions, list):
                strategies.extend([{'type': 'suggestion', 'action': item} for item in suggestions])
        
        # Extract from recommendations field
        if enhanced_context.get('recommendations'):
            recommendations = enhanced_context['recommendations']
            if isinstance(recommendations, list):
                strategies.extend([{'type': 'recommendation', 'action': item} for item in recommendations])
        
        return strategies
    
    def _apply_remediation_strategy(self, strategy: Dict[str, Any], modified_kwargs: Dict[str, Any], error: Exception) -> None:
        """Apply a specific remediation strategy suggested by Gemini."""
        strategy_type = strategy.get('type', 'unknown')
        action = strategy.get('action', '')
        details = strategy.get('details', {})
        
        logging.info(f"🎯 STRATEGY: Applying {strategy_type} - {action}")
        
        # Apply strategy based on type and action
        if strategy_type == 'error_handling':
            self._apply_error_handling_strategy(action, details, modified_kwargs, error)
        elif strategy_type == 'suggestion':
            self._apply_suggestion_strategy(action, modified_kwargs, error)
        elif strategy_type == 'recommendation':
            self._apply_recommendation_strategy(action, modified_kwargs, error)
        else:
            # Generic strategy application
            self._apply_generic_strategy(strategy, modified_kwargs, error)
    
    def _apply_error_handling_strategy(self, action: str, details: Any, modified_kwargs: Dict[str, Any], error: Exception) -> None:
        """Apply error handling strategy based on Gemini's suggestion."""
        if action == 'add_validation' or 'validation' in action.lower():
            self._apply_validation_strategy(details, modified_kwargs)
        elif action == 'add_retry' or 'retry' in action.lower():
            self._apply_retry_strategy(details, modified_kwargs)
        elif action == 'add_timeout' or 'timeout' in action.lower():
            self._apply_timeout_strategy(details, modified_kwargs)
        elif action == 'add_circuit_breaker' or 'circuit' in action.lower():
            self._apply_circuit_breaker_strategy(details, modified_kwargs)
        elif action == 'add_sanitization' or 'sanitiz' in action.lower():
            self._apply_sanitization_strategy(details, modified_kwargs)
        elif action == 'add_memory_optimization' or 'memory' in action.lower():
            self._apply_memory_optimization_strategy(details, modified_kwargs)
        else:
            # Generic error handling application
            self._apply_generic_error_handling(action, details, modified_kwargs)
    
    def _apply_validation_strategy(self, details: Any, modified_kwargs: Dict[str, Any]) -> None:
        """Apply validation strategy based on Gemini's suggestion."""
        modified_kwargs['validate_input'] = True
        modified_kwargs['clean_input'] = True
        
        # Apply specific validation rules if provided
        if isinstance(details, dict):
            for key, value in details.items():
                if key.startswith('validate_'):
                    modified_kwargs[key] = value
        
        logging.info("✅ VALIDATION STRATEGY: Applied input validation and cleaning")
    
    def _apply_retry_strategy(self, details: Any, modified_kwargs: Dict[str, Any]) -> None:
        """Apply retry strategy based on Gemini's suggestion."""
        modified_kwargs['retry_on_failure'] = True
        
        # Apply specific retry configuration if provided
        if isinstance(details, dict):
            for key, value in details.items():
                if 'retry' in key.lower() or 'attempt' in key.lower():
                    modified_kwargs[key] = value
        
        logging.info("🔄 RETRY STRATEGY: Applied retry configuration")
    
    def _apply_timeout_strategy(self, details: Any, modified_kwargs: Dict[str, Any]) -> None:
        """Apply timeout strategy based on Gemini's suggestion."""
        # Apply timeout configuration
        if isinstance(details, dict) and 'timeout' in details:
            modified_kwargs['timeout'] = details['timeout']
        else:
            modified_kwargs['timeout'] = 30  # Default timeout
        
        logging.info(f"⏱️  TIMEOUT STRATEGY: Applied timeout configuration")
    
    def _apply_circuit_breaker_strategy(self, details: Any, modified_kwargs: Dict[str, Any]) -> None:
        """Apply circuit breaker strategy based on Gemini's suggestion."""
        modified_kwargs['circuit_breaker_enabled'] = True
        
        # Apply specific circuit breaker configuration if provided
        if isinstance(details, dict):
            for key, value in details.items():
                if 'circuit' in key.lower() or 'threshold' in key.lower():
                    modified_kwargs[key] = value
        else:
            modified_kwargs['circuit_breaker_threshold'] = 3  # Default threshold
        
        logging.info("🔄 CIRCUIT BREAKER STRATEGY: Applied circuit breaker configuration")
    
    def _apply_sanitization_strategy(self, details: Any, modified_kwargs: Dict[str, Any]) -> None:
        """Apply sanitization strategy based on Gemini's suggestion."""
        modified_kwargs['sanitize_input'] = True
        
        # Apply input sanitization to string values
        for key, value in modified_kwargs.items():
            if isinstance(value, str) and key not in ['agent_scratchpad', 'intermediate_steps']:
                original_value = value
                sanitized_value = ' '.join(value.split())  # Basic sanitization
                modified_kwargs[key] = sanitized_value
                if original_value != sanitized_value:
                    logging.info(f"🧹 SANITIZATION: {key}: '{original_value}' → '{sanitized_value}'")
        
        logging.info("🧹 SANITIZATION STRATEGY: Applied input sanitization")
    
    def _apply_memory_optimization_strategy(self, details: Any, modified_kwargs: Dict[str, Any]) -> None:
        """Apply memory optimization strategy based on Gemini's suggestion."""
        # Apply memory optimization settings
        if isinstance(details, dict):
            for key, value in details.items():
                if 'batch' in key.lower() or 'memory' in key.lower():
                    modified_kwargs[key] = value
        else:
            modified_kwargs['batch_size'] = 1
            modified_kwargs['streaming'] = True
        
        logging.info("💾 MEMORY OPTIMIZATION STRATEGY: Applied memory optimization")
    
    def _apply_generic_error_handling(self, action: str, details: Any, modified_kwargs: Dict[str, Any]) -> None:
        """Apply generic error handling based on Gemini's suggestion."""
        # Convert action to parameter name
        param_name = action.lower().replace(' ', '_').replace('add_', '')
        modified_kwargs[param_name] = True
        
        # Apply details as configuration if it's a dict
        if isinstance(details, dict):
            for key, value in details.items():
                modified_kwargs[f"{param_name}_{key}"] = value
        
        logging.info(f"🔧 GENERIC ERROR HANDLING: Applied {action}")
    
    def _apply_suggestion_strategy(self, action: str, modified_kwargs: Dict[str, Any], error: Exception) -> None:
        """Apply suggestion strategy based on Gemini's suggestion."""
        # Parse suggestion and apply appropriate fix
        action_lower = action.lower()
        
        if 'timeout' in action_lower:
            modified_kwargs['timeout'] = 30
            logging.info("⏱️  SUGGESTION: Applied timeout fix")
        elif 'retry' in action_lower:
            modified_kwargs['retry_on_failure'] = True
            logging.info("🔄 SUGGESTION: Applied retry fix")
        elif 'validate' in action_lower:
            modified_kwargs['validate_input'] = True
            logging.info("✅ SUGGESTION: Applied validation fix")
        elif 'sanitize' in action_lower:
            modified_kwargs['sanitize_input'] = True
            logging.info("🧹 SUGGESTION: Applied sanitization fix")
        else:
            # Generic suggestion application
            logging.info(f"💡 SUGGESTION: {action}")
    
    def _apply_recommendation_strategy(self, action: str, modified_kwargs: Dict[str, Any], error: Exception) -> None:
        """Apply recommendation strategy based on Gemini's recommendation."""
        # Similar to suggestion strategy but with different logging
        self._apply_suggestion_strategy(action, modified_kwargs, error)
        logging.info(f"📋 RECOMMENDATION: {action}")
    
    def _apply_generic_strategy(self, strategy: Dict[str, Any], modified_kwargs: Dict[str, Any], error: Exception) -> None:
        """Apply generic strategy based on Gemini's analysis."""
        action = strategy.get('action', '')
        details = strategy.get('details', {})
        
        # Try to extract meaningful parameters from the strategy
        if isinstance(details, dict):
            for key, value in details.items():
                if key not in ['agent_scratchpad', 'intermediate_steps', 'messages', 'input_variables']:
                    modified_kwargs[key] = value
                    logging.info(f"🔧 GENERIC STRATEGY: Applied {key}={value}")
        
        logging.info(f"🎯 GENERIC STRATEGY: Applied {action}")
    
    def _apply_specific_fix(self, fix_type: str, fix_details: Any, modified_kwargs: Dict[str, Any], error: Exception) -> None:
        """Apply specific fix suggested by Gemini."""
        logging.info(f"🔧 SPECIFIC FIX: Applying {fix_type}")
        
        if isinstance(fix_details, dict):
            for key, value in fix_details.items():
                if key not in ['agent_scratchpad', 'intermediate_steps', 'messages', 'input_variables']:
                    original_value = modified_kwargs.get(key)
                    modified_kwargs[key] = value
                    logging.info(f"🔧 SPECIFIC FIX: {key}: {original_value} → {value}")
        elif isinstance(fix_details, (str, int, float, bool)):
            # Apply as a single parameter
            param_name = f"{fix_type}_fix"
            modified_kwargs[param_name] = fix_details
            logging.info(f"🔧 SPECIFIC FIX: {param_name}={fix_details}")
    
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
                from ..ai.gemini_analyzer import GeminiAnalyzer
                import os
                analyzer = GeminiAnalyzer(api_key=os.getenv("GEMINI_API_KEY"))
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
