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
    """Intelligent retry system that uses Gemini to enhance retry attempts."""
    
    def __init__(self, gemini_analyzer: GeminiAnalyzer, max_retries: int = 3, 
                 base_delay: float = 1.0, max_delay: float = 60.0):
        self.gemini_analyzer = gemini_analyzer
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.retry_history = []
        
    def retry_with_gemini_context(self, operation: Callable, *args, 
                                 error_context: Optional[ErrorContext] = None,
                                 **kwargs) -> Any:
        """Retry an operation with Gemini-enhanced context."""
        last_error = None
        retry_attempts = []
        original_kwargs = kwargs.copy()  # Preserve original parameters
        
        for attempt in range(self.max_retries + 1):  # +1 for initial attempt
            try:
                start_time = time.time()
                
                if attempt == 0:
                    # Initial attempt
                    result = operation(*args, **kwargs)
                    execution_time = time.time() - start_time
                    
                    # Log successful execution
                    self._log_retry_attempt(attempt, True, None, execution_time, error_context)
                    return result
                    
                else:
                    # Retry attempt with enhanced context
                    enhanced_result = self._retry_with_enhanced_context(
                        operation, *args, attempt=attempt, 
                        last_error=last_error, error_context=error_context, 
                        original_kwargs=original_kwargs, **kwargs
                    )
                    
                    execution_time = time.time() - start_time
                    self._log_retry_attempt(attempt, True, None, execution_time, error_context)
                    return enhanced_result
                    
            except Exception as e:
                execution_time = time.time() - start_time
                last_error = e
                
                # Log failed attempt
                self._log_retry_attempt(attempt, False, e, execution_time, error_context)
                
                if attempt < self.max_retries:
                    # Generate enhanced context for retry
                    enhanced_context = self._generate_enhanced_context(e, error_context, attempt)
                    retry_attempts.append({
                        'attempt': attempt,
                        'error': e,
                        'enhanced_context': enhanced_context,
                        'timestamp': datetime.now()
                    })
                    
                    # Wait before retry
                    delay = self._calculate_delay(attempt)
                    logging.info(f"Retry attempt {attempt + 1} in {delay:.2f}s...")
                    time.sleep(delay)
                    
                else:
                    # Max retries exceeded
                    logging.error(f"Max retries ({self.max_retries}) exceeded. Final error: {e}")
                    raise e
        
        # This should never be reached
        raise RuntimeError("Unexpected retry loop termination")
    
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
