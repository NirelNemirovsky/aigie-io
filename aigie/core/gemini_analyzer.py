"""
Gemini-powered error analysis and remediation for Aigie.
"""

import os
import json
import time
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
import logging

# Try to import both Vertex AI and Gemini API key SDKs
try:
    import vertexai
    from vertexai.generative_models import GenerativeModel as VertexGenerativeModel
    VERTEX_AVAILABLE = True
except ImportError:
    VERTEX_AVAILABLE = False

try:
    import google.generativeai as genai
    GEMINI_API_KEY_AVAILABLE = True
except ImportError:
    GEMINI_API_KEY_AVAILABLE = False

from .error_types import ErrorType, ErrorSeverity, DetectedError, ErrorContext


class GeminiAnalyzer:
    """Uses Gemini to intelligently analyze errors and generate remediation strategies."""
    
    def __init__(self, project_id: Optional[str] = None, location: str = "us-central1", api_key: Optional[str] = None):
        self.project_id = project_id or os.getenv("GOOGLE_CLOUD_PROJECT")
        self.location = location
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.backend = None  # 'vertex', 'api_key', or None
        self.model = None
        self.is_initialized = False
        
        # Prefer Vertex AI if project is set
        if self.project_id and VERTEX_AVAILABLE:
            try:
                vertexai.init(project=self.project_id, location=self.location)
                self.model = VertexGenerativeModel("gemini-2.5-flash")
                self.backend = 'vertex'
                self.is_initialized = True
                logging.info(f"Gemini (Vertex) initialized successfully for project: {self.project_id}")
            except Exception as e:
                logging.warning(f"Failed to initialize Gemini (Vertex): {e}")
                self.is_initialized = False
        # Else try API key
        elif self.api_key and GEMINI_API_KEY_AVAILABLE:
            try:
                genai.configure(api_key=self.api_key)
                # Use gemini-2.5-flash for the API key backend
                self.model = genai.GenerativeModel("gemini-2.5-flash")
                self.backend = 'api_key'
                self.is_initialized = True
                logging.info("Gemini (API key) initialized successfully")
            except Exception as e:
                logging.warning(f"Failed to initialize Gemini (API key): {e}")
                self.is_initialized = False
        else:
            logging.info("Gemini not available - using fallback error analysis")
            self.is_initialized = False
    
    def analyze_error(self, error: Exception, context: ErrorContext) -> Dict[str, Any]:
        """Analyze an error using Gemini and return enhanced analysis."""
        if not self.is_initialized:
            return self._fallback_analysis(error, context)
        
        try:
            # Create analysis prompt
            prompt = self._create_analysis_prompt(error, context)
            
            # Get Gemini analysis based on backend
            if self.backend == 'vertex':
                response = self.model.generate_content(prompt)
                text = response.text
            elif self.backend == 'api_key':
                response = self.model.generate_content(prompt)
                # google-generative-ai returns a response with .text or .candidates[0].text
                text = getattr(response, 'text', None)
                if not text and hasattr(response, 'candidates') and response.candidates:
                    text = response.candidates[0].text
            else:
                return self._fallback_analysis(error, context)
            
            analysis = self._parse_gemini_response(text)
            
            # Enhance with fallback if Gemini response is incomplete
            if not analysis.get("error_type") or not analysis.get("suggestions"):
                fallback = self._fallback_analysis(error, context)
                analysis = {**fallback, **analysis}  # Merge, Gemini takes precedence
            
            return analysis
            
        except Exception as e:
            logging.error(f"Gemini analysis failed: {e}")
            return self._fallback_analysis(error, context)
    
    def generate_remediation_strategy(self, error: Exception, context: ErrorContext, 
                                    error_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a remediation strategy using Gemini."""
        if not self.is_initialized:
            return self._fallback_remediation(error, context, error_analysis)
        
        try:
            # Create remediation prompt
            prompt = self._create_remediation_prompt(error, context, error_analysis)
            
            # Get Gemini remediation based on backend
            if self.backend == 'vertex':
                response = self.model.generate_content(prompt)
                text = response.text
            elif self.backend == 'api_key':
                response = self.model.generate_content(prompt)
                text = getattr(response, 'text', None)
                if not text and hasattr(response, 'candidates') and response.candidates:
                    text = response.candidates[0].text
            else:
                return self._fallback_remediation(error, context, error_analysis)
            
            remediation = self._parse_remediation_response(text)
            
            # Enhance with fallback if Gemini response is incomplete
            if not remediation.get("retry_strategy") or not remediation.get("enhanced_prompt"):
                fallback = self._fallback_remediation(error, context, error_analysis)
                remediation = {**fallback, **remediation}  # Merge, Gemini takes precedence
            
            return remediation
            
        except Exception as e:
            logging.error(f"Gemini remediation generation failed: {e}")
            return self._fallback_remediation(error, context, error_analysis)
    
    def _create_analysis_prompt(self, error: Exception, context: ErrorContext) -> str:
        """Create a prompt for Gemini to analyze the error."""
        prompt = f"""
You are an expert AI error analyst. Analyze the following error and provide a detailed classification.

ERROR DETAILS:
- Exception Type: {type(error).__name__}
- Error Message: {str(error)}
- Framework: {context.framework}
- Component: {context.component}
- Method: {context.method}
- Timestamp: {context.timestamp}
- Input Data: {context.input_data}
- State Data: {context.state}

ANALYSIS TASK:
1. Classify the error type from these categories:
   - RUNTIME_EXCEPTION: General runtime errors
   - API_ERROR: External API/service errors
   - STATE_ERROR: State management issues
   - VALIDATION_ERROR: Input validation problems
   - MEMORY_ERROR: Memory-related issues
   - TIMEOUT: Execution timeout issues
   - LANGCHAIN_CHAIN_ERROR: LangChain-specific errors
   - LANGGRAPH_STATE_ERROR: LangGraph-specific errors

2. Determine error severity (LOW, MEDIUM, HIGH, CRITICAL)

3. Provide 3-5 specific, actionable suggestions for fixing the error

4. Identify if this is a retryable error

RESPONSE FORMAT (JSON):
{{
    "error_type": "ERROR_TYPE_HERE",
    "severity": "SEVERITY_HERE",
    "suggestions": ["suggestion1", "suggestion2", "suggestion3"],
    "is_retryable": true/false,
    "confidence": 0.95,
    "analysis_summary": "Brief summary of what went wrong"
}}
"""
        return prompt
    
    def _create_remediation_prompt(self, error: Exception, context: ErrorContext, 
                                   error_analysis: Dict[str, Any]) -> str:
        """Create a prompt for Gemini to generate specific, actionable remediation strategies."""
        return f"""
You are an expert AI remediation specialist. Generate a SPECIFIC, actionable remediation strategy for this error.

ERROR CONTEXT:
- Error: {type(error).__name__}: {str(error)}
- Framework: {context.framework}
- Component: {context.component}
- Method: {context.method}
- Input Data: {context.input_data}
- State: {context.state}

ERROR ANALYSIS:
- Type: {error_analysis.get('error_type', 'unknown')}
- Severity: {error_analysis.get('severity', 'unknown')}
- Root Cause: {error_analysis.get('root_cause', 'unknown')}

REMEDIATION REQUIREMENTS:
Generate a SPECIFIC remediation strategy that can be automatically applied:

1. **Retry Strategy**: Specific approach for retrying the operation
2. **Parameter Modifications**: EXACT parameter values that will fix the issue
3. **Implementation Steps**: Step-by-step actions to implement the fix
4. **Confidence**: 0.0 to 1.0 based on fix certainty

REQUIRED PARAMETER MODIFICATIONS:
You MUST provide specific values for these parameters based on the error type:

{{
    "retry_strategy": {{
        "approach": "specific retry method with exact steps",
        "max_retries": number,
        "backoff_delay": number
    }},
    "parameter_modifications": {{
        "timeout": number,
        "max_wait": number,
        "batch_size": number,
        "streaming": boolean,
        "circuit_breaker_enabled": boolean,
        "circuit_breaker_threshold": number,
        "retry_on_failure": boolean,
        "connection_pool_size": number,
        "validate_input": boolean,
        "clean_input": boolean,
        "sanitize_input": boolean,
        "rate_limit_delay": number,
        "exponential_backoff": boolean,
        "reset_state": boolean,
        "state_validation": boolean,
        "max_concurrent": number,
        "synchronization": boolean
    }},
    "implementation_steps": [
        "Step 1: specific action to take",
        "Step 2: specific action to take",
        "Step 3: specific action to take"
    ],
    "confidence": number_between_0_and_1,
    "fix_description": "Brief description of what the fix does"
}}

PARAMETER MODIFICATION EXAMPLES:
- For timeout errors: {{"timeout": 30, "max_wait": 60}}
- For API errors: {{"circuit_breaker_enabled": true, "retry_on_failure": true, "connection_pool_size": 10}}
- For validation errors: {{"validate_input": true, "clean_input": true, "sanitize_input": true}}
- For memory errors: {{"batch_size": 1, "streaming": true}}
- For rate limit errors: {{"rate_limit_delay": 5.0, "exponential_backoff": true}}
- For state errors: {{"reset_state": true, "state_validation": true}}

IMPORTANT: 
1. Provide SPECIFIC, ACTIONABLE solutions that can be implemented immediately
2. Do NOT give generic advice like "review the code" or "check configuration"
3. Give exact parameter values that will resolve the specific error
4. Focus on practical, implementable solutions
5. Consider the framework and component context

Generate a concrete remediation plan now:
"""
    
    def _parse_gemini_response(self, response_text: str) -> Dict[str, Any]:
        """Parse Gemini's response into structured data."""
        try:
            # Try to extract JSON from the response
            if "{" in response_text and "}" in response_text:
                start = response_text.find("{")
                end = response_text.rfind("}") + 1
                json_str = response_text[start:end]
                
                # Clean up common JSON issues
                json_str = json_str.replace("'", '"')  # Replace single quotes
                json_str = json_str.replace("True", "true")  # Fix boolean values
                json_str = json_str.replace("False", "false")
                
                return json.loads(json_str)
            else:
                # Fallback parsing
                return self._parse_text_response(response_text)
                
        except json.JSONDecodeError as e:
            logging.warning(f"Failed to parse Gemini JSON response: {e}")
            return self._parse_text_response(response_text)
    
    def _parse_remediation_response(self, response_text: str) -> Dict[str, Any]:
        """Parse Gemini's remediation response."""
        return self._parse_gemini_response(response_text)
    
    def _parse_text_response(self, response_text: str) -> Dict[str, Any]:
        """Parse text response when JSON parsing fails."""
        # Extract key information from text
        analysis = {
            "error_type": "UNKNOWN_ERROR",
            "severity": "MEDIUM",
            "suggestions": ["Review the error message", "Check component configuration"],
            "is_retryable": True,
            "confidence": 0.5,
            "analysis_summary": "Error analysis completed with fallback parsing"
        }
        
        # Try to extract error type from text
        if "runtime" in response_text.lower():
            analysis["error_type"] = "RUNTIME_EXCEPTION"
        elif "api" in response_text.lower():
            analysis["error_type"] = "API_ERROR"
        elif "state" in response_text.lower():
            analysis["error_type"] = "STATE_ERROR"
        elif "validation" in response_text.lower():
            analysis["error_type"] = "VALIDATION_ERROR"
        
        # Try to extract severity
        if "critical" in response_text.lower():
            analysis["severity"] = "CRITICAL"
        elif "high" in response_text.lower():
            analysis["severity"] = "HIGH"
        elif "low" in response_text.lower():
            analysis["severity"] = "LOW"
        
        return analysis
    
    def _fallback_analysis(self, error: Exception, context: ErrorContext) -> Dict[str, Any]:
        """Fallback error analysis when Gemini is not available."""
        error_type = "RUNTIME_EXCEPTION"
        severity = "MEDIUM"
        suggestions = []
        
        # Basic classification based on exception type and message
        error_message = str(error).lower()
        
        if "timeout" in error_message or "timed out" in error_message:
            error_type = "TIMEOUT"
            severity = "HIGH"
            suggestions = [
                "Increase timeout configuration",
                "Check for blocking operations",
                "Consider asynchronous execution"
            ]
        elif "api" in error_message or "http" in error_message:
            error_type = "API_ERROR"
            severity = "MEDIUM"
            suggestions = [
                "Check API endpoint configuration",
                "Verify authentication credentials",
                "Review rate limiting settings"
            ]
        elif "state" in error_message or "graph" in error_message:
            error_type = "STATE_ERROR"
            severity = "MEDIUM"
            suggestions = [
                "Validate state transitions",
                "Check data type consistency",
                "Review state initialization"
            ]
        elif "memory" in error_message:
            error_type = "MEMORY_ERROR"
            severity = "HIGH"
            suggestions = [
                "Check for memory leaks",
                "Review large data structures",
                "Consider streaming for large datasets"
            ]
        else:
            # Generic suggestions
            suggestions = [
                "Review the error message for clues",
                "Check component configuration",
                "Verify input data format",
                "Review recent changes to the system"
            ]
        
        return {
            "error_type": error_type,
            "severity": severity,
            "suggestions": suggestions,
            "is_retryable": True,
            "confidence": 0.7,
            "analysis_summary": f"Fallback analysis: {type(error).__name__} error in {context.component}.{context.method}"
        }
    
    def _fallback_remediation(self, error: Exception, context: ErrorContext, 
                             error_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback remediation strategy when Gemini is not available."""
        error_message = str(error).lower()
        
        # Generate specific remediation based on error type
        if "timeout" in error_message or "timed out" in error_message:
            return {
                "retry_strategy": {
                    "approach": "Retry with increased timeout and adaptive backoff",
                    "max_retries": 3,
                    "backoff_delay": 2.0
                },
                "parameter_modifications": {
                    "timeout": 30,  # Increase timeout to 30 seconds
                    "max_wait": 60,  # Increase max wait time
                    "adaptive_timeout": True
                },
                "implementation_steps": [
                    "Increase timeout parameter to 30 seconds",
                    "Add adaptive timeout based on previous attempts",
                    "Implement exponential backoff with jitter"
                ],
                "confidence": 0.8,
                "fix_description": "Increases timeout parameters and adds adaptive backoff for slow operations"
            }
        elif "api" in error_message or "connection" in error_message:
            return {
                "retry_strategy": {
                    "approach": "Retry with circuit breaker pattern and connection pooling",
                    "max_retries": 5,
                    "backoff_delay": 1.0
                },
                "parameter_modifications": {
                    "circuit_breaker_enabled": True,
                    "circuit_breaker_threshold": 3,
                    "retry_on_failure": True,
                    "connection_pool_size": 10,
                    "max_retries": 5
                },
                "implementation_steps": [
                    "Enable retry on failure with exponential backoff",
                    "Implement circuit breaker pattern (threshold: 3)",
                    "Add connection pooling (size: 10)",
                    "Add request deduplication"
                ],
                "confidence": 0.7,
                "fix_description": "Implements circuit breaker pattern, connection pooling, and request deduplication for API failures"
            }
        elif "validation" in error_message or "format" in error_message:
            return {
                "retry_strategy": {
                    "approach": "Retry with input validation, cleaning, and sanitization",
                    "max_retries": 2,
                    "backoff_delay": 0.5
                },
                "parameter_modifications": {
                    "validate_input": True,
                    "clean_input": True,
                    "sanitize_input": True,
                    "max_retries": 2
                },
                "implementation_steps": [
                    "Add comprehensive input validation before processing",
                    "Clean and sanitize input data (remove extra whitespace, normalize)",
                    "Add input type checking and conversion",
                    "Log validation failures with detailed context"
                ],
                "confidence": 0.9,
                "fix_description": "Adds comprehensive input validation, cleaning, and sanitization to prevent format errors"
            }
        elif "memory" in error_message:
            return {
                "retry_strategy": {
                    "approach": "Retry with memory optimization and streaming",
                    "max_retries": 2,
                    "backoff_delay": 1.0
                },
                "parameter_modifications": {
                    "batch_size": 1,
                    "streaming": True,
                    "memory_optimization": True
                },
                "implementation_steps": [
                    "Reduce batch size to 1 for memory efficiency",
                    "Enable streaming to process data incrementally",
                    "Add memory monitoring and cleanup"
                ],
                "confidence": 0.8,
                "fix_description": "Optimizes memory usage with reduced batch size and streaming"
            }
        elif "rate limit" in error_message:
            return {
                "retry_strategy": {
                    "approach": "Retry with rate limiting and exponential backoff",
                    "max_retries": 3,
                    "backoff_delay": 2.0
                },
                "parameter_modifications": {
                    "rate_limit_delay": 5.0,
                    "exponential_backoff": True,
                    "max_retries": 3
                },
                "implementation_steps": [
                    "Add 5-second delay for rate limit compliance",
                    "Enable exponential backoff for subsequent retries",
                    "Implement request throttling"
                ],
                "confidence": 0.7,
                "fix_description": "Handles rate limiting with delays and exponential backoff"
            }
        else:
            # Generic remediation
            return {
                "retry_strategy": {
                    "approach": "Retry with enhanced error handling",
                    "max_retries": 3,
                    "backoff_delay": 1.0
                },
                "parameter_modifications": {},
                "implementation_steps": [
                    "Add error handling around the operation",
                    "Implement retry logic with exponential backoff",
                    "Add logging for debugging"
                ],
                "confidence": 0.6,
                "fix_description": "Generic retry with exponential backoff"
            }
    
    def is_available(self) -> bool:
        """Check if Gemini is available and initialized."""
        return self.is_initialized and self.model is not None
    
    def get_status(self) -> Dict[str, Any]:
        """Get the current status of the Gemini analyzer."""
        return {
            "is_initialized": self.is_initialized,
            "backend": self.backend,
            "project_id": self.project_id,
            "location": self.location,
            "vertex_available": VERTEX_AVAILABLE,
            "api_key_available": GEMINI_API_KEY_AVAILABLE,
            "model_loaded": self.model is not None
        }
