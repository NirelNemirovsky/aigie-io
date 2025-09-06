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

from ..types.error_types import ErrorType, ErrorSeverity, DetectedError, ErrorContext


class GeminiAnalyzer:
    """Primary error analysis engine using Gemini AI. This is the central component for all error analysis and remediation in Aigie."""
    
    def __init__(self, project_id: Optional[str] = None, location: str = "us-central1", api_key: Optional[str] = None):
        self.project_id = project_id or os.getenv("GOOGLE_CLOUD_PROJECT")
        self.location = location
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.backend = None  # 'vertex', 'api_key', or None
        self.model = None
        self.is_initialized = False
        
        # Retry configuration for robust Gemini interactions
        self.max_retries = 3
        self.retry_delay = 1.0
        self.analysis_cache = {}  # Cache successful analyses
        
        # Prefer API key authentication over Vertex AI
        if self.api_key and GEMINI_API_KEY_AVAILABLE:
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
        # Fallback to Vertex AI if API key is not available or failed
        elif self.project_id and VERTEX_AVAILABLE:
            try:
                vertexai.init(project=self.project_id, location=self.location)
                self.model = VertexGenerativeModel("gemini-2.5-flash")
                self.backend = 'vertex'
                self.is_initialized = True
                logging.info(f"Gemini (Vertex AI) initialized successfully for project: {self.project_id}")
            except Exception as e:
                logging.warning(f"Failed to initialize Gemini (Vertex AI): {e}")
                self.is_initialized = False
        else:
            logging.warning("Gemini not available - no API key or Vertex AI project configured")
            self.is_initialized = False
    
    def analyze_error(self, error: Exception, context: ErrorContext) -> Dict[str, Any]:
        """Analyze an error using Gemini with retries. This is the primary method for all error analysis in Aigie."""
        if not self.is_initialized:
            logging.error("Gemini is not initialized - this should not happen in production")
            raise RuntimeError("Gemini analyzer is not available. Please check your API key or project configuration.")
        
        # Create cache key for this error
        error_signature = f"{type(error).__name__}_{context.framework}_{context.component}_{str(error)[:100]}"
        
        # Check cache first
        if error_signature in self.analysis_cache:
            cached = self.analysis_cache[error_signature]
            if (datetime.now() - cached['timestamp']).seconds < 300:  # 5 minute cache
                logging.debug(f"Using cached analysis for {error_signature}")
                return cached['analysis']
        
        # Attempt analysis with retries
        for attempt in range(self.max_retries + 1):
            try:
                logging.info(f"Gemini error analysis attempt {attempt + 1}/{self.max_retries + 1}")
                
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
                    raise Exception(f"Unknown backend: {self.backend}")
                
                analysis = self._parse_gemini_response(text)
                
                # Validate analysis completeness
                if not self._validate_analysis(analysis):
                    raise Exception("Incomplete analysis from Gemini")
                
                # Cache successful analysis
                self.analysis_cache[error_signature] = {
                    'analysis': analysis,
                    'timestamp': datetime.now()
                }
                
                logging.info(f"Gemini analysis successful on attempt {attempt + 1}")
                return analysis
                
            except Exception as e:
                logging.warning(f"Gemini analysis attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries:
                    delay = self.retry_delay * (2 ** attempt)  # Exponential backoff
                    logging.info(f"Retrying in {delay} seconds...")
                    time.sleep(delay)
                else:
                    logging.error(f"All Gemini analysis attempts failed: {e}")
                    raise Exception(f"Gemini analysis failed after {self.max_retries + 1} attempts: {e}")
    
    def generate_remediation_strategy(self, error: Exception, context: ErrorContext, 
                                    error_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a remediation strategy using Gemini with retries."""
        if not self.is_initialized:
            logging.error("Gemini is not initialized - this should not happen in production")
            raise RuntimeError("Gemini analyzer is not available. Please check your API key or project configuration.")
        
        # Attempt remediation generation with retries
        for attempt in range(self.max_retries + 1):
            try:
                logging.info(f"Gemini remediation generation attempt {attempt + 1}/{self.max_retries + 1}")
                
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
                    raise Exception(f"Unknown backend: {self.backend}")
                
                remediation = self._parse_remediation_response(text)
                
                # Validate remediation completeness
                if not self._validate_remediation(remediation):
                    raise Exception("Incomplete remediation strategy from Gemini")
                
                logging.info(f"Gemini remediation generation successful on attempt {attempt + 1}")
                return remediation
                
            except Exception as e:
                logging.warning(f"Gemini remediation attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries:
                    delay = self.retry_delay * (2 ** attempt)  # Exponential backoff
                    logging.info(f"Retrying in {delay} seconds...")
                    time.sleep(delay)
                else:
                    logging.error(f"All Gemini remediation attempts failed: {e}")
                    raise Exception(f"Gemini remediation failed after {self.max_retries + 1} attempts: {e}")
    
    def generate_prompt_injection_remediation(self, error: Exception, context: ErrorContext, 
                                            stored_operation: Dict[str, Any], 
                                            detected_error: 'DetectedError') -> Dict[str, Any]:
        """Generate specific prompt injection remediation strategy using Gemini with retries."""
        if not self.is_initialized:
            logging.error("Gemini is not initialized - this should not happen in production")
            raise RuntimeError("Gemini analyzer is not available. Please check your API key or project configuration.")
        
        # Attempt prompt injection remediation with retries
        for attempt in range(self.max_retries + 1):
            try:
                logging.info(f"Gemini prompt injection remediation attempt {attempt + 1}/{self.max_retries + 1}")
                
                # Create specialized prompt injection prompt
                prompt = self._create_prompt_injection_prompt(error, context, stored_operation, detected_error)
                
                # Get Gemini analysis based on backend
                if self.backend == 'vertex':
                    response = self.model.generate_content(prompt)
                    text = response.text
                elif self.backend == 'api_key':
                    response = self.model.generate_content(prompt)
                    text = getattr(response, 'text', None)
                    if not text and hasattr(response, 'candidates') and response.candidates:
                        text = response.candidates[0].text
                else:
                    raise Exception(f"Unknown backend: {self.backend}")
                
                remediation = self._parse_prompt_injection_response(text)
                
                # Validate prompt injection remediation completeness
                if not self._validate_prompt_injection_remediation(remediation):
                    raise Exception("Incomplete prompt injection remediation from Gemini")
                
                logging.info(f"Gemini prompt injection remediation successful on attempt {attempt + 1}")
                return remediation
                
            except Exception as e:
                logging.warning(f"Gemini prompt injection remediation attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries:
                    delay = self.retry_delay * (2 ** attempt)  # Exponential backoff
                    logging.info(f"Retrying in {delay} seconds...")
                    time.sleep(delay)
                else:
                    logging.error(f"All Gemini prompt injection remediation attempts failed: {e}")
                    raise Exception(f"Gemini prompt injection remediation failed after {self.max_retries + 1} attempts: {e}")
    
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
    
    def _create_prompt_injection_prompt(self, error: Exception, context: ErrorContext, 
                                       stored_operation: Dict[str, Any], detected_error: 'DetectedError') -> str:
        """Create a specialized prompt for generating prompt injection remediation."""
        operation_type = stored_operation.get('operation_type', 'unknown')
        original_prompt = stored_operation.get('original_prompt', 'N/A')
        
        return f"""
You are an expert AI remediation specialist focused on PROMPT INJECTION for error recovery. 
Your task is to generate specific guidance that will be injected into the AI agent/LLM prompt to help it succeed on retry.

FAILED OPERATION DETAILS:
- Operation Type: {operation_type}
- Error: {type(error).__name__}: {str(error)}
- Framework: {context.framework}
- Component: {context.component}
- Method: {context.method}
- Timestamp: {context.timestamp}
- Original Prompt: {original_prompt[:200]}...

ERROR ANALYSIS:
- Severity: {detected_error.severity.value}
- Error Type: {detected_error.error_type.value}
- Context: {detected_error.context.framework if detected_error.context else 'unknown'}

PROMPT INJECTION TASK:
Generate SPECIFIC guidance that will be injected into the agent/LLM prompt to help it avoid the same error and succeed.

Your response must include:
1. **Specific Error Guidance**: What went wrong and how to avoid it
2. **Operation-Specific Hints**: Tailored advice for this type of operation
3. **Actionable Instructions**: Concrete steps the AI should take
4. **Alternative Approaches**: Backup strategies if the primary approach fails
5. **Confidence Level**: How confident you are this will work (0.0-1.0)

RESPONSE FORMAT (JSON):
{{
    "prompt_injection_hints": [
        "Specific hint 1 about what went wrong",
        "Specific hint 2 about how to avoid the error",
        "Specific hint 3 about alternative approaches",
        "Specific hint 4 about validation/verification"
    ],
    "operation_specific_guidance": {{
        "primary_approach": "Main strategy to try",
        "fallback_approaches": ["Alternative 1", "Alternative 2"],
        "validation_steps": ["Check 1", "Check 2"],
        "error_prevention": ["Prevention step 1", "Prevention step 2"]
    }},
    "parameter_modifications": {{
        "timeout": 30,
        "retry_attempts": 3,
        "validation_enabled": true,
        "careful_mode": true
    }},
    "confidence": 0.85,
    "reasoning": "Brief explanation of why this approach should work"
}}

IMPORTANT GUIDELINES:
- Be SPECIFIC about what the AI should do differently
- Focus on ACTIONABLE guidance that can be directly applied
- Consider the specific operation type ({operation_type})
- Provide concrete examples where relevant
- Ensure hints are clear and unambiguous
- Tailor advice to the specific error type: {detected_error.error_type.value}

Generate the prompt injection remediation now:"""
    
    def _parse_gemini_response(self, response_text: str) -> Dict[str, Any]:
        """Parse Gemini's response into structured data with robust error handling."""
        try:
            # Try to extract JSON from the response
            if "{" in response_text and "}" in response_text:
                start = response_text.find("{")
                end = response_text.rfind("}") + 1
                json_str = response_text[start:end]
                
                # More robust JSON cleaning
                json_str = self._clean_json_string(json_str)
                
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError as json_error:
                    logging.warning(f"JSON parsing failed after cleaning: {json_error}")
                    # Try to fix common JSON issues
                    json_str = self._fix_common_json_issues(json_str)
                    try:
                        return json.loads(json_str)
                    except json.JSONDecodeError:
                        logging.warning("All JSON parsing attempts failed, using fallback")
                        return self._parse_text_response(response_text)
            else:
                # Fallback parsing
                return self._parse_text_response(response_text)
                
        except Exception as e:
            logging.warning(f"Failed to parse Gemini response: {e}")
            return self._parse_text_response(response_text)
    
    def _clean_json_string(self, json_str: str) -> str:
        """Clean JSON string for better parsing."""
        # Replace single quotes with double quotes (but be careful with apostrophes)
        json_str = json_str.replace("'", '"')
        
        # Fix boolean values
        json_str = json_str.replace("True", "true")
        json_str = json_str.replace("False", "false")
        json_str = json_str.replace("None", "null")
        
        # Remove trailing commas
        import re
        json_str = re.sub(r',\s*}', '}', json_str)
        json_str = re.sub(r',\s*]', ']', json_str)
        
        return json_str
    
    def _fix_common_json_issues(self, json_str: str) -> str:
        """Fix common JSON formatting issues."""
        import re
        
        # Fix unescaped quotes in strings
        # This is a simple approach - in production you might want more sophisticated parsing
        json_str = re.sub(r'(?<!\\)"(?=.*":)', r'\"', json_str)
        
        # Fix missing quotes around keys
        json_str = re.sub(r'(\w+):', r'"\1":', json_str)
        
        # Fix missing quotes around string values (simple heuristic)
        json_str = re.sub(r':\s*([^",{\[\s][^",}\]\s]*)(?=\s*[,}\]])', r': "\1"', json_str)
        
        return json_str
    
    def _parse_remediation_response(self, response_text: str) -> Dict[str, Any]:
        """Parse Gemini's remediation response."""
        return self._parse_gemini_response(response_text)
    
    def _parse_prompt_injection_response(self, response_text: str) -> Dict[str, Any]:
        """Parse Gemini's prompt injection remediation response."""
        try:
            parsed = self._parse_gemini_response(response_text)
            
            # Ensure required fields are present
            if 'prompt_injection_hints' not in parsed:
                parsed['prompt_injection_hints'] = [
                    "Previous attempt failed, try a different approach",
                    "Be more careful and specific in your response",
                    "Consider alternative methods if the first approach doesn't work"
                ]
            
            if 'confidence' not in parsed:
                parsed['confidence'] = 0.7
            
            return parsed
            
        except Exception as e:
            logging.warning(f"Failed to parse prompt injection response: {e}")
            return {
                "prompt_injection_hints": [
                    "Previous attempt failed, try a different approach",
                    "Be more careful and specific in your response",
                    "Consider alternative methods if the first approach doesn't work"
                ],
                "operation_specific_guidance": {
                    "primary_approach": "Try the original task with more care",
                    "fallback_approaches": ["Break down the task into smaller steps"],
                    "validation_steps": ["Verify your approach before proceeding"],
                    "error_prevention": ["Double-check your work"]
                },
                "parameter_modifications": {},
                "confidence": 0.6,
                "reasoning": "Fallback guidance due to parsing error"
            }
    
    def _parse_text_response(self, response_text: str) -> Dict[str, Any]:
        """Parse text response when JSON parsing fails."""
        # Extract key information from text
        analysis = {
            "error_type": "runtime_exception",  # Use lowercase enum value
            "severity": "medium",  # Use lowercase enum value
            "suggestions": ["Review the error message", "Check component configuration"],
            "is_retryable": True,
            "confidence": 0.5,
            "analysis_summary": "Error analysis completed with fallback parsing"
        }
        
        # Try to extract error type from text
        if "runtime" in response_text.lower():
            analysis["error_type"] = "runtime_exception"
        elif "api" in response_text.lower():
            analysis["error_type"] = "api_error"
        elif "state" in response_text.lower():
            analysis["error_type"] = "state_error"
        elif "validation" in response_text.lower():
            analysis["error_type"] = "validation_error"
        elif "timeout" in response_text.lower():
            analysis["error_type"] = "timeout"
        elif "memory" in response_text.lower():
            analysis["error_type"] = "memory_error"
        elif "network" in response_text.lower():
            analysis["error_type"] = "network_error"
        elif "authentication" in response_text.lower():
            analysis["error_type"] = "authentication_error"
        
        # Try to extract severity
        if "critical" in response_text.lower():
            analysis["severity"] = "critical"
        elif "high" in response_text.lower():
            analysis["severity"] = "high"
        elif "low" in response_text.lower():
            analysis["severity"] = "low"
        
        return analysis
    
    def _validate_analysis(self, analysis: Dict[str, Any]) -> bool:
        """Validate that the analysis contains all required fields."""
        required_fields = ['error_type', 'severity', 'suggestions', 'is_retryable', 'confidence']
        for field in required_fields:
            if not analysis.get(field):
                logging.warning(f"Analysis missing required field: {field}")
                return False
        
        # Check that suggestions is a non-empty list
        if not isinstance(analysis['suggestions'], list) or len(analysis['suggestions']) == 0:
            logging.warning("Analysis missing actionable suggestions")
            return False
            
        # Check confidence is a valid number
        try:
            confidence = float(analysis['confidence'])
            if confidence < 0 or confidence > 1:
                logging.warning(f"Invalid confidence value: {confidence}")
                return False
        except (ValueError, TypeError):
            logging.warning("Confidence is not a valid number")
            return False
        
        return True
    
    def _validate_remediation(self, remediation: Dict[str, Any]) -> bool:
        """Validate that the remediation strategy contains all required fields."""
        required_fields = ['retry_strategy', 'parameter_modifications', 'implementation_steps', 'confidence']
        for field in required_fields:
            if field not in remediation:
                logging.warning(f"Remediation missing required field: {field}")
                return False
        
        # Check that implementation_steps is a non-empty list
        if not isinstance(remediation['implementation_steps'], list) or len(remediation['implementation_steps']) == 0:
            logging.warning("Remediation missing implementation steps")
            return False
            
        # Check confidence is a valid number
        try:
            confidence = float(remediation['confidence'])
            if confidence < 0 or confidence > 1:
                logging.warning(f"Invalid confidence value: {confidence}")
                return False
        except (ValueError, TypeError):
            logging.warning("Confidence is not a valid number")
            return False
        
        return True
    
    def _validate_prompt_injection_remediation(self, remediation: Dict[str, Any]) -> bool:
        """Validate that the prompt injection remediation contains all required fields."""
        required_fields = ['prompt_injection_hints', 'confidence']
        for field in required_fields:
            if field not in remediation:
                logging.warning(f"Prompt injection remediation missing required field: {field}")
                return False
        
        # Check that prompt_injection_hints is a non-empty list
        if not isinstance(remediation['prompt_injection_hints'], list) or len(remediation['prompt_injection_hints']) == 0:
            logging.warning("Prompt injection remediation missing hints")
            return False
            
        # Check confidence is a valid number
        try:
            confidence = float(remediation['confidence'])
            if confidence < 0 or confidence > 1:
                logging.warning(f"Invalid confidence value: {confidence}")
                return False
        except (ValueError, TypeError):
            logging.warning("Confidence is not a valid number")
            return False
        
        return True
    
    def is_available(self) -> bool:
        """Check if Gemini is available and initialized. In the new architecture, this should always be true."""
        return self.is_initialized and self.model is not None
    
    def ensure_available(self) -> bool:
        """Ensure Gemini is available, raising an exception if not."""
        if not self.is_available():
            raise RuntimeError(
                "Gemini analyzer is not available. Aigie requires Gemini for error analysis. "
                "Please check your GEMINI_API_KEY or GOOGLE_CLOUD_PROJECT configuration."
            )
        return True
    
    def get_status(self) -> Dict[str, Any]:
        """Get the current status of the Gemini analyzer."""
        return {
            "enabled": self.is_initialized and self.model is not None,
            "is_initialized": self.is_initialized,
            "backend": self.backend,
            "project_id": self.project_id,
            "location": self.location,
            "vertex_available": VERTEX_AVAILABLE,
            "api_key_available": GEMINI_API_KEY_AVAILABLE,
            "model_loaded": self.model is not None,
            "max_retries": self.max_retries,
            "retry_delay": self.retry_delay,
            "cache_size": len(self.analysis_cache)
        }
    
    def clear_cache(self):
        """Clear the analysis cache."""
        self.analysis_cache.clear()
        logging.info("Gemini analysis cache cleared")
    
    async def _generate_content_async(self, prompt: str) -> str:
        """Generate content asynchronously using Gemini."""
        if not self.is_initialized:
            raise RuntimeError("Gemini analyzer is not initialized")
        
        try:
            if self.backend == 'vertex':
                # For Vertex AI, we need to use the sync method in async context
                import asyncio
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(None, self.model.generate_content, prompt)
                return response.text
            elif self.backend == 'api_key':
                # For API key backend, use sync method in async context
                import asyncio
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(None, self.model.generate_content, prompt)
                # Handle response text extraction
                text = getattr(response, 'text', None)
                if not text and hasattr(response, 'candidates') and response.candidates:
                    text = response.candidates[0].text
                return text
            else:
                raise RuntimeError("No valid Gemini backend available")
        except Exception as e:
            logging.error(f"Async content generation failed: {e}")
            raise