"""
Advanced Runtime Validator - High-performance LLM-as-Judge with LangChain integration.
Consolidated version with all functionality in one file.
"""

import os
import json
import asyncio
import logging
import time
from typing import Dict, Any, Optional, List, Tuple, Union, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import weakref

# LangChain imports
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser, JsonOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import HumanMessage, SystemMessage
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except ImportError:
    # Fallback for when langchain_google_genai is not available
    ChatGoogleGenerativeAI = None
from pydantic import BaseModel, Field
from pydantic import ValidationError

from ..ai.gemini_analyzer import GeminiAnalyzer
from ..types.validation_types import (
    ExecutionStep, ValidationResult, ValidationStrategy, 
    RiskLevel, ValidationStatus
)


@dataclass
class ValidationConfig:
    """Configuration for the runtime validator."""
    
    # Performance settings
    max_concurrent_validations: int = 10
    cache_ttl_seconds: int = 300
    max_cache_size: int = 1000
    enable_streaming: bool = True
    enable_parallel_strategies: bool = True
    
    # Quality settings
    min_confidence_threshold: float = 0.7
    enable_adaptive_validation: bool = True
    enable_pattern_learning: bool = True
    
    # LLM settings
    model_name: str = "gemini-2.5-flash"
    temperature: float = 0.1
    max_tokens: int = 2048
    
    # Validation strategies
    enabled_strategies: List[ValidationStrategy] = field(default_factory=lambda: [
        ValidationStrategy.GOAL_ALIGNMENT,
        ValidationStrategy.LOGICAL_CONSISTENCY,
        ValidationStrategy.OUTPUT_QUALITY,
        ValidationStrategy.SAFETY_COMPLIANCE
    ])
    
    # Adaptive settings
    learning_window_size: int = 100
    confidence_decay_factor: float = 0.95
    pattern_similarity_threshold: float = 0.8


@dataclass
class ValidationPattern:
    """Pattern learned from validation history."""
    
    pattern_id: str
    step_signature: str
    success_rate: float
    avg_confidence: float
    common_issues: List[str]
    effective_strategies: List[ValidationStrategy]
    last_seen: datetime
    frequency: int = 1


@dataclass
class ValidationMetrics:
    """Real-time validation metrics."""
    
    total_validations: int = 0
    successful_validations: int = 0
    avg_validation_time: float = 0.0
    avg_confidence: float = 0.0
    cache_hit_rate: float = 0.0
    parallel_utilization: float = 0.0
    pattern_matches: int = 0
    
    def update(self, validation_time: float, confidence: float, cache_hit: bool, 
               parallel_used: bool, pattern_match: bool = False):
        """Update metrics with new validation data."""
        self.total_validations += 1
        if confidence >= 0.7:
            self.successful_validations += 1
        
        # Update running averages
        self.avg_validation_time = (
            (self.avg_validation_time * (self.total_validations - 1) + validation_time) 
            / self.total_validations
        )
        self.avg_confidence = (
            (self.avg_confidence * (self.total_validations - 1) + confidence) 
            / self.total_validations
        )
        
        # Update cache hit rate
        if cache_hit:
            self.cache_hit_rate = (
                (self.cache_hit_rate * (self.total_validations - 1) + 1) 
                / self.total_validations
            )
        
        # Update parallel utilization
        if parallel_used:
            self.parallel_utilization = (
                (self.parallel_utilization * (self.total_validations - 1) + 1) 
                / self.total_validations
            )
        
        if pattern_match:
            self.pattern_matches += 1


class ValidationOutput(BaseModel):
    """Structured output for validation results."""
    
    is_valid: bool = Field(description="Whether the step is valid")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score from 0.0 to 1.0")
    reasoning: str = Field(description="Detailed reasoning for the validation decision")
    issues: List[str] = Field(default_factory=list, description="List of identified issues")
    suggestions: List[str] = Field(default_factory=list, description="List of improvement suggestions")
    risk_level: str = Field(description="Risk level: low, medium, high, critical")
    strategy_scores: Dict[str, float] = Field(default_factory=dict, description="Scores for each validation strategy")
    
    class Config:
        extra = "forbid"


class ValidationCallbackHandler(BaseCallbackHandler):
    """Callback handler for validation monitoring."""
    
    def __init__(self, validator_ref):
        self.validator_ref = validator_ref
        self.validation_times = []
    
    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs):
        """Called when LLM starts."""
        self.start_time = time.time()
    
    def on_llm_end(self, response, **kwargs):
        """Called when LLM ends."""
        if hasattr(self, 'start_time'):
            validation_time = time.time() - self.start_time
            self.validation_times.append(validation_time)
            
            # Update validator metrics if reference is still valid
            validator = self.validator_ref()
            if validator and hasattr(validator, '_update_validation_time'):
                validator._update_validation_time(validation_time)


class RuntimeValidator:
    """Advanced LLM-as-Judge validator with LangChain integration and dynamic strategies."""
    
    def __init__(self, gemini_analyzer: GeminiAnalyzer, config: Optional[ValidationConfig] = None):
        self.gemini_analyzer = gemini_analyzer
        self.config = config or ValidationConfig()
        
        # Initialize LangChain components
        self._setup_langchain_components()
        
        # Caching and performance
        self.validation_cache = {}
        self.pattern_cache = {}
        self.executor = ThreadPoolExecutor(max_workers=self.config.max_concurrent_validations)
        
        # Learning and adaptation
        self.validation_patterns: Dict[str, ValidationPattern] = {}
        self.validation_history: List[Tuple[ExecutionStep, ValidationResult]] = []
        self.metrics = ValidationMetrics()
        
        # Callback handler for monitoring
        self.callback_handler = ValidationCallbackHandler(weakref.ref(self))
        
        # Strategy registry for dynamic strategy selection
        self.strategies = self._build_strategy_registry()
        
        # Legacy cache configuration for backward compatibility
        self.cache_ttl = self.config.cache_ttl_seconds
        self.max_cache_size = self.config.max_cache_size
        
        logging.info("Advanced RuntimeValidator initialized with LangChain integration and performance optimization")
    
    def _setup_langchain_components(self):
        """Setup LangChain components for validation."""
        try:
            # Initialize the LLM
            if ChatGoogleGenerativeAI is None:
                raise ImportError("langchain_google_genai not available")
            
            self.llm = ChatGoogleGenerativeAI(
                model=self.config.model_name,
                temperature=self.config.temperature,
                max_output_tokens=self.config.max_tokens,
                google_api_key=os.getenv("GEMINI_API_KEY"),
                callbacks=[self.callback_handler] if hasattr(self, 'callback_handler') else []
            )
            
            # Setup output parser
            self.output_parser = PydanticOutputParser(pydantic_object=ValidationOutput)
            
            # Create validation prompt template
            self.validation_prompt = ChatPromptTemplate.from_messages([
                SystemMessage(content=self._get_system_prompt()),
                HumanMessage(content=self._get_validation_prompt_template())
            ])
            
            # Create the validation chain
            self.validation_chain = (
                self.validation_prompt 
                | self.llm 
                | self.output_parser
            )
            
            # Create strategy-specific prompts
            self.strategy_prompts = self._create_strategy_prompts()
            
        except Exception as e:
            logging.error(f"Failed to setup LangChain components: {e}")
            # Fallback to basic setup
            self.llm = None
            self.validation_chain = None
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for validation."""
        return """You are an expert AI agent execution validator. Your role is to judge whether execution steps are correct, safe, and aligned with the agent's goals.

VALIDATION CRITERIA:
1. GOAL ALIGNMENT: Does this step advance the agent's stated goal?
2. LOGICAL CONSISTENCY: Is the step logically sound given the context?
3. OUTPUT QUALITY: Will this likely produce appropriate output?
4. STATE COHERENCE: Does this maintain consistent agent state?
5. SAFETY COMPLIANCE: Does this follow safety guidelines?
6. PERFORMANCE OPTIMALITY: Is this the most efficient approach?

Be thorough but concise. Focus on practical correctness and safety. Provide specific, actionable feedback."""

    def _get_validation_prompt_template(self) -> str:
        """Get the validation prompt template."""
        return """AGENT CONTEXT:
- Goal: {agent_goal}
- Framework: {framework}
- Component: {component}
- Operation: {operation}
- Step Reasoning: {step_reasoning}

EXECUTION DETAILS:
- Input Data: {input_summary}
- Previous Conversation: {conversation_context}
- Timestamp: {timestamp}

{format_instructions}

Please provide your validation judgment:"""

    def _create_strategy_prompts(self) -> Dict[ValidationStrategy, ChatPromptTemplate]:
        """Create strategy-specific prompts for parallel validation."""
        prompts = {}
        
        for strategy in ValidationStrategy:
            if strategy == ValidationStrategy.GOAL_ALIGNMENT:
                prompts[strategy] = ChatPromptTemplate.from_messages([
                    SystemMessage(content="Analyze goal alignment for this AI agent step."),
                    HumanMessage(content="""Agent Goal: {agent_goal}
Step: {operation} on {component}
Input: {input_summary}

Questions:
1. Is this the right tool/component for achieving the goal?
2. Does the input make sense for the stated goal?
3. Is this step in the right sequence for goal completion?
4. Are there better alternatives for this goal?

Provide a score (0.0-1.0) and reasoning.""")
                ])
            
            elif strategy == ValidationStrategy.LOGICAL_CONSISTENCY:
                prompts[strategy] = ChatPromptTemplate.from_messages([
                    SystemMessage(content="Check logical consistency for this AI agent step."),
                    HumanMessage(content="""Previous Context: {conversation_context}
Current Step: {operation}
Input: {input_summary}
Component: {component}

Look for:
1. Contradictions with previous steps
2. Missing prerequisites
3. Logical fallacies
4. Inconsistent reasoning
5. Data type mismatches

Provide a score (0.0-1.0) and reasoning.""")
                ])
            
            elif strategy == ValidationStrategy.OUTPUT_QUALITY:
                prompts[strategy] = ChatPromptTemplate.from_messages([
                    SystemMessage(content="Predict output quality for this AI agent step."),
                    HumanMessage(content="""Component: {component}
Operation: {operation}
Input: {input_summary}
Expected Output Type: {expected_output_type}

Assess:
1. Likely output format and structure
2. Completeness for the task
3. Relevance to the goal
4. Potential issues or edge cases
5. Quality of expected results

Provide a score (0.0-1.0) and reasoning.""")
                ])
            
            elif strategy == ValidationStrategy.SAFETY_COMPLIANCE:
                prompts[strategy] = ChatPromptTemplate.from_messages([
                    SystemMessage(content="Check safety compliance for this AI agent step."),
                    HumanMessage(content="""Step: {operation} on {component}
Input: {input_summary}
Goal: {agent_goal}

Safety checks:
1. No harmful content generation
2. No unauthorized data access
3. No system manipulation
4. Appropriate use of tools
5. Privacy protection
6. Ethical considerations

Provide a score (0.0-1.0) and reasoning.""")
                ])
        
        return prompts

    def _build_strategy_registry(self) -> Dict[ValidationStrategy, Callable]:
        """Build registry of validation strategies."""
        return {
            ValidationStrategy.GOAL_ALIGNMENT: self._validate_goal_alignment,
            ValidationStrategy.LOGICAL_CONSISTENCY: self._validate_logical_consistency,
            ValidationStrategy.OUTPUT_QUALITY: self._validate_output_quality,
            ValidationStrategy.STATE_COHERENCE: self._validate_state_coherence,
            ValidationStrategy.SAFETY_COMPLIANCE: self._validate_safety_compliance,
            ValidationStrategy.PERFORMANCE_OPTIMALITY: self._validate_performance_optimality
        }
    
    async def validate_step(self, step: ExecutionStep) -> ValidationResult:
        """Main validation entry point with advanced features."""
        start_time = time.time()
        
        try:
            # 1. Check cache first
            cache_key = self._get_cache_key(step)
            if cache_key in self.validation_cache:
                cached_result = self.validation_cache[cache_key]
                if self._is_cache_valid(cached_result):
                    logging.debug(f"Using cached validation for step {step.step_id}")
                    self.metrics.update(0.001, cached_result['result'].confidence, True, False)
                    return cached_result['result']
            
            # 2. Check for pattern matches
            pattern_match = self._find_pattern_match(step)
            if pattern_match:
                logging.debug(f"Pattern match found for step {step.step_id}")
                self.metrics.update(0.001, pattern_match.avg_confidence, True, False, True)
                return self._create_result_from_pattern(step, pattern_match)
            
            # 3. Determine validation strategy
            strategies_to_use = self._select_strategies(step)
            
            # 4. Perform validation
            if self.config.enable_parallel_strategies and len(strategies_to_use) > 1:
                validation_result = await self._validate_parallel(step, strategies_to_use)
            else:
                validation_result = await self._validate_sequential(step, strategies_to_use)
            
            # 5. Apply adaptive learning
            if self.config.enable_adaptive_validation:
                self._learn_from_validation(step, validation_result)
            
            # 6. Cache the result
            self._cache_result(cache_key, validation_result)
            
            # 7. Update metrics
            validation_time = time.time() - start_time
            self.metrics.update(
                validation_time, 
                validation_result.confidence, 
                False, 
                self.config.enable_parallel_strategies
            )
            
            # 8. Update step with validation results
            step.validation_status = ValidationStatus.VALID if validation_result.is_valid else ValidationStatus.INVALID
            step.validation_score = validation_result.confidence
            step.validation_reasoning = validation_result.reasoning
            step.validation_timestamp = validation_result.validation_timestamp
            
            logging.info(f"Step {step.step_id} validated in {validation_time:.3f}s - Valid: {validation_result.is_valid}, Confidence: {validation_result.confidence:.2f}")
            
            return validation_result
            
        except Exception as e:
            logging.error(f"Validation failed for step {step.step_id}: {e}")
            return ValidationResult(
                is_valid=False,
                confidence=0.0,
                reasoning=f"Validation error: {str(e)}",
                issues=[f"Validation system error: {str(e)}"],
                suggestions=["Check validation system configuration"],
                risk_level=RiskLevel.HIGH
            )
    
    async def _validate_parallel(self, step: ExecutionStep, strategies: List[ValidationStrategy]) -> ValidationResult:
        """Perform parallel validation using multiple strategies."""
        if not self.llm:
            return await self._fallback_validation(step)
        
        # Prepare context for all strategies
        context = self._prepare_validation_context(step)
        
        # Create validation tasks
        tasks = []
        for strategy in strategies:
            if strategy in self.strategy_prompts:
                task = self._validate_strategy_async(step, strategy, context)
                tasks.append(task)
        
        # Execute parallel validation
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            strategy_results = {}
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logging.warning(f"Strategy {strategies[i]} failed: {result}")
                    strategy_results[strategies[i]] = {"score": 0.5, "reasoning": f"Strategy failed: {result}"}
                else:
                    strategy_results[strategies[i]] = result
            
            # Synthesize final result
            return self._synthesize_validation_result(step, strategy_results)
            
        except Exception as e:
            logging.error(f"Parallel validation failed: {e}")
            return await self._fallback_validation(step)

    async def _validate_sequential(self, step: ExecutionStep, strategies: List[ValidationStrategy]) -> ValidationResult:
        """Perform sequential validation using multiple strategies."""
        if not self.llm:
            return await self._fallback_validation(step)
        
        # Prepare context
        context = self._prepare_validation_context(step)
        
        # Execute strategies sequentially
        strategy_results = {}
        for strategy in strategies:
            try:
                result = await self._validate_strategy_async(step, strategy, context)
                strategy_results[strategy] = result
            except Exception as e:
                logging.warning(f"Strategy {strategy} failed: {e}")
                strategy_results[strategy] = {"score": 0.5, "reasoning": f"Strategy failed: {e}"}
        
        # Synthesize final result
        return self._synthesize_validation_result(step, strategy_results)

    async def _validate_strategy_async(self, step: ExecutionStep, strategy: ValidationStrategy, context: Dict[str, Any]) -> Dict[str, Any]:
        """Validate using a specific strategy asynchronously."""
        if strategy not in self.strategy_prompts:
            return {"score": 0.5, "reasoning": "Strategy not available"}
        
        try:
            # Create strategy-specific prompt
            prompt = self.strategy_prompts[strategy].format_messages(**context)
            
            # Get LLM response
            response = await self.llm.ainvoke(prompt)
            
            # Parse response
            return self._parse_strategy_response(response.content, strategy)
            
        except Exception as e:
            logging.warning(f"Strategy {strategy} validation failed: {e}")
            return {"score": 0.5, "reasoning": f"Strategy validation failed: {e}"}

    async def _fallback_validation(self, step: ExecutionStep) -> ValidationResult:
        """Fallback validation when LangChain is not available."""
        try:
            # Use the original Gemini analyzer
            validation_prompt = self._create_basic_validation_prompt(step)
            judgment = await self.gemini_analyzer._generate_content_async(validation_prompt)
            
            # Parse judgment
            judgment_data = self._parse_judgment_response(judgment)
            
            return ValidationResult(
                is_valid=judgment_data.get("is_valid", True),
                confidence=judgment_data.get("confidence", 0.5),
                reasoning=judgment_data.get("reasoning", "Fallback validation"),
                issues=judgment_data.get("issues", []),
                suggestions=judgment_data.get("suggestions", []),
                risk_level=self._parse_risk_level(judgment_data.get("risk_level", "low"))
            )
            
        except Exception as e:
            logging.error(f"Fallback validation failed: {e}")
            return ValidationResult(
                is_valid=True,  # Conservative fallback
                confidence=0.3,
                reasoning=f"Fallback validation failed: {e}",
                issues=["Validation system error"],
                suggestions=["Check system configuration"],
                risk_level=RiskLevel.MEDIUM
            )

    def _prepare_validation_context(self, step: ExecutionStep) -> Dict[str, Any]:
        """Prepare context for validation."""
        return {
            "agent_goal": step.agent_goal or "Not specified",
            "framework": step.framework,
            "component": step.component,
            "operation": step.operation,
            "step_reasoning": step.step_reasoning or "Not provided",
            "input_summary": self._summarize_input_data(step.input_data),
            "conversation_context": self._format_conversation_context(step.conversation_history),
            "timestamp": step.timestamp.isoformat(),
            "expected_output_type": self._infer_expected_output_type(step),
            "format_instructions": self.output_parser.get_format_instructions() if hasattr(self, 'output_parser') else ""
        }

    def _select_strategies(self, step: ExecutionStep) -> List[ValidationStrategy]:
        """Dynamically select validation strategies based on step context."""
        selected_strategies = []
        
        # Always include basic strategies
        selected_strategies.extend([
            ValidationStrategy.GOAL_ALIGNMENT,
            ValidationStrategy.SAFETY_COMPLIANCE
        ])
        
        # Add context-specific strategies
        if step.agent_goal:
            selected_strategies.append(ValidationStrategy.LOGICAL_CONSISTENCY)
        
        if step.component and "llm" in step.component.lower():
            selected_strategies.append(ValidationStrategy.OUTPUT_QUALITY)
        
        if step.intermediate_state:
            selected_strategies.append(ValidationStrategy.STATE_COHERENCE)
        
        if step.execution_time and step.execution_time > 1.0:
            selected_strategies.append(ValidationStrategy.PERFORMANCE_OPTIMALITY)
        
        # Filter by enabled strategies
        return [s for s in selected_strategies if s in self.config.enabled_strategies]

    def _find_pattern_match(self, step: ExecutionStep) -> Optional[ValidationPattern]:
        """Find matching validation pattern for the step."""
        if not self.config.enable_pattern_learning:
            return None
        
        step_signature = self._create_step_signature(step)
        
        for pattern in self.validation_patterns.values():
            if self._calculate_similarity(step_signature, pattern.step_signature) > self.config.pattern_similarity_threshold:
                return pattern
        
        return None

    def _create_step_signature(self, step: ExecutionStep) -> str:
        """Create a signature for the step to enable pattern matching."""
        signature_parts = [
            step.framework,
            step.component,
            step.operation,
            str(hash(str(sorted(step.input_data.items())))),
            step.agent_goal or "no_goal"
        ]
        return "|".join(signature_parts)

    def _calculate_similarity(self, sig1: str, sig2: str) -> float:
        """Calculate similarity between two step signatures."""
        if sig1 == sig2:
            return 1.0
        
        # Simple similarity based on common parts
        parts1 = sig1.split("|")
        parts2 = sig2.split("|")
        
        if len(parts1) != len(parts2):
            return 0.0
        
        matches = sum(1 for p1, p2 in zip(parts1, parts2) if p1 == p2)
        return matches / len(parts1)

    def _create_result_from_pattern(self, step: ExecutionStep, pattern: ValidationPattern) -> ValidationResult:
        """Create validation result from matched pattern."""
        return ValidationResult(
            is_valid=pattern.success_rate > 0.7,
            confidence=pattern.avg_confidence,
            reasoning=f"Pattern-based validation (pattern: {pattern.pattern_id})",
            issues=pattern.common_issues,
            suggestions=["Consider pattern-based optimizations"],
            risk_level=RiskLevel.LOW if pattern.success_rate > 0.8 else RiskLevel.MEDIUM
        )

    def _learn_from_validation(self, step: ExecutionStep, result: ValidationResult):
        """Learn from validation results to improve future validations."""
        if not self.config.enable_adaptive_validation:
            return
        
        # Add to history
        self.validation_history.append((step, result))
        
        # Keep history within window
        if len(self.validation_history) > self.config.learning_window_size:
            self.validation_history.pop(0)
        
        # Update patterns
        step_signature = self._create_step_signature(step)
        
        if step_signature in self.validation_patterns:
            pattern = self.validation_patterns[step_signature]
            pattern.frequency += 1
            pattern.last_seen = datetime.now()
            
            # Update success rate and confidence
            success = 1 if result.is_valid else 0
            pattern.success_rate = (
                (pattern.success_rate * (pattern.frequency - 1) + success) / pattern.frequency
            )
            pattern.avg_confidence = (
                (pattern.avg_confidence * (pattern.frequency - 1) + result.confidence) / pattern.frequency
            )
            
            # Update issues and suggestions
            if result.issues:
                pattern.common_issues.extend(result.issues)
                # Keep only recent issues
                pattern.common_issues = pattern.common_issues[-10:]
        else:
            # Create new pattern
            pattern = ValidationPattern(
                pattern_id=f"pattern_{len(self.validation_patterns)}",
                step_signature=step_signature,
                success_rate=1 if result.is_valid else 0,
                avg_confidence=result.confidence,
                common_issues=result.issues.copy(),
                effective_strategies=[],  # Will be populated based on strategy results
                last_seen=datetime.now()
            )
            self.validation_patterns[step_signature] = pattern

    def _synthesize_validation_result(self, step: ExecutionStep, strategy_results: Dict[ValidationStrategy, Dict[str, Any]]) -> ValidationResult:
        """Synthesize final validation result from strategy results."""
        if not strategy_results:
            return ValidationResult(
                is_valid=True,
                confidence=0.5,
                reasoning="No validation strategies available",
                issues=[],
                suggestions=[],
                risk_level=RiskLevel.LOW
            )
        
        # Calculate overall scores
        scores = []
        all_issues = []
        all_suggestions = []
        
        for strategy, result in strategy_results.items():
            if isinstance(result, dict) and "score" in result:
                scores.append(result["score"])
                if "issues" in result:
                    all_issues.extend(result["issues"])
                if "suggestions" in result:
                    all_suggestions.extend(result["suggestions"])
        
        # Calculate overall confidence
        if scores:
            avg_score = sum(scores) / len(scores)
            confidence = max(0.0, min(1.0, avg_score))
        else:
            confidence = 0.5
        
        # Determine validity
        is_valid = confidence >= self.config.min_confidence_threshold
        
        # Determine risk level
        if confidence >= 0.9:
            risk_level = RiskLevel.LOW
        elif confidence >= 0.7:
            risk_level = RiskLevel.MEDIUM
        elif confidence >= 0.5:
            risk_level = RiskLevel.HIGH
        else:
            risk_level = RiskLevel.CRITICAL
        
        # Create reasoning
        reasoning_parts = []
        for strategy, result in strategy_results.items():
            if isinstance(result, dict) and "reasoning" in result:
                reasoning_parts.append(f"{strategy.value}: {result['reasoning']}")
        
        reasoning = "; ".join(reasoning_parts) if reasoning_parts else "Validation completed"
        
        return ValidationResult(
            is_valid=is_valid,
            confidence=confidence,
            reasoning=reasoning,
            issues=list(set(all_issues)),  # Remove duplicates
            suggestions=list(set(all_suggestions)),  # Remove duplicates
            risk_level=risk_level,
            strategy_results=strategy_results
        )

    # Strategy implementations
    async def _validate_goal_alignment(self, step: ExecutionStep, context: Dict[str, Any]) -> Dict[str, Any]:
        """Validate goal alignment using LangChain."""
        if ValidationStrategy.GOAL_ALIGNMENT not in self.strategy_prompts:
            return {"score": 0.5, "reasoning": "Goal alignment strategy not available"}
        
        try:
            prompt = self.strategy_prompts[ValidationStrategy.GOAL_ALIGNMENT].format_messages(**context)
            response = await self.llm.ainvoke(prompt)
            return self._parse_strategy_response(response.content, ValidationStrategy.GOAL_ALIGNMENT)
        except Exception as e:
            return {"score": 0.5, "reasoning": f"Goal alignment validation failed: {e}"}

    async def _validate_logical_consistency(self, step: ExecutionStep, context: Dict[str, Any]) -> Dict[str, Any]:
        """Validate logical consistency using LangChain."""
        if ValidationStrategy.LOGICAL_CONSISTENCY not in self.strategy_prompts:
            return {"score": 0.5, "reasoning": "Logical consistency strategy not available"}
        
        try:
            prompt = self.strategy_prompts[ValidationStrategy.LOGICAL_CONSISTENCY].format_messages(**context)
            response = await self.llm.ainvoke(prompt)
            return self._parse_strategy_response(response.content, ValidationStrategy.LOGICAL_CONSISTENCY)
        except Exception as e:
            return {"score": 0.5, "reasoning": f"Logical consistency validation failed: {e}"}

    async def _validate_output_quality(self, step: ExecutionStep, context: Dict[str, Any]) -> Dict[str, Any]:
        """Validate output quality using LangChain."""
        if ValidationStrategy.OUTPUT_QUALITY not in self.strategy_prompts:
            return {"score": 0.5, "reasoning": "Output quality strategy not available"}
        
        try:
            prompt = self.strategy_prompts[ValidationStrategy.OUTPUT_QUALITY].format_messages(**context)
            response = await self.llm.ainvoke(prompt)
            return self._parse_strategy_response(response.content, ValidationStrategy.OUTPUT_QUALITY)
        except Exception as e:
            return {"score": 0.5, "reasoning": f"Output quality validation failed: {e}"}

    async def _validate_state_coherence(self, step: ExecutionStep, context: Dict[str, Any]) -> Dict[str, Any]:
        """Validate state coherence using LangChain."""
        if not step.intermediate_state:
            return {"score": 1.0, "reasoning": "No state to validate"}
        
        try:
            # Create state-specific context
            state_context = context.copy()
            state_context["intermediate_state"] = str(step.intermediate_state)
            
            # Use a custom prompt for state coherence
            prompt = ChatPromptTemplate.from_messages([
                SystemMessage(content="Check state coherence for this AI agent step."),
                HumanMessage(content="""Current State: {intermediate_state}
Step: {operation}
Input: {input_summary}
Framework: {framework}

Assess:
1. State consistency before and after step
2. Proper state transitions
3. No state corruption
4. Appropriate state updates

Provide a score (0.0-1.0) and reasoning.""")
            ]).format_messages(**state_context)
            
            response = await self.llm.ainvoke(prompt)
            return self._parse_strategy_response(response.content, ValidationStrategy.STATE_COHERENCE)
        except Exception as e:
            return {"score": 0.5, "reasoning": f"State coherence validation failed: {e}"}

    async def _validate_safety_compliance(self, step: ExecutionStep, context: Dict[str, Any]) -> Dict[str, Any]:
        """Validate safety compliance using LangChain."""
        if ValidationStrategy.SAFETY_COMPLIANCE not in self.strategy_prompts:
            return {"score": 0.5, "reasoning": "Safety compliance strategy not available"}
        
        try:
            prompt = self.strategy_prompts[ValidationStrategy.SAFETY_COMPLIANCE].format_messages(**context)
            response = await self.llm.ainvoke(prompt)
            return self._parse_strategy_response(response.content, ValidationStrategy.SAFETY_COMPLIANCE)
        except Exception as e:
            return {"score": 0.5, "reasoning": f"Safety compliance validation failed: {e}"}

    async def _validate_performance_optimality(self, step: ExecutionStep, context: Dict[str, Any]) -> Dict[str, Any]:
        """Validate performance optimality using LangChain."""
        try:
            # Create performance-specific context
            perf_context = context.copy()
            perf_context["execution_time"] = step.execution_time or "Unknown"
            perf_context["memory_usage"] = step.memory_usage or "Unknown"
            perf_context["input_size"] = len(str(step.input_data))
            
            prompt = ChatPromptTemplate.from_messages([
                SystemMessage(content="Assess performance optimality for this AI agent step."),
                HumanMessage(content="""Step: {operation} on {component}
Input Size: {input_size} characters
Execution Time: {execution_time}
Memory Usage: {memory_usage}

Consider:
1. Is this the most efficient approach?
2. Are there faster alternatives?
3. Resource usage appropriateness
4. Scalability concerns
5. Bottleneck potential

Provide a score (0.0-1.0) and reasoning.""")
            ]).format_messages(**perf_context)
            
            response = await self.llm.ainvoke(prompt)
            return self._parse_strategy_response(response.content, ValidationStrategy.PERFORMANCE_OPTIMALITY)
        except Exception as e:
            return {"score": 0.5, "reasoning": f"Performance optimality validation failed: {e}"}

    def _parse_strategy_response(self, response: str, strategy: ValidationStrategy) -> Dict[str, Any]:
        """Parse strategy response to extract score and reasoning."""
        try:
            # Try to extract score and reasoning from response
            lines = response.strip().split('\n')
            score = 0.5  # Default
            reasoning = response[:200]  # First 200 chars as reasoning
            issues = []
            suggestions = []
            
            for line in lines:
                if "score" in line.lower():
                    # Try to extract numeric score
                    import re
                    numbers = re.findall(r'(\d+\.?\d*)', line)
                    if numbers:
                        score = max(0.0, min(1.0, float(numbers[0])))
                elif "reasoning" in line.lower():
                    reasoning = line.split(":", 1)[1].strip() if ":" in line else line
                elif "issue" in line.lower():
                    issues.append(line.strip())
                elif "suggestion" in line.lower():
                    suggestions.append(line.strip())
            
            return {
                "score": score,
                "reasoning": reasoning,
                "issues": issues,
                "suggestions": suggestions
            }
        except Exception as e:
            return {
                "score": 0.5,
                "reasoning": f"Failed to parse {strategy.value} response: {e}",
                "issues": ["Response parsing error"],
                "suggestions": []
            }

    def _create_basic_validation_prompt(self, step: ExecutionStep) -> str:
        """Create basic validation prompt for fallback."""
        return f"""
You are an expert AI agent execution validator. Your job is to judge whether this execution step is correct and appropriate.

AGENT CONTEXT:
- Goal: {step.agent_goal or "Not specified"}
- Framework: {step.framework}
- Component: {step.component}
- Operation: {step.operation}
- Step Reasoning: {step.step_reasoning or "Not provided"}

EXECUTION DETAILS:
- Input Data: {self._summarize_input_data(step.input_data)}
- Previous Conversation: {self._format_conversation_context(step.conversation_history)}
- Timestamp: {step.timestamp.isoformat()}

Please provide your judgment as JSON with the following structure:
{{
    "is_valid": true/false,
    "confidence": 0.0-1.0,
    "reasoning": "detailed explanation of your judgment",
    "issues": ["list of specific issues if any"],
    "suggestions": ["list of improvement suggestions"],
    "risk_level": "low/medium/high/critical"
}}

Be thorough but concise. Focus on practical correctness and safety.
"""

    def _parse_judgment_response(self, response: str) -> Dict[str, Any]:
        """Parse judgment response from LLM."""
        try:
            # Clean response - remove markdown code blocks if present
            cleaned_response = response.strip()
            if cleaned_response.startswith("```json"):
                cleaned_response = cleaned_response[7:]  # Remove ```json
            if cleaned_response.endswith("```"):
                cleaned_response = cleaned_response[:-3]  # Remove ```
            cleaned_response = cleaned_response.strip()
            
            return json.loads(cleaned_response)
        except json.JSONDecodeError:
            # Fallback parsing
            return {
                "is_valid": True,
                "confidence": 0.5,
                "reasoning": response[:200],
                "issues": [],
                "suggestions": [],
                "risk_level": "medium"
            }

    def _parse_risk_level(self, risk_level_str: str) -> RiskLevel:
        """Parse risk level string to enum."""
        risk_level_map = {
            "low": RiskLevel.LOW,
            "medium": RiskLevel.MEDIUM,
            "high": RiskLevel.HIGH,
            "critical": RiskLevel.CRITICAL
        }
        return risk_level_map.get(risk_level_str.lower(), RiskLevel.MEDIUM)
    
    def _summarize_input_data(self, input_data: Dict[str, Any]) -> str:
        """Create a concise summary of input data."""
        if not input_data:
            return "No input data"
        
        summary_parts = []
        for key, value in input_data.items():
            if isinstance(value, str):
                # Truncate long strings
                value_str = value[:100] + "..." if len(value) > 100 else value
                summary_parts.append(f"{key}: {value_str}")
            elif isinstance(value, (list, dict)):
                summary_parts.append(f"{key}: {type(value).__name__} (len={len(value)})")
            else:
                summary_parts.append(f"{key}: {str(value)[:50]}")
        
        return "; ".join(summary_parts)
    
    def _format_conversation_context(self, conversation_history: Optional[List[Dict[str, Any]]]) -> str:
        """Format conversation history for validation."""
        if not conversation_history:
            return "None"
        
        recent_messages = conversation_history[-3:]  # Last 3 messages
        return "\n".join([
            f"{msg.get('role', 'unknown')}: {msg.get('content', '')[:200]}"
            for msg in recent_messages
        ])
    
    def _infer_expected_output_type(self, step: ExecutionStep) -> str:
        """Infer expected output type based on component and operation."""
        component = step.component.lower()
        operation = step.operation.lower()
        
        if "llm" in component or "chat" in component:
            return "Text response"
        elif "tool" in component:
            return "Tool execution result"
        elif "chain" in component:
            return "Chain output"
        elif "retriever" in component:
            return "Retrieved documents"
        else:
            return "Unknown output type"
    
    def _get_cache_key(self, step: ExecutionStep) -> str:
        """Generate cache key for step."""
        key_data = {
            "framework": step.framework,
            "component": step.component,
            "operation": step.operation,
            "input_hash": hash(str(sorted(step.input_data.items()))),
            "goal": step.agent_goal
        }
        return f"validation_{hash(str(key_data))}"
    
    def _is_cache_valid(self, cached_result: Dict[str, Any]) -> bool:
        """Check if cached result is still valid."""
        cache_time = cached_result.get("timestamp", datetime.min)
        age = (datetime.now() - cache_time).total_seconds()
        return age < self.config.cache_ttl_seconds
    
    def _cache_result(self, cache_key: str, result: ValidationResult):
        """Cache validation result."""
        if len(self.validation_cache) >= self.config.max_cache_size:
            # Remove oldest entries
            oldest_key = min(self.validation_cache.keys(), 
                           key=lambda k: self.validation_cache[k]["timestamp"])
            del self.validation_cache[oldest_key]
        
        self.validation_cache[cache_key] = {
            "result": result,
            "timestamp": datetime.now()
        }
    
    def _update_validation_time(self, validation_time: float):
        """Update validation time metrics."""
        # This method is called by the callback handler
        pass

    def get_metrics(self) -> Dict[str, Any]:
        """Get current validation metrics."""
        return {
            "total_validations": self.metrics.total_validations,
            "successful_validations": self.metrics.successful_validations,
            "success_rate": self.metrics.successful_validations / max(1, self.metrics.total_validations),
            "avg_validation_time": self.metrics.avg_validation_time,
            "avg_confidence": self.metrics.avg_confidence,
            "cache_hit_rate": self.metrics.cache_hit_rate,
            "parallel_utilization": self.metrics.parallel_utilization,
            "pattern_matches": self.metrics.pattern_matches,
            "pattern_count": len(self.validation_patterns),
            "cache_size": len(self.validation_cache)
        }

    def clear_cache(self):
        """Clear validation cache."""
        self.validation_cache.clear()
        logging.info("Validation cache cleared")

    def clear_patterns(self):
        """Clear learned patterns."""
        self.validation_patterns.clear()
        self.validation_history.clear()
        logging.info("Validation patterns cleared")

    def shutdown(self):
        """Shutdown the validator and cleanup resources."""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)
        logging.info("AdvancedRuntimeValidator shutdown complete")
