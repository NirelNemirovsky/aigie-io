"""
Step Corrector - Intelligent auto-remediation for invalid execution steps.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime

from ..ai.gemini_analyzer import GeminiAnalyzer
from ..types.validation_types import (
    ExecutionStep, ValidationResult, CorrectionResult, 
    CorrectionStrategy, ValidationStatus
)


class StepCorrector:
    """Handles automatic correction of invalid steps using LLM intelligence."""
    
    def __init__(self, gemini_analyzer: GeminiAnalyzer):
        self.gemini_analyzer = gemini_analyzer
        self.correction_strategies = {
            CorrectionStrategy.PARAMETER_ADJUSTMENT: self._adjust_parameters,
            CorrectionStrategy.PROMPT_REFINEMENT: self._refine_prompts,
            CorrectionStrategy.STATE_RESTORATION: self._restore_state,
            CorrectionStrategy.TOOL_SUBSTITUTION: self._substitute_tools,
            CorrectionStrategy.LOGIC_REPAIR: self._repair_logic,
            CorrectionStrategy.GOAL_REALIGNMENT: self._realign_goal
        }
        
        # Correction history for learning
        self.correction_history = []
        self.successful_patterns = {}
        
        logging.info("StepCorrector initialized with intelligent remediation capabilities")
    
    async def correct_step(self, step: ExecutionStep, 
                          validation_result: ValidationResult) -> CorrectionResult:
        """Attempt to automatically correct an invalid step."""
        
        start_time = datetime.now()
        correction_attempts = 0
        max_attempts = 3
        
        try:
            # Analyze failure modes
            failure_analysis = await self._analyze_failure(step, validation_result)
            logging.info(f"Analyzed failure for step {step.step_id}: {failure_analysis['primary_issue']}")
            
            # Select appropriate correction strategy
            strategy = self._select_correction_strategy(failure_analysis)
            logging.info(f"Selected correction strategy: {strategy.value}")
            
            # Apply correction with retries
            corrected_step = step
            last_error = None
            
            for attempt in range(max_attempts):
                correction_attempts += 1
                
                try:
                    corrected_step = await self.correction_strategies[strategy](step, failure_analysis)
                    
                    # Validate the correction
                    validation_result = await self._validate_correction(corrected_step, step)
                    
                    if validation_result.is_valid:
                        # Success!
                        correction_time = (datetime.now() - start_time).total_seconds()
                        
                        result = CorrectionResult(
                            original_step=step,
                            corrected_step=corrected_step,
                            correction_strategy=strategy,
                            validation_result=validation_result,
                            success=True,
                            correction_reasoning=failure_analysis['correction_reasoning'],
                            correction_attempts=correction_attempts
                        )
                        
                        # Store successful pattern
                        self._store_successful_pattern(step, corrected_step, strategy)
                        
                        logging.info(f"Step {step.step_id} corrected successfully in {correction_time:.2f}s using {strategy.value}")
                        return result
                    
                    else:
                        # Correction didn't work, try different approach
                        logging.warning(f"Correction attempt {attempt + 1} failed validation: {validation_result.reasoning}")
                        last_error = validation_result.reasoning
                        
                        # Try alternative strategy
                        strategy = self._select_alternative_strategy(failure_analysis, strategy)
                        
                except Exception as e:
                    logging.warning(f"Correction attempt {attempt + 1} failed: {e}")
                    last_error = str(e)
                    continue
            
            # All attempts failed
            correction_time = (datetime.now() - start_time).total_seconds()
            
            result = CorrectionResult(
                original_step=step,
                corrected_step=None,
                correction_strategy=strategy,
                validation_result=validation_result,
                success=False,
                correction_reasoning=f"All correction attempts failed. Last error: {last_error}",
                correction_attempts=correction_attempts
            )
            
            logging.error(f"Failed to correct step {step.step_id} after {correction_attempts} attempts in {correction_time:.2f}s")
            return result
            
        except Exception as e:
            logging.error(f"Step correction failed for {step.step_id}: {e}")
            return CorrectionResult(
                original_step=step,
                corrected_step=None,
                correction_strategy=None,
                validation_result=validation_result,
                success=False,
                correction_reasoning=f"Correction system error: {str(e)}",
                correction_attempts=correction_attempts
            )
    
    async def _analyze_failure(self, step: ExecutionStep, validation_result: ValidationResult) -> Dict[str, Any]:
        """Analyze why the step failed and determine correction approach."""
        
        prompt = f"""
Analyze this failed AI agent step and determine the best correction approach:

ORIGINAL STEP:
- Framework: {step.framework}
- Component: {step.component}
- Operation: {step.operation}
- Input: {step.input_data}
- Goal: {step.agent_goal or "Not specified"}
- Reasoning: {step.step_reasoning or "Not provided"}

VALIDATION FAILURE:
- Issues: {validation_result.issues}
- Suggestions: {validation_result.suggestions}
- Risk Level: {validation_result.risk_level.value}

ANALYSIS REQUIRED:
1. What is the primary issue causing the failure?
2. What type of correction would be most effective?
3. What specific changes should be made?
4. What is the expected outcome after correction?

Provide your analysis as JSON:
{{
    "primary_issue": "description of main problem",
    "issue_category": "tool_mismatch|parameter_error|logic_error|goal_misalignment|state_error",
    "correction_approach": "parameter_adjustment|prompt_refinement|tool_substitution|logic_repair|goal_realignment|state_restoration",
    "specific_changes": {{
        "component": "suggested component",
        "operation": "suggested operation", 
        "input_data": "suggested input modifications",
        "reasoning": "suggested reasoning"
    }},
    "confidence": 0.0-1.0,
    "correction_reasoning": "detailed explanation of correction approach"
}}
"""
        
        try:
            response = await self.gemini_analyzer._generate_content_async(prompt)
            import json
            
            # Clean response - remove markdown code blocks if present
            cleaned_response = response.strip()
            if cleaned_response.startswith("```json"):
                cleaned_response = cleaned_response[7:]  # Remove ```json
            if cleaned_response.endswith("```"):
                cleaned_response = cleaned_response[:-3]  # Remove ```
            cleaned_response = cleaned_response.strip()
            
            analysis = json.loads(cleaned_response)
            
            # Validate required fields
            required_fields = ["primary_issue", "issue_category", "correction_approach", "specific_changes", "confidence", "correction_reasoning"]
            for field in required_fields:
                if field not in analysis:
                    analysis[field] = self._get_default_analysis_field(field)
            
            return analysis
            
        except Exception as e:
            logging.warning(f"Failure analysis failed: {e}")
            return self._get_fallback_analysis(step, validation_result)
    
    def _select_correction_strategy(self, failure_analysis: Dict[str, Any]) -> CorrectionStrategy:
        """Select the best correction strategy based on failure analysis."""
        
        approach = failure_analysis.get("correction_approach", "parameter_adjustment")
        
        # Map approach to strategy
        approach_map = {
            "parameter_adjustment": CorrectionStrategy.PARAMETER_ADJUSTMENT,
            "prompt_refinement": CorrectionStrategy.PROMPT_REFINEMENT,
            "tool_substitution": CorrectionStrategy.TOOL_SUBSTITUTION,
            "logic_repair": CorrectionStrategy.LOGIC_REPAIR,
            "goal_realignment": CorrectionStrategy.GOAL_REALIGNMENT,
            "state_restoration": CorrectionStrategy.STATE_RESTORATION
        }
        
        return approach_map.get(approach, CorrectionStrategy.PARAMETER_ADJUSTMENT)
    
    def _select_alternative_strategy(self, failure_analysis: Dict[str, Any], 
                                   current_strategy: CorrectionStrategy) -> CorrectionStrategy:
        """Select an alternative strategy if the current one fails."""
        
        # Priority order for alternative strategies
        alternatives = {
            CorrectionStrategy.PARAMETER_ADJUSTMENT: [
                CorrectionStrategy.PROMPT_REFINEMENT,
                CorrectionStrategy.TOOL_SUBSTITUTION,
                CorrectionStrategy.LOGIC_REPAIR
            ],
            CorrectionStrategy.PROMPT_REFINEMENT: [
                CorrectionStrategy.PARAMETER_ADJUSTMENT,
                CorrectionStrategy.TOOL_SUBSTITUTION,
                CorrectionStrategy.LOGIC_REPAIR
            ],
            CorrectionStrategy.TOOL_SUBSTITUTION: [
                CorrectionStrategy.PARAMETER_ADJUSTMENT,
                CorrectionStrategy.PROMPT_REFINEMENT,
                CorrectionStrategy.LOGIC_REPAIR
            ],
            CorrectionStrategy.LOGIC_REPAIR: [
                CorrectionStrategy.GOAL_REALIGNMENT,
                CorrectionStrategy.PARAMETER_ADJUSTMENT,
                CorrectionStrategy.TOOL_SUBSTITUTION
            ],
            CorrectionStrategy.GOAL_REALIGNMENT: [
                CorrectionStrategy.LOGIC_REPAIR,
                CorrectionStrategy.PARAMETER_ADJUSTMENT
            ],
            CorrectionStrategy.STATE_RESTORATION: [
                CorrectionStrategy.PARAMETER_ADJUSTMENT,
                CorrectionStrategy.LOGIC_REPAIR
            ]
        }
        
        alt_list = alternatives.get(current_strategy, [CorrectionStrategy.PARAMETER_ADJUSTMENT])
        return alt_list[0] if alt_list else CorrectionStrategy.PARAMETER_ADJUSTMENT
    
    async def _adjust_parameters(self, step: ExecutionStep, failure_analysis: Dict[str, Any]) -> ExecutionStep:
        """Adjust parameters of the step."""
        
        specific_changes = failure_analysis.get("specific_changes", {})
        
        # Create corrected step
        corrected_step = ExecutionStep(
            step_id=f"{step.step_id}_corrected",
            timestamp=datetime.now(),
            framework=step.framework,
            component=specific_changes.get("component", step.component),
            operation=specific_changes.get("operation", step.operation),
            input_data=self._adjust_input_data(step.input_data, specific_changes),
            agent_goal=step.agent_goal,
            conversation_history=step.conversation_history,
            step_reasoning=specific_changes.get("reasoning", step.step_reasoning),
            parent_step_id=step.step_id
        )
        
        return corrected_step
    
    async def _refine_prompts(self, step: ExecutionStep, failure_analysis: Dict[str, Any]) -> ExecutionStep:
        """Refine prompts and input data."""
        
        specific_changes = failure_analysis.get("specific_changes", {})
        
        # Refine input data based on suggestions
        refined_input = self._refine_input_data(step.input_data, failure_analysis)
        
        corrected_step = ExecutionStep(
            step_id=f"{step.step_id}_corrected",
            timestamp=datetime.now(),
            framework=step.framework,
            component=step.component,
            operation=step.operation,
            input_data=refined_input,
            agent_goal=step.agent_goal,
            conversation_history=step.conversation_history,
            step_reasoning=specific_changes.get("reasoning", step.step_reasoning),
            parent_step_id=step.step_id
        )
        
        return corrected_step
    
    async def _substitute_tools(self, step: ExecutionStep, failure_analysis: Dict[str, Any]) -> ExecutionStep:
        """Substitute the tool/component with a more appropriate one."""
        
        specific_changes = failure_analysis.get("specific_changes", {})
        
        # Map common tool substitutions
        tool_substitutions = {
            "DatabaseQueryTool": "ChatOpenAI",
            "WebSearchTool": "ChatOpenAI", 
            "CalculatorTool": "ChatOpenAI",
            "FileSystemTool": "ChatOpenAI"
        }
        
        suggested_component = specific_changes.get("component", step.component)
        if suggested_component in tool_substitutions:
            suggested_component = tool_substitutions[suggested_component]
        
        corrected_step = ExecutionStep(
            step_id=f"{step.step_id}_corrected",
            timestamp=datetime.now(),
            framework=step.framework,
            component=suggested_component,
            operation=specific_changes.get("operation", "invoke"),
            input_data=self._adapt_input_for_tool(step.input_data, suggested_component),
            agent_goal=step.agent_goal,
            conversation_history=step.conversation_history,
            step_reasoning=specific_changes.get("reasoning", step.step_reasoning),
            parent_step_id=step.step_id
        )
        
        return corrected_step
    
    async def _repair_logic(self, step: ExecutionStep, failure_analysis: Dict[str, Any]) -> ExecutionStep:
        """Repair the logical flow of the step."""
        
        specific_changes = failure_analysis.get("specific_changes", {})
        
        # Repair the reasoning and approach
        repaired_reasoning = self._repair_step_reasoning(step, failure_analysis)
        
        corrected_step = ExecutionStep(
            step_id=f"{step.step_id}_corrected",
            timestamp=datetime.now(),
            framework=step.framework,
            component=step.component,
            operation=step.operation,
            input_data=step.input_data,
            agent_goal=step.agent_goal,
            conversation_history=step.conversation_history,
            step_reasoning=repaired_reasoning,
            parent_step_id=step.step_id
        )
        
        return corrected_step
    
    async def _realign_goal(self, step: ExecutionStep, failure_analysis: Dict[str, Any]) -> ExecutionStep:
        """Realign the step with the agent's goal."""
        
        specific_changes = failure_analysis.get("specific_changes", {})
        
        # Adjust goal alignment
        realigned_goal = specific_changes.get("goal", step.agent_goal)
        realigned_reasoning = self._create_goal_aligned_reasoning(step, realigned_goal)
        
        corrected_step = ExecutionStep(
            step_id=f"{step.step_id}_corrected",
            timestamp=datetime.now(),
            framework=step.framework,
            component=step.component,
            operation=step.operation,
            input_data=step.input_data,
            agent_goal=realigned_goal,
            conversation_history=step.conversation_history,
            step_reasoning=realigned_reasoning,
            parent_step_id=step.step_id
        )
        
        return corrected_step
    
    async def _restore_state(self, step: ExecutionStep, failure_analysis: Dict[str, Any]) -> ExecutionStep:
        """Restore or fix the agent state."""
        
        # For now, just return the original step with updated reasoning
        corrected_step = ExecutionStep(
            step_id=f"{step.step_id}_corrected",
            timestamp=datetime.now(),
            framework=step.framework,
            component=step.component,
            operation=step.operation,
            input_data=step.input_data,
            agent_goal=step.agent_goal,
            conversation_history=step.conversation_history,
            step_reasoning="State restored - retrying with clean state",
            parent_step_id=step.step_id,
            intermediate_state={}  # Clear state
        )
        
        return corrected_step
    
    async def _validate_correction(self, corrected_step: ExecutionStep, 
                                 original_step: ExecutionStep) -> ValidationResult:
        """Validate that the correction is appropriate."""
        
        # Simple validation - check if the correction addresses the main issues
        # In a real implementation, this would use the RuntimeValidator
        
        # Basic checks
        if not corrected_step.component or not corrected_step.operation:
            return ValidationResult(
                is_valid=False,
                confidence=0.0,
                reasoning="Correction resulted in invalid component or operation",
                issues=["Invalid correction"],
                suggestions=["Check correction logic"]
            )
        
        # Check if correction is different from original
        if (corrected_step.component == original_step.component and 
            corrected_step.operation == original_step.operation and
            corrected_step.input_data == original_step.input_data):
            return ValidationResult(
                is_valid=False,
                confidence=0.0,
                reasoning="Correction did not change the step",
                issues=["No changes made"],
                suggestions=["Apply meaningful corrections"]
            )
        
        # Basic validation passed
        return ValidationResult(
            is_valid=True,
            confidence=0.7,
            reasoning="Correction appears to address the identified issues",
            issues=[],
            suggestions=[]
        )
    
    def _adjust_input_data(self, input_data: Dict[str, Any], 
                          specific_changes: Dict[str, Any]) -> Dict[str, Any]:
        """Adjust input data based on specific changes."""
        
        adjusted_input = input_data.copy()
        
        # Apply specific changes
        if "input_data" in specific_changes:
            changes = specific_changes["input_data"]
            if isinstance(changes, dict):
                adjusted_input.update(changes)
        
        return adjusted_input
    
    def _refine_input_data(self, input_data: Dict[str, Any], 
                          failure_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Refine input data based on failure analysis."""
        
        refined_input = input_data.copy()
        
        # Add context or improve prompts
        if "messages" in refined_input and isinstance(refined_input["messages"], list):
            for message in refined_input["messages"]:
                if message.get("role") == "user":
                    # Enhance user message with context
                    message["content"] = f"Context: {failure_analysis.get('correction_reasoning', '')}\n\n{message['content']}"
        
        return refined_input
    
    def _adapt_input_for_tool(self, input_data: Dict[str, Any], 
                             new_component: str) -> Dict[str, Any]:
        """Adapt input data for a different tool/component."""
        
        if "ChatOpenAI" in new_component:
            # Convert to chat format
            if "query" in input_data:
                return {
                    "messages": [{"role": "user", "content": input_data["query"]}]
                }
            elif "text" in input_data:
                return {
                    "messages": [{"role": "user", "content": input_data["text"]}]
                }
        
        return input_data
    
    def _repair_step_reasoning(self, step: ExecutionStep, 
                              failure_analysis: Dict[str, Any]) -> str:
        """Repair the step reasoning."""
        
        original_reasoning = step.step_reasoning or "No reasoning provided"
        correction_reasoning = failure_analysis.get("correction_reasoning", "")
        
        return f"REPAIRED: {original_reasoning}\n\nCorrection: {correction_reasoning}"
    
    def _create_goal_aligned_reasoning(self, step: ExecutionStep, goal: str) -> str:
        """Create reasoning that aligns with the goal."""
        
        return f"Goal-aligned reasoning: {goal}\n\nStep: {step.operation} on {step.component} to achieve this goal."
    
    def _store_successful_pattern(self, original_step: ExecutionStep, 
                                 corrected_step: ExecutionStep, 
                                 strategy: CorrectionStrategy):
        """Store successful correction patterns for learning."""
        
        pattern_key = f"{original_step.component}_{original_step.operation}"
        
        if pattern_key not in self.successful_patterns:
            self.successful_patterns[pattern_key] = []
        
        pattern = {
            "original_component": original_step.component,
            "corrected_component": corrected_step.component,
            "strategy": strategy.value,
            "timestamp": datetime.now(),
            "success": True
        }
        
        self.successful_patterns[pattern_key].append(pattern)
        
        # Keep only recent patterns
        if len(self.successful_patterns[pattern_key]) > 10:
            self.successful_patterns[pattern_key] = self.successful_patterns[pattern_key][-10:]
    
    def _get_default_analysis_field(self, field: str) -> Any:
        """Get default value for missing analysis field."""
        defaults = {
            "primary_issue": "Unknown issue",
            "issue_category": "logic_error",
            "correction_approach": "parameter_adjustment",
            "specific_changes": {},
            "confidence": 0.5,
            "correction_reasoning": "Default correction approach"
        }
        return defaults.get(field, None)
    
    def _get_fallback_analysis(self, step: ExecutionStep, 
                              validation_result: ValidationResult) -> Dict[str, Any]:
        """Get fallback analysis when LLM analysis fails."""
        
        return {
            "primary_issue": "Analysis failed, using fallback",
            "issue_category": "logic_error",
            "correction_approach": "parameter_adjustment",
            "specific_changes": {
                "component": step.component,
                "operation": step.operation,
                "reasoning": "Fallback correction"
            },
            "confidence": 0.3,
            "correction_reasoning": "Using fallback correction approach due to analysis failure"
        }
