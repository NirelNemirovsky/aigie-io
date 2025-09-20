"""Aigie-integrated version of the Open Deep Research agent."""

import asyncio
import os
import sys
from typing import Literal

# Add the aigie package to the path
sys.path.insert(0, '/Users/nirelnemirovsky/Documents/dev/aigie/aigie-io')

from aigie.core.error_handling.error_detector import ErrorDetector
from aigie.core.monitoring.monitoring import PerformanceMonitor
from aigie.utils.config import AigieConfig
from aigie.reporting.logger import AigieLogger

from langchain.chat_models import init_chat_model
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    filter_messages,
    get_buffer_string,
)
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from open_deep_research.configuration import (
    Configuration,
)
from open_deep_research.prompts import (
    clarify_with_user_instructions,
    compress_research_simple_human_message,
    compress_research_system_prompt,
    final_report_generation_prompt,
    lead_researcher_prompt,
    research_system_prompt,
    transform_messages_into_research_topic_prompt,
)
from open_deep_research.state import (
    AgentInputState,
    AgentState,
    ClarifyWithUser,
    ConductResearch,
    ResearchComplete,
    ResearcherOutputState,
    ResearcherState,
    ResearchQuestion,
    SupervisorState,
)
from open_deep_research.utils import (
    anthropic_websearch_called,
    get_all_tools,
    get_api_key_for_model,
    get_model_token_limit,
    get_notes_from_tool_calls,
    get_today_str,
    is_token_limit_exceeded,
    openai_websearch_called,
    remove_up_to_last_ai_message,
    think_tool,
)

# Initialize Aigie components
aigie_config = AigieConfig()
error_detector = ErrorDetector()
monitoring_system = PerformanceMonitor()
aigie_logger = AigieLogger()

# Initialize a configurable model that we will use throughout the agent
configurable_model = init_chat_model(
    configurable_fields=("model", "max_tokens", "api_key"),
)

async def aigie_enhanced_clarify_with_user(state: AgentState, config: RunnableConfig) -> Command[Literal["write_research_brief", "__end__"]]:
    """Aigie-enhanced version of clarify_with_user with runtime validation and error handling."""
    
    try:
        # Aigie: Start monitoring this step
        monitoring_system.start_step("clarify_with_user")
        
        # Step 1: Check if clarification is enabled in configuration
        configurable = Configuration.from_runnable_config(config)
        if not configurable.allow_clarification:
            # Skip clarification step and proceed directly to research
            return Command(goto="write_research_brief")
        
        # Step 2: Prepare the model for structured clarification analysis
        messages = state["messages"]
        model_config = {
            "model": configurable.research_model,
            "max_tokens": configurable.research_model_max_tokens,
            "api_key": get_api_key_for_model(configurable.research_model, config),
            "tags": ["langsmith:nostream"]
        }
        
        # Aigie: Validate model configuration
        validation_result = await runtime_validator.validate_model_config(model_config)
        if not validation_result.is_valid:
            monitoring_system.log_error("clarify_with_user", f"Model validation failed: {validation_result.errors}")
            return Command(
                goto=END, 
                update={"messages": [AIMessage(content="Error: Invalid model configuration. Please check your API keys and model settings.")]}
            )
        
        # Configure model with structured output and retry logic
        clarification_model = (
            configurable_model
            .with_structured_output(ClarifyWithUser)
            .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
            .with_config(model_config)
        )
        
        # Step 3: Analyze whether clarification is needed
        prompt_content = clarify_with_user_instructions.format(
            messages=get_buffer_string(messages), 
            date=get_today_str()
        )
        
        # Aigie: Validate prompt before sending
        prompt_validation = await runtime_validator.validate_prompt(prompt_content)
        if not prompt_validation.is_valid:
            monitoring_system.log_error("clarify_with_user", f"Prompt validation failed: {prompt_validation.errors}")
            return Command(
                goto=END, 
                update={"messages": [AIMessage(content="Error: Invalid prompt format. Please try again.")]}
            )
        
        response = await clarification_model.ainvoke([HumanMessage(content=prompt_content)])
        
        # Aigie: Validate response structure
        response_validation = await runtime_validator.validate_structured_output(response, ClarifyWithUser)
        if not response_validation.is_valid:
            monitoring_system.log_error("clarify_with_user", f"Response validation failed: {response_validation.errors}")
            return Command(
                goto=END, 
                update={"messages": [AIMessage(content="Error: Invalid response format. Please try again.")]}
            )
        
        # Step 4: Route based on clarification analysis
        if response.need_clarification:
            # End with clarifying question for user
            monitoring_system.complete_step("clarify_with_user", success=True)
            return Command(
                goto=END, 
                update={"messages": [AIMessage(content=response.question)]}
            )
        else:
            # Proceed to research with verification message
            monitoring_system.complete_step("clarify_with_user", success=True)
            return Command(
                goto="write_research_brief", 
                update={"messages": [AIMessage(content=response.verification)]}
            )
            
    except Exception as e:
        # Aigie: Enhanced error handling
        error_context = await error_detector.extract_error_context(e, state, config)
        monitoring_system.log_error("clarify_with_user", f"Unexpected error: {str(e)}", error_context)
        
        # Try to recover gracefully
        return Command(
            goto=END, 
            update={"messages": [AIMessage(content=f"I encountered an error while processing your request: {str(e)}. Please try again with a more specific question.")]}
        )

async def aigie_enhanced_write_research_brief(state: AgentState, config: RunnableConfig) -> Command[Literal["research_supervisor"]]:
    """Aigie-enhanced version of write_research_brief with runtime validation and error handling."""
    
    try:
        # Aigie: Start monitoring this step
        monitoring_system.start_step("write_research_brief")
        
        # Step 1: Set up the research model for structured output
        configurable = Configuration.from_runnable_config(config)
        research_model_config = {
            "model": configurable.research_model,
            "max_tokens": configurable.research_model_max_tokens,
            "api_key": get_api_key_for_model(configurable.research_model, config),
            "tags": ["langsmith:nostream"]
        }
        
        # Aigie: Validate model configuration
        validation_result = await runtime_validator.validate_model_config(research_model_config)
        if not validation_result.is_valid:
            monitoring_system.log_error("write_research_brief", f"Model validation failed: {validation_result.errors}")
            return Command(
                goto=END, 
                update={"messages": [AIMessage(content="Error: Invalid model configuration. Please check your API keys and model settings.")]}
            )
        
        # Configure model for structured research question generation
        research_model = (
            configurable_model
            .with_structured_output(ResearchQuestion)
            .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
            .with_config(research_model_config)
        )
        
        # Step 2: Generate structured research brief from user messages
        prompt_content = transform_messages_into_research_topic_prompt.format(
            messages=get_buffer_string(state.get("messages", [])),
            date=get_today_str()
        )
        
        # Aigie: Validate prompt before sending
        prompt_validation = await runtime_validator.validate_prompt(prompt_content)
        if not prompt_validation.is_valid:
            monitoring_system.log_error("write_research_brief", f"Prompt validation failed: {prompt_validation.errors}")
            return Command(
                goto=END, 
                update={"messages": [AIMessage(content="Error: Invalid prompt format. Please try again.")]}
            )
        
        response = await research_model.ainvoke([HumanMessage(content=prompt_content)])
        
        # Aigie: Validate response structure
        response_validation = await runtime_validator.validate_structured_output(response, ResearchQuestion)
        if not response_validation.is_valid:
            monitoring_system.log_error("write_research_brief", f"Response validation failed: {response_validation.errors}")
            return Command(
                goto=END, 
                update={"messages": [AIMessage(content="Error: Invalid response format. Please try again.")]}
            )
        
        # Step 3: Initialize supervisor with research brief and instructions
        supervisor_system_prompt = lead_researcher_prompt.format(
            date=get_today_str(),
            max_concurrent_research_units=configurable.max_concurrent_research_units,
            max_researcher_iterations=configurable.max_researcher_iterations
        )
        
        monitoring_system.complete_step("write_research_brief", success=True)
        return Command(
            goto="research_supervisor", 
            update={
                "research_brief": response.research_brief,
                "supervisor_messages": {
                    "type": "override",
                    "value": [
                        SystemMessage(content=supervisor_system_prompt),
                        HumanMessage(content=response.research_brief)
                    ]
                }
            }
        )
        
    except Exception as e:
        # Aigie: Enhanced error handling
        error_context = await error_detector.extract_error_context(e, state, config)
        monitoring_system.log_error("write_research_brief", f"Unexpected error: {str(e)}", error_context)
        
        # Try to recover gracefully
        return Command(
            goto=END, 
            update={"messages": [AIMessage(content=f"I encountered an error while creating the research brief: {str(e)}. Please try again with a more specific question.")]}
        )

async def aigie_enhanced_supervisor(state: SupervisorState, config: RunnableConfig) -> Command[Literal["supervisor_tools"]]:
    """Aigie-enhanced version of supervisor with runtime validation and error handling."""
    
    try:
        # Aigie: Start monitoring this step
        monitoring_system.start_step("supervisor")
        
        # Step 1: Configure the supervisor model with available tools
        configurable = Configuration.from_runnable_config(config)
        research_model_config = {
            "model": configurable.research_model,
            "max_tokens": configurable.research_model_max_tokens,
            "api_key": get_api_key_for_model(configurable.research_model, config),
            "tags": ["langsmith:nostream"]
        }
        
        # Aigie: Validate model configuration
        validation_result = await runtime_validator.validate_model_config(research_model_config)
        if not validation_result.is_valid:
            monitoring_system.log_error("supervisor", f"Model validation failed: {validation_result.errors}")
            return Command(
                goto=END, 
                update={"messages": [AIMessage(content="Error: Invalid model configuration. Please check your API keys and model settings.")]}
            )
        
        # Available tools: research delegation, completion signaling, and strategic thinking
        lead_researcher_tools = [ConductResearch, ResearchComplete, think_tool]
        
        # Configure model with tools, retry logic, and model settings
        research_model = (
            configurable_model
            .bind_tools(lead_researcher_tools)
            .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
            .with_config(research_model_config)
        )
        
        # Step 2: Generate supervisor response based on current context
        supervisor_messages = state.get("supervisor_messages", [])
        
        # Aigie: Validate messages before processing
        messages_validation = await runtime_validator.validate_messages(supervisor_messages)
        if not messages_validation.is_valid:
            monitoring_system.log_error("supervisor", f"Messages validation failed: {messages_validation.errors}")
            return Command(
                goto=END, 
                update={"messages": [AIMessage(content="Error: Invalid message format. Please try again.")]}
            )
        
        response = await research_model.ainvoke(supervisor_messages)
        
        # Aigie: Validate response
        response_validation = await runtime_validator.validate_ai_response(response)
        if not response_validation.is_valid:
            monitoring_system.log_error("supervisor", f"Response validation failed: {response_validation.errors}")
            return Command(
                goto=END, 
                update={"messages": [AIMessage(content="Error: Invalid response format. Please try again.")]}
            )
        
        # Step 3: Update state and proceed to tool execution
        monitoring_system.complete_step("supervisor", success=True)
        return Command(
            goto="supervisor_tools",
            update={
                "supervisor_messages": [response],
                "research_iterations": state.get("research_iterations", 0) + 1
            }
        )
        
    except Exception as e:
        # Aigie: Enhanced error handling
        error_context = await error_detector.extract_error_context(e, state, config)
        monitoring_system.log_error("supervisor", f"Unexpected error: {str(e)}", error_context)
        
        # Try to recover gracefully
        return Command(
            goto=END, 
            update={"messages": [AIMessage(content=f"I encountered an error while supervising research: {str(e)}. Please try again.")]}
        )

async def aigie_enhanced_final_report_generation(state: AgentState, config: RunnableConfig):
    """Aigie-enhanced version of final_report_generation with runtime validation and error handling."""
    
    try:
        # Aigie: Start monitoring this step
        monitoring_system.start_step("final_report_generation")
        
        # Step 1: Extract research findings and prepare state cleanup
        notes = state.get("notes", [])
        cleared_state = {"notes": {"type": "override", "value": []}}
        findings = "\n".join(notes)
        
        # Aigie: Validate findings before processing
        findings_validation = await runtime_validator.validate_content(findings)
        if not findings_validation.is_valid:
            monitoring_system.log_error("final_report_generation", f"Findings validation failed: {findings_validation.errors}")
            return {
                "final_report": "Error: Invalid research findings format. Please try again.",
                "messages": [AIMessage(content="Report generation failed due to invalid findings")],
                **cleared_state
            }
        
        # Step 2: Configure the final report generation model
        configurable = Configuration.from_runnable_config(config)
        writer_model_config = {
            "model": configurable.final_report_model,
            "max_tokens": configurable.final_report_model_max_tokens,
            "api_key": get_api_key_for_model(configurable.final_report_model, config),
            "tags": ["langsmith:nostream"]
        }
        
        # Aigie: Validate model configuration
        validation_result = await runtime_validator.validate_model_config(writer_model_config)
        if not validation_result.is_valid:
            monitoring_system.log_error("final_report_generation", f"Model validation failed: {validation_result.errors}")
            return {
                "final_report": "Error: Invalid model configuration. Please check your API keys and model settings.",
                "messages": [AIMessage(content="Report generation failed due to model configuration error")],
                **cleared_state
            }
        
        # Step 3: Attempt report generation with token limit retry logic
        max_retries = 3
        current_retry = 0
        findings_token_limit = None
        
        while current_retry <= max_retries:
            try:
                # Create comprehensive prompt with all research context
                final_report_prompt = final_report_generation_prompt.format(
                    research_brief=state.get("research_brief", ""),
                    messages=get_buffer_string(state.get("messages", [])),
                    findings=findings,
                    date=get_today_str()
                )
                
                # Aigie: Validate prompt before sending
                prompt_validation = await runtime_validator.validate_prompt(final_report_prompt)
                if not prompt_validation.is_valid:
                    monitoring_system.log_error("final_report_generation", f"Prompt validation failed: {prompt_validation.errors}")
                    return {
                        "final_report": "Error: Invalid prompt format. Please try again.",
                        "messages": [AIMessage(content="Report generation failed due to invalid prompt")],
                        **cleared_state
                    }
                
                # Generate the final report
                final_report = await configurable_model.with_config(writer_model_config).ainvoke([
                    HumanMessage(content=final_report_prompt)
                ])
                
                # Aigie: Validate final report
                report_validation = await runtime_validator.validate_content(final_report.content)
                if not report_validation.is_valid:
                    monitoring_system.log_error("final_report_generation", f"Report validation failed: {report_validation.errors}")
                    return {
                        "final_report": "Error: Generated report failed validation. Please try again.",
                        "messages": [AIMessage(content="Report generation failed due to validation error")],
                        **cleared_state
                    }
                
                # Return successful report generation
                monitoring_system.complete_step("final_report_generation", success=True)
                return {
                    "final_report": final_report.content, 
                    "messages": [final_report],
                    **cleared_state
                }
                
            except Exception as e:
                # Handle token limit exceeded errors with progressive truncation
                if is_token_limit_exceeded(e, configurable.final_report_model):
                    current_retry += 1
                    
                    if current_retry == 1:
                        # First retry: determine initial truncation limit
                        model_token_limit = get_model_token_limit(configurable.final_report_model)
                        if not model_token_limit:
                            monitoring_system.log_error("final_report_generation", "Could not determine model token limit")
                            return {
                                "final_report": f"Error generating final report: Token limit exceeded, however, we could not determine the model's maximum context length. Please update the model map in deep_researcher/utils.py with this information. {e}",
                                "messages": [AIMessage(content="Report generation failed due to token limits")],
                                **cleared_state
                            }
                        # Use 4x token limit as character approximation for truncation
                        findings_token_limit = model_token_limit * 4
                    else:
                        # Subsequent retries: reduce by 10% each time
                        findings_token_limit = int(findings_token_limit * 0.9)
                    
                    # Truncate findings and retry
                    findings = findings[:findings_token_limit]
                    continue
                else:
                    # Non-token-limit error: return error immediately
                    monitoring_system.log_error("final_report_generation", f"Non-token-limit error: {str(e)}")
                    return {
                        "final_report": f"Error generating final report: {e}",
                        "messages": [AIMessage(content="Report generation failed due to an error")],
                        **cleared_state
                    }
        
        # Step 4: Return failure result if all retries exhausted
        monitoring_system.log_error("final_report_generation", "Maximum retries exceeded")
        return {
            "final_report": "Error generating final report: Maximum retries exceeded",
            "messages": [AIMessage(content="Report generation failed after maximum retries")],
            **cleared_state
        }
        
    except Exception as e:
        # Aigie: Enhanced error handling
        error_context = await error_detector.extract_error_context(e, state, config)
        monitoring_system.log_error("final_report_generation", f"Unexpected error: {str(e)}", error_context)
        
        # Try to recover gracefully
        return {
            "final_report": f"I encountered an error while generating the final report: {str(e)}. Please try again.",
            "messages": [AIMessage(content="Report generation failed due to unexpected error")],
            **cleared_state
        }

# Main Aigie-Enhanced Deep Researcher Graph Construction
# Creates the complete deep research workflow with Aigie integration
aigie_deep_researcher_builder = StateGraph(
    AgentState, 
    input=AgentInputState, 
    config_schema=Configuration
)

# Add main workflow nodes for the complete research process with Aigie enhancements
aigie_deep_researcher_builder.add_node("clarify_with_user", aigie_enhanced_clarify_with_user)
aigie_deep_researcher_builder.add_node("write_research_brief", aigie_enhanced_write_research_brief)
aigie_deep_researcher_builder.add_node("final_report_generation", aigie_enhanced_final_report_generation)

# Define main workflow edges for sequential execution
aigie_deep_researcher_builder.add_edge(START, "clarify_with_user")
aigie_deep_researcher_builder.add_edge("final_report_generation", END)

# Compile the complete Aigie-enhanced deep researcher workflow
aigie_deep_researcher = aigie_deep_researcher_builder.compile()

# Export the monitoring system for external access
def get_monitoring_system():
    """Get the monitoring system instance for external access."""
    return monitoring_system

def get_aigie_instance():
    """Get the Aigie instance for external access."""
    return aigie
