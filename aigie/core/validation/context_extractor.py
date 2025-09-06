"""
Intelligent context extraction using Gemini as an LLM agent.
This module uses LangChain and Gemini to intelligently infer agent goals and context.
"""

import json
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging

from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from ..types.validation_types import ExecutionStep
from ..ai.gemini_analyzer import GeminiAnalyzer


class ContextExtractor:
    """Intelligent context extraction using Gemini as an LLM agent."""
    
    def __init__(self, gemini_analyzer: Optional[GeminiAnalyzer] = None, enable_llm: bool = True):
        self.logger = logging.getLogger(__name__)
        self.gemini_analyzer = gemini_analyzer
        self.agent_executor = None
        
        # Initialize LangChain agent for context extraction if enabled
        if enable_llm:
            self._initialize_agent()
        else:
            self.logger.info("Context extraction agent disabled - will use fallback mode")
    
    def _initialize_agent(self):
        """Initialize the LangChain agent for context extraction."""
        try:
            # Initialize Gemini LLM with API key authentication
            import os
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY not found in environment")
                
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                temperature=0.1,  # Low temperature for consistent analysis
                convert_system_message_to_human=True,
                google_api_key=api_key
            )
            
            # Create context extraction tools
            self.tools = self._create_context_tools()
            
            # Create the agent prompt
            self.prompt = self._create_agent_prompt()
            
            # Create the ReAct agent
            self.agent = create_react_agent(
                llm=self.llm,
                tools=self.tools,
                prompt=self.prompt
            )
            
            # Create agent executor
            self.agent_executor = AgentExecutor(
                agent=self.agent,
                tools=self.tools,
                verbose=False,
                handle_parsing_errors=True,
                max_iterations=3
            )
            
            self.logger.info("Context extraction agent initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize context extraction agent: {e}")
            self.agent_executor = None
    
    def extract_context(self, step: ExecutionStep) -> ExecutionStep:
        """Extract and infer context using the LLM agent."""
        try:
            if not self.agent_executor:
                self.logger.warning("Context extraction agent not available, using fallback")
                return self._fallback_context_extraction(step)
            
            # Prepare input for the agent
            agent_input = {
                "framework": step.framework,
                "component": step.component,
                "operation": step.operation,
                "input_data": json.dumps(step.input_data, default=str),
                "conversation_history": json.dumps(step.conversation_history or [], default=str),
                "step_reasoning": step.step_reasoning or "Not provided"
            }
            
            # Use the agent to analyze the context
            result = self.agent_executor.invoke(agent_input)
            
            # Parse the agent's response
            analysis = self._parse_agent_response(result.get("output", ""))
            
            # Update the step with extracted context
            step.inferred_goal = analysis.get("inferred_goal", "general_assistance")
            step.context_clues = analysis.get("context_clues", [])
            step.operation_pattern = analysis.get("operation_pattern", "unknown")
            step.auto_confidence = analysis.get("confidence", 0.5)
            
            # If no explicit goal was provided, use the inferred one
            if not step.agent_goal and step.inferred_goal:
                step.agent_goal = step.inferred_goal
                self.logger.debug(f"Auto-inferred agent goal: {step.inferred_goal}")
            
            return step
            
        except Exception as e:
            self.logger.warning(f"LLM-based context extraction failed: {e}")
            return self._fallback_context_extraction(step)
    
    def _fallback_context_extraction(self, step: ExecutionStep) -> ExecutionStep:
        """Fallback context extraction when LLM agent is not available."""
        # Simple fallback based on component type
        if "Chat" in step.component or "LLM" in step.component:
            step.inferred_goal = "Provide conversational AI assistance"
        elif "Tool" in step.component:
            if "Database" in step.component:
                step.inferred_goal = "Query and retrieve data from databases"
            elif "Search" in step.component:
                step.inferred_goal = "Search for information"
            elif "Calculator" in step.component:
                step.inferred_goal = "Perform mathematical calculations"
            else:
                step.inferred_goal = "Execute tool operations"
        else:
            step.inferred_goal = "Provide general assistance"
        
        step.context_clues = [f"component:{step.component}", f"operation:{step.operation}"]
        step.operation_pattern = "unknown"
        step.auto_confidence = 0.3
        
        if not step.agent_goal:
            step.agent_goal = step.inferred_goal
        
        return step
    
    def _create_context_tools(self) -> List[Tool]:
        """Create tools for the context extraction agent."""
        
        def analyze_input_data(input_data: str) -> str:
            """Analyze the input data to understand the user's intent."""
            try:
                data = json.loads(input_data)
                
                # Extract key information
                analysis = {
                    "has_messages": "messages" in data,
                    "message_count": len(data.get("messages", [])),
                    "message_content": [],
                    "other_parameters": {},
                    "intent_indicators": []
                }
                
                # Analyze messages
                if "messages" in data and isinstance(data["messages"], list):
                    for msg in data["messages"]:
                        if isinstance(msg, dict) and "content" in msg:
                            content = str(msg["content"])
                            analysis["message_content"].append({
                                "role": msg.get("role", "unknown"),
                                "content": content[:200],  # Truncate for analysis
                                "length": len(content)
                            })
                            
                            # Look for intent indicators
                            content_lower = content.lower()
                            if any(word in content_lower for word in ["what", "how", "why", "when", "where", "who"]):
                                analysis["intent_indicators"].append("question_asking")
                            if any(word in content_lower for word in ["write", "create", "generate", "build"]):
                                analysis["intent_indicators"].append("content_creation")
                            if any(word in content_lower for word in ["calculate", "compute", "math"]):
                                analysis["intent_indicators"].append("computation")
                            if any(word in content_lower for word in ["hello", "hi", "greeting"]):
                                analysis["intent_indicators"].append("conversation")
                
                # Analyze other parameters
                for key, value in data.items():
                    if key != "messages":
                        analysis["other_parameters"][key] = str(value)[:100]
                
                return json.dumps(analysis, indent=2)
                
            except Exception as e:
                return f"Error analyzing input data: {str(e)}"
        
        def analyze_component_context(component: str, operation: str, framework: str) -> str:
            """Analyze the component and operation to understand the context."""
            try:
                # Component analysis
                component_analysis = {
                    "component": component,
                    "operation": operation,
                    "framework": framework,
                    "component_type": "unknown",
                    "operation_type": "unknown",
                    "typical_purpose": "unknown"
                }
                
                # Analyze component type
                if "Chat" in component or "LLM" in component:
                    component_analysis["component_type"] = "language_model"
                    component_analysis["typical_purpose"] = "conversational_ai"
                elif "Tool" in component:
                    component_analysis["component_type"] = "tool"
                    if "Database" in component:
                        component_analysis["typical_purpose"] = "data_querying"
                    elif "Search" in component:
                        component_analysis["typical_purpose"] = "information_retrieval"
                    elif "Calculator" in component:
                        component_analysis["typical_purpose"] = "mathematical_computation"
                elif "Chain" in component:
                    component_analysis["component_type"] = "chain"
                    component_analysis["typical_purpose"] = "workflow_execution"
                elif "Graph" in component:
                    component_analysis["component_type"] = "graph"
                    component_analysis["typical_purpose"] = "workflow_orchestration"
                
                # Analyze operation type
                if operation in ["invoke", "ainvoke"]:
                    component_analysis["operation_type"] = "direct_execution"
                elif operation in ["run", "arun"]:
                    component_analysis["operation_type"] = "workflow_execution"
                elif operation in ["query", "search"]:
                    component_analysis["operation_type"] = "information_retrieval"
                elif operation in ["calculate", "execute"]:
                    component_analysis["operation_type"] = "computation"
                
                return json.dumps(component_analysis, indent=2)
                
            except Exception as e:
                return f"Error analyzing component context: {str(e)}"
        
        def extract_conversation_history(conversation_data: str) -> str:
            """Extract and analyze conversation history."""
            try:
                if not conversation_data or conversation_data == "[]":
                    return "No conversation history available"
                
                history = json.loads(conversation_data)
                
                analysis = {
                    "message_count": len(history),
                    "conversation_flow": [],
                    "topics_discussed": [],
                    "user_intent_evolution": []
                }
                
                for msg in history:
                    if isinstance(msg, dict):
                        role = msg.get("role", "unknown")
                        content = str(msg.get("content", ""))
                        
                        analysis["conversation_flow"].append({
                            "role": role,
                            "content_preview": content[:100],
                            "timestamp": msg.get("timestamp", "unknown")
                        })
                        
                        # Analyze topics
                        content_lower = content.lower()
                        if any(word in content_lower for word in ["question", "ask", "what", "how"]):
                            analysis["topics_discussed"].append("question_answering")
                        if any(word in content_lower for word in ["code", "program", "function"]):
                            analysis["topics_discussed"].append("programming")
                        if any(word in content_lower for word in ["data", "analyze", "process"]):
                            analysis["topics_discussed"].append("data_analysis")
                
                # Remove duplicates
                analysis["topics_discussed"] = list(set(analysis["topics_discussed"]))
                
                return json.dumps(analysis, indent=2)
                
            except Exception as e:
                return f"Error analyzing conversation history: {str(e)}"
        
        return [
            Tool(
                name="analyze_input_data",
                description="Analyze the input data to understand user intent and content",
                func=analyze_input_data
            ),
            Tool(
                name="analyze_component_context",
                description="Analyze the component, operation, and framework to understand the execution context",
                func=analyze_component_context
            ),
            Tool(
                name="extract_conversation_history",
                description="Extract and analyze conversation history to understand context",
                func=extract_conversation_history
            )
        ]
    
    def _create_agent_prompt(self) -> ChatPromptTemplate:
        """Create the agent prompt for context extraction."""
        
        from langchain.prompts import PromptTemplate
        
        template = """You are an expert AI context extraction agent. Your job is to intelligently analyze agent execution steps and infer their goals, context, and purpose.

You have access to the following tools:
{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: You must respond with a JSON object containing:
{{
    "inferred_goal": "Clear, specific description of what the agent is trying to accomplish",
    "context_clues": ["list", "of", "relevant", "context", "clues"],
    "operation_pattern": "Pattern describing the type of operation",
    "confidence": 0.0-1.0,
    "reasoning": "Detailed explanation of your analysis and reasoning",
    "goal_category": "question_answering|code_generation|data_analysis|conversation|task_completion|creative_writing|computation|information_retrieval|workflow_orchestration|general_assistance",
    "suggestions": ["list", "of", "suggestions", "for", "improvement"]
}}

Begin!

Question: Analyze this execution step:

Framework: {framework}
Component: {component}
Operation: {operation}
Input Data: {input_data}
Conversation History: {conversation_history}
Step Reasoning: {step_reasoning}

Use your tools to analyze the context and provide a comprehensive analysis.
{agent_scratchpad}"""

        return PromptTemplate(
            template=template,
            input_variables=["tools", "tool_names", "framework", "component", "operation", "input_data", "conversation_history", "step_reasoning", "agent_scratchpad"]
        )
    
    def _parse_agent_response(self, response: str) -> Dict[str, Any]:
        """Parse the agent's response into structured data."""
        try:
            # Try to extract JSON from the response
            if "{" in response and "}" in response:
                start = response.find("{")
                end = response.rfind("}") + 1
                json_str = response[start:end]
                
                # Clean the JSON string
                json_str = json_str.replace("'", '"')  # Replace single quotes with double quotes
                json_str = json_str.replace("True", "true").replace("False", "false").replace("None", "null")
                
                return json.loads(json_str)
            else:
                # Fallback if no JSON found
                return {
                    "inferred_goal": "general_assistance",
                    "context_clues": ["agent_response_parsing_failed"],
                    "operation_pattern": "unknown",
                    "confidence": 0.3,
                    "reasoning": "Failed to parse agent response",
                    "goal_category": "general_assistance",
                    "suggestions": ["Check agent response format"]
                }
                
        except Exception as e:
            self.logger.warning(f"Failed to parse agent response: {e}")
            return {
                "inferred_goal": "general_assistance",
                "context_clues": ["parsing_error"],
                "operation_pattern": "unknown",
                "confidence": 0.2,
                "reasoning": f"Error parsing agent response: {str(e)}",
                "goal_category": "general_assistance",
                "suggestions": ["Review agent response format"]
            }
    
    def get_goal_suggestions(self, step: ExecutionStep) -> List[str]:
        """Get alternative goal suggestions based on the step context."""
        if not step.context_clues:
            return ["No context clues available"]
        
        suggestions = []
        for clue in step.context_clues[:3]:  # Limit to 3 suggestions
            if ":" in clue:
                suggestions.append(f"Context suggests: {clue.split(':', 1)[1]}")
            else:
                suggestions.append(f"Context clue: {clue}")
        
        return suggestions
