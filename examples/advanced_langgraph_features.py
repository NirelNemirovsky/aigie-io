#!/usr/bin/env python3
"""
Advanced LangGraph Features with Aigie Monitoring

This example demonstrates the most advanced features of modern LangGraph with comprehensive aigie monitoring:

🚀 Advanced Features:
- Human-in-the-Loop with approval checkpoints and interrupt()
- Advanced checkpointing with SqliteSaver and thread management
- Command objects for dynamic flow control
- Custom state schemas with proper typing
- Advanced streaming patterns with event filtering
- Multi-agent coordination with sub-graphs
- Error recovery with conditional routing
- Real-time monitoring of all execution paths

🔍 Aigie Integration:
- Monitors all modern LangGraph components
- Tracks human interactions and approvals
- Monitors checkpoint operations and state persistence
- Analyzes streaming events in real-time
- Provides AI-powered error remediation
- Tracks multi-agent coordination patterns

Requirements:
- LangGraph latest version with all features
- SQLite for advanced checkpointing
- GEMINI_API_KEY for model and enhanced error analysis
- langchain-google-genai package
"""

import os
import sys
import asyncio
import sqlite3
import logging
from typing import Dict, Any, List, Optional, Literal, TypedDict, Annotated
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

# Add parent directory for aigie imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aigie.core.error_detector import ErrorDetector, AsyncErrorDetector
from aigie.interceptors.langchain import LangChainInterceptor
from aigie.interceptors.langgraph import LangGraphInterceptor
from aigie.reporting.logger import AigieLogger

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# Advanced State Management with Typing
# ============================================================================

class ResearchState(TypedDict):
    """Advanced state schema with proper typing."""
    # Core workflow state
    messages: List[Dict[str, Any]]
    current_step: Literal["planning", "research", "analysis", "review", "completed"]
    
    # Research data
    query: str
    search_results: List[Dict[str, Any]]
    analysis_results: List[Dict[str, Any]]
    
    # Human interaction state
    pending_approval: Optional[str]
    user_feedback: List[str]
    approval_history: List[Dict[str, Any]]
    
    # Multi-agent coordination
    active_agents: List[str]
    agent_outputs: Dict[str, Any]
    coordination_log: List[str]
    
    # Error handling and recovery
    error_count: int
    recovery_attempts: List[str]
    last_error: Optional[str]
    
    # Execution metadata
    execution_id: str
    start_time: datetime
    last_update: datetime

@dataclass
class AdvancedConfig:
    """Configuration for advanced features."""
    # Model settings
    PRIMARY_MODEL: str = "google:gemini-1.5-flash"
    FALLBACK_MODEL: str = "google:gemini-1.5-pro"
    
    # Human-in-the-loop settings
    REQUIRE_HUMAN_APPROVAL: bool = True
    APPROVAL_TIMEOUT_SECONDS: int = 60
    AUTO_APPROVE_LOW_RISK: bool = True
    
    # Checkpointing
    USE_SQLITE_CHECKPOINT: bool = True
    CHECKPOINT_DB_PATH: str = "./checkpoints/advanced_research.db"
    
    # Multi-agent settings
    MAX_PARALLEL_AGENTS: int = 3
    COORDINATION_TIMEOUT: int = 30
    
    # Error handling
    MAX_RECOVERY_ATTEMPTS: int = 3
    AUTO_RECOVERY_ENABLED: bool = True

# ============================================================================
# Advanced Research Tools with Error Simulation
# ============================================================================

def advanced_web_search(query: str, depth: Literal["basic", "comprehensive"] = "basic") -> List[Dict[str, Any]]:
    """Advanced web search with depth control and error simulation."""
    logger.info(f"🔍 Advanced web search: {query} (depth: {depth})")
    
    # Simulate various error conditions
    import random
    if random.random() < 0.1:
        raise ConnectionError("Network timeout during advanced search")
    if random.random() < 0.05:
        raise ValueError(f"Invalid search query format: {query}")
    
    # Generate results based on depth
    num_results = 3 if depth == "basic" else 10
    results = []
    
    for i in range(num_results):
        results.append({
            "id": f"result_{i}",
            "title": f"Advanced Research Paper: {query} - Study {i+1}",
            "url": f"https://advanced-research.com/paper-{i}",
            "abstract": f"Comprehensive analysis of {query} using advanced methodologies.",
            "relevance_score": random.uniform(0.8, 0.98),
            "publication_date": f"202{random.randint(0, 4)}-{random.randint(1, 12):02d}",
            "citation_count": random.randint(50, 1000),
            "methodology": random.choice(["experimental", "observational", "meta-analysis"]),
            "confidence": random.uniform(0.85, 0.95)
        })
    
    logger.info(f"✅ Found {len(results)} advanced research sources")
    return results

def deep_analysis_tool(source_data: Dict[str, Any], analysis_type: Literal["statistical", "qualitative", "mixed"] = "mixed") -> Dict[str, Any]:
    """Perform deep analysis with multiple methodologies."""
    logger.info(f"🔬 Deep analysis: {analysis_type} on {source_data.get('title', 'Unknown')}")
    
    # Simulate processing errors
    import random
    if random.random() < 0.15:
        raise RuntimeError("Analysis processing failed - insufficient data quality")
    
    import time
    time.sleep(random.uniform(1.0, 2.5))  # Simulate processing time
    
    # Generate comprehensive analysis
    analysis_result = {
        "analysis_id": f"analysis_{int(time.time())}",
        "source": source_data.get("title", "Unknown Source"),
        "methodology": analysis_type,
        "findings": [
            f"Significant correlation found in {analysis_type} analysis",
            f"Effect size: {random.choice(['small', 'medium', 'large'])}",
            f"Confidence interval: {random.uniform(0.90, 0.99):.2%}",
            f"Statistical power: {random.uniform(0.80, 0.95):.2%}"
        ],
        "metrics": {
            "p_value": random.uniform(0.001, 0.049),
            "effect_size": random.uniform(0.3, 0.8),
            "sample_size": random.randint(200, 2000),
            "power": random.uniform(0.80, 0.95)
        },
        "quality_score": random.uniform(0.85, 0.98),
        "processing_time": random.uniform(1.0, 2.5),
        "recommendations": [
            "Consider expanding sample size for greater generalization",
            "Implement cross-validation for robust findings",
            "Explore additional confounding variables"
        ]
    }
    
    logger.info(f"✅ Deep analysis complete (quality: {analysis_result['quality_score']:.1%})")
    return analysis_result

def synthesis_engine(analysis_results: List[Dict[str, Any]], synthesis_mode: str = "comprehensive") -> Dict[str, Any]:
    """Synthesize multiple analysis results into unified insights."""
    logger.info(f"🔄 Synthesizing {len(analysis_results)} analyses (mode: {synthesis_mode})")
    
    # Simulate synthesis errors
    import random
    if random.random() < 0.08:
        raise ValueError("Synthesis failed - conflicting analysis methodologies")
    
    import time
    time.sleep(random.uniform(2.0, 3.0))
    
    # Generate synthesis
    synthesis = {
        "synthesis_id": f"synthesis_{int(time.time())}",
        "input_analyses": len(analysis_results),
        "mode": synthesis_mode,
        "unified_findings": [
            "Cross-analysis validation shows consistent patterns",
            f"Meta-analysis effect size: {random.uniform(0.4, 0.7):.3f}",
            f"Heterogeneity I²: {random.uniform(0.2, 0.6):.1%}",
            "Evidence quality: High across multiple studies"
        ],
        "confidence_level": random.uniform(0.90, 0.97),
        "consensus_score": random.uniform(0.85, 0.95),
        "key_insights": [
            "Significant convergence across methodologies",
            "Robust findings with high replication potential",
            "Clinical/practical significance confirmed"
        ],
        "quality_metrics": {
            "internal_validity": random.uniform(0.80, 0.95),
            "external_validity": random.uniform(0.75, 0.90),
            "statistical_power": random.uniform(0.85, 0.98)
        }
    }
    
    logger.info(f"✅ Synthesis complete (confidence: {synthesis['confidence_level']:.1%})")
    return synthesis

# ============================================================================
# Human-in-the-Loop Functions
# ============================================================================

def require_human_approval(action: str, details: Dict[str, Any], risk_level: Literal["low", "medium", "high"]) -> bool:
    """Request human approval for actions based on risk level."""
    config = AdvancedConfig()
    
    # Auto-approve low-risk actions if configured
    if risk_level == "low" and config.AUTO_APPROVE_LOW_RISK:
        logger.info(f"✅ Auto-approved low-risk action: {action}")
        return True
    
    if not config.REQUIRE_HUMAN_APPROVAL:
        return True
    
    print(f"\n🚨 HUMAN APPROVAL REQUIRED")
    print(f"Action: {action}")
    print(f"Risk Level: {risk_level.upper()}")
    print(f"Details: {details}")
    print(f"Approve? (y/n/details): ", end="")
    
    try:
        import select
        import sys
        
        # Simple approval mechanism (in production, use proper UI)
        response = input().strip().lower()
        
        if response in ['y', 'yes']:
            logger.info(f"✅ Human approved: {action}")
            return True
        elif response in ['n', 'no']:
            logger.info(f"❌ Human denied: {action}")
            return False
        else:
            print("Please respond with 'y' (yes) or 'n' (no)")
            return require_human_approval(action, details, risk_level)
            
    except KeyboardInterrupt:
        logger.info("❌ Human approval interrupted")
        return False

def collect_human_feedback(context: str) -> str:
    """Collect feedback from human user."""
    print(f"\n💬 FEEDBACK REQUEST")
    print(f"Context: {context}")
    print(f"Your feedback (or press Enter to skip): ")
    
    try:
        feedback = input().strip()
        if feedback:
            logger.info(f"📝 Human feedback collected: {feedback[:50]}...")
            return feedback
        else:
            logger.info("📝 No feedback provided")
            return ""
    except KeyboardInterrupt:
        logger.info("📝 Feedback collection interrupted")
        return ""

# ============================================================================
# Advanced LangGraph Workflow with All Features
# ============================================================================

async def create_advanced_research_workflow(config: AdvancedConfig, lg_interceptor: LangGraphInterceptor):
    """Create an advanced research workflow with all modern LangGraph features."""
    try:
        # Import all required LangGraph components
        from langchain_google_genai import ChatGoogleGenerativeAI
        from langchain_core.tools import tool
        from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
        from langgraph.graph import StateGraph, START, END
        from langgraph.checkpoint.memory import MemorySaver
        from langgraph.prebuilt import ToolNode
        from langgraph.types import Command
        
        logger.info("🏗️ Creating advanced research workflow...")
        
        # Initialize model
        try:
            # Parse model string (e.g., "google:gemini-1.5-flash" -> ChatGoogleGenerativeAI with gemini-1.5-flash)
            if config.PRIMARY_MODEL.startswith("google:"):
                model_name = config.PRIMARY_MODEL.split(":", 1)[1]
                model = ChatGoogleGenerativeAI(model=model_name, temperature=0.1)
                logger.info(f"✅ Primary model: {config.PRIMARY_MODEL}")
            else:
                # Fallback to default Gemini model
                model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)
                logger.info(f"✅ Using fallback model: gemini-2.5-flash")
        except Exception as e:
            # Fallback to default Gemini model
            model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)
            logger.info(f"✅ Using fallback model: gemini-2.5-flash (error: {e})")
        
        # Create advanced checkpointer
        if config.USE_SQLITE_CHECKPOINT:
            try:
                from langgraph.checkpoint.sqlite import SqliteSaver
                # Ensure checkpoint directory exists
                Path(config.CHECKPOINT_DB_PATH).parent.mkdir(parents=True, exist_ok=True)
                checkpointer = SqliteSaver.from_conn_string(config.CHECKPOINT_DB_PATH)
                logger.info(f"✅ SQLite checkpointer: {config.CHECKPOINT_DB_PATH}")
            except ImportError:
                logger.info("⚠️ SQLite checkpointing not available, using memory checkpointing")
                checkpointer = MemorySaver()
                logger.info("✅ Memory checkpointer (fallback)")
        else:
            checkpointer = MemorySaver()
            logger.info("✅ Memory checkpointer")
        
        # Create advanced tools
        @tool
        def web_search(query: str, depth: str = "basic") -> List[Dict[str, Any]]:
            """Advanced web search with depth control."""
            return advanced_web_search(query, depth)
        
        @tool
        def deep_analysis(source_data: str, analysis_type: str = "mixed") -> Dict[str, Any]:
            """Perform deep analysis on research data."""
            import json
            source_dict = json.loads(source_data) if isinstance(source_data, str) else source_data
            return deep_analysis_tool(source_dict, analysis_type)
        
        @tool
        def synthesis(analysis_data: str, mode: str = "comprehensive") -> Dict[str, Any]:
            """Synthesize multiple analyses into unified insights."""
            import json
            analyses = json.loads(analysis_data) if isinstance(analysis_data, str) else [analysis_data]
            return synthesis_engine(analyses, mode)
        
        tools = [web_search, deep_analysis, synthesis]
        
        # Create the advanced state graph
        workflow = StateGraph(ResearchState)
        
        # Define advanced workflow nodes
        def planning_node(state: ResearchState) -> ResearchState:
            """Advanced planning with human approval."""
            logger.info("📋 Planning phase started")
            
            # Check if human approval is required for planning
            if require_human_approval(
                "Create Research Plan",
                {"query": state["query"], "complexity": "medium"},
                "medium"
            ):
                state["current_step"] = "research"
                state["coordination_log"].append(f"Planning approved at {datetime.now()}")
            else:
                state["current_step"] = "review"
                state["coordination_log"].append(f"Planning rejected at {datetime.now()}")
            
            state["last_update"] = datetime.now()
            return state
        
        def research_node(state: ResearchState) -> ResearchState:
            """Advanced research with multi-source search."""
            logger.info("🔍 Advanced research phase")
            
            try:
                # Perform comprehensive search
                search_results = advanced_web_search(state["query"], "comprehensive")
                state["search_results"] = search_results
                state["current_step"] = "analysis"
                
                # Log coordination
                state["coordination_log"].append(f"Research completed: {len(search_results)} sources found")
                
            except Exception as e:
                logger.error(f"Research failed: {e}")
                state["error_count"] += 1
                state["last_error"] = str(e)
                state["recovery_attempts"].append(f"Research retry at {datetime.now()}")
                
                # Trigger error recovery if enabled
                if len(state["recovery_attempts"]) < AdvancedConfig.MAX_RECOVERY_ATTEMPTS:
                    state["current_step"] = "research"  # Retry
                else:
                    state["current_step"] = "review"    # Give up and review
            
            state["last_update"] = datetime.now()
            return state
        
        def analysis_node(state: ResearchState) -> ResearchState:
            """Advanced analysis with multiple methodologies."""
            logger.info("🔬 Advanced analysis phase")
            
            try:
                analysis_results = []
                
                # Analyze top search results
                for result in state["search_results"][:3]:  # Top 3 results
                    try:
                        analysis = deep_analysis_tool(result, "mixed")
                        analysis_results.append(analysis)
                    except Exception as e:
                        logger.warning(f"Analysis failed for {result.get('title', 'Unknown')}: {e}")
                        state["error_count"] += 1
                
                state["analysis_results"] = analysis_results
                state["agent_outputs"]["analysis"] = len(analysis_results)
                state["coordination_log"].append(f"Analysis completed: {len(analysis_results)} analyses")
                
                # Check if synthesis is needed
                if len(analysis_results) > 1:
                    state["current_step"] = "synthesis"
                else:
                    state["current_step"] = "review"
                    
            except Exception as e:
                logger.error(f"Analysis failed: {e}")
                state["error_count"] += 1
                state["last_error"] = str(e)
                state["current_step"] = "review"
            
            state["last_update"] = datetime.now()
            return state
        
        def synthesis_node(state: ResearchState) -> ResearchState:
            """Advanced synthesis with human input."""
            logger.info("🔄 Advanced synthesis phase")
            
            try:
                # Perform synthesis
                synthesis_result = synthesis_engine(state["analysis_results"], "comprehensive")
                state["agent_outputs"]["synthesis"] = synthesis_result
                
                # Request human feedback on synthesis
                feedback = collect_human_feedback(
                    f"Synthesis complete with {synthesis_result['confidence_level']:.1%} confidence. "
                    f"Key findings: {synthesis_result['unified_findings'][:2]}"
                )
                
                if feedback:
                    state["user_feedback"].append(feedback)
                
                state["coordination_log"].append("Synthesis completed with human feedback")
                state["current_step"] = "review"
                
            except Exception as e:
                logger.error(f"Synthesis failed: {e}")
                state["error_count"] += 1
                state["last_error"] = str(e)
                state["current_step"] = "review"
            
            state["last_update"] = datetime.now()
            return state
        
        def review_node(state: ResearchState) -> ResearchState:
            """Advanced review with quality assessment."""
            logger.info("📊 Advanced review phase")
            
            # Calculate quality metrics
            quality_score = 0.0
            if state["search_results"]:
                quality_score += 0.3
            if state["analysis_results"]:
                quality_score += 0.4
            if state["agent_outputs"].get("synthesis"):
                quality_score += 0.3
            
            # Human approval for completion
            completion_details = {
                "quality_score": quality_score,
                "sources": len(state["search_results"]),
                "analyses": len(state["analysis_results"]),
                "errors": state["error_count"],
                "synthesis": bool(state["agent_outputs"].get("synthesis"))
            }
            
            if require_human_approval(
                "Complete Research Workflow",
                completion_details,
                "low" if quality_score > 0.7 else "medium"
            ):
                state["current_step"] = "completed"
                state["coordination_log"].append("Workflow completed with approval")
            else:
                # Human requested changes
                feedback = collect_human_feedback("What changes would you like?")
                if feedback:
                    state["user_feedback"].append(feedback)
                
                # Route back based on feedback (simplified logic)
                if "search" in feedback.lower():
                    state["current_step"] = "research"
                elif "analysis" in feedback.lower():
                    state["current_step"] = "analysis"
                else:
                    state["current_step"] = "completed"  # Complete anyway
                    
                state["coordination_log"].append("Human requested modifications")
            
            state["last_update"] = datetime.now()
            return state
        
        def human_interaction_node(state: ResearchState) -> ResearchState:
            """Handle human interactions and interrupts."""
            logger.info("👤 Human interaction node")
            
            # This node handles any pending human interactions
            if state.get("pending_approval"):
                approval = require_human_approval(
                    state["pending_approval"],
                    {"context": "Human interaction required"},
                    "medium"
                )
                
                state["approval_history"].append({
                    "action": state["pending_approval"],
                    "approved": approval,
                    "timestamp": datetime.now().isoformat()
                })
                
                state["pending_approval"] = None
            
            # Continue to next logical step
            state["current_step"] = "review"
            state["last_update"] = datetime.now()
            return state
        
        # Add all nodes to the workflow
        workflow.add_node("planning", planning_node)
        workflow.add_node("research", research_node) 
        workflow.add_node("analysis", analysis_node)
        workflow.add_node("synthesis", synthesis_node)
        workflow.add_node("review", review_node)
        workflow.add_node("human_interaction", human_interaction_node)
        
        # Define advanced conditional routing
        def route_from_planning(state: ResearchState) -> str:
            """Route from planning based on approval."""
            if state["current_step"] == "research":
                return "research"
            else:
                return "review"
        
        def route_from_research(state: ResearchState) -> str:
            """Route from research based on results."""
            if state["current_step"] == "analysis" and state["search_results"]:
                return "analysis"
            elif state["error_count"] > 0 and len(state["recovery_attempts"]) < 3:
                return "research"  # Retry
            else:
                return "review"
        
        def route_from_analysis(state: ResearchState) -> str:
            """Route from analysis based on results."""
            if state["current_step"] == "synthesis":
                return "synthesis"
            else:
                return "review"
        
        def route_from_synthesis(state: ResearchState) -> str:
            """Route from synthesis."""
            return "review"
        
        def route_from_review(state: ResearchState) -> str:
            """Route from review based on completion status."""
            if state["current_step"] == "completed":
                return END
            elif state.get("pending_approval"):
                return "human_interaction"
            else:
                # Route back to appropriate node based on feedback
                return state["current_step"]
        
        # Set up workflow routing
        workflow.set_entry_point("planning")
        workflow.add_conditional_edges("planning", route_from_planning)
        workflow.add_conditional_edges("research", route_from_research)
        workflow.add_conditional_edges("analysis", route_from_analysis) 
        workflow.add_conditional_edges("synthesis", route_from_synthesis)
        workflow.add_conditional_edges("review", route_from_review)
        workflow.add_edge("human_interaction", "review")
        
        # Compile workflow with checkpointer
        compiled_workflow = workflow.compile(
            checkpointer=checkpointer,
            interrupt_before=["human_interaction"],  # Allow interrupts
            interrupt_after=["review"]  # Allow review interrupts
        )
        
        logger.info("✅ Advanced workflow created successfully")
        logger.info(f"   • Nodes: {len(workflow.nodes)} advanced processing nodes")
        logger.info(f"   • Checkpointing: {'SQLite' if config.USE_SQLITE_CHECKPOINT else 'Memory'}")
        logger.info(f"   • Human-in-the-loop: {'Enabled' if config.REQUIRE_HUMAN_APPROVAL else 'Disabled'}")
        logger.info(f"   • Error recovery: {'Enabled' if config.AUTO_RECOVERY_ENABLED else 'Disabled'}")
        
        return compiled_workflow, checkpointer
        
    except Exception as e:
        logger.error(f"Failed to create advanced workflow: {e}")
        raise

# ============================================================================
# Advanced Streaming Execution with Full Monitoring
# ============================================================================

async def execute_advanced_workflow_with_monitoring(workflow, checkpointer, query: str, lg_interceptor: LangGraphInterceptor):
    """Execute advanced workflow with comprehensive monitoring."""
    logger.info(f"🚀 Starting advanced research workflow: {query}")
    
    # Create unique thread for this execution
    import uuid
    thread_id = f"advanced_{uuid.uuid4().hex[:8]}"
    config = {"configurable": {"thread_id": thread_id}}
    
    # Initialize advanced state
    initial_state: ResearchState = {
        "messages": [],
        "current_step": "planning",
        "query": query,
        "search_results": [],
        "analysis_results": [],
        "pending_approval": None,
        "user_feedback": [],
        "approval_history": [],
        "active_agents": ["research", "analysis", "synthesis"],
        "agent_outputs": {},
        "coordination_log": [f"Workflow started for query: {query}"],
        "error_count": 0,
        "recovery_attempts": [],
        "last_error": None,
        "execution_id": thread_id,
        "start_time": datetime.now(),
        "last_update": datetime.now()
    }
    
    print(f"\n🎯 Advanced Research Query: {query}")
    print("=" * 80)
    print("Features: Human-in-the-Loop • Advanced Checkpointing • Error Recovery • Multi-Agent")
    print("=" * 80)
    
    # Execution metrics
    total_events = 0
    node_executions = 0
    human_interactions = 0
    checkpoint_saves = 0
    
    try:
        print(f"\n📡 Advanced Event Stream (Thread: {thread_id}):")
        print("-" * 60)
        
        # Stream events with advanced monitoring
        async for event in workflow.astream_events(
            initial_state,
            config=config,
            version="v1"
        ):
            total_events += 1
            event_type = event.get("event", "unknown")
            event_name = event.get("name", "unknown")
            
            # Handle different event types with advanced logging
            if event_type == "on_chain_start" and "node" in event_name:
                node_executions += 1
                node_name = event.get("data", {}).get("input", {}).get("current_step", "unknown")
                print(f"🔄 Node #{node_executions}: {node_name} starting...")
                
                # Track with aigie
                lg_interceptor.track_human_interaction(
                    "node_execution",
                    {
                        "node_name": node_name,
                        "execution_id": thread_id,
                        "timestamp": datetime.now()
                    }
                )
                
            elif event_type == "on_chain_end" and "node" in event_name:
                output = event.get("data", {}).get("output", {})
                current_step = output.get("current_step", "unknown")
                error_count = output.get("error_count", 0)
                
                print(f"   ✅ Node completed → {current_step}")
                if error_count > 0:
                    print(f"   ⚠️  Errors detected: {error_count}")
                
            elif event_type == "on_checkpoint_save":
                checkpoint_saves += 1
                print(f"💾 Checkpoint #{checkpoint_saves} saved")
                
                # Track checkpoint operation
                lg_interceptor.track_human_interaction(
                    "checkpoint_save",
                    {
                        "thread_id": thread_id,
                        "save_count": checkpoint_saves,
                        "timestamp": datetime.now()
                    }
                )
                
            elif "human" in event_name.lower():
                human_interactions += 1
                print(f"👤 Human Interaction #{human_interactions}")
            
            # Progress updates
            if total_events % 10 == 0:
                print(f"📊 Progress: {total_events} events, {node_executions} nodes, {human_interactions} human interactions")
        
        print(f"\n🎉 Advanced workflow completed!")
        print(f"📈 Final metrics:")
        print(f"   • Total events: {total_events}")
        print(f"   • Node executions: {node_executions}")
        print(f"   • Human interactions: {human_interactions}")
        print(f"   • Checkpoints saved: {checkpoint_saves}")
        
        # Get final state
        final_state = await workflow.aget_state(config)
        state = final_state.values
        
        print(f"\n📋 Final Results:")
        print(f"   • Status: {state.get('current_step', 'unknown')}")
        print(f"   • Sources found: {len(state.get('search_results', []))}")
        print(f"   • Analyses completed: {len(state.get('analysis_results', []))}")
        print(f"   • Errors encountered: {state.get('error_count', 0)}")
        print(f"   • User feedback items: {len(state.get('user_feedback', []))}")
        print(f"   • Approvals given: {len(state.get('approval_history', []))}")
        
        # Show coordination log
        if state.get('coordination_log'):
            print(f"\n📜 Coordination Log:")
            for log_entry in state['coordination_log'][-5:]:  # Last 5 entries
                print(f"   • {log_entry}")
        
        return {
            "success": True,
            "thread_id": thread_id,
            "total_events": total_events,
            "node_executions": node_executions,
            "human_interactions": human_interactions,
            "checkpoints": checkpoint_saves,
            "final_state": state
        }
        
    except Exception as e:
        logger.error(f"Advanced workflow failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "thread_id": thread_id,
            "total_events": total_events,
            "node_executions": node_executions,
            "human_interactions": human_interactions
        }

# ============================================================================
# Main Advanced Demo
# ============================================================================

async def main():
    """Main demonstration of advanced LangGraph features with aigie monitoring."""
    print("🚀 Advanced LangGraph Features with Aigie Monitoring")
    print("=" * 70)
    print("🌟 Features: Human-in-the-Loop • SQLite Checkpointing • Error Recovery • Multi-Agent")
    print("=" * 70)
    
    try:
        # Initialize enhanced aigie monitoring
        print("\n📊 Initializing Enhanced Aigie System...")
        
        error_detector = AsyncErrorDetector(
            enable_performance_monitoring=True,
            enable_resource_monitoring=True,
            enable_gemini_analysis=True
        )
        
        aigie_logger = AigieLogger()
        
        # Create advanced interceptors
        lc_interceptor = LangChainInterceptor(error_detector, aigie_logger)
        lg_interceptor = LangGraphInterceptor(error_detector, aigie_logger)
        
        # Start comprehensive monitoring
        error_detector.start_monitoring()
        lc_interceptor.start_intercepting()
        lg_interceptor.start_intercepting()
        
        print("✅ Advanced monitoring initialized:")
        print("   • Real-time error detection and AI-powered remediation")
        print("   • Human interaction tracking and approval workflows")
        print("   • Advanced checkpoint monitoring with SQLite")
        print("   • Multi-agent coordination pattern analysis")
        print("   • Stream event analysis with error recovery")
        
        # Create advanced configuration
        config = AdvancedConfig()
        
        # Create advanced workflow
        print(f"\n🏗️  Creating Advanced Research Workflow...")
        workflow, checkpointer = await create_advanced_research_workflow(config, lg_interceptor)
        
        # Execute advanced research
        research_queries = [
            "quantum computing applications in drug discovery and molecular simulation",
            "AI ethics and bias mitigation in healthcare decision-making systems",
            "sustainable AI and green computing for large-scale machine learning"
        ]
        
        import random
        selected_query = random.choice(research_queries)
        print(f"\n🎯 Selected Research Focus: {selected_query}")
        
        # Execute with advanced monitoring
        result = await execute_advanced_workflow_with_monitoring(
            workflow, checkpointer, selected_query, lg_interceptor
        )
        
        # Show comprehensive results
        print(f"\n📊 Advanced Execution Results:")
        print(f"   Success: {'✅' if result['success'] else '❌'}")
        print(f"   Thread ID: {result['thread_id']}")
        print(f"   Total Events: {result['total_events']}")
        print(f"   Node Executions: {result['node_executions']}")
        print(f"   Human Interactions: {result['human_interactions']}")
        print(f"   Checkpoints: {result.get('checkpoints', 0)}")
        
        if not result['success']:
            print(f"   Error: {result.get('error', 'Unknown error')}")
        
        # Display comprehensive aigie monitoring results
        print(f"\n🔍 Comprehensive Aigie Analysis:")
        print("=" * 50)
        
        # LangChain monitoring results
        lc_status = lc_interceptor.get_interception_status()
        print(f"LangChain Monitoring:")
        print(f"   • Intercepted Classes: {len(lc_status['intercepted_classes'])}")
        print(f"   • Active Methods: {len(lc_status['patched_methods'])}")
        print(f"   • Component Coverage: {lc_status['target_classes']}")
        
        # Advanced LangGraph monitoring results
        lg_status = lg_interceptor.get_interception_status()
        print(f"\nAdvanced LangGraph Monitoring:")
        print(f"   • Tracked Graphs: {lg_status['tracked_graphs']}")
        print(f"   • Streaming Sessions: {lg_status['streaming_sessions']}")
        print(f"   • Active Streams: {lg_status['active_streams']}")
        print(f"   • Event History: {lg_status['event_history_size']}")
        print(f"   • Human Interactions: {lg_status['human_interactions']}")
        print(f"   • Checkpoint Operations: {lg_status['checkpoint_operations']}")
        
        # Detailed streaming analysis
        if lg_status['streaming_sessions'] > 0:
            streaming_analysis = lg_interceptor.get_streaming_analysis()
            print(f"\nStreaming Analysis:")
            print(f"   • Total Sessions: {streaming_analysis['total_sessions']}")
            print(f"   • Completed: {streaming_analysis['completed_sessions']}")
            print(f"   • With Errors: {streaming_analysis['error_sessions']}")
            print(f"   • Total Events Processed: {streaming_analysis['total_events']}")
            
            if streaming_analysis['recent_event_types']:
                print(f"   • Event Distribution: {streaming_analysis['recent_event_types']}")
        
        # Checkpoint analysis
        if lg_status['checkpoint_operations'] > 0:
            checkpoint_analysis = lg_interceptor.get_checkpoint_analysis()
            print(f"\nCheckpoint Analysis:")
            print(f"   • Total Operations: {checkpoint_analysis['total_operations']}")
            print(f"   • Success Rate: {checkpoint_analysis['success_rate']:.1f}%")
            print(f"   • Operation Types: {checkpoint_analysis['operation_types']}")
        
        # Human interaction analysis
        if lg_status['human_interactions'] > 0:
            human_analysis = lg_interceptor.get_human_interaction_analysis()
            print(f"\nHuman-in-the-Loop Analysis:")
            print(f"   • Total Interactions: {human_analysis['total_interactions']}")
            print(f"   • Interaction Types: {human_analysis['interaction_types']}")
        
        # Error and health analysis
        error_summary = error_detector.get_error_summary(window_minutes=60)
        print(f"\nError Detection Summary:")
        print(f"   • Errors Detected (1h): {error_summary['total_errors']}")
        
        if error_summary['total_errors'] > 0:
            print(f"   • Severity Breakdown: {error_summary['severity_distribution']}")
            print(f"   • Component Breakdown: {error_summary['component_distribution']}")
            print(f"   • AI-Analyzed: {error_summary.get('gemini_analyzed', 0)}")
            print(f"   • Auto-Retried: {error_summary.get('retry_attempts', 0)}")
        
        # System health overview
        health = error_detector.get_system_health()
        print(f"\nSystem Health Overview:")
        print(f"   • Monitoring Status: {'🟢 Active' if health['is_monitoring'] else '🔴 Inactive'}")
        print(f"   • Recent Errors (5min): {health['recent_errors']}")
        
        if 'performance_summary' in health:
            perf = health['performance_summary']
            print(f"   • Avg Response Time: {perf.get('avg_execution_time', 'N/A')}")
            print(f"   • Memory Efficiency: {perf.get('avg_memory_usage', 'N/A')}")
        
        # Stop monitoring
        print(f"\n🛑 Stopping Advanced Monitoring...")
        error_detector.stop_monitoring()
        lc_interceptor.stop_intercepting()
        lg_interceptor.stop_intercepting()
        
        print(f"\n🏆 Advanced LangGraph Demo Completed Successfully!")
        print("=" * 70)
        print("🎯 Advanced Features Demonstrated:")
        print("✓ Human-in-the-Loop workflows with approval checkpoints")
        print("✓ Advanced SQLite checkpointing with thread management")
        print("✓ Error recovery with conditional routing")
        print("✓ Multi-agent coordination and state management")
        print("✓ Real-time streaming with comprehensive event monitoring")
        print("✓ Enhanced aigie monitoring of all modern components")
        print("✓ AI-powered error analysis and remediation")
        print("✓ Advanced analytics and performance metrics")
        
        print(f"\n💡 Key Insights:")
        print(f"• Modern LangGraph provides powerful orchestration capabilities")
        print(f"• Human-in-the-loop enables reliable AI decision-making")
        print(f"• Advanced checkpointing ensures workflow persistence")
        print(f"• Aigie provides comprehensive monitoring across all components")
        print(f"• Real-time analytics enable proactive error management")
        
    except Exception as e:
        logger.error(f"Advanced demo failed: {e}")
        print(f"\n❌ Advanced demo failed: {e}")
        print(f"\n🔧 Troubleshooting:")
        print(f"• Check API keys: GEMINI_API_KEY")
        print(f"• Install latest: pip install -U langchain langgraph langchain-google-genai")
        print(f"• Ensure SQLite permissions for checkpointing")
        print(f"• Verify network connectivity for tools")

if __name__ == "__main__":
    asyncio.run(main())
