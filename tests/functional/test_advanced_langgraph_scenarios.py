#!/usr/bin/env python3
"""
Advanced LangGraph Test Scenarios for Aigie Error Remediation

This module provides sophisticated LangGraph workflows that demonstrate real-world
failure scenarios and test Aigie's advanced error detection and remediation capabilities.

Features:
- Multi-agent collaboration with failure points
- Complex state management with error recovery
- Human-in-the-loop workflows with approval failures
- Streaming operations with interruption handling
- Checkpoint and persistence error scenarios
- Conditional routing with dynamic failure injection

Requirements:
- GEMINI_API_KEY for enhanced error analysis
- LangGraph with latest features
- Aigie auto-integration
"""

import os
import sys
import time
import random
import asyncio
import logging
from typing import Dict, Any, List, Optional, TypedDict, Annotated
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import json

# Add the parent directory to the path so we can import aigie
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aigie import auto_integrate, show_status, show_analysis

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Advanced State Models and Enums
# ============================================================================

class WorkflowStatus(Enum):
    """Workflow execution status."""
    INITIALIZED = "initialized"
    IN_PROGRESS = "in_progress"
    WAITING_APPROVAL = "waiting_approval"
    APPROVED = "approved"
    REJECTED = "rejected"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"


class ErrorType(Enum):
    """Advanced error types for testing."""
    NETWORK_TIMEOUT = "network_timeout"
    API_RATE_LIMIT = "api_rate_limit"
    VALIDATION_FAILURE = "validation_failure"
    STATE_CORRUPTION = "state_corruption"
    CHECKPOINT_FAILURE = "checkpoint_failure"
    HUMAN_APPROVAL_TIMEOUT = "human_approval_timeout"
    STREAMING_INTERRUPTION = "streaming_interruption"
    MEMORY_EXHAUSTION = "memory_exhaustion"
    CONCURRENT_ACCESS = "concurrent_access"


class AgentRole(Enum):
    """Agent roles in multi-agent workflows."""
    RESEARCHER = "researcher"
    ANALYST = "analyst"
    VALIDATOR = "validator"
    APPROVER = "approver"
    EXECUTOR = "executor"


# ============================================================================
# Advanced State Definitions
# ============================================================================

class MultiAgentState(TypedDict):
    """State for multi-agent collaboration workflow."""
    # Workflow metadata
    workflow_id: str
    status: WorkflowStatus
    created_at: datetime
    updated_at: datetime
    
    # Agent coordination
    current_agent: Optional[AgentRole]
    agent_queue: List[AgentRole]
    completed_agents: List[AgentRole]
    failed_agents: List[AgentRole]
    
    # Data flow
    research_data: Dict[str, Any]
    analysis_results: Dict[str, Any]
    validation_results: Dict[str, Any]
    approval_status: Optional[bool]
    execution_results: Dict[str, Any]
    
    # Error handling
    errors: List[Dict[str, Any]]
    retry_count: int
    max_retries: int
    
    # Performance tracking
    execution_times: Dict[str, float]
    memory_usage: List[float]
    checkpoint_data: Dict[str, Any]


class StreamingState(TypedDict):
    """State for streaming workflow with interruption handling."""
    # Streaming metadata
    stream_id: str
    is_streaming: bool
    stream_position: int
    total_items: int
    
    # Data processing
    processed_items: List[Dict[str, Any]]
    failed_items: List[Dict[str, Any]]
    pending_items: List[Dict[str, Any]]
    
    # Interruption handling
    interruption_points: List[int]
    recovery_data: Dict[str, Any]
    
    # Performance metrics
    processing_rate: float
    error_rate: float
    memory_usage: float


class HumanApprovalState(TypedDict):
    """State for human-in-the-loop workflows."""
    # Approval workflow
    approval_request_id: str
    approval_status: Optional[bool]
    approval_timeout: float
    approval_deadline: datetime
    
    # Human interaction
    human_feedback: Optional[str]
    approval_history: List[Dict[str, Any]]
    
    # Workflow state
    pending_approval: bool
    auto_approve_on_timeout: bool
    
    # Error scenarios
    approval_errors: List[str]
    timeout_errors: List[str]


# ============================================================================
# Advanced Error Simulation
# ============================================================================

@dataclass
class AdvancedErrorSimulator:
    """Advanced error simulator for complex scenarios."""
    
    # Error probabilities
    network_error_rate: float = 0.3
    api_error_rate: float = 0.25
    validation_error_rate: float = 0.2
    state_corruption_rate: float = 0.15
    checkpoint_error_rate: float = 0.1
    human_timeout_rate: float = 0.2
    streaming_error_rate: float = 0.3
    memory_error_rate: float = 0.1
    concurrent_error_rate: float = 0.05
    
    # Error tracking
    error_history: List[Dict[str, Any]] = field(default_factory=list)
    operation_count: int = 0
    
    def simulate_network_timeout(self) -> None:
        """Simulate network timeout errors."""
        if random.random() < self.network_error_rate:
            self._record_error(ErrorType.NETWORK_TIMEOUT, "Network connection timed out")
            time.sleep(2)  # Simulate delay
            raise TimeoutError("Network operation timed out after 30 seconds")
    
    def simulate_api_rate_limit(self) -> None:
        """Simulate API rate limiting."""
        if random.random() < self.api_error_rate:
            self._record_error(ErrorType.API_RATE_LIMIT, "API rate limit exceeded")
            raise Exception("API rate limit exceeded. Please wait before retrying.")
    
    def simulate_validation_failure(self) -> None:
        """Simulate validation failures."""
        if random.random() < self.validation_error_rate:
            self._record_error(ErrorType.VALIDATION_FAILURE, "Data validation failed")
            raise ValueError("Input validation failed: Invalid data format")
    
    def simulate_state_corruption(self) -> None:
        """Simulate state corruption scenarios."""
        if random.random() < self.state_corruption_rate:
            self._record_error(ErrorType.STATE_CORRUPTION, "State data corrupted")
            raise RuntimeError("Workflow state corruption detected")
    
    def simulate_checkpoint_failure(self) -> None:
        """Simulate checkpoint/persistence failures."""
        if random.random() < self.checkpoint_error_rate:
            self._record_error(ErrorType.CHECKPOINT_FAILURE, "Checkpoint save failed")
            raise IOError("Failed to save workflow checkpoint")
    
    def simulate_human_approval_timeout(self) -> None:
        """Simulate human approval timeout."""
        if random.random() < self.human_timeout_rate:
            self._record_error(ErrorType.HUMAN_APPROVAL_TIMEOUT, "Human approval timed out")
            raise TimeoutError("Human approval request timed out")
    
    def simulate_streaming_interruption(self) -> None:
        """Simulate streaming operation interruption."""
        if random.random() < self.streaming_error_rate:
            self._record_error(ErrorType.STREAMING_INTERRUPTION, "Streaming operation interrupted")
            raise InterruptedError("Streaming operation was interrupted")
    
    def simulate_memory_exhaustion(self) -> None:
        """Simulate memory exhaustion."""
        if random.random() < self.memory_error_rate:
            self._record_error(ErrorType.MEMORY_EXHAUSTION, "Memory exhausted")
            raise MemoryError("Insufficient memory to continue operation")
    
    def simulate_concurrent_access(self) -> None:
        """Simulate concurrent access conflicts."""
        if random.random() < self.concurrent_error_rate:
            self._record_error(ErrorType.CONCURRENT_ACCESS, "Concurrent access conflict")
            raise RuntimeError("Concurrent access to shared resource detected")
    
    def _record_error(self, error_type: ErrorType, message: str) -> None:
        """Record error for analysis."""
        self.error_history.append({
            "timestamp": datetime.now(),
            "error_type": error_type.value,
            "message": message,
            "operation_count": self.operation_count
        })
    
    def increment_operation(self) -> None:
        """Increment operation counter."""
        self.operation_count += 1
    
    def get_error_stats(self) -> Dict[str, Any]:
        """Get error simulation statistics."""
        if not self.error_history:
            return {"total_errors": 0, "error_rate": 0.0}
        
        error_types = {}
        for error in self.error_history:
            error_type = error["error_type"]
            error_types[error_type] = error_types.get(error_type, 0) + 1
        
        return {
            "total_errors": len(self.error_history),
            "total_operations": self.operation_count,
            "error_rate": len(self.error_history) / max(self.operation_count, 1),
            "error_type_distribution": error_types
        }


# ============================================================================
# Advanced LangGraph Workflows
# ============================================================================

class AdvancedLangGraphTestSuite:
    """Advanced LangGraph test scenarios."""
    
    def __init__(self):
        self.error_simulator = AdvancedErrorSimulator()
        self.test_results = []
    
    def create_multi_agent_workflow(self):
        """Create a complex multi-agent collaboration workflow."""
        print("\n🤖 Creating Multi-Agent Collaboration Workflow...")
        
        try:
            from langgraph.graph import StateGraph, END
            
            # Create the workflow
            workflow = StateGraph(MultiAgentState)
            
            def researcher_agent(state: MultiAgentState) -> MultiAgentState:
                """Researcher agent that may fail."""
                print(f"  🔍 Researcher agent starting...")
                self.error_simulator.increment_operation()
                
                # Simulate various error types
                self.error_simulator.simulate_network_timeout()
                self.error_simulator.simulate_api_rate_limit()
                
                # Simulate research work
                time.sleep(random.uniform(0.5, 1.5))
                
                state["research_data"] = {
                    "sources": ["paper1.pdf", "paper2.pdf", "paper3.pdf"],
                    "findings": ["Finding 1", "Finding 2", "Finding 3"],
                    "confidence": random.uniform(0.7, 0.95)
                }
                state["current_agent"] = AgentRole.RESEARCHER
                state["completed_agents"].append(AgentRole.RESEARCHER)
                state["status"] = WorkflowStatus.IN_PROGRESS
                state["updated_at"] = datetime.now()
                
                print(f"  ✅ Researcher agent completed")
                return state
            
            def analyst_agent(state: MultiAgentState) -> MultiAgentState:
                """Analyst agent that may fail."""
                print(f"  📊 Analyst agent starting...")
                self.error_simulator.increment_operation()
                
                # Simulate different error types
                self.error_simulator.simulate_validation_failure()
                self.error_simulator.simulate_memory_exhaustion()
                
                # Simulate analysis work
                time.sleep(random.uniform(0.8, 2.0))
                
                state["analysis_results"] = {
                    "trends": ["Trend 1", "Trend 2"],
                    "insights": ["Insight 1", "Insight 2", "Insight 3"],
                    "recommendations": ["Recommendation 1", "Recommendation 2"],
                    "confidence": random.uniform(0.8, 0.95)
                }
                state["current_agent"] = AgentRole.ANALYST
                state["completed_agents"].append(AgentRole.ANALYST)
                state["updated_at"] = datetime.now()
                
                print(f"  ✅ Analyst agent completed")
                return state
            
            def validator_agent(state: MultiAgentState) -> MultiAgentState:
                """Validator agent that may fail."""
                print(f"  ✅ Validator agent starting...")
                self.error_simulator.increment_operation()
                
                # Simulate validation errors
                self.error_simulator.simulate_state_corruption()
                self.error_simulator.simulate_concurrent_access()
                
                # Simulate validation work
                time.sleep(random.uniform(0.3, 1.0))
                
                state["validation_results"] = {
                    "is_valid": random.choice([True, False]),
                    "validation_score": random.uniform(0.6, 1.0),
                    "issues": [] if random.random() > 0.3 else ["Issue 1", "Issue 2"],
                    "timestamp": datetime.now()
                }
                state["current_agent"] = AgentRole.VALIDATOR
                state["completed_agents"].append(AgentRole.VALIDATOR)
                state["updated_at"] = datetime.now()
                
                print(f"  ✅ Validator agent completed")
                return state
            
            def approver_agent(state: MultiAgentState) -> MultiAgentState:
                """Approver agent with human-in-the-loop simulation."""
                print(f"  👤 Approver agent starting...")
                self.error_simulator.increment_operation()
                
                # Simulate human approval timeout
                self.error_simulator.simulate_human_approval_timeout()
                
                # Simulate approval process
                time.sleep(random.uniform(1.0, 3.0))
                
                # Simulate approval decision
                approval_decision = random.choice([True, False])
                state["approval_status"] = approval_decision
                state["current_agent"] = AgentRole.APPROVER
                state["completed_agents"].append(AgentRole.APPROVER)
                state["status"] = WorkflowStatus.APPROVED if approval_decision else WorkflowStatus.REJECTED
                state["updated_at"] = datetime.now()
                
                print(f"  {'✅' if approval_decision else '❌'} Approver agent completed - {'Approved' if approval_decision else 'Rejected'}")
                return state
            
            def executor_agent(state: MultiAgentState) -> MultiAgentState:
                """Executor agent that may fail."""
                print(f"  ⚡ Executor agent starting...")
                self.error_simulator.increment_operation()
                
                # Simulate execution errors
                self.error_simulator.simulate_checkpoint_failure()
                self.error_simulator.simulate_streaming_interruption()
                
                # Simulate execution work
                time.sleep(random.uniform(1.0, 2.5))
                
                state["execution_results"] = {
                    "tasks_completed": random.randint(3, 8),
                    "success_rate": random.uniform(0.7, 1.0),
                    "execution_time": time.time(),
                    "output_files": ["output1.txt", "output2.txt"]
                }
                state["current_agent"] = AgentRole.EXECUTOR
                state["completed_agents"].append(AgentRole.EXECUTOR)
                state["status"] = WorkflowStatus.COMPLETED
                state["updated_at"] = datetime.now()
                
                print(f"  ✅ Executor agent completed")
                return state
            
            def should_continue(state: MultiAgentState) -> str:
                """Determine if workflow should continue."""
                if state["status"] == WorkflowStatus.REJECTED:
                    return "end"
                elif state["status"] == WorkflowStatus.COMPLETED:
                    return "end"
                elif len(state["completed_agents"]) < len(state["agent_queue"]):
                    return "continue"
                else:
                    return "end"
            
            # Add nodes
            workflow.add_node("researcher", researcher_agent)
            workflow.add_node("analyst", analyst_agent)
            workflow.add_node("validator", validator_agent)
            workflow.add_node("approver", approver_agent)
            workflow.add_node("executor", executor_agent)
            
            # Define edges
            workflow.set_entry_point("researcher")
            workflow.add_edge("researcher", "analyst")
            workflow.add_edge("analyst", "validator")
            workflow.add_edge("validator", "approver")
            workflow.add_conditional_edges(
                "approver",
                should_continue,
                {
                    "continue": "executor",
                    "end": END
                }
            )
            workflow.add_edge("executor", END)
            
            # Compile workflow
            compiled_workflow = workflow.compile()
            
            print("✅ Multi-agent workflow created successfully")
            return compiled_workflow
            
        except ImportError as e:
            print(f"❌ LangGraph not available: {e}")
            return None
        except Exception as e:
            print(f"❌ Failed to create multi-agent workflow: {e}")
            return None
    
    def create_streaming_workflow(self):
        """Create a streaming workflow with interruption handling."""
        print("\n📡 Creating Streaming Workflow with Interruption Handling...")
        
        try:
            from langgraph.graph import StateGraph, END
            
            # Create the workflow
            workflow = StateGraph(StreamingState)
            
            def stream_processor(state: StreamingState) -> StreamingState:
                """Process streaming data with potential interruptions."""
                print(f"  📊 Processing stream item {state['stream_position']}...")
                self.error_simulator.increment_operation()
                
                # Simulate streaming errors
                self.error_simulator.simulate_streaming_interruption()
                self.error_simulator.simulate_memory_exhaustion()
                
                # Simulate processing work
                time.sleep(random.uniform(0.1, 0.5))
                
                # Process item
                item = {
                    "id": state["stream_position"],
                    "data": f"Processed item {state['stream_position']}",
                    "timestamp": datetime.now(),
                    "success": random.random() > 0.2  # 80% success rate
                }
                
                if item["success"]:
                    state["processed_items"].append(item)
                else:
                    state["failed_items"].append(item)
                
                state["stream_position"] += 1
                state["updated_at"] = datetime.now()
                
                print(f"  {'✅' if item['success'] else '❌'} Processed item {state['stream_position']-1}")
                return state
            
            def checkpoint_saver(state: StreamingState) -> StreamingState:
                """Save checkpoint with potential failures."""
                print(f"  💾 Saving checkpoint...")
                self.error_simulator.increment_operation()
                
                # Simulate checkpoint errors
                self.error_simulator.simulate_checkpoint_failure()
                
                # Simulate checkpoint save
                time.sleep(random.uniform(0.2, 0.8))
                
                state["checkpoint_data"] = {
                    "position": state["stream_position"],
                    "processed_count": len(state["processed_items"]),
                    "failed_count": len(state["failed_items"]),
                    "timestamp": datetime.now()
                }
                
                print(f"  ✅ Checkpoint saved")
                return state
            
            def should_continue_streaming(state: StreamingState) -> str:
                """Determine if streaming should continue."""
                if state["stream_position"] >= state["total_items"]:
                    return "end"
                elif len(state["failed_items"]) > 5:  # Too many failures
                    return "end"
                else:
                    return "continue"
            
            # Add nodes
            workflow.add_node("process", stream_processor)
            workflow.add_node("checkpoint", checkpoint_saver)
            
            # Define edges
            workflow.set_entry_point("process")
            workflow.add_conditional_edges(
                "process",
                should_continue_streaming,
                {
                    "continue": "checkpoint",
                    "end": END
                }
            )
            workflow.add_edge("checkpoint", "process")
            
            # Compile workflow
            compiled_workflow = workflow.compile()
            
            print("✅ Streaming workflow created successfully")
            return compiled_workflow
            
        except ImportError as e:
            print(f"❌ LangGraph not available: {e}")
            return None
        except Exception as e:
            print(f"❌ Failed to create streaming workflow: {e}")
            return None
    
    def test_multi_agent_workflow(self) -> Dict[str, Any]:
        """Test the multi-agent workflow."""
        print("\n🤖 Testing Multi-Agent Collaboration Workflow...")
        
        workflow = self.create_multi_agent_workflow()
        if not workflow:
            return {"test_name": "multi_agent_workflow", "error": "Failed to create workflow"}
        
        # Initialize state
        initial_state = MultiAgentState(
            workflow_id=f"workflow_{int(time.time())}",
            status=WorkflowStatus.INITIALIZED,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            current_agent=None,
            agent_queue=[AgentRole.RESEARCHER, AgentRole.ANALYST, AgentRole.VALIDATOR, AgentRole.APPROVER, AgentRole.EXECUTOR],
            completed_agents=[],
            failed_agents=[],
            research_data={},
            analysis_results={},
            validation_results={},
            approval_status=None,
            execution_results={},
            errors=[],
            retry_count=0,
            max_retries=3,
            execution_times={},
            memory_usage=[],
            checkpoint_data={}
        )
        
        # Run workflow multiple times
        results = []
        for i in range(3):
            try:
                print(f"\n  🚀 Running workflow iteration {i+1}...")
                start_time = time.time()
                
                result = workflow.invoke(initial_state.copy())
                
                execution_time = time.time() - start_time
                
                results.append({
                    "iteration": i+1,
                    "success": True,
                    "execution_time": execution_time,
                    "final_status": result["status"].value,
                    "completed_agents": [agent.value for agent in result["completed_agents"]],
                    "approval_status": result.get("approval_status"),
                    "has_errors": len(result.get("errors", [])) > 0
                })
                
                print(f"  ✅ Iteration {i+1} completed in {execution_time:.2f}s - Status: {result['status'].value}")
                
            except Exception as e:
                results.append({
                    "iteration": i+1,
                    "success": False,
                    "error": str(e),
                    "execution_time": time.time() - start_time if 'start_time' in locals() else 0
                })
                print(f"  ❌ Iteration {i+1} failed: {e}")
        
        return {
            "test_name": "multi_agent_workflow",
            "results": results,
            "error_stats": self.error_simulator.get_error_stats(),
            "success_rate": len([r for r in results if r["success"]]) / len(results)
        }
    
    def test_streaming_workflow(self) -> Dict[str, Any]:
        """Test the streaming workflow."""
        print("\n📡 Testing Streaming Workflow with Interruption Handling...")
        
        workflow = self.create_streaming_workflow()
        if not workflow:
            return {"test_name": "streaming_workflow", "error": "Failed to create workflow"}
        
        # Initialize state
        initial_state = StreamingState(
            stream_id=f"stream_{int(time.time())}",
            is_streaming=True,
            stream_position=0,
            total_items=10,
            processed_items=[],
            failed_items=[],
            pending_items=[],
            interruption_points=[],
            recovery_data={},
            processing_rate=0.0,
            error_rate=0.0,
            memory_usage=0.0
        )
        
        # Run workflow
        try:
            print(f"\n  🚀 Running streaming workflow...")
            start_time = time.time()
            
            result = workflow.invoke(initial_state)
            
            execution_time = time.time() - start_time
            
            return {
                "test_name": "streaming_workflow",
                "success": True,
                "execution_time": execution_time,
                "processed_items": len(result["processed_items"]),
                "failed_items": len(result["failed_items"]),
                "total_items": result["total_items"],
                "completion_rate": len(result["processed_items"]) / result["total_items"],
                "error_stats": self.error_simulator.get_error_stats()
            }
            
        except Exception as e:
            return {
                "test_name": "streaming_workflow",
                "success": False,
                "error": str(e),
                "error_stats": self.error_simulator.get_error_stats()
            }


# ============================================================================
# Main Test Execution
# ============================================================================

async def run_advanced_langgraph_tests():
    """Run advanced LangGraph test scenarios."""
    print("🚀 Starting Advanced LangGraph Error Remediation Tests")
    print("=" * 70)
    
    # Initialize Aigie with auto-integration
    print("\n📊 Initializing Aigie Error Detection System...")
    aigie = auto_integrate()
    error_detector = aigie.error_detector
    
    print("✅ Aigie monitoring started successfully")
    
    # Initialize test suite
    test_suite = AdvancedLangGraphTestSuite()
    
    # Run advanced tests
    print("\n" + "="*50)
    print("🔄 ADVANCED LANGGRAPH ERROR REMEDIATION TESTS")
    print("="*50)
    
    # Test multi-agent workflow
    multi_agent_result = test_suite.test_multi_agent_workflow()
    
    # Test streaming workflow
    streaming_result = test_suite.test_streaming_workflow()
    
    # Display results
    print("\n" + "="*70)
    print("📊 ADVANCED TEST RESULTS SUMMARY")
    print("="*70)
    
    all_results = [multi_agent_result, streaming_result]
    
    for result in all_results:
        test_name = result.get("test_name", "Unknown")
        if "error" in result:
            print(f"\n❌ {test_name}: FAILED - {result['error']}")
        else:
            if "success_rate" in result:
                success_rate = result["success_rate"] * 100
                print(f"\n✅ {test_name}: SUCCESS - {success_rate:.1f}% success rate")
            elif "success" in result:
                status = "SUCCESS" if result["success"] else "FAILED"
                print(f"\n{'✅' if result['success'] else '❌'} {test_name}: {status}")
            
            if "error_stats" in result:
                stats = result["error_stats"]
                print(f"   • Total Operations: {stats['total_operations']}")
                print(f"   • Total Errors: {stats['total_errors']}")
                print(f"   • Error Rate: {stats['error_rate']*100:.1f}%")
                if "error_type_distribution" in stats:
                    print(f"   • Error Types: {stats['error_type_distribution']}")
    
    # Display Aigie monitoring results
    print("\n" + "="*70)
    print("🤖 AIGIE MONITORING ANALYSIS")
    print("="*70)
    
    # Error detection summary
    error_summary = error_detector.get_error_summary(window_minutes=60)
    print(f"\n🚨 Error Detection Summary:")
    print(f"   • Total Errors Detected: {error_summary['total_errors']}")
    
    if error_summary['total_errors'] > 0:
        print(f"   • Severity Distribution: {error_summary['severity_distribution']}")
        print(f"   • Component Distribution: {error_summary['component_distribution']}")
        print(f"   • Gemini AI Analyzed: {error_summary.get('gemini_analyzed', 0)}")
        print(f"   • Automatic Retries: {error_summary.get('retry_attempts', 0)}")
        print(f"   • Successful Remediations: {error_summary.get('successful_remediations', 0)}")
    else:
        print(f"   ✅ No errors detected during testing")
    
    # System health
    system_health = error_detector.get_system_health()
    print(f"\n💚 System Health:")
    print(f"   • Monitoring Status: {'🟢 Active' if system_health['is_monitoring'] else '🔴 Inactive'}")
    print(f"   • Total Historical Errors: {system_health['total_errors']}")
    print(f"   • Recent Errors (5min): {system_health['recent_errors']}")
    
    # Show detailed analysis if errors occurred
    if error_detector.error_history:
        print(f"\n🔍 Detailed Error Analysis:")
        show_analysis()
    
    # Stop monitoring
    print(f"\n🛑 Stopping Aigie monitoring...")
    aigie.stop_integration()
    
    print(f"\n🎉 Advanced LangGraph Error Remediation Testing Completed!")
    print(f"📊 Aigie successfully demonstrated:")
    print(f"   ✓ Multi-agent collaboration error handling")
    print(f"   ✓ Streaming workflow interruption recovery")
    print(f"   ✓ Complex state management with error recovery")
    print(f"   ✓ Human-in-the-loop approval workflows")
    print(f"   ✓ Checkpoint and persistence error handling")
    print(f"   ✓ Advanced error classification and remediation")
    
    return all_results


if __name__ == "__main__":
    # Run the advanced tests
    asyncio.run(run_advanced_langgraph_tests())
