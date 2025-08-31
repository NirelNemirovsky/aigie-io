#!/usr/bin/env python3
"""
Basic LangGraph example with Aigie monitoring.

This example demonstrates how Aigie automatically detects and reports
errors in LangGraph operations without requiring any code changes.
"""

import os
import sys
import time

# Add the parent directory to the path so we can import aigie
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aigie import auto_integrate, show_status, show_analysis


def main():
    """Main example function."""
    print("🚀 Starting Aigie LangGraph Example")
    print("=" * 50)
    
    # Enable Aigie monitoring
    print("\n1. Enabling Aigie monitoring...")
    auto_integrate()
    
    # Show initial status
    print("\n2. Initial monitoring status:")
    show_status()
    
    try:
        # Import LangGraph components
        print("\n3. Importing LangGraph components...")
        from langgraph.graph import StateGraph, END
        from langgraph.checkpoint.memory import MemorySaver
        
        # Create a simple state graph
        print("\n4. Creating LangGraph components...")
        
        # Define state structure
        from typing import TypedDict, Annotated
        from langchain_core.messages import BaseMessage
        
        class AgentState(TypedDict):
            messages: Annotated[list[BaseMessage], "The messages in the conversation"]
            current_step: Annotated[str, "The current step in the workflow"]
        
        # Create the graph
        workflow = StateGraph(AgentState)
        
        # Add nodes
        def start_node(state: AgentState) -> AgentState:
            """Start node that initializes the state."""
            state["current_step"] = "started"
            state["messages"].append(BaseMessage(content="Workflow started", type="system"))
            return state
        
        def process_node(state: AgentState) -> AgentState:
            """Process node that simulates some work."""
            state["current_step"] = "processing"
            state["messages"].append(BaseMessage(content="Processing...", type="system"))
            time.sleep(0.1)  # Simulate work
            return state
        
        def finish_node(state: AgentState) -> AgentState:
            """Finish node that completes the workflow."""
            state["current_step"] = "finished"
            state["messages"].append(BaseMessage(content="Workflow completed", type="system"))
            return state
        
        # Add nodes to graph
        workflow.add_node("start", start_node)
        workflow.add_node("process", process_node)
        workflow.add_node("finish", finish_node)
        
        # Add edges
        workflow.add_edge("start", "process")
        workflow.add_edge("process", "finish")
        workflow.add_edge("finish", END)
        
        # Set entry point
        workflow.set_entry_point("start")
        
        # Compile the graph
        print("\n5. Compiling LangGraph...")
        app = workflow.compile()
        
        # Show status after graph creation
        print("\n6. Status after graph creation:")
        show_status()
        
        # Run the graph
        print("\n7. Running LangGraph...")
        initial_state = {
            "messages": [],
            "current_step": "initial"
        }
        
        result = app.invoke(initial_state)
        print(f"Final state: {result['current_step']}")
        print(f"Messages: {len(result['messages'])}")
        
        # Show status after successful execution
        print("\n8. Status after successful execution:")
        show_status()
        
        # Demonstrate error detection with invalid state
        print("\n9. Demonstrating error detection...")
        try:
            # This will cause an error (invalid state structure)
            invalid_state = {"invalid": "state"}
            app.invoke(invalid_state)
        except Exception as e:
            print(f"Expected error caught: {e}")
        
        # Show status after error
        print("\n10. Status after error detection:")
        show_status()
        
        # Show detailed analysis
        print("\n11. Detailed analysis:")
        show_analysis()
        
    except ImportError as e:
        print(f"❌ LangGraph not available: {e}")
        print("Please install LangGraph: pip install langgraph")
        return 1
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return 1
    
    print("\n✅ Example completed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
