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


class MockMessage:
    """Simple mock message for testing."""
    
    def __init__(self, content, type="system"):
        self.content = content
        self.type = type


class MockStateGraph:
    """Simple mock state graph for testing."""
    
    def __init__(self, state_type):
        self.state_type = state_type
        self.nodes = {}
        self.edges = {}
        self.entry_point = None
    
    def add_node(self, name, func):
        """Add a node to the graph."""
        self.nodes[name] = func
    
    def add_edge(self, from_node, to_node):
        """Add an edge to the graph."""
        if from_node not in self.edges:
            self.edges[from_node] = []
        self.edges[from_node].append(to_node)
    
    def set_entry_point(self, node_name):
        """Set the entry point of the graph."""
        self.entry_point = node_name
    
    def compile(self, checkpointer=None):
        """Compile the graph into an executable app."""
        return MockCompiledGraph(self)


class MockCompiledGraph:
    """Simple mock compiled graph for testing."""
    
    def __init__(self, graph):
        self.graph = graph
    
    def invoke(self, state):
        """Execute the graph with the given state."""
        current_node = self.graph.entry_point
        
        while current_node and current_node in self.graph.nodes:
            # Execute the current node
            state = self.graph.nodes[current_node](state)
            
            # Move to next node
            if current_node in self.graph.edges:
                current_node = self.graph.edges[current_node][0]
            else:
                current_node = None
        
        return state


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
        
        # Create a simple state graph
        print("\n4. Creating LangGraph components...")
        
        # Define state structure
        from typing import TypedDict, Annotated
        
        class AgentState(TypedDict):
            messages: Annotated[list, "The messages in the conversation"]
            current_step: Annotated[str, "The current step in the workflow"]
        
        # Create the graph
        workflow = MockStateGraph(AgentState)
        
        # Add nodes
        def start_node(state: AgentState) -> AgentState:
            """Start node that initializes the state."""
            state["current_step"] = "started"
            state["messages"].append(MockMessage("Workflow started", "system"))
            return state
        
        def process_node(state: AgentState) -> AgentState:
            """Process node that simulates some work."""
            state["current_step"] = "processing"
            state["messages"].append(MockMessage("Processing...", "system"))
            time.sleep(0.1)  # Simulate work
            return state
        
        def finish_node(state: AgentState) -> AgentState:
            """Finish node that completes the workflow."""
            state["current_step"] = "finished"
            state["messages"].append(MockMessage("Workflow completed", "system"))
            return state
        
        # Add nodes to graph
        workflow.add_node("start", start_node)
        workflow.add_node("process", process_node)
        workflow.add_node("finish", finish_node)
        
        # Add edges
        workflow.add_edge("start", "process")
        workflow.add_edge("process", "finish")
        
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
