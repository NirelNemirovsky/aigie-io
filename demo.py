#!/usr/bin/env python3
"""
Aigie Demonstration Script

This script demonstrates Aigie's core capabilities without requiring
external dependencies like LangChain or LangGraph.
"""

import time
import sys
import os

# Add the current directory to the path so we can import aigie
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from aigie import auto_integrate, show_status, show_analysis
from aigie.utils.decorators import monitor_execution, monitor_langchain, monitor_langgraph


def simulate_langchain_operation():
    """Simulate a LangChain operation that might fail."""
    print("  🔗 Simulating LangChain operation...")
    time.sleep(0.1)  # Simulate processing time
    
    # Simulate an error
    raise Exception("Simulated LangChain API error: Rate limit exceeded")


def simulate_langgraph_operation():
    """Simulate a LangGraph operation that might fail."""
    print("  🕸️  Simulating LangGraph operation...")
    time.sleep(0.1)  # Simulate processing time
    
    # Simulate an error
    raise Exception("Simulated LangGraph state error: Invalid state transition")


@monitor_execution(framework="demo", component="DemoComponent", method="demo_method")
def monitored_function():
    """A function monitored by Aigie."""
    print("  📊 Running monitored function...")
    time.sleep(0.2)  # Simulate work
    return "Function completed successfully"


@monitor_langchain(component="SimulatedChain", method="run")
def simulated_langchain_chain():
    """Simulate a LangChain chain with monitoring."""
    print("  🔗 Running simulated LangChain chain...")
    time.sleep(0.15)  # Simulate processing
    return "Chain executed successfully"


@monitor_langgraph(component="SimulatedGraph", method="invoke")
def simulated_langgraph_graph():
    """Simulate a LangGraph graph with monitoring."""
    print("  🕸️  Running simulated LangGraph graph...")
    time.sleep(0.15)  # Simulate processing
    return "Graph executed successfully"


def main():
    """Main demonstration function."""
    print("🚀 Aigie Demonstration")
    print("=" * 50)
    print("This demo shows Aigie's real-time error detection and monitoring capabilities.")
    print()
    
    # Step 1: Enable Aigie monitoring
    print("1️⃣  Enabling Aigie monitoring...")
    auto_integrate()
    print("✅ Aigie monitoring enabled!")
    print()
    
    # Step 2: Show initial status
    print("2️⃣  Initial monitoring status:")
    show_status()
    print()
    
    # Step 3: Demonstrate successful operations
    print("3️⃣  Demonstrating successful operations...")
    
    try:
        result = monitored_function()
        print(f"   ✅ Result: {result}")
    except Exception as e:
        print(f"   ❌ Unexpected error: {e}")
    
    try:
        result = simulated_langchain_chain()
        print(f"   ✅ Result: {result}")
    except Exception as e:
        print(f"   ❌ Unexpected error: {e}")
    
    try:
        result = simulated_langgraph_graph()
        print(f"   ✅ Result: {result}")
    except Exception as e:
        print(f"   ❌ Unexpected error: {e}")
    
    print()
    
    # Step 4: Show status after successful operations
    print("4️⃣  Status after successful operations:")
    show_status()
    print()
    
    # Step 5: Demonstrate error detection
    print("5️⃣  Demonstrating error detection...")
    
    print("   Simulating LangChain error...")
    try:
        simulate_langchain_operation()
    except Exception as e:
        print(f"   ❌ Error caught: {e}")
    
    print("   Simulating LangGraph error...")
    try:
        simulate_langgraph_operation()
    except Exception as e:
        print(f"   ❌ Error caught: {e}")
    
    print()
    
    # Step 6: Show status after errors
    print("6️⃣  Status after error detection:")
    show_status()
    print()
    
    # Step 7: Show detailed analysis
    print("7️⃣  Detailed analysis:")
    show_analysis()
    print()
    
    # Step 8: Demonstrate performance monitoring
    print("8️⃣  Demonstrating performance monitoring...")
    
    # Run a slow operation
    print("   Running slow operation...")
    start_time = time.time()
    
    @monitor_execution(framework="demo", component="SlowComponent", method="slow_method")
    def slow_operation():
        time.sleep(0.5)  # Simulate slow operation
        return "Slow operation completed"
    
    try:
        result = slow_operation()
        print(f"   ✅ {result}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    execution_time = time.time() - start_time
    print(f"   ⏱️  Total execution time: {execution_time:.2f}s")
    print()
    
    # Step 9: Final status
    print("9️⃣  Final monitoring status:")
    show_status()
    print()
    
    # Step 10: Summary
    print("🎯 Demo Summary:")
    print("   • Aigie monitoring was enabled automatically")
    print("   • Multiple operations were monitored")
    print("   • Errors were detected and classified in real-time")
    print("   • Performance metrics were collected")
    print("   • Rich console output was provided")
    print()
    print("✨ Aigie provides seamless monitoring without code changes!")
    print("   Users can continue using LangChain and LangGraph normally.")
    print("   All errors, performance issues, and system health are monitored automatically.")
    
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n🛑 Demo interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Demo failed with error: {e}")
        sys.exit(1)
