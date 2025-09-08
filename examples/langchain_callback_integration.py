"""
Example demonstrating Aigie's LangChain callback integration for runtime error handling.

This example shows how Aigie's enhanced callback system integrates with LangChain
to provide real-time error detection and automatic remediation.
"""

import os
import sys
from typing import Dict, Any

# Add the parent directory to the path to import aigie
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aigie import auto_integrate, get_langchain_callback_handler, register_callback_with_langchain_component
from aigie.utils.config import AigieConfig

# Example LangChain components (these would be your actual LangChain components)
class MockChatModel:
    """Mock ChatModel for demonstration."""
    
    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        self.model_name = model_name
        self.callbacks = None
    
    def invoke(self, messages, **kwargs):
        """Mock invoke method that can fail."""
        if "error" in str(messages):
            raise Exception("Mock LLM error: Invalid input")
        return {"content": f"Response from {self.model_name}"}

class MockTool:
    """Mock Tool for demonstration."""
    
    def __init__(self, name: str = "mock_tool"):
        self.name = name
        self.callbacks = None
    
    def invoke(self, input_str: str, **kwargs):
        """Mock tool invoke that can fail."""
        if "fail" in input_str.lower():
            raise Exception("Mock tool error: Tool execution failed")
        return f"Tool {self.name} processed: {input_str}"

class MockChain:
    """Mock Chain for demonstration."""
    
    def __init__(self, name: str = "mock_chain"):
        self.name = name
        self.callbacks = None
    
    def invoke(self, inputs: Dict[str, Any], **kwargs):
        """Mock chain invoke that can fail."""
        if "error" in str(inputs):
            raise Exception("Mock chain error: Chain execution failed")
        return {"output": f"Chain {self.name} processed: {inputs}"}

def demonstrate_callback_integration():
    """Demonstrate Aigie's LangChain callback integration."""
    
    print("🚀 Aigie LangChain Callback Integration Demo")
    print("=" * 50)
    
    # Configure Aigie with enhanced error handling
    config = AigieConfig(
        enable_gemini_analysis=True,
        enable_automatic_retry=True,
        log_level="INFO"
    )
    
    # Initialize Aigie
    print("\n1. Initializing Aigie...")
    integrator = auto_integrate(config)
    
    # Get the callback handler
    print("\n2. Getting LangChain callback handler...")
    callback_handler = get_langchain_callback_handler()
    
    if callback_handler:
        print("✅ Callback handler obtained successfully")
        print(f"   Handler type: {type(callback_handler).__name__}")
    else:
        print("❌ Failed to get callback handler")
        return
    
    # Create mock LangChain components
    print("\n3. Creating mock LangChain components...")
    chat_model = MockChatModel("gpt-3.5-turbo")
    tool = MockTool("search_tool")
    chain = MockChain("research_chain")
    
    # Register callback handler with components
    print("\n4. Registering callback handler with components...")
    
    # Register with chat model
    success = register_callback_with_langchain_component(chat_model)
    print(f"   ChatModel registration: {'✅ Success' if success else '❌ Failed'}")
    
    # Register with tool
    success = register_callback_with_langchain_component(tool)
    print(f"   Tool registration: {'✅ Success' if success else '❌ Failed'}")
    
    # Register with chain
    success = register_callback_with_langchain_component(chain)
    print(f"   Chain registration: {'✅ Success' if success else '❌ Failed'}")
    
    # Test successful operations
    print("\n5. Testing successful operations...")
    
    try:
        # Test successful chat model
        result = chat_model.invoke("Hello, how are you?")
        print(f"   ChatModel success: {result}")
    except Exception as e:
        print(f"   ChatModel error: {e}")
    
    try:
        # Test successful tool
        result = tool.invoke("Search for information about AI")
        print(f"   Tool success: {result}")
    except Exception as e:
        print(f"   Tool error: {e}")
    
    try:
        # Test successful chain
        result = chain.invoke({"query": "What is machine learning?"})
        print(f"   Chain success: {result}")
    except Exception as e:
        print(f"   Chain error: {e}")
    
    # Test error scenarios (these should trigger Aigie's error handling)
    print("\n6. Testing error scenarios (should trigger Aigie error handling)...")
    
    try:
        # Test chat model error
        result = chat_model.invoke("This will cause an error")
        print(f"   ChatModel result: {result}")
    except Exception as e:
        print(f"   ChatModel error caught: {e}")
    
    try:
        # Test tool error
        result = tool.invoke("This will fail")
        print(f"   Tool result: {result}")
    except Exception as e:
        print(f"   Tool error caught: {e}")
    
    try:
        # Test chain error
        result = chain.invoke({"query": "This will cause an error"})
        print(f"   Chain result: {result}")
    except Exception as e:
        print(f"   Chain error caught: {e}")
    
    # Show analysis
    print("\n7. Aigie Analysis:")
    from aigie import show_analysis
    show_analysis()
    
    # Show status
    print("\n8. Aigie Status:")
    from aigie import show_status
    show_status()
    
    print("\n✅ Demo completed!")
    print("\nKey Features Demonstrated:")
    print("• LangChain callback handler integration")
    print("• Automatic error detection and classification")
    print("• Real-time error remediation attempts")
    print("• Enhanced error context with LangChain-specific data")
    print("• Comprehensive error analysis and reporting")

def demonstrate_manual_callback_usage():
    """Demonstrate manual callback handler usage."""
    
    print("\n" + "=" * 50)
    print("🔧 Manual Callback Handler Usage")
    print("=" * 50)
    
    # Get the callback handler
    callback_handler = get_langchain_callback_handler()
    
    if not callback_handler:
        print("❌ No callback handler available")
        return
    
    print("✅ Callback handler available for manual use")
    print(f"   Handler methods: {[method for method in dir(callback_handler) if method.startswith('on_')]}")
    
    # Demonstrate manual error handling
    print("\nManual error handling example:")
    
    # Simulate a tool error
    try:
        error = Exception("Simulated tool error")
        serialized = {"name": "test_tool", "id": "test_id"}
        
        # Manually call the callback handler
        result = callback_handler.on_tool_error(
            error,
            serialized=serialized,
            run_id="test_run",
            parent_run_id=None,
            tags=["test"],
            metadata={"test": True}
        )
        
        print(f"   Error handling result: {result}")
        
    except Exception as e:
        print(f"   Error in manual handling: {e}")

if __name__ == "__main__":
    # Set up environment
    os.environ.setdefault("GEMINI_API_KEY", "your-gemini-api-key-here")
    
    try:
        demonstrate_callback_integration()
        demonstrate_manual_callback_usage()
    except KeyboardInterrupt:
        print("\n\n⏹️  Demo interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up
        from aigie import disable_monitoring
        disable_monitoring()
        print("\n🧹 Cleanup completed")
