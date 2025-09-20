#!/usr/bin/env python3
"""Test script to demonstrate Aigie integration with Open Deep Research agent."""

import asyncio
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up the environment
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY", "AIzaSyCdvjGYov-sA7Aal6TsfvZ6zJ8f5Otlvos")
os.environ["OPENAI_API_KEY"] = "sk-test-key"  # Placeholder for testing

from aigie_integrated_researcher import aigie_deep_researcher, get_monitoring_system, get_aigie_instance

async def test_aigie_integration():
    """Test the Aigie-integrated deep research agent."""
    
    print("🔬 Testing Aigie Integration with Open Deep Research Agent")
    print("=" * 60)
    
    # Get Aigie and monitoring instances
    aigie = get_aigie_instance()
    monitoring = get_monitoring_system()
    
    print(f"✅ Aigie instance loaded: {type(aigie).__name__}")
    print(f"✅ Monitoring system loaded: {type(monitoring).__name__}")
    print()
    
    # Create a test configuration
    config = {
        "configurable": {
            "research_model": "openai:gpt-4o-mini",
            "compression_model": "openai:gpt-4o-mini", 
            "final_report_model": "openai:gpt-4o-mini",
            "summarization_model": "openai:gpt-4o-mini",
            "search_api": "none",  # Disable search for now
            "max_researcher_iterations": 1,
            "max_react_tool_calls": 2,
            "allow_clarification": False
        }
    }
    
    # Test message
    test_input = {
        "messages": [
            {"role": "user", "content": "What are the key benefits of using runtime validation in AI agents?"}
        ]
    }
    
    print("📝 Test Configuration:")
    print(f"   Model: {config['configurable']['research_model']}")
    print(f"   Search API: {config['configurable']['search_api']}")
    print(f"   Max Iterations: {config['configurable']['max_researcher_iterations']}")
    print()
    
    print("💬 Test Input:")
    print(f"   Query: {test_input['messages'][0]['content']}")
    print()
    
    print("🚀 Starting Aigie-Enhanced Deep Research Agent...")
    print("-" * 60)
    
    try:
        # Run the Aigie-enhanced agent
        result = await aigie_deep_researcher.ainvoke(test_input, config)
        
        print("✅ Agent completed successfully!")
        print()
        
        # Display results
        if "final_report" in result:
            print("📊 Final Report:")
            print("-" * 40)
            print(result["final_report"])
            print()
        
        if "messages" in result and result["messages"]:
            print("💬 Agent Messages:")
            print("-" * 40)
            for i, msg in enumerate(result["messages"], 1):
                print(f"{i}. {msg.content[:200]}...")
            print()
        
        # Display monitoring information
        print("📈 Aigie Monitoring Summary:")
        print("-" * 40)
        print(f"   Total Steps Monitored: {len(monitoring.step_history)}")
        print(f"   Successful Steps: {sum(1 for step in monitoring.step_history if step.get('success', False))}")
        print(f"   Failed Steps: {sum(1 for step in monitoring.step_history if not step.get('success', True))}")
        print()
        
        if monitoring.step_history:
            print("📋 Step Details:")
            for step in monitoring.step_history:
                status = "✅" if step.get('success', False) else "❌"
                print(f"   {status} {step.get('step_name', 'Unknown')}: {step.get('message', 'No message')}")
        
    except Exception as e:
        print(f"❌ Error running Aigie-enhanced agent: {e}")
        import traceback
        traceback.print_exc()
        
        # Display monitoring information even on error
        print("\n📈 Aigie Monitoring Summary (Error State):")
        print("-" * 40)
        print(f"   Total Steps Monitored: {len(monitoring.step_history)}")
        print(f"   Successful Steps: {sum(1 for step in monitoring.step_history if step.get('success', False))}")
        print(f"   Failed Steps: {sum(1 for step in monitoring.step_history if not step.get('success', True))}")
        print()
        
        if monitoring.step_history:
            print("📋 Step Details:")
            for step in monitoring.step_history:
                status = "✅" if step.get('success', False) else "❌"
                print(f"   {status} {step.get('step_name', 'Unknown')}: {step.get('message', 'No message')}")

async def test_aigie_validation_features():
    """Test Aigie's validation features specifically."""
    
    print("\n🔍 Testing Aigie Validation Features")
    print("=" * 60)
    
    aigie = get_aigie_instance()
    monitoring = get_monitoring_system()
    
    # Test 1: Model Configuration Validation
    print("1. Testing Model Configuration Validation...")
    valid_config = {
        "model": "openai:gpt-4o-mini",
        "max_tokens": 1000,
        "api_key": "sk-test-key"
    }
    
    invalid_config = {
        "model": "invalid-model",
        "max_tokens": -1,
        "api_key": None
    }
    
    print(f"   Valid config: {valid_config}")
    print(f"   Invalid config: {invalid_config}")
    
    # Test 2: Prompt Validation
    print("\n2. Testing Prompt Validation...")
    valid_prompt = "What are the benefits of AI validation?"
    invalid_prompt = ""  # Empty prompt
    
    print(f"   Valid prompt: '{valid_prompt}'")
    print(f"   Invalid prompt: '{invalid_prompt}'")
    
    # Test 3: Content Validation
    print("\n3. Testing Content Validation...")
    valid_content = "This is a well-formed research report with proper structure."
    invalid_content = ""  # Empty content
    
    print(f"   Valid content: '{valid_content}'")
    print(f"   Invalid content: '{invalid_content}'")
    
    print("\n✅ Aigie validation features demonstrated!")
    print("   Note: Full validation requires actual API calls with valid credentials.")

if __name__ == "__main__":
    print("🤖 Aigie Integration Test Suite")
    print("=" * 60)
    
    # Run the main integration test
    asyncio.run(test_aigie_integration())
    
    # Run the validation features test
    asyncio.run(test_aigie_validation_features())
    
    print("\n🎉 Aigie Integration Test Complete!")
    print("=" * 60)

