#!/usr/bin/env python3
"""
Integration tests for Aigie auto-integration with real agents.
"""

import asyncio
import os
import sys
import logging
import time
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add the project root to the path
sys.path.insert(0, '/Users/nirelnemirovsky/Documents/dev/aigie/aigie-io')

# Import Aigie's auto-integration
from aigie.auto_integration import auto_integrate

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_langchain_agent_integration():
    """Test with a real LangChain agent."""
    
    print("🤖 TESTING LANGCHAIN AGENT WITH AIGIE")
    print("=" * 60)
    
    try:
        # Create a simple LangChain agent
        from langchain.agents import AgentExecutor, create_tool_calling_agent
        from langchain.tools import tool
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_google_genai import ChatGoogleGenerativeAI
        
        # Define a simple tool
        @tool
        def get_weather(city: str) -> str:
            """Get the current weather for a city."""
            # Mock weather data
            weather_data = {
                "New York": "Sunny, 72°F",
                "London": "Cloudy, 65°F", 
                "Tokyo": "Rainy, 68°F",
                "Paris": "Partly cloudy, 70°F"
            }
            return weather_data.get(city, f"Weather data not available for {city}")
        
        @tool
        def calculate_math(expression: str) -> str:
            """Calculate a mathematical expression safely."""
            try:
                # Simple safe evaluation
                allowed_chars = set('0123456789+-*/.() ')
                if all(c in allowed_chars for c in expression):
                    result = eval(expression)
                    return f"The result of {expression} is {result}"
                else:
                    return "Invalid mathematical expression"
            except:
                return "Error calculating expression"
        
        # Create the agent
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.1,
            google_api_key=os.getenv("GEMINI_API_KEY")
        )
        
        tools = [get_weather, calculate_math]
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant with access to weather and math tools."),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}")
        ])
        
        agent = create_tool_calling_agent(llm, tools, prompt)
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
        
        # Initialize Aigie with auto-integration
        print("   Initializing Aigie with auto-integration...")
        aigie = auto_integrate()
        
        # Test the agent with Aigie monitoring
        print("\n   Testing agent execution with Aigie...")
        
        test_queries = [
            "What's the weather in New York?",
            "Calculate 15 * 8 + 42",
            "Tell me about the weather in London and calculate 100 / 4",
            "What's 2 to the power of 10?",
            "Get weather for Tokyo and Paris"
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n   Query {i}: {query}")
            
            start_time = time.time()
            
            # Execute the agent (Aigie will automatically monitor it)
            result = agent_executor.invoke({"input": query})
            
            execution_time = time.time() - start_time
            
            print(f"      ✅ Result: {result.get('output', 'No output')[:100]}...")
            print(f"      ⚡ Time: {execution_time:.3f}s")
            
            # Get Aigie status
            try:
                status = aigie.get_integration_status()
                print(f"      📊 Aigie status: {status.get('is_integrated', False)}")
                
                # Check for error detection
                error_info = status.get('error_detector', {})
                print(f"      🎯 Errors detected: {error_info.get('total_errors', 0)}")
            except Exception as e:
                print(f"      ⚠️  Could not get Aigie status: {e}")
        
        print("\n   ✅ LangChain agent test completed successfully!")
        return True
        
    except Exception as e:
        print(f"   ❌ LangChain agent test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_langgraph_workflow_integration():
    """Test with a real LangGraph workflow."""
    
    print("\n🔄 TESTING LANGGRAPH WORKFLOW WITH AIGIE")
    print("=" * 60)
    
    try:
        # Create a simple LangGraph workflow
        from langgraph.graph import StateGraph, END
        from langchain_core.messages import HumanMessage, AIMessage
        from langchain_google_genai import ChatGoogleGenerativeAI
        
        # Define the workflow state
        from typing import TypedDict, List
        from langchain_core.messages import BaseMessage
        
        class WorkflowState(TypedDict):
            messages: List[BaseMessage]
            search_results: str
            summary: str
        
        # Create the workflow
        def process_node(state: WorkflowState) -> WorkflowState:
            messages = state["messages"]
            last_message = messages[-1]
            content = last_message.content.lower()
            
            if "search" in content:
                search_results = f"Search results for '{last_message.content}': Found 5 relevant articles about {last_message.content}"
                return {
                    **state,
                    "search_results": search_results,
                    "messages": messages + [AIMessage(content=f"I found: {search_results}")]
                }
            elif "summarize" in content:
                text_to_summarize = state.get("search_results", last_message.content)
                summary = f"Summary: {text_to_summarize[:100]}... (This is a mock summary)"
                return {
                    **state,
                    "summary": summary,
                    "messages": messages + [AIMessage(content=f"Summary: {summary}")]
                }
            else:
                return {
                    **state,
                    "messages": messages + [AIMessage(content="I processed your request")]
                }
        
        # Build the graph
        workflow = StateGraph(WorkflowState)
        workflow.add_node("process", process_node)
        
        # Set entry point
        workflow.set_entry_point("process")
        
        # Add edge to end
        workflow.add_edge("process", END)
        
        app = workflow.compile()
        
        # Initialize Aigie (may already be integrated from previous test)
        print("   Using Aigie for LangGraph workflow...")
        aigie = auto_integrate()
        
        # Test the workflow with Aigie monitoring
        print("\n   Testing workflow execution with Aigie...")
        
        test_inputs = [
            {"messages": [HumanMessage(content="Search for information about AI")]},
            {"messages": [HumanMessage(content="Summarize the search results")]},
            {"messages": [HumanMessage(content="Search for machine learning trends")]},
            {"messages": [HumanMessage(content="Summarize what you found")]},
        ]
        
        for i, input_data in enumerate(test_inputs, 1):
            print(f"\n   Workflow Step {i}: {input_data['messages'][0].content}")
            
            start_time = time.time()
            
            # Execute the workflow (Aigie will automatically monitor it)
            result = app.invoke(input_data)
            
            execution_time = time.time() - start_time
            
            print(f"      ✅ Result: {len(result.get('messages', []))} messages")
            print(f"      ⚡ Time: {execution_time:.3f}s")
            
            # Get Aigie status
            try:
                status = aigie.get_integration_status()
                error_info = status.get('error_detector', {})
                print(f"      🎯 Errors detected: {error_info.get('total_errors', 0)}")
            except Exception as e:
                print(f"      ⚠️  Could not get Aigie status: {e}")
        
        print("\n   ✅ LangGraph workflow test completed successfully!")
        return True
        
    except Exception as e:
        print(f"   ❌ LangGraph workflow test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_aigie_monitoring_integration():
    """Test Aigie's monitoring capabilities."""
    
    print("\n📊 TESTING AIGIE MONITORING")
    print("=" * 60)
    
    try:
        # Initialize Aigie (may already be integrated from previous test)
        aigie = auto_integrate()
        
        # Create a simple test agent that will be monitored
        class TestAgent:
            def invoke(self, input_data: dict) -> dict:
                # Simulate some processing
                time.sleep(0.1)
                return {"output": f"Processed: {input_data.get('input', '')}"}
        
        test_agent = TestAgent()
        
        # Run multiple executions to generate monitoring data
        print("   Running multiple agent executions...")
        
        for i in range(5):
            test_agent.invoke({"input": f"Test execution {i}"})
        
        # Test monitoring data collection
        print("\n   Testing monitoring data collection...")
        
        status = aigie.get_integration_status()
        print(f"      ✅ Integration status: {status.get('is_integrated', False)}")
        
        # Test detailed analysis
        print("\n   Testing detailed analysis...")
        
        analysis = aigie.get_detailed_analysis()
        print(f"      ✅ Analysis available: {len(analysis)} categories")
        
        for category, data in analysis.items():
            if isinstance(data, dict):
                print(f"         {category}: {len(data)} items")
            else:
                print(f"         {category}: {data}")
        
        # Test error summary
        print("\n   Testing error summary...")
        
        error_summary = analysis.get('error_summary', {})
        print(f"      ✅ Total errors: {error_summary.get('total_errors', 0)}")
        
        if error_summary.get('severity_distribution'):
            for severity, count in error_summary['severity_distribution'].items():
                print(f"         {severity}: {count}")
        
        print("\n   ✅ Aigie monitoring test completed successfully!")
        return True
        
    except Exception as e:
        print(f"   ❌ Aigie monitoring test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run all integration tests with Aigie."""
    
    print("🚀 AIGIE INTEGRATION TEST SUITE")
    print("=" * 60)
    print("This test validates Aigie with actual LangChain and LangGraph agents")
    print("using auto-integration, just like in production.")
    print()
    
    tests = [
        ("LangChain Agent", test_langchain_agent_integration),
        ("LangGraph Workflow", test_langgraph_workflow_integration),
        ("Aigie Monitoring", test_aigie_monitoring_integration)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*60}")
        print(f"RUNNING: {test_name}")
        print('='*60)
        
        try:
            start_time = time.time()
            result = await test_func()
            test_time = time.time() - start_time
            
            results.append((test_name, result, test_time))
            
            if result:
                print(f"\n✅ {test_name} PASSED ({test_time:.2f}s)")
            else:
                print(f"\n❌ {test_name} FAILED ({test_time:.2f}s)")
                
        except Exception as e:
            print(f"\n💥 {test_name} CRASHED: {e}")
            results.append((test_name, False, 0))
    
    # Summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print('='*60)
    
    passed = sum(1 for _, result, _ in results if result)
    total = len(results)
    total_time = sum(time for _, _, time in results)
    
    for test_name, result, test_time in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name:<30} ({test_time:.2f}s)")
    
    print(f"\nOverall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    print(f"Total time: {total_time:.2f}s")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Aigie works correctly with real agents.")
    else:
        print(f"\n⚠️  {total-passed} tests failed. Please review the issues above.")
    
    return passed == total


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
