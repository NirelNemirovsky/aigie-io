#!/usr/bin/env python3
"""Test script to run the Open Deep Research agent directly."""

import asyncio
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up the environment
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY", "AIzaSyCdvjGYov-sA7Aal6TsfvZ6zJ8f5Otlvos")
os.environ["OPENAI_API_KEY"] = "sk-test-key"  # Placeholder for testing

from src.open_deep_research.deep_researcher import deep_researcher
from src.open_deep_research.configuration import Configuration

async def test_agent():
    """Test the deep research agent with a simple query."""
    
    # Create a test configuration using Google AI (Gemini) via direct API
    config = {
        "configurable": {
            "research_model": "google_genai:gemini-1.5-flash",
            "compression_model": "google_genai:gemini-1.5-flash", 
            "final_report_model": "google_genai:gemini-1.5-flash",
            "summarization_model": "google_genai:gemini-1.5-flash",
            "search_api": "tavily",  # Enable Tavily web search
            "max_researcher_iterations": 4,
            "max_react_tool_calls": 6,
            "allow_clarification": False,
            # Token limits for Gemini
            "research_model_max_tokens": 8192,
            "compression_model_max_tokens": 8192,
            "final_report_model_max_tokens": 8192,
            "summarization_model_max_tokens": 4096,
        }
    }
    
    # Test message - comprehensive research query
    test_input = {
        "messages": [
            {"role": "user", "content": "Research the latest breakthroughs in quantum computing and its applications in AI, machine learning, and cryptography. Focus on developments from 2024-2025, including both theoretical advances and practical implementations."}
        ]
    }
    
    print("Starting Deep Research Agent test with Google AI (Gemini) + Tavily Web Search...")
    print(f"Configuration: {config}")
    print(f"Input: {test_input}")
    print("-" * 50)
    
    try:
        # Run the agent
        result = await deep_researcher.ainvoke(test_input, config)
        print("Agent completed successfully!")
        print(f"Result: {result}")
        
    except Exception as e:
        print(f"Error running agent: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_agent())
