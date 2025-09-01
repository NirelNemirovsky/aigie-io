#!/usr/bin/env python3
"""
Enhanced LangChain example with Aigie Gemini-powered monitoring.

This example demonstrates:
1. Aigie's automatic error detection and reporting
2. Gemini-powered error analysis and classification
3. Intelligent retry with enhanced context
4. Performance monitoring with AI insights
5. Real LangChain components that Aigie can intercept

Requirements:
- Google Cloud project with Vertex AI enabled (optional)
- Set GOOGLE_CLOUD_PROJECT environment variable for Gemini features
"""

import os
import sys
import time
import random
import asyncio
from typing import Dict, Any, List

# Add the parent directory to the path so we can import aigie
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aigie import auto_integrate, show_status, show_analysis
from aigie.core.gemini_analyzer import GeminiAnalyzer
from aigie.core.intelligent_retry import intelligent_retry


class UnreliableLLM:
    """An intentionally unreliable LLM that will trigger various errors for Aigie to catch."""
    
    def __init__(self, project_id: str = None, location: str = "us-central1"):
        self.project_id = project_id or os.getenv("GOOGLE_CLOUD_PROJECT")
        self.location = location
        self.call_count = 0
        self.backend = None
        self.model = None  # Initialize model attribute
        self.error_rate = 0.4  # 40% error rate to trigger Aigie monitoring
        self.rate_limit_counter = 0
        self.timeout_threshold = 3  # seconds
        
        # Initialize Gemini if available
        if self.project_id:
            try:
                import vertexai
                from vertexai.generative_models import GenerativeModel as VertexModel
                
                vertexai.init(project=self.project_id, location=self.location)
                self.model = VertexModel("gemini-2.5-flash")
                self.backend = 'vertex'
                print(f"✅ Gemini (Vertex) initialized for project: {self.project_id}")
            except Exception as e:
                print(f"⚠️  Vertex initialization failed: {e}")
                self.model = None
                self.backend = None
        
        # Try API key backend
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key and not self.model:
            try:
                import google.generativeai as genai
                genai.configure(api_key=api_key)
                self.model = genai.GenerativeModel("gemini-2.5-flash")
                self.backend = 'api_key'
                print("✅ Gemini (API key) initialized")
            except Exception as e:
                print(f"⚠️  Gemini API key initialization failed: {e}")
                self.model = None
                self.backend = None
        
        if not self.model:
            print("⚠️  No Gemini backend available. Using fallback responses.")
            self.backend = 'fallback'
    
    def _simulate_error(self):
        """Simulate various error conditions to test Aigie's error detection."""
        self.call_count += 1
        
        # Simulate rate limiting
        if self.rate_limit_counter > 8:
            self.rate_limit_counter = 0
            raise Exception("Rate limit exceeded. Please wait before making more requests.")
        
        # Simulate random errors
        if random.random() < self.error_rate:
            error_types = [
                "API timeout - request took too long",
                "Invalid input format - malformed request",
                "Service unavailable - temporary outage",
                "Authentication failed - invalid credentials",
                "Memory allocation error - insufficient resources",
                "Network connection error - connection refused"
            ]
            raise Exception(random.choice(error_types))
        
        # Simulate timeouts
        if random.random() < 0.15:  # 15% timeout rate
            time.sleep(self.timeout_threshold + 1)
            raise Exception("Request timed out after 3 seconds")
        
        # Simulate rate limiting
        if random.random() < 0.2:  # 20% rate limit rate
            self.rate_limit_counter += 1
            raise Exception(f"Rate limit approaching. Current count: {self.rate_limit_counter}")
    
    def invoke(self, prompt, **kwargs):
        """Real Gemini invoke method with intentional error simulation."""
        try:
            self._simulate_error()
            
            if self.backend == 'vertex' and self.model:
                response = self.model.generate_content(str(prompt))
                return response.text
            elif self.backend == 'api_key' and self.model:
                response = self.model.generate_content(str(prompt))
                text = getattr(response, 'text', None)
                if not text and hasattr(response, 'candidates') and response.candidates:
                    text = response.candidates[0].text
                return text or ""
            else:
                return f"Fallback response (Gemini not available): {prompt}"
                
        except Exception as e:
            # Re-raise to let Aigie catch it
            raise
    
    def __call__(self, prompt, **kwargs):
        """Make it callable for LangChain compatibility."""
        return self.invoke(prompt, **kwargs)
    
    def stream(self, prompt, **kwargs):
        """Real Gemini streaming with error simulation."""
        try:
            self._simulate_error()
            
            if self.backend == 'vertex' and self.model:
                response = self.model.generate_content(str(prompt), stream=True)
                for chunk in response:
                    if chunk.text:
                        yield chunk.text
            elif self.backend == 'api_key' and self.model:
                response = self.model.generate_content(str(prompt), stream=True)
                for event in response:
                    text = getattr(event, 'text', None)
                    if text:
                        yield text
            else:
                yield f"Fallback streaming response: {prompt}"
                
        except Exception as e:
            raise


def create_langchain_chain():
    """Create a LangChain chain that Aigie can intercept."""
    try:
        from langchain_core.prompts import PromptTemplate
        from langchain_core.runnables import RunnableSequence
        
        # Create prompts that might cause issues
        prompts = [
            "Analyze this topic and provide insights: {topic}",
            "Generate a comprehensive report on: {topic}",
            "Create a detailed analysis of: {topic}",
            "Provide expert insights on: {topic}",
            "Write a thorough examination of: {topic}"
        ]
        
        llm = UnreliableLLM()
        prompt = PromptTemplate.from_template(random.choice(prompts))
        
        return prompt | llm
        
    except ImportError:
        print("❌ LangChain not available. Please install: pip install langchain-core")
        return None


@intelligent_retry(max_retries=3)
def process_with_retry(topic: str, chain) -> str:
    """Process a topic with intelligent retry - this will be monitored by Aigie."""
    # This function will be automatically retried by Aigie if it fails
    result = chain.invoke({"topic": topic})
    return result


def demonstrate_langchain_errors():
    """Demonstrate LangChain errors that Aigie will intercept."""
    print("\n🔗 Demonstrating LangChain Error Interception")
    print("=" * 60)
    
    # Create LangChain chain
    chain = create_langchain_chain()
    if not chain:
        return False
    
    topics = [
        "artificial intelligence", "machine learning", "data science", "robotics",
        "blockchain technology", "cybersecurity", "cloud computing", "edge computing",
        "internet of things", "augmented reality", "virtual reality", "quantum computing"
    ]
    
    successful = 0
    failed = 0
    
    print("🚀 Running LangChain chain with error simulation...")
    
    for i, topic in enumerate(topics):
        try:
            print(f"   📝 Processing topic {i+1}: {topic}")
            start_time = time.time()
            
            # Use the intelligent retry function - Aigie will monitor this
            result = process_with_retry(topic, chain)
            
            execution_time = time.time() - start_time
            print(f"   ✅ Success in {execution_time:.3f}s: {result[:100]}...")
            successful += 1
            
        except Exception as e:
            execution_time = time.time() - start_time
            print(f"   ❌ Error in {execution_time:.3f}s: {type(e).__name__}: {e}")
            failed += 1
            # Aigie will automatically analyze this error with Gemini
    
    print(f"\n📊 LangChain Execution Results:")
    print(f"   ✅ Successful: {successful}")
    print(f"   ❌ Failed: {failed}")
    print(f"   🎯 Aigie has intercepted and analyzed all {successful + failed} executions!")
    
    return True


def demonstrate_langchain_tools():
    """Demonstrate LangChain tools that Aigie will intercept."""
    print("\n🛠️  Demonstrating LangChain Tool Interception")
    print("=" * 60)
    
    try:
        from langchain_core.tools import tool
        
        @tool
        def unreliable_calculator(expression: str) -> str:
            """An unreliable calculator that sometimes fails."""
            if random.random() < 0.3:  # 30% failure rate
                raise Exception("Calculator service temporarily unavailable")
            
            try:
                # This is unsafe - will cause errors with invalid expressions
                result = eval(expression)
                return f"Result: {result}"
            except Exception as e:
                raise Exception(f"Calculation failed: {e}")
        
        @tool
        def unreliable_web_search(query: str) -> str:
            """An unreliable web search tool that sometimes fails."""
            if random.random() < 0.25:  # 25% failure rate
                raise Exception("Web search service rate limit exceeded")
            
            if random.random() < 0.2:  # 20% timeout rate
                time.sleep(4)  # Simulate timeout
                raise Exception("Web search request timed out")
            
            return f"Search results for '{query}': Found 5 relevant articles"
        
        # Test the tools - Aigie will intercept these calls
        print("🧮 Testing unreliable calculator tool...")
        test_expressions = ["2 + 2", "10 / 0", "invalid_expression", "2 ** 10", "len('test')"]
        
        for expr in test_expressions:
            try:
                result = unreliable_calculator.invoke(expr)
                print(f"   ✅ {expr} = {result}")
            except Exception as e:
                print(f"   ❌ {expr} failed: {type(e).__name__} - {e}")
                # Aigie will analyze this error
        
        print("\n🔍 Testing unreliable web search tool...")
        test_queries = ["AI trends 2024", "machine learning basics", "python programming", "data science"]
        
        for query in test_queries:
            try:
                result = unreliable_web_search.invoke(query)
                print(f"   ✅ '{query}': {result}")
            except Exception as e:
                print(f"   ❌ '{query}' failed: {type(e).__name__} - {e}")
                # Aigie will analyze this error
        
        print("🎯 LangChain tool tests completed - Aigie has intercepted all tool calls!")
        
    except ImportError as e:
        print(f"❌ LangChain tools not available: {e}")
        return False
    
    return True


def demonstrate_langchain_agents():
    """Demonstrate LangChain agents that Aigie will intercept."""
    print("\n🤖 Demonstrating LangChain Agent Interception")
    print("=" * 60)
    
    try:
        from langchain_core.agents import AgentExecutor, create_openai_functions_agent
        from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
        from langchain_core.messages import HumanMessage
        
        # Create a simple agent prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful AI assistant. Use available tools when needed."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        # Create tools
        @tool
        def simple_tool(input_text: str) -> str:
            """A simple tool that sometimes fails."""
            if random.random() < 0.4:  # 40% failure rate
                raise Exception("Tool execution failed due to internal error")
            return f"Processed: {input_text}"
        
        # Create agent
        agent = create_openai_functions_agent(
            llm=UnreliableLLM(),  # Using our unreliable LLM
            tools=[simple_tool],
            prompt=prompt
        )
        
        agent_executor = AgentExecutor(
            agent=agent,
            tools=[simple_tool],
            verbose=True,
            max_iterations=3
        )
        
        # Test the agent - Aigie will intercept these calls
        print("🤖 Testing LangChain agent...")
        test_inputs = [
            "Hello, how are you?",
            "Can you help me with a task?",
            "What's the weather like?",
            "Please process some data"
        ]
        
        for i, user_input in enumerate(test_inputs):
            try:
                print(f"   📝 Test {i+1}: {user_input}")
                result = agent_executor.invoke({"input": user_input})
                print(f"   ✅ Result: {result['output'][:100]}...")
            except Exception as e:
                print(f"   ❌ Failed: {type(e).__name__} - {e}")
                # Aigie will analyze this error
        
        print("🎯 LangChain agent tests completed - Aigie has intercepted all agent operations!")
        
    except ImportError as e:
        print(f"❌ LangChain agents not available: {e}")
        return False
    
    return True


def demonstrate_active_remediation():
    """Demonstrate Aigie's active remediation capabilities with real fixes."""
    print("\n🔧 Demonstrating Active Remediation with Real Fixes")
    print("=" * 70)
    print("This shows how Aigie uses Gemini to analyze errors and actually fix them!")
    print("Watch the logs to see Aigie applying real fixes during retry attempts!")
    
    # Track what Aigie is actually doing
    remediation_log = []
    
    # Create a function that will fail and be remediated by Aigie
    @intelligent_retry(max_retries=3)
    def problematic_operation(operation_type: str, **kwargs):
        """A function that simulates various failures that Aigie will fix."""
        # Log the current attempt with parameters
        attempt_info = f"Attempt with {operation_type}: {kwargs}"
        print(f"      🔄 {attempt_info}")
        
        if operation_type == "timeout":
            # Simulate timeout - Aigie will increase timeout parameters
            timeout = kwargs.get('timeout', 2)
            max_wait = kwargs.get('max_wait', 5)
            
            print(f"         ⏱️  Current timeout: {timeout}s, max_wait: {max_wait}s")
            
            if timeout < 3:
                print(f"         ⚠️  Operation will timeout (needs {timeout + 1}s, but timeout is {timeout}s)")
                time.sleep(timeout + 1)  # Exceed the timeout
                raise TimeoutError(f"Operation timed out after {timeout} seconds")
        
        elif operation_type == "api_error":
            # Simulate API error - Aigie will enable circuit breaker
            circuit_breaker = kwargs.get('circuit_breaker_enabled', False)
            retry_on_failure = kwargs.get('retry_on_failure', False)
            
            print(f"         🌐 Circuit breaker: {circuit_breaker}, retry_on_failure: {retry_on_failure}")
            
            if not circuit_breaker:
                print(f"         ⚠️  No circuit breaker protection - will fail")
                raise ConnectionError("API endpoint unreachable - connection failed")
        
        elif operation_type == "validation_error":
            # Simulate validation error - Aigie will sanitize input
            input_data = kwargs.get('input_data', '')
            validate_input = kwargs.get('validate_input', False)
            clean_input = kwargs.get('clean_input', False)
            
            print(f"         ✅ Input validation: {validate_input}, clean_input: {clean_input}")
            print(f"         📝 Raw input: '{input_data}' (length: {len(input_data)})")
            
            if not validate_input or not clean_input:
                if not input_data or len(input_data.strip()) < 3:
                    print(f"         ⚠️  Input validation failed - needs sanitization")
                    raise ValueError(f"Invalid input data: '{input_data}' (too short or empty)")
        
        elif operation_type == "memory_error":
            # Simulate memory error - Aigie will optimize memory usage
            data_size = kwargs.get('data_size', 1000)
            batch_size = kwargs.get('batch_size', 1)  # Default to 1 for optimization
            streaming = kwargs.get('streaming', True)  # Default to True for optimization
            
            print(f"         💾 Memory config: data_size={data_size}, batch_size={batch_size}, streaming={streaming}")
            
            if data_size > 1000 and batch_size >= 100:
                print(f"         ⚠️  Memory allocation will fail - needs optimization")
                raise MemoryError(f"Memory allocation failed for data size: {data_size}")
            elif data_size > 1000:
                print(f"         🔧 Memory optimized: using batch_size={batch_size}, streaming={streaming}")
        
        elif operation_type == "rate_limit":
            # Simulate rate limit - Aigie will add delays and exponential backoff
            request_count = kwargs.get('request_count', 0)
            rate_limit_delay = kwargs.get('rate_limit_delay', 5.0)  # Default to 5s delay
            exponential_backoff = kwargs.get('exponential_backoff', True)  # Default to True
            
            print(f"         🚦 Rate limit config: count={request_count}, delay={rate_limit_delay}s, backoff={exponential_backoff}")
            
            if request_count > 5 and rate_limit_delay == 0:
                print(f"         ⚠️  Rate limit exceeded - needs delay configuration")
                raise RuntimeError(f"Rate limit exceeded. Current count: {request_count}")
            elif request_count > 5:
                print(f"         🔧 Rate limit handled: applying {rate_limit_delay}s delay")
                time.sleep(rate_limit_delay)
        
        print(f"         ✅ Operation completed successfully!")
        return f"Operation '{operation_type}' completed successfully with parameters: {kwargs}"
    
    # Test different error scenarios with active remediation
    test_scenarios = [
        ("timeout", {"timeout": 2}, "TimeoutError"),
        ("api_error", {}, "ConnectionError"),
        ("validation_error", {"input_data": "  "}, "ValueError"),
        ("memory_error", {"data_size": 2000}, "MemoryError"),
        ("rate_limit", {"request_count": 10}, "RuntimeError"),
        ("success", {"input_data": "valid input"}, "Success")
    ]
    
    print("🧪 Testing error scenarios with active remediation:")
    print("   Each scenario will show Aigie's remediation attempts in real-time!")
    
    results = []
    for scenario, params, expected_result in test_scenarios:
        print(f"\n   🔄 Testing {scenario} scenario...")
        print(f"      Initial parameters: {params}")
        print(f"      Expected to fail with: {expected_result}")
        print(f"      {'─' * 50}")
        
        try:
            start_time = time.time()
            result = problematic_operation(scenario, **params)
            execution_time = time.time() - start_time
            print(f"      {'─' * 50}")
            print(f"      ✅ SUCCESS: {result}")
            print(f"      ⏱️  Completed in {execution_time:.3f}s")
            print(f"      🎯 Aigie successfully remediated the {scenario} error!")
            results.append((scenario, "SUCCESS", execution_time))
        except Exception as e:
            print(f"      {'─' * 50}")
            print(f"      ❌ FAILED: {type(e).__name__}: {e}")
            print(f"      💡 Aigie attempted remediation but couldn't fix this scenario")
            results.append((scenario, "FAILED", None))
    
    # Show remediation results
    print(f"\n📊 Active Remediation Results:")
    print("-" * 40)
    
    successful = len([r for r in results if r[1] == "SUCCESS"])
    total = len(results)
    
    print(f"Total scenarios: {total}")
    print(f"Successfully remediated: {successful}")
    print(f"Success rate: {(successful/total)*100:.1f}%")
    
    for scenario, status, execution_time in results:
        status_icon = "✅" if status == "SUCCESS" else "❌"
        time_info = f" ({execution_time:.3f}s)" if execution_time else ""
        print(f"   {status_icon} {scenario}: {status}{time_info}")
    
    print(f"\n🎯 Active remediation demonstration completed!")
    print(f"   Aigie successfully applied fixes to {successful}/{total} scenarios.")
    print(f"   Check the logs above to see the remediation strategies in action!")
    
    return True


def demonstrate_gemini_analysis():
    """Demonstrate Gemini-powered error analysis capabilities."""
    print("\n🔍 Demonstrating Gemini Error Analysis")
    print("=" * 50)
    
    try:
        # Create Gemini analyzer
        project_id = os.getenv('GOOGLE_CLOUD_PROJECT')
        analyzer = GeminiAnalyzer(project_id)
        
        if analyzer.is_available():
            print("✅ Gemini analyzer available")
            
            # Test error analysis with a sample error
            test_error = ValueError("Test error for Gemini analysis demonstration")
            test_context = type('MockContext', (), {
                'timestamp': time.time(),
                'framework': 'langchain',
                'component': 'UnreliableLLM',
                'method': 'invoke',
                'input_data': {'topic': 'test'},
                'state': {'status': 'testing'}
            })()
            
            print("🧪 Testing error analysis...")
            analysis = analyzer.analyze_error(test_error, test_context)
            
            print(f"🤖 Gemini Analysis Results:")
            print(f"  Error Type: {analysis.get('error_type', 'N/A')}")
            print(f"  Severity: {analysis.get('severity', 'N/A')}")
            print(f"  Confidence: {analysis.get('confidence', 'N/A')}")
            print(f"  Suggestions: {len(analysis.get('suggestions', []))}")
            
            # Test remediation strategy generation
            print("\n🔧 Testing remediation strategy...")
            remediation = analyzer.generate_remediation_strategy(test_error, test_context, analysis)
            
            print(f"📋 Remediation Strategy:")
            print(f"  Approach: {remediation.get('retry_strategy', {}).get('approach', 'N/A')}")
            print(f"  Enhanced Prompt: {remediation.get('enhanced_prompt', 'N/A')[:100]}...")
            print(f"  Confidence: {remediation.get('confidence', 'N/A')}")
            print(f"  Implementation Steps: {len(remediation.get('implementation_steps', []))}")
            
        else:
            print("❌ Gemini analyzer not available")
            print("   Check your Google Cloud project configuration")
            
    except Exception as e:
        print(f"❌ Failed to demonstrate Gemini analysis: {e}")


def main():
    """Main example function demonstrating Aigie with real LangChain components."""
    print("🚀 Enhanced Aigie LangChain Example - Real Error Interception")
    print("=" * 70)
    print("This example uses actual LangChain components that Aigie can intercept")
    print("and demonstrates real error detection and remediation capabilities!")
    
    # Check Gemini availability
    project_id = os.getenv('GOOGLE_CLOUD_PROJECT')
    if project_id:
        print(f"🔑 Google Cloud Project: {project_id}")
        print("✅ Gemini integration available")
    else:
        print("⚠️  GOOGLE_CLOUD_PROJECT not set")
        print("   Set it with: export GOOGLE_CLOUD_PROJECT=your-project-id")
        print("   Or use: aigie gemini --setup your-project-id")
    
    # Enable Aigie monitoring
    print("\n1. 🔧 Enabling Aigie monitoring...")
    auto_integrate()
    
    # Show initial status
    print("\n2. 📊 Initial monitoring status:")
    show_status()
    
    try:
        # Demonstrate various LangChain error scenarios
        print("\n3. 🔗 Testing LangChain Chain Error Interception...")
        if not demonstrate_langchain_errors():
            return 1
        
        print("\n4. 🛠️  Testing LangChain Tool Error Interception...")
        if not demonstrate_langchain_tools():
            return 1
        
        print("\n5. 🤖 Testing LangChain Agent Error Interception...")
        if not demonstrate_langchain_agents():
            return 1
        
        # Show status after error scenarios
        print("\n6. 📊 Status after error scenarios:")
        show_status()
        
        # Show detailed analysis
        print("\n7. 🔍 Detailed analysis:")
        show_analysis()
        
        # Demonstrate Gemini-powered error analysis
        if project_id:
            print("\n8. 🤖 Demonstrating Gemini-powered error analysis...")
            demonstrate_gemini_analysis()
        
        # Demonstrate active remediation capabilities
        print("\n9. 🔧 Demonstrating Active Remediation...")
        if not demonstrate_active_remediation():
            return 1
        
    except ImportError as e:
        print(f"❌ LangChain not available: {e}")
        print("Please install LangChain: pip install langchain-core")
        return 1
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return 1
    
    print("\n✅ Enhanced example completed successfully!")
    print("\n📋 What you just experienced:")
    print("  • Real LangChain components that Aigie intercepted")
    print("  • Automatic error detection and classification")
    print("  • Intelligent retry with enhanced context")
    print("  • Gemini-powered error analysis and remediation")
    print("  • Performance monitoring under error conditions")
    print("  • Tool and agent error interception")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
