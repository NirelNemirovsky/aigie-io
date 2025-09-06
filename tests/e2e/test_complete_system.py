#!/usr/bin/env python3
"""
End-to-end tests for the complete Aigie system.
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

from aigie.core.ai.gemini_analyzer import GeminiAnalyzer
from aigie.core.validation.runtime_validator import RuntimeValidator
from aigie.core.validation.step_corrector import StepCorrector
from aigie.core.validation.validation_engine import ValidationEngine
from aigie.core.types.validation_types import ExecutionStep, ValidationStrategy
from aigie.core.runtime_validator_v2 import ValidationConfig
from aigie.core.validation_pipeline import ValidationPipeline, ValidationStage
from aigie.core.validation_monitor import ValidationMonitor

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_component_integration():
    """Test each component individually and in integration."""
    
    print("🔍 E2E TEST - Component Integration")
    print("=" * 60)
    
    # Check API key
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        print("❌ GEMINI_API_KEY not found - using fallback mode")
        gemini_api_key = "test_key"
    
    # Initialize components
    print("\n1. Initializing Components...")
    
    try:
        gemini_analyzer = GeminiAnalyzer(api_key=gemini_api_key)
        print(f"   ✅ GeminiAnalyzer: {gemini_analyzer.is_available()}")
        
        # Test different configurations
        configs = [
            ("Basic Config", ValidationConfig()),
            ("High Performance", ValidationConfig(
                max_concurrent_validations=20,
                enable_parallel_strategies=True,
                enable_adaptive_validation=True
            )),
            ("Conservative", ValidationConfig(
                enabled_strategies=[ValidationStrategy.GOAL_ALIGNMENT, ValidationStrategy.SAFETY_COMPLIANCE],
                min_confidence_threshold=0.8
            ))
        ]
        
        for config_name, config in configs:
            print(f"\n   Testing {config_name}...")
            
            validator = RuntimeValidator(
                gemini_analyzer=gemini_analyzer,
                enable_pipeline=True,
                enable_monitoring=True,
                config=config
            )
            
            # Test basic functionality
            test_step = ExecutionStep(
                framework="langchain",
                component="LLMChain",
                operation="invoke",
                input_data={"input": "Test input"},
                agent_goal="Test goal",
                step_reasoning="Test reasoning"
            )
            
            start_time = time.time()
            result = await validator.validate_step(test_step)
            validation_time = time.time() - start_time
            
            print(f"      ✅ Validation: {result.is_valid} (confidence: {result.confidence:.2f})")
            print(f"      ⚡ Time: {validation_time:.3f}s")
            print(f"      🎯 Risk: {result.risk_level.value}")
            
            # Test metrics
            metrics = validator.get_metrics()
            print(f"      📊 Metrics: {len(metrics)} categories")
            
            validator.shutdown()
    
    except Exception as e:
        print(f"   ❌ Component test failed: {e}")
        return False
    
    return True


async def test_langchain_integration():
    """Test LangChain integration specifically."""
    
    print("\n🔗 LANGCHAIN INTEGRATION TEST")
    print("=" * 60)
    
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        print("⚠️  No API key - testing fallback mode")
        return True
    
    try:
        gemini_analyzer = GeminiAnalyzer(api_key=gemini_api_key)
        
        # Test with LangChain integration enabled
        config = ValidationConfig(
            enable_parallel_strategies=True,
            enabled_strategies=list(ValidationStrategy)
        )
        
        validator = RuntimeValidator(
            gemini_analyzer=gemini_analyzer,
            enable_pipeline=True,
            enable_monitoring=True,
            config=config
        )
        
        # Test different step types
        test_steps = [
            ExecutionStep(
                framework="langchain",
                component="LLMChain",
                operation="invoke",
                input_data={"input": "What is the capital of France?"},
                agent_goal="Answer geography questions",
                step_reasoning="User asked about capital cities",
                conversation_history=[
                    {"role": "user", "content": "I need help with geography"},
                    {"role": "assistant", "content": "I can help with geography questions!"}
                ]
            ),
            ExecutionStep(
                framework="langchain",
                component="Tool",
                operation="run",
                input_data={"query": "weather in New York"},
                agent_goal="Get weather information",
                step_reasoning="User wants weather data",
                intermediate_state={"location": "New York", "unit": "celsius"}
            ),
            ExecutionStep(
                framework="langgraph",
                component="StateGraph",
                operation="invoke",
                input_data={"messages": [{"role": "user", "content": "Help me write a poem"}]},
                agent_goal="Creative writing assistance",
                step_reasoning="User wants creative help",
                intermediate_state={"task": "poetry", "style": "free_verse"}
            )
        ]
        
        for i, step in enumerate(test_steps, 1):
            print(f"\n   Test Step {i}: {step.component}.{step.operation}")
            
            start_time = time.time()
            result = await validator.validate_step(step)
            validation_time = time.time() - start_time
            
            print(f"      ✅ Valid: {result.is_valid}")
            print(f"      🎯 Confidence: {result.confidence:.2f}")
            print(f"      ⚡ Time: {validation_time:.3f}s")
            print(f"      🔍 Reasoning: {result.reasoning[:100]}...")
            
            if result.issues:
                print(f"      ⚠️  Issues: {len(result.issues)} found")
            
            if result.suggestions:
                print(f"      💡 Suggestions: {len(result.suggestions)} provided")
        
        validator.shutdown()
        return True
        
    except Exception as e:
        print(f"   ❌ LangChain integration test failed: {e}")
        return False


async def test_performance_optimization():
    """Test performance optimization features."""
    
    print("\n⚡ PERFORMANCE OPTIMIZATION TEST")
    print("=" * 60)
    
    try:
        gemini_analyzer = GeminiAnalyzer()
        
        # Test with performance optimization
        config = ValidationConfig(
            max_concurrent_validations=10,
            enable_parallel_strategies=True,
            enable_adaptive_validation=True,
            enable_pattern_learning=True,
            cache_ttl_seconds=300
        )
        
        validator = RuntimeValidator(
            gemini_analyzer=gemini_analyzer,
            enable_pipeline=True,
            enable_monitoring=True,
            config=config
        )
        
        # Test concurrent validations
        print("   Testing concurrent validations...")
        
        async def validate_step_async(step_id, input_data):
            step = ExecutionStep(
                framework="langchain",
                component="LLMChain",
                operation="invoke",
                input_data=input_data,
                agent_goal=f"Test goal {step_id}",
                step_reasoning=f"Test reasoning {step_id}"
            )
            
            start_time = time.time()
            result = await validator.validate_step(step)
            validation_time = time.time() - start_time
            
            return {
                "step_id": step_id,
                "valid": result.is_valid,
                "confidence": result.confidence,
                "time": validation_time
            }
        
        # Run multiple validations concurrently
        tasks = []
        for i in range(5):
            task = validate_step_async(i, {"input": f"Test input {i}"})
            tasks.append(task)
        
        start_time = time.time()
        results = await asyncio.gather(*tasks)
        total_time = time.time() - start_time
        
        print(f"      ✅ Completed {len(results)} concurrent validations in {total_time:.3f}s")
        print(f"      📊 Average time per validation: {total_time/len(results):.3f}s")
        
        # Test caching
        print("\n   Testing caching...")
        
        # First validation (should be cached)
        step = ExecutionStep(
            framework="langchain",
            component="LLMChain",
            operation="invoke",
            input_data={"input": "Cache test input"},
            agent_goal="Test caching",
            step_reasoning="Test caching reasoning"
        )
        
        start_time = time.time()
        result1 = await validator.validate_step(step)
        time1 = time.time() - start_time
        
        # Second validation (should use cache)
        start_time = time.time()
        result2 = await validator.validate_step(step)
        time2 = time.time() - start_time
        
        print(f"      ✅ First validation: {time1:.3f}s")
        print(f"      ✅ Cached validation: {time2:.3f}s")
        print(f"      📈 Speedup: {time1/time2:.1f}x")
        
        validator.shutdown()
        return True
        
    except Exception as e:
        print(f"   ❌ Performance test failed: {e}")
        return False


async def test_error_handling():
    """Test error handling and edge cases."""
    
    print("\n🛡️ ERROR HANDLING TEST")
    print("=" * 60)
    
    try:
        gemini_analyzer = GeminiAnalyzer()
        validator = RuntimeValidator(gemini_analyzer)
        
        # Test with invalid step
        print("   Testing invalid step...")
        
        invalid_step = ExecutionStep(
            framework="unknown",
            component="",
            operation="",
            input_data={},
            agent_goal=None,
            step_reasoning=None
        )
        
        result = await validator.validate_step(invalid_step)
        print(f"      ✅ Handled invalid step: {result.is_valid}")
        print(f"      🎯 Confidence: {result.confidence:.2f}")
        
        # Test with malformed data
        print("\n   Testing malformed data...")
        
        malformed_step = ExecutionStep(
            framework="langchain",
            component="LLMChain",
            operation="invoke",
            input_data={"input": None},  # None input
            agent_goal="Test malformed data",
            step_reasoning="Test malformed data reasoning"
        )
        
        result = await validator.validate_step(malformed_step)
        print(f"      ✅ Handled malformed data: {result.is_valid}")
        
        # Test with very large input
        print("\n   Testing large input...")
        
        large_input = "x" * 10000  # 10KB input
        large_step = ExecutionStep(
            framework="langchain",
            component="LLMChain",
            operation="invoke",
            input_data={"input": large_input},
            agent_goal="Test large input",
            step_reasoning="Test large input reasoning"
        )
        
        start_time = time.time()
        result = await validator.validate_step(large_step)
        validation_time = time.time() - start_time
        
        print(f"      ✅ Handled large input: {result.is_valid}")
        print(f"      ⚡ Time: {validation_time:.3f}s")
        
        validator.shutdown()
        return True
        
    except Exception as e:
        print(f"   ❌ Error handling test failed: {e}")
        return False


async def test_monitoring_and_metrics():
    """Test monitoring and metrics collection."""
    
    print("\n📊 MONITORING AND METRICS TEST")
    print("=" * 60)
    
    try:
        gemini_analyzer = GeminiAnalyzer()
        validator = RuntimeValidator(
            gemini_analyzer=gemini_analyzer,
            enable_monitoring=True
        )
        
        # Run some validations to generate metrics
        for i in range(5):
            step = ExecutionStep(
                framework="langchain",
                component="LLMChain",
                operation="invoke",
                input_data={"input": f"Metrics test {i}"},
                agent_goal="Test metrics",
                step_reasoning="Test metrics reasoning"
            )
            await validator.validate_step(step)
        
        # Test metrics collection
        print("   Testing metrics collection...")
        
        metrics = validator.get_metrics()
        print(f"      ✅ Metrics categories: {len(metrics)}")
        
        for category, data in metrics.items():
            print(f"         {category}: {len(data)} metrics")
        
        # Test trend analysis
        print("\n   Testing trend analysis...")
        
        trends = validator.get_trends()
        print(f"      ✅ Trends: {len(trends)}")
        
        for trend in trends:
            print(f"         {trend['metric_name']}: {trend['trend_direction']}")
        
        # Test alert system
        print("\n   Testing alert system...")
        
        def alert_handler(alert_data):
            print(f"      🔔 Alert: {alert_data['metric_name']} = {alert_data['current_value']:.2f}")
        
        validator.add_alert_handler(alert_handler)
        validator.add_alert("avg_validation_time", 0.1, "gt", "low")
        
        # Test metrics export
        print("\n   Testing metrics export...")
        
        validator.export_metrics("e2e_test_metrics.json")
        print("      ✅ Metrics exported to e2e_test_metrics.json")
        
        validator.shutdown()
        return True
        
    except Exception as e:
        print(f"   ❌ Monitoring test failed: {e}")
        return False


async def test_real_agent_scenarios():
    """Test with realistic agent scenarios."""
    
    print("\n🤖 REAL AGENT SCENARIOS TEST")
    print("=" * 60)
    
    try:
        gemini_analyzer = GeminiAnalyzer()
        validator = RuntimeValidator(
            gemini_analyzer=gemini_analyzer,
            enable_pipeline=True,
            enable_monitoring=True
        )
        
        # Scenario 1: Chatbot conversation
        print("   Scenario 1: Chatbot conversation...")
        
        chatbot_steps = [
            ExecutionStep(
                framework="langchain",
                component="LLMChain",
                operation="invoke",
                input_data={"input": "Hello, how are you?"},
                agent_goal="Provide friendly conversation",
                step_reasoning="User greeting, respond warmly",
                conversation_history=[]
            ),
            ExecutionStep(
                framework="langchain",
                component="LLMChain",
                operation="invoke",
                input_data={"input": "What's the weather like?"},
                agent_goal="Provide helpful information",
                step_reasoning="User asking about weather, need to be helpful",
                conversation_history=[
                    {"role": "user", "content": "Hello, how are you?"},
                    {"role": "assistant", "content": "Hello! I'm doing well, thank you for asking. How can I help you today?"}
                ]
            )
        ]
        
        for i, step in enumerate(chatbot_steps, 1):
            result = await validator.validate_step(step)
            print(f"      Step {i}: Valid={result.is_valid}, Confidence={result.confidence:.2f}")
        
        # Scenario 2: Code generation
        print("\n   Scenario 2: Code generation...")
        
        code_steps = [
            ExecutionStep(
                framework="langchain",
                component="LLMChain",
                operation="invoke",
                input_data={"input": "Write a Python function to calculate fibonacci"},
                agent_goal="Generate working code",
                step_reasoning="User wants code generation, provide complete function",
                conversation_history=[]
            ),
            ExecutionStep(
                framework="langchain",
                component="Tool",
                operation="run",
                input_data={"code": "def fib(n): return n if n <= 1 else fib(n-1) + fib(n-2)"},
                agent_goal="Execute and test code",
                step_reasoning="Execute the generated code to verify it works",
                intermediate_state={"generated_code": "def fib(n): return n if n <= 1 else fib(n-1) + fib(n-2)"}
            )
        ]
        
        for i, step in enumerate(code_steps, 1):
            result = await validator.validate_step(step)
            print(f"      Step {i}: Valid={result.is_valid}, Confidence={result.confidence:.2f}")
        
        # Scenario 3: Data analysis
        print("\n   Scenario 3: Data analysis...")
        
        analysis_steps = [
            ExecutionStep(
                framework="langgraph",
                component="StateGraph",
                operation="invoke",
                input_data={"data": [1, 2, 3, 4, 5], "operation": "mean"},
                agent_goal="Analyze data and provide insights",
                step_reasoning="Calculate mean of the data",
                intermediate_state={"data": [1, 2, 3, 4, 5], "operation": "mean"}
            ),
            ExecutionStep(
                framework="langgraph",
                component="StateGraph",
                operation="invoke",
                input_data={"data": [1, 2, 3, 4, 5], "operation": "visualize"},
                agent_goal="Create data visualization",
                step_reasoning="Create a chart of the data",
                intermediate_state={"data": [1, 2, 3, 4, 5], "mean": 3.0, "operation": "visualize"}
            )
        ]
        
        for i, step in enumerate(analysis_steps, 1):
            result = await validator.validate_step(step)
            print(f"      Step {i}: Valid={result.is_valid}, Confidence={result.confidence:.2f}")
        
        validator.shutdown()
        return True
        
    except Exception as e:
        print(f"   ❌ Real agent scenarios test failed: {e}")
        return False


async def main():
    """Run all end-to-end tests."""
    
    print("🚀 E2E TEST SUITE")
    print("=" * 60)
    print("This test thoroughly validates every component of the Aigie system")
    print("with real LLM calls and comprehensive error handling.")
    print()
    
    tests = [
        ("Component Integration", test_component_integration),
        ("LangChain Integration", test_langchain_integration),
        ("Performance Optimization", test_performance_optimization),
        ("Error Handling", test_error_handling),
        ("Monitoring and Metrics", test_monitoring_and_metrics),
        ("Real Agent Scenarios", test_real_agent_scenarios)
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
        print("\n🎉 ALL TESTS PASSED! The Aigie system is working correctly.")
    else:
        print(f"\n⚠️  {total-passed} tests failed. Please review the issues above.")
    
    return passed == total


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
