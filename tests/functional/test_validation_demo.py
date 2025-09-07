#!/usr/bin/env python3
"""
Simple validation demo to test the reorganized Aigie structure.
This demonstrates that all imports work correctly and basic functionality is available.
"""

import sys
import os

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Add the current directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all imports work with the new structure."""
    print("🧪 Testing imports with new organized structure...")
    
    try:
        # Test core imports
        from aigie.core import ErrorDetector, PerformanceMonitor, GeminiAnalyzer
        from aigie.core import RuntimeValidator, StepCorrector, ValidationEngine, ContextExtractor
        from aigie.core import ExecutionStep, ValidationResult, ValidationStatus
        print("✅ Core imports working")
        
        # Test specific module imports
        from aigie.core.error_handling.error_detector import ErrorDetector as ED
        from aigie.core.ai.gemini_analyzer import GeminiAnalyzer as GA
        from aigie.core.validation.runtime_validator import RuntimeValidator as RV
        from aigie.core.types.validation_types import ExecutionStep as ES
        print("✅ Specific module imports working")
        
        # Test interceptor imports
        from aigie.interceptors.langchain import LangChainInterceptor
        from aigie.interceptors.langgraph import LangGraphInterceptor
        print("✅ Interceptor imports working")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality without requiring API keys."""
    print("\n🔧 Testing basic functionality...")
    
    try:
        # Test ErrorDetector without Gemini (for testing)
        from aigie.core.error_handling.error_detector import ErrorDetector
        detector = ErrorDetector(enable_gemini_analysis=False)
        print("✅ ErrorDetector created (testing mode)")
        
        # Test PerformanceMonitor
        from aigie.core.monitoring.monitoring import PerformanceMonitor
        monitor = PerformanceMonitor()
        print("✅ PerformanceMonitor created")
        
        # Test ExecutionStep creation
        from aigie.core.types.validation_types import ExecutionStep, ValidationStatus
        step = ExecutionStep(
            framework="langchain",
            component="ChatOpenAI",
            operation="invoke",
            input_data={"messages": [{"role": "user", "content": "Hello"}]},
            agent_goal="Answer user questions"
        )
        print("✅ ExecutionStep created")
        
        # Test copy method
        step_copy = step.copy()
        print("✅ ExecutionStep copy method working")
        
        # Test ValidationResult
        from aigie.core.types.validation_types import ValidationResult, RiskLevel
        result = ValidationResult(
            is_valid=True,
            confidence=0.8,
            reasoning="Test validation",
            risk_level=RiskLevel.LOW
        )
        print("✅ ValidationResult created")
        
        return True
        
    except Exception as e:
        print(f"❌ Functionality test failed: {e}")
        return False

def test_validation_system():
    """Test the validation system components."""
    print("\n🧠 Testing validation system components...")
    
    try:
        # Test ContextExtractor (should work without Gemini)
        from aigie.core.validation.context_extractor import ContextExtractor
        extractor = ContextExtractor(enable_llm=False)
        print("✅ ContextExtractor created (testing mode)")
        
        # Test context extraction
        from aigie.core.types.validation_types import ExecutionStep
        step = ExecutionStep(
            framework="langchain",
            component="ChatOpenAI",
            operation="invoke",
            input_data={"messages": [{"role": "user", "content": "What is 2+2?"}]}
        )
        
        # Extract context (should use fallback since no Gemini)
        step_with_context = extractor.extract_context(step)
        print(f"✅ Context extraction working - inferred goal: {step_with_context.inferred_goal}")
        
        return True
        
    except Exception as e:
        print(f"❌ Validation system test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Aigie Reorganization Validation Test")
    print("=" * 50)
    
    success = True
    
    # Test imports
    if not test_imports():
        success = False
    
    # Test basic functionality
    if not test_basic_functionality():
        success = False
    
    # Test validation system
    if not test_validation_system():
        success = False
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 All tests passed! Aigie reorganization is working correctly.")
        print("\n📋 Summary:")
        print("✅ All imports work with new organized structure")
        print("✅ Core functionality is available")
        print("✅ Validation system components work")
        print("✅ Error handling works in testing mode")
        print("✅ Context extraction works with fallback")
        print("\n🚀 Aigie is ready for use with the new organized structure!")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
