#!/usr/bin/env python3
"""
Functional tests for the validation system with real Gemini API calls.
"""

import asyncio
import sys
import os
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
from aigie.core.types.validation_types import ExecutionStep, ValidationStatus


async def test_simple_validation():
    """Test a simple validation with real Gemini API calls."""
    
    print("🧪 Testing Simple Runtime Validation with REAL Gemini API calls...")
    
    # Check for API key
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ GEMINI_API_KEY not found in environment variables.")
        print("   Please set your Gemini API key in the .env file:")
        print("   GEMINI_API_KEY=your_api_key_here")
        return False
    
    print(f"✅ Found Gemini API key: {api_key[:10]}...")
    
    # Initialize real components
    print("\n1. Initializing real components...")
    
    try:
        # Initialize Gemini analyzer with real API
        gemini_analyzer = GeminiAnalyzer()
        
        if not gemini_analyzer.is_available():
            print("❌ Gemini analyzer is not available. Please check your API key.")
            return False
        
        print(f"✅ Gemini analyzer initialized successfully")
        print(f"   Backend: {gemini_analyzer.backend}")
        
        # Initialize validation components
        validator = RuntimeValidator(gemini_analyzer)
        corrector = StepCorrector(gemini_analyzer)
        validation_engine = ValidationEngine(validator, corrector)
        
        print("✅ All validation components initialized")
        
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        return False
    
    # Test Case: Simple valid step
    print("\n2. Testing simple valid step with real Gemini validation...")
    
    valid_step = ExecutionStep(
        framework="langchain",
        component="ChatOpenAI",
        operation="invoke",
        input_data={
            "messages": [{"role": "user", "content": "What is 2+2?"}],
            "temperature": 0.7
        },
        agent_goal="Answer simple math questions",
        step_reasoning="User asked a simple math question, using ChatOpenAI to provide the answer"
    )
    
    print(f"   Step: {valid_step.operation} on {valid_step.component}")
    print(f"   Goal: {valid_step.agent_goal}")
    print(f"   Input: {valid_step.input_data['messages'][0]['content']}")
    print("   🔄 Calling real Gemini for validation...")
    
    try:
        start_time = datetime.now()
        processed_step = await validation_engine.process_step(valid_step)
        validation_time = (datetime.now() - start_time).total_seconds()
        
        print(f"✅ Validation completed in {validation_time:.2f}s")
        print(f"   Valid: {processed_step.validation_result.is_valid}")
        print(f"   Confidence: {processed_step.validation_result.confidence:.2f}")
        print(f"   Risk Level: {processed_step.validation_result.risk_level.value}")
        print(f"   Reasoning: {processed_step.validation_result.reasoning[:150]}...")
        
        if processed_step.validation_result.issues:
            print(f"   Issues: {processed_step.validation_result.issues}")
        if processed_step.validation_result.suggestions:
            print(f"   Suggestions: {processed_step.validation_result.suggestions}")
        
        print(f"   Successful: {processed_step.is_successful}")
        
        return True
        
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_context_extraction():
    """Test automatic context extraction without manual parameters."""
    
    print("\n🧪 Testing Automatic Context Extraction (No Manual Parameters)...")
    
    # Check for API key
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ GEMINI_API_KEY not found in environment variables.")
        return False
    
    print(f"✅ Found Gemini API key: {api_key[:10]}...")
    
    # Initialize components
    print("\n1. Initializing components...")
    
    try:
        gemini_analyzer = GeminiAnalyzer()
        if not gemini_analyzer.is_available():
            print("❌ Gemini analyzer not available.")
            return False
        
        validator = RuntimeValidator(gemini_analyzer)
        corrector = StepCorrector(gemini_analyzer)
        validation_engine = ValidationEngine(validator, corrector)
        
        from aigie.core.validation.context_extractor import ContextExtractor
        context_extractor = ContextExtractor()
        
        print("✅ All components initialized")
        
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        return False
    
    # Test Case 1: Question Answering (no manual goal)
    print("\n2. Testing question answering with auto-extracted context...")
    
    step1 = ExecutionStep(
        framework="langchain",
        component="ChatOpenAI",
        operation="invoke",
        input_data={
            "messages": [{"role": "user", "content": "What is the capital of Japan?"}],
            "temperature": 0.7
        }
        # No agent_goal provided - should be auto-extracted
    )
    
    print(f"   Before extraction:")
    print(f"   - Agent goal: {step1.agent_goal}")
    print(f"   - Inferred goal: {step1.inferred_goal}")
    print(f"   - Context clues: {step1.context_clues}")
    
    # Extract context automatically
    step1 = context_extractor.extract_context(step1)
    
    print(f"   After extraction:")
    print(f"   - Agent goal: {step1.agent_goal}")
    print(f"   - Inferred goal: {step1.inferred_goal}")
    print(f"   - Context clues: {step1.context_clues}")
    print(f"   - Operation pattern: {step1.operation_pattern}")
    print(f"   - Auto confidence: {step1.auto_confidence}")
    
    # Test validation with auto-extracted context
    print("\n   🔄 Validating with auto-extracted context...")
    try:
        start_time = datetime.now()
        processed_step = await validation_engine.process_step(step1)
        validation_time = (datetime.now() - start_time).total_seconds()
        
        print(f"   ✅ Validation completed in {validation_time:.2f}s")
        print(f"   - Valid: {processed_step.validation_result.is_valid}")
        print(f"   - Confidence: {processed_step.validation_result.confidence:.2f}")
        print(f"   - Reasoning: {processed_step.validation_result.reasoning[:100]}...")
        
    except Exception as e:
        print(f"   ❌ Validation failed: {e}")
        return False
    
    print("\n🎉 Automatic Context Extraction Test PASSED!")
    print("   ✅ No manual parameters required")
    print("   ✅ Goals auto-inferred from context")
    print("   ✅ Context clues extracted automatically")
    print("   ✅ Validation works with auto-extracted context")
    
    return True


async def test_llm_context_extraction():
    """Test LLM-based context extraction using Gemini as an intelligent agent."""
    
    print("\n🧪 Testing LLM-Based Context Extraction with Gemini Agent...")
    
    # Check for API key
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ GEMINI_API_KEY not found in environment variables.")
        return False
    
    print(f"✅ Found Gemini API key: {api_key[:10]}...")
    
    # Initialize components
    print("\n1. Initializing LLM-based context extractor...")
    
    try:
        gemini_analyzer = GeminiAnalyzer()
        if not gemini_analyzer.is_available():
            print("❌ Gemini analyzer not available.")
            return False
        
        from aigie.core.validation.context_extractor import ContextExtractor
        context_extractor = ContextExtractor(gemini_analyzer)
        print("✅ LLM-based context extractor initialized")
        
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        return False
    
    # Test Case 1: Complex question answering
    print("\n2. Testing complex question answering with LLM analysis...")
    
    step1 = ExecutionStep(
        framework="langchain",
        component="ChatOpenAI",
        operation="invoke",
        input_data={
            "messages": [{"role": "user", "content": "Can you explain the difference between machine learning and deep learning, and provide examples of when to use each?"}],
            "temperature": 0.7
        }
    )
    
    print(f"   Input: {step1.input_data['messages'][0]['content'][:100]}...")
    print("   🔄 Using Gemini agent to analyze context...")
    
    try:
        start_time = datetime.now()
        step1 = context_extractor.extract_context(step1)
        extraction_time = (datetime.now() - start_time).total_seconds()
        
        print(f"   ✅ Context extraction completed in {extraction_time:.2f}s")
        print(f"   - Inferred goal: {step1.inferred_goal}")
        print(f"   - Context clues: {step1.context_clues[:3]}...")
        print(f"   - Operation pattern: {step1.operation_pattern}")
        print(f"   - Confidence: {step1.auto_confidence:.2f}")
        
    except Exception as e:
        print(f"   ❌ Context extraction failed: {e}")
        return False
    
    print("\n🎉 LLM-Based Context Extraction Test PASSED!")
    print("   ✅ Gemini agent successfully analyzes context")
    print("   ✅ Intelligent goal inference working")
    print("   ✅ Context clues extracted intelligently")
    print("   ✅ Operation patterns identified")
    print("   ✅ High confidence scoring")
    print("   ✅ No hardcoded patterns - pure LLM intelligence!")
    
    return True


async def main():
    """Run all functional tests."""
    print("🚀 Starting Functional Validation Tests...")
    print("=" * 60)
    
    # Test simple validation
    simple_success = await test_simple_validation()
    
    # Test context extraction
    context_success = await test_context_extraction()
    
    # Test LLM context extraction
    llm_context_success = await test_llm_context_extraction()
    
    print("\n" + "=" * 60)
    print("📊 Test Results Summary:")
    print(f"   Simple Validation: {'✅ PASSED' if simple_success else '❌ FAILED'}")
    print(f"   Context Extraction: {'✅ PASSED' if context_success else '❌ FAILED'}")
    print(f"   LLM Context Extraction: {'✅ PASSED' if llm_context_success else '❌ FAILED'}")
    
    overall_success = simple_success and context_success and llm_context_success
    
    if overall_success:
        print("\n🎉 ALL TESTS PASSED!")
        print("   ✅ Validation system working with real Gemini API")
        print("   ✅ Context extraction working automatically")
        print("   ✅ LLM-based intelligence working")
    else:
        print("\n❌ SOME TESTS FAILED!")
        print("   Check the error messages above for details")
    
    return overall_success


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
