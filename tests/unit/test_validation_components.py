#!/usr/bin/env python3
"""
Unit tests for validation components.
"""

import unittest
import sys
import os
import asyncio
from unittest.mock import Mock, patch

# Add the parent directory to the path so we can import aigie
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from aigie.core.types.validation_types import ExecutionStep, ValidationResult, ValidationStatus, RiskLevel
from aigie.core.types.error_types import ErrorContext
from aigie.core.ai.gemini_analyzer import GeminiAnalyzer
from aigie.core.validation.runtime_validator import RuntimeValidator
from aigie.core.validation.step_corrector import StepCorrector
from aigie.core.validation.validation_engine import ValidationEngine


class TestValidationTypes(unittest.TestCase):
    """Test validation type definitions."""
    
    def test_execution_step_creation(self):
        """Test ExecutionStep creation."""
        step = ExecutionStep(
            framework="langchain",
            component="ChatOpenAI",
            operation="invoke",
            input_data={"input": "test"},
            agent_goal="Test goal",
            step_reasoning="Test reasoning"
        )
        
        self.assertEqual(step.framework, "langchain")
        self.assertEqual(step.component, "ChatOpenAI")
        self.assertEqual(step.operation, "invoke")
        self.assertEqual(step.input_data, {"input": "test"})
        self.assertEqual(step.agent_goal, "Test goal")
        self.assertEqual(step.step_reasoning, "Test reasoning")
    
    def test_validation_result_creation(self):
        """Test ValidationResult creation."""
        result = ValidationResult(
            is_valid=True,
            confidence=0.85,
            reasoning="Test reasoning",
            risk_level=RiskLevel.LOW
        )
        
        self.assertTrue(result.is_valid)
        self.assertEqual(result.confidence, 0.85)
        self.assertEqual(result.reasoning, "Test reasoning")
        self.assertEqual(result.risk_level, RiskLevel.LOW)
    
    def test_validation_status_enum(self):
        """Test ValidationStatus enum."""
        self.assertIn(ValidationStatus.VALID, ValidationStatus)
        self.assertIn(ValidationStatus.INVALID, ValidationStatus)
        self.assertIn(ValidationStatus.PENDING, ValidationStatus)
    
    def test_risk_level_enum(self):
        """Test RiskLevel enum."""
        self.assertIn(RiskLevel.LOW, RiskLevel)
        self.assertIn(RiskLevel.MEDIUM, RiskLevel)
        self.assertIn(RiskLevel.HIGH, RiskLevel)
        self.assertIn(RiskLevel.CRITICAL, RiskLevel)


class TestGeminiAnalyzer(unittest.TestCase):
    """Test Gemini analyzer functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = GeminiAnalyzer()
    
    def test_analyzer_creation(self):
        """Test analyzer creation."""
        self.assertIsNotNone(self.analyzer)
    
    def test_analyzer_availability(self):
        """Test analyzer availability check."""
        # This will depend on whether API key is available
        availability = self.analyzer.is_available()
        self.assertIsInstance(availability, bool)
    
    @patch('aigie.core.ai.gemini_analyzer.genai')
    def test_analyze_with_mock(self, mock_genai):
        """Test analyzer with mocked Gemini API."""
        # Mock the Gemini response
        mock_genai.configure.return_value = None
        mock_model = Mock()
        mock_model.generate_content.return_value.text = "Test analysis"
        mock_genai.GenerativeModel.return_value = mock_model
        
        # Mock the analyzer to be initialized
        self.analyzer.is_initialized = True
        self.analyzer.model = mock_model
        self.analyzer.backend = "api_key"  # Set a valid backend
        
        # Test analysis
        result = self.analyzer.analyze_error(Exception("Test error"), ErrorContext("test", "test", "test", "test"))
        self.assertIsNotNone(result)


class TestRuntimeValidator(unittest.TestCase):
    """Test runtime validator functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = Mock(spec=GeminiAnalyzer)
        self.analyzer.is_available.return_value = True
        self.validator = RuntimeValidator(self.analyzer)
    
    def test_validator_creation(self):
        """Test validator creation."""
        self.assertIsNotNone(self.validator)
        self.assertEqual(self.validator.gemini_analyzer, self.analyzer)
    
    @patch('aigie.core.runtime_validator.asyncio')
    async def test_validate_step(self, mock_asyncio):
        """Test step validation."""
        step = ExecutionStep(
            framework="langchain",
            component="ChatOpenAI",
            operation="invoke",
            input_data={"input": "test"},
            agent_goal="Test goal",
            step_reasoning="Test reasoning"
        )
        
        # Mock the validation result
        expected_result = ValidationResult(
            is_valid=True,
            confidence=0.8,
            reasoning="Test reasoning",
            risk_level=RiskLevel.LOW
        )
        
        # Mock the analyzer response
        self.analyzer.analyze.return_value = expected_result
        
        result = await self.validator.validate_step(step)
        
        self.assertIsInstance(result, ValidationResult)
        self.assertTrue(result.is_valid)


class TestStepCorrector(unittest.TestCase):
    """Test step corrector functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = Mock(spec=GeminiAnalyzer)
        self.analyzer.is_available.return_value = True
        self.corrector = StepCorrector(self.analyzer)
    
    def test_corrector_creation(self):
        """Test corrector creation."""
        self.assertIsNotNone(self.corrector)
        self.assertEqual(self.corrector.gemini_analyzer, self.analyzer)
    
    async def test_correct_step(self):
        """Test step correction."""
        step = ExecutionStep(
            framework="langchain",
            component="WrongTool",
            operation="run",
            input_data={"input": "test"},
            agent_goal="Answer questions",
            step_reasoning="Test reasoning"
        )
        
        # Create a mock validation result
        validation_result = ValidationResult(
            is_valid=False,
            confidence=0.3,
            reasoning="Wrong tool for the task",
            risk_level=RiskLevel.MEDIUM
        )
        
        # Mock the correction response
        self.analyzer.analyze_error.return_value = {
            "primary_issue": "tool_mismatch",
            "suggested_fix": "Use ChatOpenAI for answering questions",
            "confidence": 0.8
        }
        
        result = await self.corrector.correct_step(step, validation_result)
        
        self.assertIsNotNone(result)
        self.assertTrue(result.is_successful)


class TestValidationEngine(unittest.TestCase):
    """Test validation engine functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.validator = Mock(spec=RuntimeValidator)
        self.corrector = Mock(spec=StepCorrector)
        self.engine = ValidationEngine(self.validator, self.corrector)
    
    def test_engine_creation(self):
        """Test engine creation."""
        self.assertIsNotNone(self.engine)
        self.assertEqual(self.engine.validator, self.validator)
        self.assertEqual(self.engine.corrector, self.corrector)
    
    @patch('aigie.core.validation_engine.asyncio')
    async def test_process_step(self, mock_asyncio):
        """Test step processing."""
        step = ExecutionStep(
            framework="langchain",
            component="ChatOpenAI",
            operation="invoke",
            input_data={"input": "test"},
            agent_goal="Test goal",
            step_reasoning="Test reasoning"
        )
        
        # Mock validation result
        validation_result = ValidationResult(
            is_valid=True,
            confidence=0.8,
            reasoning="Test reasoning",
            risk_level=RiskLevel.LOW
        )
        
        self.validator.validate_step.return_value = validation_result
        
        result = await self.engine.process_step(step)
        
        self.assertIsNotNone(result)
        self.assertTrue(result.is_successful)


if __name__ == "__main__":
    unittest.main()
