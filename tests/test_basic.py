#!/usr/bin/env python3
"""
Basic tests for Aigie core functionality.
"""

import unittest
import sys
import os

# Add the parent directory to the path so we can import aigie
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aigie.core.error_types import ErrorType, ErrorSeverity, ErrorContext, DetectedError, classify_error, determine_severity
from aigie.core.monitoring import PerformanceMonitor, ResourceMonitor
from aigie.core.error_detector import ErrorDetector
from aigie.reporting.logger import AigieLogger
from aigie.utils.config import AigieConfig


class TestErrorTypes(unittest.TestCase):
    """Test error type classification and severity determination."""
    
    def test_error_type_enum(self):
        """Test that error types are properly defined."""
        self.assertIn(ErrorType.RUNTIME_EXCEPTION, ErrorType)
        self.assertIn(ErrorType.API_ERROR, ErrorType)
        self.assertIn(ErrorType.TIMEOUT, ErrorType)
        self.assertIn(ErrorType.MEMORY_ERROR, ErrorType)
    
    def test_error_severity_enum(self):
        """Test that error severities are properly defined."""
        self.assertIn(ErrorSeverity.LOW, ErrorSeverity)
        self.assertIn(ErrorSeverity.MEDIUM, ErrorSeverity)
        self.assertIn(ErrorSeverity.HIGH, ErrorSeverity)
        self.assertIn(ErrorSeverity.CRITICAL, ErrorSeverity)
    
    def test_error_context_creation(self):
        """Test error context creation."""
        context = ErrorContext(
            timestamp=None,  # Will be set by the system
            framework="test",
            component="TestComponent",
            method="test_method"
        )
        
        self.assertEqual(context.framework, "test")
        self.assertEqual(context.component, "TestComponent")
        self.assertEqual(context.method, "test_method")
    
    def test_detected_error_creation(self):
        """Test detected error creation."""
        context = ErrorContext(
            timestamp=None,
            framework="test",
            component="TestComponent",
            method="test_method"
        )
        
        error = DetectedError(
            error_type=ErrorType.RUNTIME_EXCEPTION,
            severity=ErrorSeverity.HIGH,
            message="Test error message",
            context=context
        )
        
        self.assertEqual(error.error_type, ErrorType.RUNTIME_EXCEPTION)
        self.assertEqual(error.severity, ErrorSeverity.HIGH)
        self.assertEqual(error.message, "Test error message")
        self.assertEqual(error.context, context)
    
    def test_error_classification(self):
        """Test error classification logic."""
        context = ErrorContext(
            timestamp=None,
            framework="langchain",
            component="LLMChain",
            method="run"
        )
        
        # Test timeout error classification
        timeout_exception = Exception("Request timed out after 30 seconds")
        error_type = classify_error(timeout_exception, context)
        self.assertEqual(error_type, ErrorType.TIMEOUT)
        
        # Test API error classification
        api_exception = Exception("HTTP 500 Internal Server Error")
        error_type = classify_error(api_exception, context)
        self.assertEqual(error_type, ErrorType.API_ERROR)
    
    def test_severity_determination(self):
        """Test severity determination logic."""
        context = ErrorContext(
            timestamp=None,
            framework="langchain",
            component="LLMChain",
            method="run"
        )
        
        # Test critical severity
        severity = determine_severity(ErrorType.MEMORY_OVERFLOW, context)
        self.assertEqual(severity, ErrorSeverity.CRITICAL)
        
        # Test high severity
        severity = determine_severity(ErrorType.API_ERROR, context)
        self.assertEqual(severity, ErrorSeverity.HIGH)
        
        # Test medium severity
        severity = determine_severity(ErrorType.RATE_LIMIT, context)
        self.assertEqual(severity, ErrorSeverity.MEDIUM)


class TestPerformanceMonitor(unittest.TestCase):
    """Test performance monitoring functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.monitor = PerformanceMonitor()
    
    def test_monitor_creation(self):
        """Test that monitor is created correctly."""
        self.assertIsNotNone(self.monitor)
        self.assertTrue(self.monitor.enable_memory_monitoring)
        self.assertTrue(self.monitor.enable_cpu_monitoring)
    
    def test_monitoring_context_manager(self):
        """Test monitoring context manager."""
        with self.monitor.monitor_execution("TestComponent", "test_method") as metrics:
            # Simulate some work
            import time
            time.sleep(0.1)
        
        self.assertIsNotNone(metrics.execution_time)
        self.assertGreater(metrics.execution_time, 0)
    
    def test_performance_issue_detection(self):
        """Test performance issue detection."""
        # Create metrics that would trigger warnings
        metrics = type('MockMetrics', (), {
            'execution_time': 60.0,  # Over threshold
            'memory_delta': 2048.0,  # Over threshold
            'cpu_delta': 90.0  # Over threshold
        })()
        
        warnings = self.monitor.check_performance_issues(metrics)
        self.assertGreater(len(warnings), 0)
        
        # Check that warnings contain expected content
        warning_text = " ".join(warnings).lower()
        self.assertIn("slow execution", warning_text)
        self.assertIn("memory", warning_text)
        self.assertIn("cpu", warning_text)


class TestResourceMonitor(unittest.TestCase):
    """Test resource monitoring functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.monitor = ResourceMonitor()
    
    def test_monitor_creation(self):
        """Test that monitor is created correctly."""
        self.assertIsNotNone(self.monitor)
        self.assertEqual(self.monitor.memory_threshold, 0.9)
        self.assertEqual(self.monitor.cpu_threshold, 0.8)
        self.assertEqual(self.monitor.disk_threshold, 0.9)
    
    def test_system_health_check(self):
        """Test system health checking."""
        health = self.monitor.check_system_health()
        
        self.assertIn("timestamp", health)
        self.assertIn("memory", health)
        self.assertIn("cpu", health)
        self.assertIn("disk", health)
        self.assertIn("overall_status", health)
        
        # Check that overall status is valid
        valid_statuses = ["healthy", "warning", "critical", "unknown"]
        self.assertIn(health["overall_status"], valid_statuses)


class TestErrorDetector(unittest.TestCase):
    """Test error detection functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.detector = ErrorDetector()
    
    def test_detector_creation(self):
        """Test that detector is created correctly."""
        self.assertIsNotNone(self.detector)
        self.assertFalse(self.detector.is_monitoring)
        self.assertIsNotNone(self.detector.performance_monitor)
        self.assertIsNotNone(self.detector.resource_monitor)
    
    def test_monitoring_control(self):
        """Test monitoring start/stop functionality."""
        self.assertFalse(self.detector.is_monitoring)
        
        self.detector.start_monitoring()
        self.assertTrue(self.detector.is_monitoring)
        
        self.detector.stop_monitoring()
        self.assertFalse(self.detector.is_monitoring)
    
    def test_error_handler_registration(self):
        """Test error handler registration."""
        handler_called = False
        
        def test_handler(error):
            nonlocal handler_called
            handler_called = True
        
        self.detector.add_error_handler(test_handler)
        self.assertIn(test_handler, self.detector.error_handlers)
    
    def test_monitoring_context_manager(self):
        """Test monitoring context manager."""
        self.detector.start_monitoring()
        
        with self.detector.monitor_execution("test", "TestComponent", "test_method"):
            # Simulate some work
            import time
            time.sleep(0.1)
        
        # Check that metrics were collected
        self.assertGreater(len(self.detector.performance_monitor.metrics_history), 0)


class TestAigieLogger(unittest.TestCase):
    """Test logging functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.logger = AigieLogger(enable_console=False, enable_file=False)
    
    def test_logger_creation(self):
        """Test that logger is created correctly."""
        self.assertIsNotNone(self.logger)
        self.assertFalse(self.logger.enable_console)
        self.assertFalse(self.logger.enable_file)
    
    def test_error_logging(self):
        """Test error logging functionality."""
        context = ErrorContext(
            timestamp=None,
            framework="test",
            component="TestComponent",
            method="test_method"
        )
        
        error = DetectedError(
            error_type=ErrorType.RUNTIME_EXCEPTION,
            severity=ErrorSeverity.HIGH,
            message="Test error message",
            context=context
        )
        
        # This should not raise an exception
        self.logger.log_error(error)
        
        # Check that error was logged
        self.assertEqual(self.logger.error_counts["total"], 1)
        self.assertEqual(self.logger.error_counts["high"], 1)


class TestAigieConfig(unittest.TestCase):
    """Test configuration functionality."""
    
    def test_default_config(self):
        """Test default configuration creation."""
        config = AigieConfig()
        
        self.assertEqual(config.log_level, "INFO")
        self.assertTrue(config.enable_console)
        self.assertFalse(config.enable_file)
        self.assertTrue(config.enable_performance_monitoring)
        self.assertTrue(config.enable_resource_monitoring)
    
    def test_config_validation(self):
        """Test configuration validation."""
        # Test invalid log level
        with self.assertRaises(ValueError):
            AigieConfig(log_level="INVALID")
        
        # Test invalid timeout threshold
        with self.assertRaises(ValueError):
            AigieConfig(timeout_threshold=-1)
        
        # Test invalid CPU threshold
        with self.assertRaises(ValueError):
            AigieConfig(cpu_threshold=150)  # Over 100%
    
    def test_config_to_dict(self):
        """Test configuration serialization."""
        config = AigieConfig()
        config_dict = config.to_dict()
        
        self.assertIn("log_level", config_dict)
        self.assertIn("enable_console", config_dict)
        self.assertIn("enable_performance_monitoring", config_dict)
        self.assertEqual(config_dict["log_level"], "INFO")


if __name__ == "__main__":
    unittest.main()
