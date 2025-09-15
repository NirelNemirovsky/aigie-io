#!/usr/bin/env python3
"""
Comprehensive Validation System Demo with Real-Time Monitoring

This demo shows the validation system in action with:
1. Detailed logging of each validation step
2. Real-time monitoring of agent execution
3. Intervention demonstrations when validation fails
4. Performance metrics and trend analysis
5. Visual dashboard of validation pipeline

The demo creates realistic agent execution scenarios and shows how aigie:
- Intercepts agent execution at each step
- Validates agent reasoning and actions
- Provides intelligent corrections when issues are detected
- Monitors performance and system health
- Learns from validation patterns

Requirements:
- GEMINI_API_KEY for AI-powered validation
- Rich console for visual output
- Real agent execution scenarios
"""

import asyncio
import sys
import os
import time
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add the project root to the path
sys.path.insert(0, '/Users/nirelnemirovsky/Documents/dev/aigie/aigie-io')

from aigie.core.ai.gemini_analyzer import GeminiAnalyzer
from aigie.core.validation.runtime_validator import RuntimeValidator, ValidationConfig
from aigie.core.validation.step_corrector import StepCorrector
from aigie.core.validation.validation_engine import ValidationEngine
from aigie.core.validation.validation_pipeline import ValidationPipeline
from aigie.core.validation.validation_monitor import ValidationMonitor, PerformanceAlert
from aigie.core.types.validation_types import (
    ExecutionStep, ValidationStatus, ValidationStrategy, RiskLevel
)
from aigie.reporting.logger import AigieLogger

# Configure detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('validation_demo.log')
    ]
)
logger = logging.getLogger(__name__)


class ValidationDemo:
    """Comprehensive validation system demonstration."""
    
    def __init__(self):
        self.gemini_analyzer = None
        self.validator = None
        self.corrector = None
        self.engine = None
        self.pipeline = None
        self.monitor = None
        self.logger = None
        self.demo_scenarios = []
        
    async def initialize_system(self):
        """Initialize the complete validation system."""
        print("🚀 Initializing Comprehensive Validation System...")
        
        # Check for API key
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            print("❌ GEMINI_API_KEY not found in environment variables.")
            print("   Please set your Gemini API key in the .env file:")
            print("   GEMINI_API_KEY=your_api_key_here")
            return False
        
        print(f"✅ Found Gemini API key: {api_key[:10]}...")
        
        try:
            # Initialize Gemini analyzer
            print("\n1. Initializing Gemini Analyzer...")
            self.gemini_analyzer = GeminiAnalyzer()
            
            if not self.gemini_analyzer.is_available():
                print("❌ Gemini analyzer is not available. Please check your API key.")
                return False
            
            print("✅ Gemini Analyzer initialized successfully")
            
            # Initialize validation components
            print("\n2. Initializing Validation Components...")
            
            # Configure validation with all strategies enabled
            config = ValidationConfig(
                enabled_strategies=list(ValidationStrategy),
                enable_parallel_strategies=True,
                enable_adaptive_validation=True,
                enable_pattern_learning=True,
                cache_ttl_seconds=300,
                max_concurrent_validations=5
            )
            
            self.validator = RuntimeValidator(self.gemini_analyzer, config)
            self.corrector = StepCorrector(self.gemini_analyzer)
            self.engine = ValidationEngine(self.validator, self.corrector)
            self.pipeline = ValidationPipeline(self.validator)
            self.monitor = ValidationMonitor(self.validator, self.pipeline)
            
            # Initialize logger
            self.logger = AigieLogger(
                log_level="DEBUG",
                enable_console=True,
                enable_file=True,
                log_file_path="validation_demo.log"
            )
            
            print("✅ All validation components initialized successfully")
            
            # Start monitoring
            print("\n3. Starting Real-Time Monitoring...")
            self.monitor.start_monitoring()
            
            # Add performance alerts
            self._setup_performance_alerts()
            
            print("✅ Real-time monitoring started")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to initialize validation system: {e}")
            logger.error(f"Initialization failed: {e}")
            return False
    
    def _setup_performance_alerts(self):
        """Setup performance monitoring alerts."""
        alerts = [
            PerformanceAlert(
                metric_name="avg_validation_time",
                threshold=5.0,
                operator="gt",
                severity="medium",
                enabled=True
            ),
            PerformanceAlert(
                metric_name="error_rate",
                threshold=0.2,
                operator="gt",
                severity="high",
                enabled=True
            ),
            PerformanceAlert(
                metric_name="memory_usage_mb",
                threshold=500.0,
                operator="gt",
                severity="medium",
                enabled=True
            )
        ]
        
        for alert in alerts:
            self.monitor.add_alert(alert)
        
        # Add alert handler
        self.monitor.add_alert_handler(self._handle_performance_alert)
    
    def _handle_performance_alert(self, alert_data: Dict[str, Any]):
        """Handle performance alerts."""
        print(f"🚨 PERFORMANCE ALERT: {alert_data['metric_name']} = {alert_data['current_value']}")
        self.logger.log_performance_issue(
            f"Performance alert: {alert_data['metric_name']} exceeded threshold",
            alert_data
        )
    
    def create_demo_scenarios(self):
        """Create realistic agent execution scenarios for demonstration."""
        self.demo_scenarios = [
            {
                "name": "Successful Research Agent",
                "step": ExecutionStep(
                    step_id="research_001",
                    framework="langchain",
                    component="research_agent",
                    operation="search_and_analyze",
                    input_data={
                        "query": "latest developments in AI safety",
                        "sources": ["arxiv", "google_scholar"],
                        "max_results": 10
                    },
                    agent_goal="Find and analyze recent research papers on AI safety",
                    step_reasoning="I need to search for recent papers on AI safety to provide comprehensive information to the user"
                ),
                "should_pass": True
            },
            {
                "name": "Problematic Code Generation",
                "step": ExecutionStep(
                    step_id="code_002",
                    framework="langgraph",
                    component="code_generator",
                    operation="generate_python_code",
                    input_data={
                        "task": "Create a function that divides by zero",
                        "language": "python",
                        "requirements": ["error_handling", "logging"]
                    },
                    agent_goal="Generate Python code for mathematical operations",
                    step_reasoning="The user wants a division function, I'll create a simple one"
                ),
                "should_pass": False
            },
            {
                "name": "Incomplete Data Processing",
                "step": ExecutionStep(
                    step_id="data_003",
                    framework="langchain",
                    component="data_processor",
                    operation="process_dataset",
                    input_data={
                        "dataset_path": "/path/to/data.csv",
                        "operations": ["clean", "transform", "analyze"]
                    },
                    agent_goal="Process and analyze the dataset",
                    step_reasoning="I'll process the data step by step"
                ),
                "should_pass": False
            },
            {
                "name": "Safe Content Generation",
                "step": ExecutionStep(
                    step_id="content_004",
                    framework="langgraph",
                    component="content_generator",
                    operation="generate_article",
                    input_data={
                        "topic": "Introduction to Machine Learning",
                        "length": "1000 words",
                        "style": "educational"
                    },
                    agent_goal="Create educational content about machine learning",
                    step_reasoning="I'll write a comprehensive introduction to ML concepts"
                ),
                "should_pass": True
            }
        ]
    
    async def run_comprehensive_demo(self):
        """Run the complete validation system demonstration."""
        print("\n" + "="*80)
        print("🎯 COMPREHENSIVE VALIDATION SYSTEM DEMONSTRATION")
        print("="*80)
        
        # Initialize system
        if not await self.initialize_system():
            return False
        
        # Create demo scenarios
        self.create_demo_scenarios()
        
        print(f"\n📋 Running {len(self.demo_scenarios)} validation scenarios...")
        
        # Run each scenario
        for i, scenario in enumerate(self.demo_scenarios, 1):
            print(f"\n{'='*60}")
            print(f"🎬 SCENARIO {i}: {scenario['name']}")
            print(f"{'='*60}")
            
            await self._run_scenario(scenario)
            
            # Show metrics after each scenario
            self._display_current_metrics()
            
            # Small delay for demonstration
            await asyncio.sleep(2)
        
        # Final comprehensive report
        await self._generate_final_report()
        
        return True
    
    async def _run_scenario(self, scenario: Dict[str, Any]):
        """Run a single validation scenario with detailed logging."""
        step = scenario["step"]
        expected_result = scenario["should_pass"]
        
        print(f"\n🔍 Validating Step: {step.step_id}")
        print(f"   Component: {step.component}")
        print(f"   Operation: {step.operation}")
        print(f"   Goal: {step.agent_goal}")
        print(f"   Expected to {'PASS' if expected_result else 'FAIL'}")
        
        # Show input data
        print(f"\n📥 Input Data:")
        for key, value in step.input_data.items():
            print(f"   {key}: {value}")
        
        # Show reasoning
        print(f"\n🧠 Agent Reasoning:")
        print(f"   {step.step_reasoning}")
        
        start_time = time.time()
        
        try:
            # Process through validation pipeline
            print(f"\n⚙️  Processing through Validation Pipeline...")
            
            # Stage 1: Pre-validation
            print("   🔸 Pre-validation stage...")
            pre_result = await self.pipeline._pre_validation_stage(step, None)
            print(f"      Result: {'✅ PASS' if pre_result.is_valid else '❌ FAIL'}")
            print(f"      Confidence: {pre_result.confidence:.2f}")
            if pre_result.issues:
                print(f"      Issues: {pre_result.issues}")
            
            # Stage 2: Fast validation
            print("   🔸 Fast validation stage...")
            fast_result = await self.pipeline._fast_validation_stage(step, pre_result)
            print(f"      Result: {'✅ PASS' if fast_result.is_valid else '❌ FAIL'}")
            print(f"      Confidence: {fast_result.confidence:.2f}")
            if fast_result.issues:
                print(f"      Issues: {fast_result.issues}")
            
            # Stage 3: Deep validation
            print("   🔸 Deep validation stage...")
            deep_result = await self.pipeline._deep_validation_stage(step, fast_result)
            print(f"      Result: {'✅ PASS' if deep_result.is_valid else '❌ FAIL'}")
            print(f"      Confidence: {deep_result.confidence:.2f}")
            if deep_result.issues:
                print(f"      Issues: {deep_result.issues}")
            
            # Stage 4: Post-validation
            print("   🔸 Post-validation stage...")
            final_result = await self.pipeline._post_validation_stage(step, deep_result)
            print(f"      Result: {'✅ PASS' if final_result.is_valid else '❌ FAIL'}")
            print(f"      Confidence: {final_result.confidence:.2f}")
            if final_result.issues:
                print(f"      Issues: {final_result.issues}")
            
            validation_time = time.time() - start_time
            
            # Record in monitor
            self.monitor.record_validation(
                step, final_result, validation_time,
                cache_hit=False, parallel_used=True
            )
            
            # Show final result
            print(f"\n📊 FINAL VALIDATION RESULT:")
            print(f"   Status: {'✅ VALID' if final_result.is_valid else '❌ INVALID'}")
            print(f"   Confidence: {final_result.confidence:.2f}")
            print(f"   Risk Level: {final_result.risk_level.value}")
            print(f"   Validation Time: {validation_time:.3f}s")
            
            if final_result.issues:
                print(f"   Issues Found: {len(final_result.issues)}")
                for issue in final_result.issues:
                    print(f"     • {issue}")
            
            if final_result.suggestions:
                print(f"   Suggestions: {len(final_result.suggestions)}")
                for suggestion in final_result.suggestions:
                    print(f"     • {suggestion}")
            
            # Check if result matches expectation
            if final_result.is_valid == expected_result:
                print(f"   ✅ Result matches expectation!")
            else:
                print(f"   ⚠️  Result differs from expectation!")
            
            # If validation failed, show correction attempt
            if not final_result.is_valid:
                await self._demonstrate_correction(step, final_result)
            
        except Exception as e:
            print(f"❌ Validation failed with error: {e}")
            logger.error(f"Validation error for step {step.step_id}: {e}")
    
    async def _demonstrate_correction(self, step: ExecutionStep, validation_result):
        """Demonstrate automatic correction when validation fails."""
        print(f"\n🔧 ATTEMPTING AUTOMATIC CORRECTION...")
        
        try:
            correction_result = await self.corrector.correct_step(step, validation_result)
            
            print(f"   Correction Strategy: {correction_result.correction_strategy.value if correction_result.correction_strategy else 'None'}")
            print(f"   Success: {'✅ YES' if correction_result.success else '❌ NO'}")
            print(f"   Attempts: {correction_result.correction_attempts}")
            
            if correction_result.success:
                print(f"   Corrected Step:")
                print(f"     Goal: {correction_result.corrected_step.agent_goal}")
                print(f"     Reasoning: {correction_result.corrected_step.step_reasoning}")
                
                # Re-validate the corrected step
                print(f"\n🔄 Re-validating corrected step...")
                revalidation_result = await self.validator.validate_step(correction_result.corrected_step)
                print(f"   Re-validation Result: {'✅ VALID' if revalidation_result.is_valid else '❌ INVALID'}")
                print(f"   New Confidence: {revalidation_result.confidence:.2f}")
            else:
                print(f"   Correction failed: {correction_result.failure_reason}")
                if correction_result.suggestions:
                    print(f"   Manual suggestions:")
                    for suggestion in correction_result.suggestions:
                        print(f"     • {suggestion}")
        
        except Exception as e:
            print(f"❌ Correction failed with error: {e}")
            logger.error(f"Correction error for step {step.step_id}: {e}")
    
    def _display_current_metrics(self):
        """Display current validation metrics."""
        metrics = self.monitor.get_metrics()
        
        print(f"\n📈 CURRENT METRICS:")
        print(f"   Total Validations: {metrics['basic_metrics']['total_validations']}")
        print(f"   Success Rate: {metrics['basic_metrics']['success_rate']:.1%}")
        print(f"   Avg Validation Time: {metrics['timing_metrics']['avg_validation_time']:.3f}s")
        print(f"   Avg Confidence: {metrics['quality_metrics']['avg_confidence']:.2f}")
        print(f"   Cache Hit Rate: {metrics['performance_metrics']['cache_hit_rate']:.1%}")
        print(f"   Error Rate: {metrics['error_metrics']['error_rate']:.1%}")
    
    async def _generate_final_report(self):
        """Generate comprehensive final report."""
        print(f"\n" + "="*80)
        print("📊 COMPREHENSIVE VALIDATION REPORT")
        print("="*80)
        
        # Get final metrics
        metrics = self.monitor.get_metrics()
        trends = self.monitor.get_trends()
        
        print(f"\n🎯 SUMMARY STATISTICS:")
        print(f"   Total Validations: {metrics['basic_metrics']['total_validations']}")
        print(f"   Successful Validations: {metrics['basic_metrics']['successful_validations']}")
        print(f"   Failed Validations: {metrics['basic_metrics']['failed_validations']}")
        print(f"   Overall Success Rate: {metrics['basic_metrics']['success_rate']:.1%}")
        
        print(f"\n⏱️  PERFORMANCE METRICS:")
        print(f"   Average Validation Time: {metrics['timing_metrics']['avg_validation_time']:.3f}s")
        print(f"   Min Validation Time: {metrics['timing_metrics']['min_validation_time']:.3f}s")
        print(f"   Max Validation Time: {metrics['timing_metrics']['max_validation_time']:.3f}s")
        print(f"   95th Percentile: {metrics['timing_metrics']['p95_validation_time']:.3f}s")
        
        print(f"\n🎯 QUALITY METRICS:")
        print(f"   Average Confidence: {metrics['quality_metrics']['avg_confidence']:.2f}")
        print(f"   High Confidence Rate: {metrics['quality_metrics']['high_confidence_rate']:.1%}")
        print(f"   Low Confidence Rate: {metrics['quality_metrics']['low_confidence_rate']:.1%}")
        
        print(f"\n🚀 PERFORMANCE METRICS:")
        print(f"   Cache Hit Rate: {metrics['performance_metrics']['cache_hit_rate']:.1%}")
        print(f"   Parallel Utilization: {metrics['performance_metrics']['parallel_utilization']:.1%}")
        print(f"   Memory Usage: {metrics['performance_metrics']['memory_usage_mb']:.1f} MB")
        print(f"   CPU Usage: {metrics['performance_metrics']['cpu_usage_percent']:.1f}%")
        
        print(f"\n❌ ERROR METRICS:")
        print(f"   Error Rate: {metrics['error_metrics']['error_rate']:.1%}")
        print(f"   Timeout Rate: {metrics['error_metrics']['timeout_rate']:.1%}")
        print(f"   Retry Rate: {metrics['error_metrics']['retry_rate']:.1%}")
        
        if trends:
            print(f"\n📈 TREND ANALYSIS:")
            for trend in trends:
                print(f"   {trend.metric_name}: {trend.trend_direction} (strength: {trend.trend_strength:.2f})")
        
        # Export metrics
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        metrics_file = f"validation_metrics_{timestamp}.json"
        self.monitor.export_metrics(metrics_file)
        print(f"\n💾 Metrics exported to: {metrics_file}")
        
        # Stop monitoring
        self.monitor.stop_monitoring()
        print(f"\n✅ Validation system demonstration completed!")


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
        print("\n2. Initializing validation components...")
        
        # Create validation config
        config = ValidationConfig(
            enabled_strategies=[
                ValidationStrategy.GOAL_ALIGNMENT,
                ValidationStrategy.SAFETY_COMPLIANCE,
                ValidationStrategy.OUTPUT_QUALITY
            ],
            enable_parallel_strategies=True,
            cache_ttl_seconds=60
        )
        
        validator = RuntimeValidator(gemini_analyzer, config)
        corrector = StepCorrector(gemini_analyzer)
        engine = ValidationEngine(validator, corrector)
        
        print("✅ Validation components initialized successfully")
        
        # Create test execution step
        print("\n3. Creating test execution step...")
        
        test_step = ExecutionStep(
            step_id="test_001",
            framework="langchain",
            component="test_agent",
            operation="test_operation",
            input_data={
                "query": "What is the capital of France?",
                "context": "geography_quiz"
            },
            agent_goal="Answer the user's geography question accurately",
            step_reasoning="I need to provide the correct answer to the geography question"
        )
        
        print("✅ Test execution step created")
        print(f"   Step ID: {test_step.step_id}")
        print(f"   Component: {test_step.component}")
        print(f"   Goal: {test_step.agent_goal}")
        
        # Run validation
        print("\n4. Running validation...")
        
        start_time = time.time()
        validation_result = await validator.validate_step(test_step)
        validation_time = time.time() - start_time
        
        print(f"✅ Validation completed in {validation_time:.3f}s")
        print(f"   Result: {'✅ VALID' if validation_result.is_valid else '❌ INVALID'}")
        print(f"   Confidence: {validation_result.confidence:.2f}")
        print(f"   Reasoning: {validation_result.reasoning}")
        
        if validation_result.issues:
            print(f"   Issues: {validation_result.issues}")
        
        if validation_result.suggestions:
            print(f"   Suggestions: {validation_result.suggestions}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        logger.error(f"Simple validation test failed: {e}")
        return False


async def main():
    """Main function to run the validation system demonstration."""
    print("🎯 AIGIE VALIDATION SYSTEM DEMONSTRATION")
    print("="*60)
    
    # Run comprehensive demo
    demo = ValidationDemo()
    success = await demo.run_comprehensive_demo()
    
    if success:
        print("\n🎉 Validation system demonstration completed successfully!")
    else:
        print("\n❌ Validation system demonstration failed!")
    
    return success


if __name__ == "__main__":
    # Run the comprehensive demo
    asyncio.run(main())
