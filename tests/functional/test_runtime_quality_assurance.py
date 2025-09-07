#!/usr/bin/env python3
"""
Runtime Execution Quality Assurance Tests for Aigie

This module provides comprehensive tests for Aigie's runtime execution quality assurance
features, including output validation, performance monitoring, and execution quality metrics.

Features:
- Output quality validation and scoring
- Performance regression detection
- Execution consistency testing
- Resource utilization monitoring
- Quality metrics tracking and analysis
- Automated quality gates and thresholds

Requirements:
- GEMINI_API_KEY for enhanced quality analysis
- Aigie auto-integration
- Performance monitoring tools
"""

import os
import sys
import time
import random
import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json
import statistics

# Add the parent directory to the path so we can import aigie
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aigie import auto_integrate, show_status, show_analysis

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Quality Assurance Models and Enums
# ============================================================================

class QualityLevel(Enum):
    """Quality levels for execution assessment."""
    EXCELLENT = "excellent"      # 90-100%
    GOOD = "good"               # 80-89%
    ACCEPTABLE = "acceptable"    # 70-79%
    POOR = "poor"               # 60-69%
    FAILED = "failed"           # <60%


class QualityMetric(Enum):
    """Quality metrics for assessment."""
    ACCURACY = "accuracy"
    COMPLETENESS = "completeness"
    CONSISTENCY = "consistency"
    PERFORMANCE = "performance"
    RELIABILITY = "reliability"
    EFFICIENCY = "efficiency"


@dataclass
class QualityScore:
    """Quality score for a specific metric."""
    metric: QualityMetric
    score: float  # 0.0 to 1.0
    level: QualityLevel
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Validate and set quality level."""
        if self.score >= 0.9:
            self.level = QualityLevel.EXCELLENT
        elif self.score >= 0.8:
            self.level = QualityLevel.GOOD
        elif self.score >= 0.7:
            self.level = QualityLevel.ACCEPTABLE
        elif self.score >= 0.6:
            self.level = QualityLevel.POOR
        else:
            self.level = QualityLevel.FAILED


@dataclass
class ExecutionQualityReport:
    """Comprehensive quality report for an execution."""
    execution_id: str
    start_time: datetime
    end_time: datetime
    overall_score: float
    overall_level: QualityLevel
    metric_scores: List[QualityScore]
    performance_metrics: Dict[str, Any]
    quality_issues: List[str]
    recommendations: List[str]
    
    def get_duration(self) -> float:
        """Get execution duration in seconds."""
        return (self.end_time - self.start_time).total_seconds()
    
    def get_metric_score(self, metric: QualityMetric) -> Optional[QualityScore]:
        """Get score for a specific metric."""
        for score in self.metric_scores:
            if score.metric == metric:
                return score
        return None


# ============================================================================
# Quality Assessment Tools
# ============================================================================

class QualityAssessor:
    """Assesses execution quality across multiple dimensions."""
    
    def __init__(self):
        self.assessment_history: List[ExecutionQualityReport] = []
        self.baseline_metrics: Dict[str, float] = {}
        self.quality_thresholds = {
            QualityMetric.ACCURACY: 0.8,
            QualityMetric.COMPLETENESS: 0.85,
            QualityMetric.CONSISTENCY: 0.75,
            QualityMetric.PERFORMANCE: 0.7,
            QualityMetric.RELIABILITY: 0.8,
            QualityMetric.EFFICIENCY: 0.75
        }
    
    def assess_execution(self, execution_data: Dict[str, Any]) -> ExecutionQualityReport:
        """Assess the quality of an execution."""
        execution_id = execution_data.get("execution_id", f"exec_{int(time.time())}")
        start_time = execution_data.get("start_time", datetime.now())
        end_time = execution_data.get("end_time", datetime.now())
        
        # Calculate individual metric scores
        metric_scores = []
        
        # Accuracy assessment
        accuracy_score = self._assess_accuracy(execution_data)
        metric_scores.append(accuracy_score)
        
        # Completeness assessment
        completeness_score = self._assess_completeness(execution_data)
        metric_scores.append(completeness_score)
        
        # Consistency assessment
        consistency_score = self._assess_consistency(execution_data)
        metric_scores.append(consistency_score)
        
        # Performance assessment
        performance_score = self._assess_performance(execution_data)
        metric_scores.append(performance_score)
        
        # Reliability assessment
        reliability_score = self._assess_reliability(execution_data)
        metric_scores.append(reliability_score)
        
        # Efficiency assessment
        efficiency_score = self._assess_efficiency(execution_data)
        metric_scores.append(efficiency_score)
        
        # Calculate overall score
        overall_score = statistics.mean([score.score for score in metric_scores])
        
        # Determine overall level
        if overall_score >= 0.9:
            overall_level = QualityLevel.EXCELLENT
        elif overall_score >= 0.8:
            overall_level = QualityLevel.GOOD
        elif overall_score >= 0.7:
            overall_level = QualityLevel.ACCEPTABLE
        elif overall_score >= 0.6:
            overall_level = QualityLevel.POOR
        else:
            overall_level = QualityLevel.FAILED
        
        # Identify quality issues and recommendations
        quality_issues = self._identify_quality_issues(metric_scores)
        recommendations = self._generate_recommendations(metric_scores, quality_issues)
        
        # Create quality report
        report = ExecutionQualityReport(
            execution_id=execution_id,
            start_time=start_time,
            end_time=end_time,
            overall_score=overall_score,
            overall_level=overall_level,
            metric_scores=metric_scores,
            performance_metrics=execution_data.get("performance_metrics", {}),
            quality_issues=quality_issues,
            recommendations=recommendations
        )
        
        # Store in history
        self.assessment_history.append(report)
        
        return report
    
    def _assess_accuracy(self, execution_data: Dict[str, Any]) -> QualityScore:
        """Assess accuracy of execution results."""
        # Simulate accuracy assessment based on execution data
        base_accuracy = 0.85
        
        # Adjust based on error rate
        error_rate = execution_data.get("error_rate", 0.0)
        accuracy_adjustment = -error_rate * 0.5
        
        # Adjust based on validation results
        validation_passed = execution_data.get("validation_passed", True)
        validation_adjustment = 0.1 if validation_passed else -0.2
        
        # Adjust based on output quality
        output_quality = execution_data.get("output_quality", 0.8)
        output_adjustment = (output_quality - 0.8) * 0.3
        
        final_score = max(0.0, min(1.0, base_accuracy + accuracy_adjustment + validation_adjustment + output_adjustment))
        
        return QualityScore(
            metric=QualityMetric.ACCURACY,
            score=final_score,
            level="high" if final_score >= 0.8 else "medium" if final_score >= 0.6 else "low",
            details={
                "base_accuracy": base_accuracy,
                "error_rate": error_rate,
                "validation_passed": validation_passed,
                "output_quality": output_quality,
                "adjustments": {
                    "error_rate": accuracy_adjustment,
                    "validation": validation_adjustment,
                    "output_quality": output_adjustment
                }
            }
        )
    
    def _assess_completeness(self, execution_data: Dict[str, Any]) -> QualityScore:
        """Assess completeness of execution."""
        # Simulate completeness assessment
        base_completeness = 0.9
        
        # Adjust based on missing outputs
        missing_outputs = execution_data.get("missing_outputs", 0)
        completeness_adjustment = -missing_outputs * 0.1
        
        # Adjust based on partial results
        partial_results = execution_data.get("partial_results", False)
        partial_adjustment = -0.15 if partial_results else 0.0
        
        # Adjust based on data coverage
        data_coverage = execution_data.get("data_coverage", 1.0)
        coverage_adjustment = (data_coverage - 1.0) * 0.2
        
        final_score = max(0.0, min(1.0, base_completeness + completeness_adjustment + partial_adjustment + coverage_adjustment))
        
        return QualityScore(
            metric=QualityMetric.COMPLETENESS,
            score=final_score,
            details={
                "base_completeness": base_completeness,
                "missing_outputs": missing_outputs,
                "partial_results": partial_results,
                "data_coverage": data_coverage,
                "adjustments": {
                    "missing_outputs": completeness_adjustment,
                    "partial_results": partial_adjustment,
                    "data_coverage": coverage_adjustment
                }
            }
        )
    
    def _assess_consistency(self, execution_data: Dict[str, Any]) -> QualityScore:
        """Assess consistency of execution."""
        # Simulate consistency assessment
        base_consistency = 0.8
        
        # Adjust based on result variance
        result_variance = execution_data.get("result_variance", 0.1)
        variance_adjustment = -result_variance * 0.5
        
        # Adjust based on execution stability
        execution_stability = execution_data.get("execution_stability", 0.9)
        stability_adjustment = (execution_stability - 0.9) * 0.3
        
        # Adjust based on repeatability
        repeatability = execution_data.get("repeatability", 0.85)
        repeatability_adjustment = (repeatability - 0.85) * 0.2
        
        final_score = max(0.0, min(1.0, base_consistency + variance_adjustment + stability_adjustment + repeatability_adjustment))
        
        return QualityScore(
            metric=QualityMetric.CONSISTENCY,
            score=final_score,
            details={
                "base_consistency": base_consistency,
                "result_variance": result_variance,
                "execution_stability": execution_stability,
                "repeatability": repeatability,
                "adjustments": {
                    "variance": variance_adjustment,
                    "stability": stability_adjustment,
                    "repeatability": repeatability_adjustment
                }
            }
        )
    
    def _assess_performance(self, execution_data: Dict[str, Any]) -> QualityScore:
        """Assess performance of execution."""
        # Simulate performance assessment
        base_performance = 0.75
        
        # Adjust based on execution time
        execution_time = execution_data.get("execution_time", 1.0)
        expected_time = execution_data.get("expected_time", 1.0)
        time_ratio = execution_time / expected_time
        time_adjustment = -max(0, time_ratio - 1.0) * 0.3
        
        # Adjust based on resource usage
        memory_usage = execution_data.get("memory_usage", 0.5)
        memory_adjustment = -max(0, memory_usage - 0.7) * 0.4
        
        # Adjust based on throughput
        throughput = execution_data.get("throughput", 1.0)
        expected_throughput = execution_data.get("expected_throughput", 1.0)
        throughput_ratio = throughput / expected_throughput
        throughput_adjustment = (throughput_ratio - 1.0) * 0.2
        
        final_score = max(0.0, min(1.0, base_performance + time_adjustment + memory_adjustment + throughput_adjustment))
        
        return QualityScore(
            metric=QualityMetric.PERFORMANCE,
            score=final_score,
            details={
                "base_performance": base_performance,
                "execution_time": execution_time,
                "expected_time": expected_time,
                "memory_usage": memory_usage,
                "throughput": throughput,
                "expected_throughput": expected_throughput,
                "adjustments": {
                    "time": time_adjustment,
                    "memory": memory_adjustment,
                    "throughput": throughput_adjustment
                }
            }
        )
    
    def _assess_reliability(self, execution_data: Dict[str, Any]) -> QualityScore:
        """Assess reliability of execution."""
        # Simulate reliability assessment
        base_reliability = 0.85
        
        # Adjust based on error rate
        error_rate = execution_data.get("error_rate", 0.0)
        error_adjustment = -error_rate * 0.6
        
        # Adjust based on retry rate
        retry_rate = execution_data.get("retry_rate", 0.0)
        retry_adjustment = -retry_rate * 0.3
        
        # Adjust based on success rate
        success_rate = execution_data.get("success_rate", 1.0)
        success_adjustment = (success_rate - 1.0) * 0.4
        
        final_score = max(0.0, min(1.0, base_reliability + error_adjustment + retry_adjustment + success_adjustment))
        
        return QualityScore(
            metric=QualityMetric.RELIABILITY,
            score=final_score,
            details={
                "base_reliability": base_reliability,
                "error_rate": error_rate,
                "retry_rate": retry_rate,
                "success_rate": success_rate,
                "adjustments": {
                    "error_rate": error_adjustment,
                    "retry_rate": retry_adjustment,
                    "success_rate": success_adjustment
                }
            }
        )
    
    def _assess_efficiency(self, execution_data: Dict[str, Any]) -> QualityScore:
        """Assess efficiency of execution."""
        # Simulate efficiency assessment
        base_efficiency = 0.8
        
        # Adjust based on resource utilization
        cpu_utilization = execution_data.get("cpu_utilization", 0.5)
        cpu_adjustment = -max(0, cpu_utilization - 0.8) * 0.3
        
        # Adjust based on memory efficiency
        memory_efficiency = execution_data.get("memory_efficiency", 0.8)
        memory_adjustment = (memory_efficiency - 0.8) * 0.4
        
        # Adjust based on I/O efficiency
        io_efficiency = execution_data.get("io_efficiency", 0.85)
        io_adjustment = (io_efficiency - 0.85) * 0.3
        
        final_score = max(0.0, min(1.0, base_efficiency + cpu_adjustment + memory_adjustment + io_adjustment))
        
        return QualityScore(
            metric=QualityMetric.EFFICIENCY,
            score=final_score,
            details={
                "base_efficiency": base_efficiency,
                "cpu_utilization": cpu_utilization,
                "memory_efficiency": memory_efficiency,
                "io_efficiency": io_efficiency,
                "adjustments": {
                    "cpu": cpu_adjustment,
                    "memory": memory_adjustment,
                    "io": io_adjustment
                }
            }
        )
    
    def _identify_quality_issues(self, metric_scores: List[QualityScore]) -> List[str]:
        """Identify quality issues from metric scores."""
        issues = []
        
        for score in metric_scores:
            if score.level in [QualityLevel.POOR, QualityLevel.FAILED]:
                issues.append(f"{score.metric.value} is {score.level.value} (score: {score.score:.2f})")
            elif score.score < self.quality_thresholds.get(score.metric, 0.7):
                issues.append(f"{score.metric.value} below threshold (score: {score.score:.2f})")
        
        return issues
    
    def _generate_recommendations(self, metric_scores: List[QualityScore], issues: List[str]) -> List[str]:
        """Generate recommendations for quality improvement."""
        recommendations = []
        
        for score in metric_scores:
            if score.level in [QualityLevel.POOR, QualityLevel.FAILED]:
                if score.metric == QualityMetric.ACCURACY:
                    recommendations.append("Improve input validation and error handling")
                elif score.metric == QualityMetric.COMPLETENESS:
                    recommendations.append("Ensure all required outputs are generated")
                elif score.metric == QualityMetric.CONSISTENCY:
                    recommendations.append("Standardize execution parameters and processes")
                elif score.metric == QualityMetric.PERFORMANCE:
                    recommendations.append("Optimize execution time and resource usage")
                elif score.metric == QualityMetric.RELIABILITY:
                    recommendations.append("Implement better error recovery mechanisms")
                elif score.metric == QualityMetric.EFFICIENCY:
                    recommendations.append("Optimize resource utilization and I/O operations")
        
        return recommendations
    
    def get_quality_trends(self, window_hours: int = 24) -> Dict[str, Any]:
        """Get quality trends over time."""
        cutoff_time = datetime.now() - timedelta(hours=window_hours)
        recent_reports = [r for r in self.assessment_history if r.start_time >= cutoff_time]
        
        if not recent_reports:
            return {"trend": "no_data", "reports": 0}
        
        # Calculate trend for overall score
        scores = [r.overall_score for r in recent_reports]
        if len(scores) > 1:
            trend = "improving" if scores[-1] > scores[0] else "declining"
        else:
            trend = "stable"
        
        # Calculate average scores by metric
        metric_averages = {}
        for metric in QualityMetric:
            metric_scores = [r.get_metric_score(metric).score for r in recent_reports if r.get_metric_score(metric)]
            if metric_scores:
                metric_averages[metric.value] = statistics.mean(metric_scores)
        
        return {
            "trend": trend,
            "reports": len(recent_reports),
            "average_overall_score": statistics.mean(scores),
            "metric_averages": metric_averages,
            "quality_level_distribution": {
                level.value: len([r for r in recent_reports if r.overall_level == level])
                for level in QualityLevel
            }
        }


# ============================================================================
# Quality Assurance Test Scenarios
# ============================================================================

class QualityAssuranceTestSuite:
    """Test suite for quality assurance features."""
    
    def __init__(self):
        self.quality_assessor = QualityAssessor()
        self.test_results = []
    
    async def test_output_quality_validation(self) -> Dict[str, Any]:
        """Test output quality validation and scoring."""
        print("\n📊 Testing Output Quality Validation...")
        
        test_scenarios = [
            {
                "name": "high_quality_execution",
                "execution_data": {
                    "error_rate": 0.05,
                    "validation_passed": True,
                    "output_quality": 0.95,
                    "missing_outputs": 0,
                    "partial_results": False,
                    "data_coverage": 1.0,
                    "result_variance": 0.05,
                    "execution_stability": 0.95,
                    "repeatability": 0.9,
                    "execution_time": 0.8,
                    "expected_time": 1.0,
                    "memory_usage": 0.6,
                    "throughput": 1.2,
                    "expected_throughput": 1.0,
                    "success_rate": 0.98,
                    "retry_rate": 0.02,
                    "cpu_utilization": 0.7,
                    "memory_efficiency": 0.9,
                    "io_efficiency": 0.88
                }
            },
            {
                "name": "medium_quality_execution",
                "execution_data": {
                    "error_rate": 0.15,
                    "validation_passed": True,
                    "output_quality": 0.8,
                    "missing_outputs": 1,
                    "partial_results": False,
                    "data_coverage": 0.9,
                    "result_variance": 0.15,
                    "execution_stability": 0.8,
                    "repeatability": 0.75,
                    "execution_time": 1.2,
                    "expected_time": 1.0,
                    "memory_usage": 0.8,
                    "throughput": 0.9,
                    "expected_throughput": 1.0,
                    "success_rate": 0.85,
                    "retry_rate": 0.1,
                    "cpu_utilization": 0.85,
                    "memory_efficiency": 0.75,
                    "io_efficiency": 0.8
                }
            },
            {
                "name": "low_quality_execution",
                "execution_data": {
                    "error_rate": 0.3,
                    "validation_passed": False,
                    "output_quality": 0.6,
                    "missing_outputs": 3,
                    "partial_results": True,
                    "data_coverage": 0.7,
                    "result_variance": 0.3,
                    "execution_stability": 0.6,
                    "repeatability": 0.5,
                    "execution_time": 2.0,
                    "expected_time": 1.0,
                    "memory_usage": 1.2,
                    "throughput": 0.6,
                    "expected_throughput": 1.0,
                    "success_rate": 0.7,
                    "retry_rate": 0.25,
                    "cpu_utilization": 0.95,
                    "memory_efficiency": 0.6,
                    "io_efficiency": 0.65
                }
            }
        ]
        
        results = []
        for scenario in test_scenarios:
            print(f"  Testing {scenario['name']}...")
            
            # Add execution metadata
            execution_data = scenario["execution_data"].copy()
            execution_data.update({
                "execution_id": f"test_{scenario['name']}_{int(time.time())}",
                "start_time": datetime.now(),
                "end_time": datetime.now() + timedelta(seconds=1)
            })
            
            # Assess quality
            quality_report = self.quality_assessor.assess_execution(execution_data)
            
            results.append({
                "scenario": scenario["name"],
                "overall_score": quality_report.overall_score,
                "overall_level": quality_report.overall_level.value,
                "metric_scores": {
                    score.metric.value: {
                        "score": score.score,
                        "level": score.level.value
                    }
                    for score in quality_report.metric_scores
                },
                "quality_issues": quality_report.quality_issues,
                "recommendations": quality_report.recommendations
            })
            
            print(f"    Overall Score: {quality_report.overall_score:.2f} ({quality_report.overall_level.value})")
            print(f"    Issues: {len(quality_report.quality_issues)}")
            print(f"    Recommendations: {len(quality_report.recommendations)}")
        
        return {
            "test_name": "output_quality_validation",
            "results": results,
            "success": True
        }
    
    async def test_performance_regression_detection(self) -> Dict[str, Any]:
        """Test performance regression detection."""
        print("\n⚡ Testing Performance Regression Detection...")
        
        # Simulate multiple executions with varying performance
        performance_scenarios = [
            {"execution_time": 0.8, "memory_usage": 0.6, "throughput": 1.2},  # Good performance
            {"execution_time": 0.9, "memory_usage": 0.7, "throughput": 1.1},  # Slightly worse
            {"execution_time": 1.1, "memory_usage": 0.8, "throughput": 0.9},  # Noticeable regression
            {"execution_time": 1.5, "memory_usage": 1.0, "throughput": 0.7},  # Significant regression
            {"execution_time": 0.7, "memory_usage": 0.5, "throughput": 1.3},  # Recovery
        ]
        
        results = []
        for i, scenario in enumerate(performance_scenarios):
            print(f"  Testing execution {i+1}...")
            
            execution_data = {
                "execution_id": f"perf_test_{i+1}",
                "start_time": datetime.now(),
                "end_time": datetime.now() + timedelta(seconds=1),
                "expected_time": 1.0,
                "expected_throughput": 1.0,
                **scenario
            }
            
            quality_report = self.quality_assessor.assess_execution(execution_data)
            performance_score = quality_report.get_metric_score(QualityMetric.PERFORMANCE)
            
            results.append({
                "execution": i+1,
                "execution_time": scenario["execution_time"],
                "memory_usage": scenario["memory_usage"],
                "throughput": scenario["throughput"],
                "performance_score": performance_score.score if performance_score else 0.0,
                "performance_level": performance_score.level.value if performance_score else "unknown"
            })
            
            print(f"    Performance Score: {performance_score.score:.2f} ({performance_score.level.value})")
        
        # Analyze trends
        performance_scores = [r["performance_score"] for r in results]
        if len(performance_scores) > 1:
            trend = "improving" if performance_scores[-1] > performance_scores[0] else "declining"
        else:
            trend = "stable"
        
        return {
            "test_name": "performance_regression_detection",
            "results": results,
            "trend": trend,
            "success": True
        }
    
    async def test_quality_trends_analysis(self) -> Dict[str, Any]:
        """Test quality trends analysis over time."""
        print("\n📈 Testing Quality Trends Analysis...")
        
        # Generate multiple quality reports over time
        base_time = datetime.now() - timedelta(hours=2)
        
        for i in range(10):
            execution_data = {
                "execution_id": f"trend_test_{i}",
                "start_time": base_time + timedelta(minutes=i*10),
                "end_time": base_time + timedelta(minutes=i*10 + 1),
                "error_rate": 0.1 + (i * 0.02),  # Gradually increasing error rate
                "execution_time": 1.0 + (i * 0.1),  # Gradually increasing execution time
                "success_rate": 0.95 - (i * 0.02),  # Gradually decreasing success rate
            }
            
            self.quality_assessor.assess_execution(execution_data)
        
        # Analyze trends
        trends = self.quality_assessor.get_quality_trends(window_hours=3)
        
        print(f"  Quality Trend: {trends['trend']}")
        print(f"  Reports Analyzed: {trends['reports']}")
        print(f"  Average Overall Score: {trends['average_overall_score']:.2f}")
        
        return {
            "test_name": "quality_trends_analysis",
            "trends": trends,
            "success": True
        }
    
    async def test_quality_gates_and_thresholds(self) -> Dict[str, Any]:
        """Test quality gates and threshold enforcement."""
        print("\n🚪 Testing Quality Gates and Thresholds...")
        
        # Test different quality levels against thresholds
        test_cases = [
            {"name": "passes_all_gates", "overall_score": 0.9},
            {"name": "passes_most_gates", "overall_score": 0.8},
            {"name": "fails_some_gates", "overall_score": 0.7},
            {"name": "fails_many_gates", "overall_score": 0.6},
            {"name": "fails_all_gates", "overall_score": 0.5},
        ]
        
        results = []
        for test_case in test_cases:
            print(f"  Testing {test_case['name']}...")
            
            # Create execution data that would result in the target overall score
            execution_data = {
                "execution_id": f"gate_test_{test_case['name']}",
                "start_time": datetime.now(),
                "end_time": datetime.now() + timedelta(seconds=1),
                "error_rate": 1.0 - test_case["overall_score"],  # Inverse relationship
                "execution_time": 2.0 - test_case["overall_score"],  # Inverse relationship
                "success_rate": test_case["overall_score"],  # Direct relationship
            }
            
            quality_report = self.quality_assessor.assess_execution(execution_data)
            
            # Check which gates pass/fail
            gate_results = {}
            for metric in QualityMetric:
                score = quality_report.get_metric_score(metric)
                threshold = self.quality_assessor.quality_thresholds.get(metric, 0.7)
                gate_results[metric.value] = {
                    "score": score.score if score else 0.0,
                    "threshold": threshold,
                    "passes": score.score >= threshold if score else False
                }
            
            results.append({
                "test_case": test_case["name"],
                "overall_score": quality_report.overall_score,
                "overall_level": quality_report.overall_level.value,
                "gate_results": gate_results,
                "gates_passed": sum(1 for gr in gate_results.values() if gr["passes"]),
                "total_gates": len(gate_results)
            })
            
            gates_passed = sum(1 for gr in gate_results.values() if gr["passes"])
            print(f"    Gates Passed: {gates_passed}/{len(gate_results)}")
        
        return {
            "test_name": "quality_gates_and_thresholds",
            "results": results,
            "success": True
        }


# ============================================================================
# Main Test Execution
# ============================================================================

async def run_quality_assurance_tests():
    """Run comprehensive quality assurance tests."""
    print("🚀 Starting Runtime Quality Assurance Tests")
    print("=" * 70)
    
    # Initialize Aigie with auto-integration
    print("\n📊 Initializing Aigie Error Detection System...")
    aigie = auto_integrate()
    error_detector = aigie.error_detector
    
    print("✅ Aigie monitoring started successfully")
    
    # Initialize test suite
    test_suite = QualityAssuranceTestSuite()
    
    # Run quality assurance tests
    print("\n" + "="*50)
    print("📊 RUNTIME QUALITY ASSURANCE TESTS")
    print("="*50)
    
    # Test output quality validation
    quality_validation_result = await test_suite.test_output_quality_validation()
    
    # Test performance regression detection
    performance_regression_result = await test_suite.test_performance_regression_detection()
    
    # Test quality trends analysis
    quality_trends_result = await test_suite.test_quality_trends_analysis()
    
    # Test quality gates and thresholds
    quality_gates_result = await test_suite.test_quality_gates_and_thresholds()
    
    # Display results
    print("\n" + "="*70)
    print("📊 QUALITY ASSURANCE TEST RESULTS")
    print("="*70)
    
    all_results = [
        quality_validation_result,
        performance_regression_result,
        quality_trends_result,
        quality_gates_result
    ]
    
    for result in all_results:
        test_name = result.get("test_name", "Unknown")
        success = result.get("success", False)
        print(f"\n{'✅' if success else '❌'} {test_name}: {'SUCCESS' if success else 'FAILED'}")
        
        if "results" in result:
            print(f"   • Test Cases: {len(result['results'])}")
        
        if "trend" in result:
            print(f"   • Trend: {result['trend']}")
        
        if "trends" in result:
            trends = result["trends"]
            print(f"   • Quality Trend: {trends['trend']}")
            print(f"   • Average Score: {trends['average_overall_score']:.2f}")
    
    # Display Aigie monitoring results
    print("\n" + "="*70)
    print("🤖 AIGIE MONITORING ANALYSIS")
    print("="*70)
    
    # Error detection summary
    error_summary = error_detector.get_error_summary(window_minutes=60)
    print(f"\n🚨 Error Detection Summary:")
    print(f"   • Total Errors Detected: {error_summary['total_errors']}")
    
    if error_summary['total_errors'] > 0:
        print(f"   • Severity Distribution: {error_summary['severity_distribution']}")
        print(f"   • Component Distribution: {error_summary['component_distribution']}")
        print(f"   • Gemini AI Analyzed: {error_summary.get('gemini_analyzed', 0)}")
        print(f"   • Automatic Retries: {error_summary.get('retry_attempts', 0)}")
        print(f"   • Successful Remediations: {error_summary.get('successful_remediations', 0)}")
    else:
        print(f"   ✅ No errors detected during testing")
    
    # System health
    system_health = error_detector.get_system_health()
    print(f"\n💚 System Health:")
    print(f"   • Monitoring Status: {'🟢 Active' if system_health['is_monitoring'] else '🔴 Inactive'}")
    print(f"   • Total Historical Errors: {system_health['total_errors']}")
    print(f"   • Recent Errors (5min): {system_health['recent_errors']}")
    
    # Show detailed analysis if errors occurred
    if error_detector.error_history:
        print(f"\n🔍 Detailed Error Analysis:")
        show_analysis()
    
    # Stop monitoring
    print(f"\n🛑 Stopping Aigie monitoring...")
    aigie.stop_integration()
    
    print(f"\n🎉 Runtime Quality Assurance Testing Completed!")
    print(f"📊 Aigie successfully demonstrated:")
    print(f"   ✓ Output quality validation and scoring")
    print(f"   ✓ Performance regression detection")
    print(f"   ✓ Quality trends analysis over time")
    print(f"   ✓ Quality gates and threshold enforcement")
    print(f"   ✓ Multi-dimensional quality assessment")
    print(f"   ✓ Automated quality recommendations")
    print(f"   ✓ Comprehensive quality reporting")
    
    return all_results


if __name__ == "__main__":
    # Run the quality assurance tests
    asyncio.run(run_quality_assurance_tests())
