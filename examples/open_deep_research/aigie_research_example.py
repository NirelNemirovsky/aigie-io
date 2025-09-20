#!/usr/bin/env python3
"""
Comprehensive example demonstrating Aigie integration with Open Deep Research.

This example shows how to:
1. Initialize Aigie with the research agent
2. Configure runtime validation and error handling
3. Monitor agent execution with detailed logging
4. Handle errors gracefully with intelligent recovery
5. Generate comprehensive reports with validation
"""

import asyncio
import os
import sys
from typing import Dict, Any, List
from dotenv import load_dotenv

# Add the aigie package to the path
sys.path.insert(0, '/Users/nirelnemirovsky/Documents/dev/aigie/aigie-io')

from aigie import Aigie
from aigie.core.validation.runtime_validator import RuntimeValidator
from aigie.core.error_handling.error_detector import ErrorDetector
from aigie.core.monitoring.monitoring import MonitoringSystem
from aigie.core.ai.gemini_analyzer import GeminiAnalyzer

# Load environment variables
load_dotenv()

class AigieResearchAgent:
    """Aigie-enhanced research agent with comprehensive monitoring and validation."""
    
    def __init__(self):
        """Initialize the Aigie research agent."""
        self.aigie = Aigie()
        self.runtime_validator = RuntimeValidator()
        self.error_detector = ErrorDetector()
        self.monitoring_system = MonitoringSystem()
        self.gemini_analyzer = GeminiAnalyzer()
        
        # Set up environment
        os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY", "AIzaSyCdvjGYov-sA7Aal6TsfvZ6zJ8f5Otlvos")
        
        print("🔬 Aigie Research Agent Initialized")
        print(f"   ✅ Aigie: {type(self.aigie).__name__}")
        print(f"   ✅ Runtime Validator: {type(self.runtime_validator).__name__}")
        print(f"   ✅ Error Detector: {type(self.error_detector).__name__}")
        print(f"   ✅ Monitoring System: {type(self.monitoring_system).__name__}")
        print(f"   ✅ Gemini Analyzer: {type(self.gemini_analyzer).__name__}")
    
    async def validate_research_query(self, query: str) -> Dict[str, Any]:
        """Validate a research query using Aigie's validation system."""
        print(f"\n🔍 Validating Research Query: '{query}'")
        
        validation_result = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "suggestions": []
        }
        
        # Check query length
        if len(query) < 10:
            validation_result["is_valid"] = False
            validation_result["errors"].append("Query too short. Please provide more details.")
        
        if len(query) > 1000:
            validation_result["warnings"].append("Query very long. Consider breaking into smaller topics.")
        
        # Check for common issues
        if query.lower().strip() in ["", "?", "help", "what"]:
            validation_result["is_valid"] = False
            validation_result["errors"].append("Query too vague. Please be more specific.")
        
        # Check for research-worthy content
        research_keywords = ["research", "study", "analysis", "investigate", "explore", "compare", "evaluate"]
        if not any(keyword in query.lower() for keyword in research_keywords):
            validation_result["suggestions"].append("Consider adding research-oriented keywords for better results.")
        
        # Use Gemini analyzer for advanced validation
        try:
            gemini_analysis = await self.gemini_analyzer.analyze_query_quality(query)
            if gemini_analysis:
                validation_result["gemini_analysis"] = gemini_analysis
        except Exception as e:
            validation_result["warnings"].append(f"Advanced analysis unavailable: {str(e)}")
        
        return validation_result
    
    async def execute_research_with_monitoring(self, query: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute research with comprehensive Aigie monitoring."""
        
        print(f"\n🚀 Starting Research Execution for: '{query}'")
        
        # Step 1: Validate the query
        query_validation = await self.validate_research_query(query)
        if not query_validation["is_valid"]:
            return {
                "success": False,
                "error": "Query validation failed",
                "validation_result": query_validation,
                "monitoring_data": self.monitoring_system.get_summary()
            }
        
        # Step 2: Start monitoring the research process
        self.monitoring_system.start_research_session(query)
        
        try:
            # Step 3: Simulate research execution with monitoring
            research_steps = [
                "query_analysis",
                "source_identification", 
                "information_gathering",
                "data_validation",
                "synthesis",
                "report_generation"
            ]
            
            results = {}
            
            for step in research_steps:
                print(f"   📋 Executing step: {step}")
                
                # Start monitoring this step
                self.monitoring_system.start_step(step)
                
                try:
                    # Simulate step execution
                    await asyncio.sleep(0.5)  # Simulate processing time
                    
                    # Validate step results
                    step_result = await self._simulate_research_step(step, query)
                    
                    # Log successful completion
                    self.monitoring_system.complete_step(step, success=True, data=step_result)
                    results[step] = step_result
                    
                    print(f"   ✅ {step} completed successfully")
                    
                except Exception as e:
                    # Log step failure
                    error_context = await self.error_detector.extract_error_context(e, {"step": step, "query": query}, config)
                    self.monitoring_system.log_error(step, str(e), error_context)
                    print(f"   ❌ {step} failed: {str(e)}")
                    
                    # Try to recover
                    recovery_result = await self._attempt_recovery(step, e)
                    if recovery_result["success"]:
                        print(f"   🔄 {step} recovered successfully")
                        results[step] = recovery_result["data"]
                    else:
                        print(f"   💥 {step} recovery failed")
                        return {
                            "success": False,
                            "error": f"Step {step} failed and could not recover",
                            "monitoring_data": self.monitoring_system.get_summary()
                        }
            
            # Step 4: Generate final report with validation
            print("\n📊 Generating Final Report...")
            final_report = await self._generate_final_report(results, query)
            
            # Validate the final report
            report_validation = await self.runtime_validator.validate_content(final_report)
            if not report_validation.is_valid:
                print("   ⚠️  Report validation failed, but continuing...")
            
            # Complete the research session
            self.monitoring_system.complete_research_session(success=True)
            
            return {
                "success": True,
                "query": query,
                "final_report": final_report,
                "step_results": results,
                "monitoring_data": self.monitoring_system.get_summary(),
                "validation_result": query_validation
            }
            
        except Exception as e:
            # Log overall failure
            error_context = await self.error_detector.extract_error_context(e, {"query": query}, config)
            self.monitoring_system.log_error("research_execution", str(e), error_context)
            self.monitoring_system.complete_research_session(success=False)
            
            return {
                "success": False,
                "error": str(e),
                "monitoring_data": self.monitoring_system.get_summary()
            }
    
    async def _simulate_research_step(self, step: str, query: str) -> Dict[str, Any]:
        """Simulate a research step with realistic data."""
        
        step_data = {
            "query_analysis": {
                "keywords": ["AI", "validation", "research"],
                "complexity": "medium",
                "estimated_sources": 15
            },
            "source_identification": {
                "academic_papers": 8,
                "industry_reports": 4,
                "news_articles": 3
            },
            "information_gathering": {
                "sources_processed": 15,
                "data_points_collected": 150,
                "quality_score": 0.85
            },
            "data_validation": {
                "sources_verified": 12,
                "accuracy_score": 0.92,
                "bias_detected": False
            },
            "synthesis": {
                "key_findings": 5,
                "confidence_level": 0.88,
                "contradictions_found": 1
            },
            "report_generation": {
                "sections": 4,
                "word_count": 1200,
                "readability_score": 0.75
            }
        }
        
        return step_data.get(step, {"status": "completed", "data": "Simulated data"})
    
    async def _attempt_recovery(self, step: str, error: Exception) -> Dict[str, Any]:
        """Attempt to recover from a failed step."""
        
        recovery_strategies = {
            "query_analysis": {"strategy": "simplify_query", "success_rate": 0.8},
            "source_identification": {"strategy": "use_backup_sources", "success_rate": 0.7},
            "information_gathering": {"strategy": "reduce_scope", "success_rate": 0.6},
            "data_validation": {"strategy": "manual_verification", "success_rate": 0.9},
            "synthesis": {"strategy": "use_partial_data", "success_rate": 0.5},
            "report_generation": {"strategy": "template_fallback", "success_rate": 0.8}
        }
        
        strategy = recovery_strategies.get(step, {"strategy": "generic_retry", "success_rate": 0.3})
        
        # Simulate recovery attempt
        await asyncio.sleep(0.2)
        
        # Simulate success based on strategy success rate
        import random
        success = random.random() < strategy["success_rate"]
        
        if success:
            return {
                "success": True,
                "strategy": strategy["strategy"],
                "data": f"Recovered using {strategy['strategy']}"
            }
        else:
            return {
                "success": False,
                "strategy": strategy["strategy"],
                "error": f"Recovery failed using {strategy['strategy']}"
            }
    
    async def _generate_final_report(self, results: Dict[str, Any], query: str) -> str:
        """Generate a final research report."""
        
        report = f"""
# Research Report: {query}

## Executive Summary
This research was conducted using Aigie-enhanced validation and monitoring systems to ensure quality and reliability.

## Key Findings
Based on the analysis of {results.get('source_identification', {}).get('academic_papers', 0)} academic papers and {results.get('source_identification', {}).get('industry_reports', 0)} industry reports:

1. **Data Quality**: Achieved {results.get('information_gathering', {}).get('quality_score', 0):.1%} quality score
2. **Source Verification**: {results.get('data_validation', {}).get('sources_verified', 0)} sources verified with {results.get('data_validation', {}).get('accuracy_score', 0):.1%} accuracy
3. **Confidence Level**: {results.get('synthesis', {}).get('confidence_level', 0):.1%} confidence in findings

## Methodology
- Query Analysis: {results.get('query_analysis', {}).get('complexity', 'unknown')} complexity
- Source Identification: {results.get('source_identification', {}).get('academic_papers', 0)} academic sources
- Data Validation: {results.get('data_validation', {}).get('bias_detected', 'unknown')} bias detected

## Conclusion
The research process was successfully completed with Aigie's validation and monitoring systems ensuring high-quality results.

---
*Report generated with Aigie Runtime Validation System*
"""
        
        return report.strip()
    
    def get_monitoring_summary(self) -> Dict[str, Any]:
        """Get a summary of the monitoring data."""
        return self.monitoring_system.get_summary()

async def main():
    """Main function to demonstrate Aigie integration."""
    
    print("🔬 Aigie Research Agent Demo")
    print("=" * 50)
    
    # Initialize the Aigie research agent
    agent = AigieResearchAgent()
    
    # Test queries
    test_queries = [
        "What are the benefits of runtime validation in AI systems?",
        "How does error handling improve agent reliability?",
        "What monitoring techniques are most effective for AI agents?"
    ]
    
    # Configuration
    config = {
        "max_iterations": 3,
        "validation_strict_mode": True,
        "enable_recovery": True
    }
    
    # Process each query
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*60}")
        print(f"Query {i}: {query}")
        print('='*60)
        
        # Execute research with monitoring
        result = await agent.execute_research_with_monitoring(query, config)
        
        # Display results
        if result["success"]:
            print(f"\n✅ Research completed successfully!")
            print(f"📊 Final Report Preview:")
            print("-" * 40)
            print(result["final_report"][:300] + "...")
        else:
            print(f"\n❌ Research failed: {result.get('error', 'Unknown error')}")
        
        # Display monitoring summary
        monitoring = result.get("monitoring_data", {})
        print(f"\n📈 Monitoring Summary:")
        print(f"   Steps Executed: {monitoring.get('total_steps', 0)}")
        print(f"   Successful Steps: {monitoring.get('successful_steps', 0)}")
        print(f"   Failed Steps: {monitoring.get('failed_steps', 0)}")
        print(f"   Recovery Attempts: {monitoring.get('recovery_attempts', 0)}")
    
    # Final summary
    print(f"\n{'='*60}")
    print("🎉 Aigie Integration Demo Complete!")
    print('='*60)
    
    final_summary = agent.get_monitoring_summary()
    print(f"📊 Overall Performance:")
    print(f"   Total Research Sessions: {final_summary.get('total_sessions', 0)}")
    print(f"   Success Rate: {final_summary.get('success_rate', 0):.1%}")
    print(f"   Average Steps per Session: {final_summary.get('avg_steps_per_session', 0):.1f}")

if __name__ == "__main__":
    asyncio.run(main())

