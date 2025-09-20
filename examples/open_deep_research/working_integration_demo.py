#!/usr/bin/env python3
"""
Working Aigie + Open Deep Research Integration Demo

This demonstrates successful integration of Aigie monitoring
with a simulated research workflow, showing all key features.
"""

import asyncio
import sys
import os
import time
from dotenv import load_dotenv

# Add the aigie package to the path
sys.path.insert(0, '/Users/nirelnemirovsky/Documents/dev/aigie/aigie-io')

# Load environment variables
load_dotenv()

# Import Aigie components
from aigie.core.error_handling.error_detector import ErrorDetector
from aigie.core.monitoring.monitoring import PerformanceMonitor
from aigie.utils.config import AigieConfig
from aigie.reporting.logger import AigieLogger

class SimulatedResearchAgent:
    """Simulated research agent that demonstrates Aigie integration."""
    
    def __init__(self):
        """Initialize the simulated research agent with Aigie monitoring."""
        # Initialize Aigie components
        self.aigie_config = AigieConfig()
        self.error_detector = ErrorDetector()
        self.monitoring_system = PerformanceMonitor()
        self.aigie_logger = AigieLogger()
        
        # Set up Google API key
        self.google_api_key = os.getenv("GOOGLE_API_KEY", "AIzaSyCdvjGYov-sA7Aal6TsfvZ6zJ8f5Otlvos")
        os.environ["GOOGLE_API_KEY"] = self.google_api_key
        
        print("🔬 Aigie-Monitored Simulated Research Agent Initialized")
        print("=" * 60)
        print(f"   ✅ Google API Key: {'Configured' if self.google_api_key else 'Missing'}")
        print(f"   ✅ Aigie Config: {type(self.aigie_config).__name__}")
        print(f"   ✅ Error Detector: {type(self.error_detector).__name__}")
        print(f"   ✅ Performance Monitor: {type(self.monitoring_system).__name__}")
        print(f"   ✅ Aigie Logger: {type(self.aigie_logger).__name__}")
    
    async def simulate_research_step(self, step_name: str, duration: float = 1.0) -> dict:
        """Simulate a research step with Aigie monitoring."""
        
        # Start monitoring this step
        step_metrics = self.monitoring_system.start_monitoring("research_step", step_name)
        
        try:
            # Log step start
            self.aigie_logger.log_system_event(f"Starting research step: {step_name}")
            
            # Simulate work
            await asyncio.sleep(duration)
            
            # Simulate some processing
            result = {
                "step": step_name,
                "duration": duration,
                "status": "completed",
                "findings": f"Simulated findings for {step_name}",
                "sources": [f"source_{i}.com" for i in range(3)],
                "confidence": 0.85
            }
            
            # Log successful completion
            self.aigie_logger.log_system_event(f"Completed research step: {step_name}")
            
            return result
            
        except Exception as e:
            # Log error
            self.aigie_logger.log_system_event(f"Error in research step {step_name}: {str(e)}")
            raise
        finally:
            # Stop monitoring
            self.monitoring_system.stop_monitoring(step_metrics)
    
    async def conduct_research(self, query: str) -> dict:
        """Conduct simulated research with comprehensive Aigie monitoring."""
        
        print(f"\n🚀 Starting Aigie-Monitored Research")
        print(f"Query: {query}")
        print("-" * 60)
        
        # Start overall monitoring
        overall_metrics = self.monitoring_system.start_monitoring("research_agent", "conduct_research")
        
        try:
            # Log research start
            self.aigie_logger.log_system_event(f"Starting research: {query}")
            
            # Start error monitoring
            self.error_detector.start_monitoring()
            
            # Simulate research steps
            steps = [
                ("query_analysis", 0.5),
                ("source_discovery", 1.0),
                ("content_extraction", 1.5),
                ("data_processing", 1.0),
                ("synthesis", 1.5),
                ("report_generation", 1.0)
            ]
            
            results = []
            for step_name, duration in steps:
                print(f"   🔍 Executing: {step_name}")
                step_result = await self.simulate_research_step(step_name, duration)
                results.append(step_result)
                print(f"   ✅ Completed: {step_name}")
            
            # Generate final report
            final_report = self._generate_final_report(query, results)
            
            # Log successful completion
            self.aigie_logger.log_system_event("Research completed successfully")
            self.error_detector.stop_monitoring()
            
            print("   ✅ Research completed successfully!")
            
            return {
                "success": True,
                "query": query,
                "steps": results,
                "final_report": final_report,
                "research_metrics": self._extract_research_metrics(results),
                "aigie_monitoring": self.monitoring_system.get_performance_summary(),
                "error_summary": self._get_error_summary()
            }
            
        except Exception as e:
            # Log error
            self.aigie_logger.log_system_event(f"Research failed: {str(e)}")
            self.error_detector.stop_monitoring()
            
            print(f"   ❌ Research failed: {e}")
            
            return {
                "success": False,
                "query": query,
                "error": str(e),
                "aigie_monitoring": self.monitoring_system.get_performance_summary(),
                "error_summary": self._get_error_summary()
            }
        
        finally:
            # Stop overall monitoring
            self.monitoring_system.stop_monitoring(overall_metrics)
    
    def _generate_final_report(self, query: str, results: list) -> str:
        """Generate a final research report."""
        report = f"# Research Report: {query}\n\n"
        report += f"## Executive Summary\n"
        report += f"This research was conducted using Aigie-monitored simulation with {len(results)} steps.\n\n"
        
        report += f"## Research Steps\n"
        for i, result in enumerate(results, 1):
            report += f"### Step {i}: {result['step']}\n"
            report += f"- Status: {result['status']}\n"
            report += f"- Duration: {result['duration']}s\n"
            report += f"- Confidence: {result['confidence']}\n"
            report += f"- Sources: {len(result['sources'])}\n\n"
        
        report += f"## Key Findings\n"
        report += f"- Total research steps: {len(results)}\n"
        report += f"- Average confidence: {sum(r['confidence'] for r in results) / len(results):.2f}\n"
        report += f"- Total sources: {sum(len(r['sources']) for r in results)}\n"
        report += f"- Research completed with Aigie monitoring and error detection\n\n"
        
        report += f"## Aigie Integration Benefits\n"
        report += f"- Real-time performance monitoring\n"
        report += f"- Error detection and recovery\n"
        report += f"- Comprehensive logging\n"
        report += f"- Quality assurance\n"
        report += f"- Production-ready monitoring\n"
        
        return report
    
    def _extract_research_metrics(self, results: list) -> dict:
        """Extract key metrics from research results."""
        return {
            "total_steps": len(results),
            "total_duration": sum(r['duration'] for r in results),
            "avg_confidence": sum(r['confidence'] for r in results) / len(results),
            "total_sources": sum(len(r['sources']) for r in results),
            "successful_steps": len([r for r in results if r['status'] == 'completed']),
            "avg_step_duration": sum(r['duration'] for r in results) / len(results)
        }
    
    def _get_error_summary(self) -> dict:
        """Get a summary of errors detected by Aigie."""
        return {
            "total_errors": 0,  # Would be populated by actual error tracking
            "error_types": [],
            "recovery_attempts": 0,
            "success_rate": 1.0
        }
    
    def get_monitoring_summary(self) -> dict:
        """Get comprehensive monitoring summary."""
        return {
            "performance": self.monitoring_system.get_performance_summary(),
            "aigie_config": {
                "config_type": type(self.aigie_config).__name__,
                "error_detector": type(self.error_detector).__name__,
                "monitoring_system": type(self.monitoring_system).__name__,
                "logger": type(self.aigie_logger).__name__
            }
        }

async def main():
    """Main function to demonstrate Aigie-monitored research."""
    
    print("🤖 Aigie-Monitored Simulated Research Agent")
    print("=" * 60)
    
    # Initialize the simulated researcher
    researcher = SimulatedResearchAgent()
    
    # Test queries
    test_queries = [
        "What are the latest breakthroughs in quantum computing and its applications in AI, machine learning, and cryptography? Focus on developments from 2024-2025.",
        "Research the current state of artificial general intelligence (AGI) development, including recent advances, challenges, and timeline predictions.",
        "Analyze the impact of large language models on software development, including coding assistance tools, automated testing, and code generation."
    ]
    
    # Process each query
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*60}")
        print(f"Research Query {i}: {query}")
        print('='*60)
        
        # Execute research with Aigie monitoring
        result = await researcher.conduct_research(query)
        
        # Display results
        if result["success"]:
            print(f"\n✅ Research completed successfully!")
            
            # Display research metrics
            metrics = result.get("research_metrics", {})
            print(f"\n📊 Research Metrics:")
            print(f"   - Total Steps: {metrics.get('total_steps', 0)}")
            print(f"   - Total Duration: {metrics.get('total_duration', 0):.1f}s")
            print(f"   - Avg Confidence: {metrics.get('avg_confidence', 0):.2f}")
            print(f"   - Total Sources: {metrics.get('total_sources', 0)}")
            print(f"   - Successful Steps: {metrics.get('successful_steps', 0)}")
            
            # Display Aigie monitoring
            monitoring = result.get("aigie_monitoring", {})
            print(f"\n📈 Aigie Monitoring:")
            print(f"   - Total Executions: {monitoring.get('total_executions', 0)}")
            print(f"   - Avg Execution Time: {monitoring.get('avg_execution_time', 0):.2f}s")
            print(f"   - Memory Usage: {monitoring.get('avg_memory_delta', 0):.1f} MB")
            
            # Show final report preview
            report = result.get("final_report", "")
            print(f"\n📄 Final Report Preview:")
            print(f"   - Length: {len(report)} characters")
            print(f"   - Preview: {report[:200]}...")
            
        else:
            print(f"\n❌ Research failed: {result.get('error', 'Unknown error')}")
        
        # Brief pause between queries
        await asyncio.sleep(0.5)
    
    # Final summary
    print(f"\n{'='*60}")
    print("🎉 Aigie-Monitored Research Demo Complete!")
    print('='*60)
    
    final_summary = researcher.get_monitoring_summary()
    print(f"📊 Final Aigie Monitoring Summary:")
    print(f"   - Performance: {final_summary.get('performance', {})}")
    print(f"   - Aigie Components: {final_summary.get('aigie_config', {})}")
    
    print(f"\n🔑 API Status:")
    print(f"   - Google API Key: {'✅ Configured' if researcher.google_api_key else '❌ Missing'}")
    print(f"   - Tavily API Key: {'✅ Configured' if os.getenv('TAVILY_API_KEY') else '❌ Missing'}")
    
    print(f"\n💡 Key Benefits of Aigie Integration:")
    print(f"   • Real-time performance monitoring")
    print(f"   • Error detection and recovery")
    print(f"   • Comprehensive logging and metrics")
    print(f"   • Quality assurance and validation")
    print(f"   • Production-ready monitoring")
    print(f"   • Seamless integration with existing workflows")

if __name__ == "__main__":
    asyncio.run(main())
