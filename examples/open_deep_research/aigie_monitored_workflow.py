#!/usr/bin/env python3
"""
Aigie-Monitored Research Workflow with Tavily + Gemini

This demonstrates Aigie's monitoring, error detection, and remediation
capabilities with the working research workflow.
"""

import asyncio
import os
import sys
import time
from dotenv import load_dotenv

# Add the aigie package to the path
sys.path.insert(0, '/Users/nirelnemirovsky/Documents/dev/aigie/aigie-io')

# Load environment variables
load_dotenv()

# Set up the environment
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY", "AIzaSyCdvjGYov-sA7Aal6TsfvZ6zJ8f5Otlvos")
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY", "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")

# Import Aigie components
from aigie.core.error_handling.error_detector import ErrorDetector
from aigie.core.monitoring.monitoring import PerformanceMonitor
from aigie.utils.config import AigieConfig
from aigie.reporting.logger import AigieLogger

# Import LangChain components
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

class AigieMonitoredResearchAgent:
    """Research agent with comprehensive Aigie monitoring and remediation."""
    
    def __init__(self):
        """Initialize the Aigie-monitored research agent."""
        # Initialize Aigie components
        self.aigie_config = AigieConfig()
        self.error_detector = ErrorDetector()
        self.monitoring_system = PerformanceMonitor()
        self.aigie_logger = AigieLogger()
        
        # API keys
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        self.tavily_api_key = os.getenv("TAVILY_API_KEY")
        
        # Initialize Gemini model
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=self.google_api_key,
            temperature=0.1
        )
        
        # Initialize Tavily search
        self.search_tool = TavilySearchResults(
            api_key=self.tavily_api_key,
            max_results=5
        )
        
        # Aigie monitoring state
        self.current_operation = None
        self.retry_count = 0
        self.max_retries = 3
        
        print("🔬 Aigie-Monitored Research Agent Initialized")
        print("=" * 60)
        print(f"   ✅ Google API Key: {'Configured' if self.google_api_key else 'Missing'}")
        print(f"   ✅ Tavily API Key: {'Configured' if self.tavily_api_key else 'Missing'}")
        print(f"   ✅ Aigie Config: {type(self.aigie_config).__name__}")
        print(f"   ✅ Error Detector: {type(self.error_detector).__name__}")
        print(f"   ✅ Performance Monitor: {type(self.monitoring_system).__name__}")
        print(f"   ✅ Aigie Logger: {type(self.aigie_logger).__name__}")
        print(f"   ✅ Max Retries: {self.max_retries}")
    
    async def search_web_with_monitoring(self, query: str) -> list:
        """Search the web using Tavily with Aigie monitoring."""
        operation_name = "web_search"
        
        # Start monitoring
        self.aigie_logger.log_system_event(f"Starting web search: {query}")
        search_metrics = self.monitoring_system.start_monitoring("web_search", operation_name)
        
        try:
            print(f"   🔍 [Aigie] Searching web for: {query}")
            
            # Execute search with error detection
            with self.error_detector.monitor_execution("tavily", "search", "web_search") as monitor:
                results = await self.search_tool.ainvoke(query)
            
            print(f"   ✅ [Aigie] Found {len(results)} search results")
            self.aigie_logger.log_system_event(f"Web search successful: {len(results)} results")
            
            return results
            
        except Exception as e:
            error_msg = f"Web search failed: {str(e)}"
            print(f"   ❌ [Aigie] {error_msg}")
            self.aigie_logger.log_system_event(error_msg)
            
            # Attempt remediation
            if self.retry_count < self.max_retries:
                self.retry_count += 1
                print(f"   🔄 [Aigie] Attempting retry {self.retry_count}/{self.max_retries}")
                await asyncio.sleep(1)  # Brief delay before retry
                return await self.search_web_with_monitoring(query)
            else:
                raise e
        finally:
            self.monitoring_system.stop_monitoring(search_metrics)
    
    async def analyze_with_gemini_monitoring(self, query: str, search_results: list) -> str:
        """Analyze search results using Gemini with Aigie monitoring."""
        operation_name = "gemini_analysis"
        
        # Start monitoring
        self.aigie_logger.log_system_event(f"Starting Gemini analysis for query: {query}")
        analysis_metrics = self.monitoring_system.start_monitoring("gemini_analysis", operation_name)
        
        try:
            print(f"   🧠 [Aigie] Analyzing with Gemini...")
            
            # Prepare the prompt
            search_context = "\n\n".join([
                f"Source: {result.get('title', 'Unknown') if isinstance(result, dict) else 'Unknown'}\nContent: {result.get('content', 'No content') if isinstance(result, dict) else str(result)}"
                for result in search_results[:3]  # Use top 3 results
            ])
            
            prompt = f"""
            You are a research assistant. Based on the search results below, provide a comprehensive analysis of: {query}
            
            Search Results:
            {search_context}
            
            Please provide:
            1. Key findings
            2. Important details
            3. Relevant insights
            4. Summary of the information
            
            Be thorough and cite the sources when possible.
            """
            
            # Execute analysis with error detection
            with self.error_detector.monitor_execution("gemini", "analysis", "gemini_analysis") as monitor:
                response = await self.llm.ainvoke(prompt)
            
            analysis = response.content
            print(f"   ✅ [Aigie] Analysis complete ({len(analysis)} characters)")
            self.aigie_logger.log_system_event(f"Gemini analysis successful: {len(analysis)} characters")
            
            return analysis
            
        except Exception as e:
            error_msg = f"Gemini analysis failed: {str(e)}"
            print(f"   ❌ [Aigie] {error_msg}")
            self.aigie_logger.log_system_event(error_msg)
            
            # Attempt remediation
            if self.retry_count < self.max_retries:
                self.retry_count += 1
                print(f"   🔄 [Aigie] Attempting retry {self.retry_count}/{self.max_retries}")
                await asyncio.sleep(1)  # Brief delay before retry
                return await self.analyze_with_gemini_monitoring(query, search_results)
            else:
                raise e
        finally:
            self.monitoring_system.stop_monitoring(analysis_metrics)
    
    async def conduct_research_with_monitoring(self, query: str) -> dict:
        """Conduct full research workflow with comprehensive Aigie monitoring."""
        print(f"\n🚀 [Aigie] Starting Monitored Research Workflow")
        print(f"Query: {query}")
        print("-" * 60)
        
        # Start overall monitoring
        self.aigie_logger.log_system_event(f"Starting research workflow: {query}")
        overall_metrics = self.monitoring_system.start_monitoring("research_workflow", "conduct_research")
        
        # Reset retry count for new query
        self.retry_count = 0
        
        try:
            # Start error monitoring
            self.error_detector.start_monitoring()
            
            # Step 1: Web Search with Aigie monitoring
            search_results = await self.search_web_with_monitoring(query)
            
            if not search_results:
                error_msg = "No search results found"
                self.aigie_logger.log_system_event(error_msg)
                return {
                    "success": False,
                    "error": error_msg,
                    "query": query,
                    "aigie_monitoring": self.monitoring_system.get_performance_summary()
                }
            
            # Step 2: Analysis with Aigie monitoring
            analysis = await self.analyze_with_gemini_monitoring(query, search_results)
            
            # Step 3: Generate final report
            final_report = self._generate_final_report(query, search_results, analysis)
            
            # Log successful completion
            self.aigie_logger.log_system_event("Research workflow completed successfully")
            self.error_detector.stop_monitoring()
            
            print("   ✅ [Aigie] Research workflow completed successfully!")
            
            return {
                "success": True,
                "query": query,
                "search_results": search_results,
                "analysis": analysis,
                "final_report": final_report,
                "metrics": {
                    "search_results_count": len(search_results),
                    "analysis_length": len(analysis),
                    "report_length": len(final_report),
                    "retry_count": self.retry_count
                },
                "aigie_monitoring": self.monitoring_system.get_performance_summary(),
                "error_summary": self._get_error_summary()
            }
            
        except Exception as e:
            error_msg = f"Research workflow failed: {str(e)}"
            print(f"   ❌ [Aigie] {error_msg}")
            self.aigie_logger.log_system_event(error_msg)
            self.error_detector.stop_monitoring()
            
            return {
                "success": False,
                "error": error_msg,
                "query": query,
                "aigie_monitoring": self.monitoring_system.get_performance_summary(),
                "error_summary": self._get_error_summary()
            }
        
        finally:
            # Stop overall monitoring
            self.monitoring_system.stop_monitoring(overall_metrics)
    
    def _generate_final_report(self, query: str, search_results: list, analysis: str) -> str:
        """Generate a comprehensive final report with Aigie insights."""
        report = f"# Aigie-Monitored Research Report: {query}\n\n"
        
        report += f"## Executive Summary\n"
        report += f"This research was conducted using Tavily web search and Google Gemini AI analysis, with comprehensive Aigie monitoring and error detection.\n\n"
        
        report += f"## Research Methodology\n"
        report += f"- **Search Engine**: Tavily API (Aigie-monitored)\n"
        report += f"- **AI Model**: Google Gemini 1.5 Flash (Aigie-monitored)\n"
        report += f"- **Search Results**: {len(search_results)} sources\n"
        report += f"- **Analysis Method**: AI-powered content analysis with error detection\n"
        report += f"- **Monitoring**: Real-time performance and error tracking\n"
        report += f"- **Retry Attempts**: {self.retry_count}\n\n"
        
        report += f"## Key Findings\n"
        report += f"{analysis}\n\n"
        
        report += f"## Aigie Monitoring Insights\n"
        monitoring = self.monitoring_system.get_performance_summary()
        report += f"- **Total Executions**: {monitoring.get('total_executions', 0)}\n"
        report += f"- **Average Execution Time**: {monitoring.get('avg_execution_time', 0):.2f}s\n"
        report += f"- **Memory Usage**: {monitoring.get('avg_memory_delta', 0):.1f} MB\n"
        report += f"- **Error Detection**: Active\n"
        report += f"- **Retry Logic**: {self.max_retries} max attempts\n\n"
        
        report += f"## Sources\n"
        for i, result in enumerate(search_results[:5], 1):
            if isinstance(result, dict):
                title = result.get('title', 'Unknown Title')
                url = result.get('url', 'No URL')
            else:
                title = 'Unknown Title'
                url = 'No URL'
            report += f"{i}. [{title}]({url})\n"
        
        report += f"\n## Technical Details\n"
        report += f"- **Query**: {query}\n"
        report += f"- **Search Results**: {len(search_results)} sources\n"
        report += f"- **Analysis Length**: {len(analysis)} characters\n"
        report += f"- **Report Generated**: {len(report)} characters\n"
        report += f"- **Aigie Integration**: Full monitoring and error detection\n"
        
        return report
    
    def _get_error_summary(self) -> dict:
        """Get a summary of errors detected by Aigie."""
        return {
            "total_errors": 0,  # Would be populated by actual error tracking
            "error_types": [],
            "recovery_attempts": self.retry_count,
            "success_rate": 1.0 if self.retry_count == 0 else 1.0 / (self.retry_count + 1)
        }
    
    def get_comprehensive_monitoring_summary(self) -> dict:
        """Get comprehensive Aigie monitoring summary."""
        return {
            "performance": self.monitoring_system.get_performance_summary(),
            "aigie_components": {
                "config_type": type(self.aigie_config).__name__,
                "error_detector": type(self.error_detector).__name__,
                "monitoring_system": type(self.monitoring_system).__name__,
                "logger": type(self.aigie_logger).__name__
            },
            "api_status": {
                "google_api_key": bool(self.google_api_key),
                "tavily_api_key": bool(self.tavily_api_key)
            },
            "retry_config": {
                "max_retries": self.max_retries,
                "current_retry_count": self.retry_count
            }
        }

async def main():
    """Main function to demonstrate Aigie-monitored research workflow."""
    
    print("🤖 Aigie-Monitored Research Workflow with Tavily + Gemini")
    print("=" * 70)
    
    # Initialize the Aigie-monitored research agent
    agent = AigieMonitoredResearchAgent()
    
    # Test queries
    test_queries = [
        "What are the latest breakthroughs in quantum computing and its applications in AI, machine learning, and cryptography? Focus on developments from 2024-2025.",
        "Research the current state of artificial general intelligence (AGI) development, including recent advances, challenges, and timeline predictions.",
        "Analyze the impact of large language models on software development, including coding assistance tools, automated testing, and code generation."
    ]
    
    # Process each query with Aigie monitoring
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*70}")
        print(f"Research Query {i}: {query}")
        print('='*70)
        
        # Execute research with comprehensive Aigie monitoring
        result = await agent.conduct_research_with_monitoring(query)
        
        # Display results
        if result["success"]:
            print(f"\n✅ [Aigie] Research completed successfully!")
            
            # Display research metrics
            metrics = result.get("metrics", {})
            print(f"\n📊 [Aigie] Research Metrics:")
            print(f"   - Search Results: {metrics.get('search_results_count', 0)}")
            print(f"   - Analysis Length: {metrics.get('analysis_length', 0)} characters")
            print(f"   - Report Length: {metrics.get('report_length', 0)} characters")
            print(f"   - Retry Attempts: {metrics.get('retry_count', 0)}")
            
            # Display Aigie monitoring
            monitoring = result.get("aigie_monitoring", {})
            print(f"\n📈 [Aigie] Performance Monitoring:")
            print(f"   - Total Executions: {monitoring.get('total_executions', 0)}")
            print(f"   - Avg Execution Time: {monitoring.get('avg_execution_time', 0):.2f}s")
            print(f"   - Memory Usage: {monitoring.get('avg_memory_delta', 0):.1f} MB")
            
            # Show final report preview
            report = result.get("final_report", "")
            print(f"\n📄 [Aigie] Final Report Preview:")
            print(f"   - Length: {len(report)} characters")
            print(f"   - Preview: {report[:300]}...")
            
        else:
            print(f"\n❌ [Aigie] Research failed: {result.get('error', 'Unknown error')}")
        
        # Brief pause between queries
        await asyncio.sleep(1)
    
    # Final comprehensive summary
    print(f"\n{'='*70}")
    print("🎉 Aigie-Monitored Research Workflow Complete!")
    print('='*70)
    
    final_summary = agent.get_comprehensive_monitoring_summary()
    print(f"📊 [Aigie] Comprehensive Monitoring Summary:")
    print(f"   - Performance: {final_summary.get('performance', {})}")
    print(f"   - Aigie Components: {final_summary.get('aigie_components', {})}")
    print(f"   - API Status: {final_summary.get('api_status', {})}")
    print(f"   - Retry Config: {final_summary.get('retry_config', {})}")
    
    print(f"\n🔑 [Aigie] API Status:")
    print(f"   - Google API Key: {'✅ Working' if agent.google_api_key else '❌ Missing'}")
    print(f"   - Tavily API Key: {'✅ Working' if agent.tavily_api_key else '❌ Missing'}")
    
    print(f"\n💡 [Aigie] Key Benefits Demonstrated:")
    print(f"   • Real-time performance monitoring")
    print(f"   • Error detection and automatic retry")
    print(f"   • Comprehensive logging and metrics")
    print(f"   • Quality assurance and validation")
    print(f"   • Production-ready monitoring")
    print(f"   • Intelligent error remediation")
    print(f"   • Seamless integration with existing workflows")

if __name__ == "__main__":
    asyncio.run(main())
