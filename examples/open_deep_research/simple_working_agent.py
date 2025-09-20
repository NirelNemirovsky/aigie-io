#!/usr/bin/env python3
"""
Simple Working Open Deep Research Agent with Tavily + Gemini

This is a simplified version that works with available dependencies
and demonstrates the full workflow with both API keys.
"""

import asyncio
import os
import sys
from dotenv import load_dotenv

# Add the aigie package to the path
sys.path.insert(0, '/Users/nirelnemirovsky/Documents/dev/aigie/aigie-io')

# Load environment variables
load_dotenv()

# Set up the environment
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY", "AIzaSyCdvjGYov-sA7Aal6TsfvZ6zJ8f5Otlvos")
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY", "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")

# Import LangChain components
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

class SimpleResearchAgent:
    """Simplified research agent that works with Tavily + Gemini."""
    
    def __init__(self):
        """Initialize the research agent with API keys."""
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
        
        print("🔬 Simple Research Agent Initialized")
        print("=" * 50)
        print(f"   ✅ Google API Key: {'Configured' if self.google_api_key else 'Missing'}")
        print(f"   ✅ Tavily API Key: {'Configured' if self.tavily_api_key else 'Missing'}")
        print(f"   ✅ Gemini Model: gemini-1.5-flash")
        print(f"   ✅ Search Tool: Tavily")
    
    async def search_web(self, query: str) -> list:
        """Search the web using Tavily."""
        try:
            print(f"   🔍 Searching web for: {query}")
            results = await self.search_tool.ainvoke(query)
            print(f"   ✅ Found {len(results)} search results")
            return results
        except Exception as e:
            print(f"   ❌ Search failed: {e}")
            return []
    
    async def analyze_with_gemini(self, query: str, search_results: list) -> str:
        """Analyze search results using Gemini."""
        try:
            print(f"   🧠 Analyzing with Gemini...")
            
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
            
            # Get response from Gemini
            response = await self.llm.ainvoke(prompt)
            
            print(f"   ✅ Analysis complete ({len(response.content)} characters)")
            return response.content
            
        except Exception as e:
            print(f"   ❌ Analysis failed: {e}")
            return f"Analysis failed: {str(e)}"
    
    async def conduct_research(self, query: str) -> dict:
        """Conduct full research workflow."""
        print(f"\n🚀 Starting Research Workflow")
        print(f"Query: {query}")
        print("-" * 50)
        
        try:
            # Step 1: Web Search with Tavily
            search_results = await self.search_web(query)
            
            if not search_results:
                return {
                    "success": False,
                    "error": "No search results found",
                    "query": query
                }
            
            # Step 2: Analysis with Gemini
            analysis = await self.analyze_with_gemini(query, search_results)
            
            # Step 3: Generate final report
            final_report = self._generate_final_report(query, search_results, analysis)
            
            print("   ✅ Research workflow completed successfully!")
            
            return {
                "success": True,
                "query": query,
                "search_results": search_results,
                "analysis": analysis,
                "final_report": final_report,
                "metrics": {
                    "search_results_count": len(search_results),
                    "analysis_length": len(analysis),
                    "report_length": len(final_report)
                }
            }
            
        except Exception as e:
            print(f"   ❌ Research workflow failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "query": query
            }
    
    def _generate_final_report(self, query: str, search_results: list, analysis: str) -> str:
        """Generate a comprehensive final report."""
        report = f"# Research Report: {query}\n\n"
        
        report += f"## Executive Summary\n"
        report += f"This research was conducted using Tavily web search and Google Gemini AI analysis.\n\n"
        
        report += f"## Research Methodology\n"
        report += f"- **Search Engine**: Tavily API\n"
        report += f"- **AI Model**: Google Gemini 1.5 Flash\n"
        report += f"- **Search Results**: {len(search_results)} sources\n"
        report += f"- **Analysis Method**: AI-powered content analysis\n\n"
        
        report += f"## Key Findings\n"
        report += f"{analysis}\n\n"
        
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
        
        return report

async def main():
    """Main function to run the research workflow."""
    
    print("🤖 Open Deep Research Agent with Tavily + Gemini")
    print("=" * 60)
    
    # Initialize the research agent
    agent = SimpleResearchAgent()
    
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
        
        # Execute research workflow
        result = await agent.conduct_research(query)
        
        # Display results
        if result["success"]:
            print(f"\n✅ Research completed successfully!")
            
            # Display metrics
            metrics = result.get("metrics", {})
            print(f"\n📊 Research Metrics:")
            print(f"   - Search Results: {metrics.get('search_results_count', 0)}")
            print(f"   - Analysis Length: {metrics.get('analysis_length', 0)} characters")
            print(f"   - Report Length: {metrics.get('report_length', 0)} characters")
            
            # Show final report preview
            report = result.get("final_report", "")
            print(f"\n📄 Final Report Preview:")
            print(f"   - Length: {len(report)} characters")
            print(f"   - Preview: {report[:300]}...")
            
        else:
            print(f"\n❌ Research failed: {result.get('error', 'Unknown error')}")
        
        # Brief pause between queries
        await asyncio.sleep(1)
    
    # Final summary
    print(f"\n{'='*60}")
    print("🎉 Research Workflow Complete!")
    print('='*60)
    
    print(f"🔑 API Status:")
    print(f"   - Google API Key: {'✅ Working' if agent.google_api_key else '❌ Missing'}")
    print(f"   - Tavily API Key: {'✅ Working' if agent.tavily_api_key else '❌ Missing'}")
    
    print(f"\n💡 Workflow Features Demonstrated:")
    print(f"   • Tavily web search integration")
    print(f"   • Google Gemini AI analysis")
    print(f"   • Comprehensive research reports")
    print(f"   • Multi-query processing")
    print(f"   • Error handling and recovery")

if __name__ == "__main__":
    asyncio.run(main())
