#!/usr/bin/env python3
"""
Comprehensive AI Research Assistant with Aigie Error Detection and Monitoring

This example demonstrates aigie's capabilities with a comprehensive AI research assistant that:
1. Uses multiple sophisticated tools (web search, document analysis, code generation)
2. Implements complex LangGraph orchestration for real-world workflows
3. Triggers extensive error conditions that aigie can detect and handle intelligently
4. Shows real-time monitoring, performance tracking, and intelligent retry capabilities

The agent helps researchers by:
- Searching for relevant papers and information with comprehensive error simulation
- Analyzing documents and extracting key insights with detailed processing
- Generating code snippets for data analysis in multiple programming languages
- Managing complex research workflows with state management and error handling

Key Features:
- Comprehensive error simulation to test aigie's detection capabilities
- Detailed tool implementations with realistic processing patterns
- Complex workflow orchestration with state management
- Extensive aigie integration showing monitoring of all execution steps
- Real-world complexity demonstrating production-ready patterns
- Both modern and legacy patterns for educational comparison

This example serves as the main demonstration of aigie's capabilities in a realistic,
complex workflow with comprehensive error conditions and monitoring integration.

Requirements:
- LangChain and LangGraph (compatible with multiple versions)
- Model provider API key (OPENAI_API_KEY, ANTHROPIC_API_KEY, or GOOGLE_API_KEY)  
- GEMINI_API_KEY for enhanced error analysis (optional but recommended)
- Internet connection for web search functionality
"""

import os
import sys
import time
import random
import asyncio
from typing import Dict, Any, List, TypedDict, Optional, Protocol
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging
from abc import ABC, abstractmethod

# Add the parent directory to the path so we can import aigie
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aigie import auto_integrate, show_status, show_analysis
from aigie.core.error_detector import ErrorDetector
from aigie.core.gemini_analyzer import GeminiAnalyzer
from aigie.core.intelligent_retry import IntelligentRetry
from aigie.interceptors.langchain import LangChainInterceptor
from aigie.interceptors.langgraph import LangGraphInterceptor
from aigie.reporting.logger import AigieLogger

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Configuration and Constants
# ============================================================================

@dataclass
class Config:
    """Configuration for the research assistant."""
    # Error simulation rates
    NETWORK_ERROR_RATE: float = 0.15
    API_ERROR_RATE: float = 0.10
    TIMEOUT_ERROR_RATE: float = 0.05
    PROCESSING_ERROR_RATE: float = 0.20
    CODE_GENERATION_ERROR_RATE: float = 0.15
    MEMORY_LEAK_RATE: float = 0.10
    
    # Limits and thresholds
    RATE_LIMIT_THRESHOLD: int = 5
    MAX_MEMORY_MB: int = 100
    MAX_SEARCH_RESULTS: int = 5
    TIMEOUT_SECONDS: int = 5
    
    # Default values
    DEFAULT_LANGUAGE: str = "python"
    DEFAULT_RESEARCH_TOPIC: str = "machine learning in healthcare"


class ErrorType(Enum):
    """Types of errors that can be simulated."""
    NETWORK_ERROR = "network_error"
    API_ERROR = "api_error"
    TIMEOUT_ERROR = "timeout_error"
    RATE_LIMIT_ERROR = "rate_limit_error"
    PROCESSING_ERROR = "processing_error"
    MEMORY_ERROR = "memory_error"
    CODE_GENERATION_ERROR = "code_generation_error"


class WorkflowStep(Enum):
    """Workflow step identifiers."""
    INITIALIZED = "initialized"
    SEARCH_COMPLETED = "search_completed"
    SEARCH_FAILED = "search_failed"
    ANALYSIS_COMPLETED = "analysis_completed"
    ANALYSIS_FAILED = "analysis_failed"
    CODE_GENERATION_COMPLETED = "code_generation_completed"
    CODE_GENERATION_FAILED = "code_generation_failed"
    COMPLETED = "completed"
    SUMMARY_FAILED = "summary_failed"


# ============================================================================
# State and Data Models
# ============================================================================

class ResearchState(TypedDict):
    """State for the research workflow."""
    current_step: str
    research_topic: str
    search_results: List[Dict[str, Any]]
    documents: List[Dict[str, Any]]
    code_snippets: List[Dict[str, Any]]
    insights: List[str]
    errors: List[str]
    execution_count: int
    memory_usage: int
    start_time: float
    last_update: float


@dataclass
class ResearchDocument:
    """Represents a research document."""
    title: str
    content: str
    source: str
    timestamp: datetime
    relevance_score: float
    
    def __post_init__(self):
        """Validate document data."""
        if not 0.0 <= self.relevance_score <= 1.0:
            raise ValueError(f"Relevance score must be between 0 and 1, got {self.relevance_score}")
        if not self.title.strip():
            raise ValueError("Document title cannot be empty")


@dataclass
class CodeSnippet:
    """Represents a generated code snippet."""
    language: str
    code: str
    description: str
    dependencies: List[str]
    timestamp: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Validate code snippet data."""
        if not self.language.strip():
            raise ValueError("Language cannot be empty")
        if not self.code.strip():
            raise ValueError("Code cannot be empty")


# ============================================================================
# Tool Interfaces and Base Classes
# ============================================================================

class ResearchTool(ABC):
    """Abstract base class for research tools."""
    
    def __init__(self, config: Config):
        self.config = config
        self.error_count = 0
        self.operation_count = 0
    
    @abstractmethod
    def execute(self, *args, **kwargs) -> Any:
        """Execute the tool's main functionality."""
        pass
    
    def _simulate_error(self, error_type: ErrorType, error_rate: float, error_message: str) -> None:
        """Simulate an error based on probability."""
        if random.random() < error_rate:
            self.error_count += 1
            if error_type == ErrorType.TIMEOUT_ERROR:
                time.sleep(self.config.TIMEOUT_SECONDS)
            raise Exception(error_message)
    
    def get_stats(self) -> Dict[str, int]:
        """Get tool usage statistics."""
        return {
            "operation_count": self.operation_count,
            "error_count": self.error_count,
            "success_rate": (self.operation_count - self.error_count) / max(self.operation_count, 1)
        }


# ============================================================================
# Research Tools Implementation
# ============================================================================

class WebSearchTool(ResearchTool):
    """Tool for searching the web for research information."""
    
    def __init__(self, config: Config):
        super().__init__(config)
        self.search_engines = [
            "https://api.duckduckgo.com/",
            "https://serpapi.com/search",
            "https://api.bing.com/search"
        ]
        self.current_engine = 0
        self.rate_limit_counter = 0
    
    def execute(self, query: str, max_results: Optional[int] = None) -> List[Dict[str, Any]]:
        """Execute web search for research information."""
        return self.search(query, max_results or self.config.MAX_SEARCH_RESULTS)
    
    def search(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Search the web for research information."""
        try:
            self.operation_count += 1
            
            # Validate input
            if not query.strip():
                raise ValueError("Search query cannot be empty")
            
            # Simulate various error conditions for aigie to catch
            self._simulate_search_errors()
            
            # Mock search results (in real implementation, this would call actual APIs)
            results = []
            actual_results = min(max_results, self.config.MAX_SEARCH_RESULTS)
            
            for i in range(actual_results):
                results.append({
                    "title": f"Research paper on {query} - Part {i+1}",
                    "url": f"https://example.com/paper-{i+1}",
                    "snippet": f"This paper discusses various aspects of {query} including methodology and findings.",
                    "source": "academic_database",
                    "relevance": random.uniform(0.7, 0.95)
                })
            
            logger.info(f"Successfully retrieved {len(results)} search results for query: {query}")
            return results
            
        except Exception as e:
            logger.error(f"Web search failed for query '{query}': {e}")
            raise
    
    def _simulate_search_errors(self) -> None:
        """Simulate various error conditions to test aigie's error detection."""
        # Increment rate limit counter
        self.rate_limit_counter += 1
        
        # Simulate rate limiting
        if self.rate_limit_counter > self.config.RATE_LIMIT_THRESHOLD:
            self.rate_limit_counter = 0
            self._simulate_error(
                ErrorType.RATE_LIMIT_ERROR,
                1.0,  # Always trigger when threshold is exceeded
                "Rate limit exceeded. Please wait before making more requests."
            )
        
        # Simulate network errors
        self._simulate_error(
            ErrorType.NETWORK_ERROR,
            self.config.NETWORK_ERROR_RATE,
            "Network connection error - connection refused"
        )
        
        # Simulate API errors
        self._simulate_error(
            ErrorType.API_ERROR,
            self.config.API_ERROR_RATE,
            "API error - service temporarily unavailable"
        )
        
        # Simulate timeouts
        self._simulate_error(
            ErrorType.TIMEOUT_ERROR,
            self.config.TIMEOUT_ERROR_RATE,
            "Request timeout - operation took too long"
        )


class DocumentAnalysisTool(ResearchTool):
    """Tool for analyzing research documents and extracting insights."""
    
    def __init__(self, config: Config):
        super().__init__(config)
        self.processing_queue: List[ResearchDocument] = []
        self.memory_usage = 0.0
    
    def execute(self, document: ResearchDocument) -> Dict[str, Any]:
        """Execute document analysis."""
        return self.analyze_document(document)
    
    def analyze_document(self, document: ResearchDocument) -> Dict[str, Any]:
        """Analyze a research document and extract key insights."""
        try:
            self.operation_count += 1
            
            # Validate input
            if not isinstance(document, ResearchDocument):
                raise TypeError("Expected ResearchDocument instance")
            
            # Simulate memory-intensive processing
            self._simulate_memory_usage()
            
            # Simulate processing errors
            self._simulate_error(
                ErrorType.PROCESSING_ERROR,
                self.config.PROCESSING_ERROR_RATE,
                "Document processing failed - corrupted content detected"
            )
            
            # Extract key insights (mock implementation)
            methodologies = ['quantitative', 'qualitative', 'mixed']
            insights = [
                f"Key finding: {document.title} contains valuable information about the research topic",
                f"Methodology: The paper uses {random.choice(methodologies)} research methods",
                f"Results: Significant findings with p-value < 0.05",
                f"Limitations: Sample size may affect generalizability"
            ]
            
            processing_time = random.uniform(0.5, 2.0)
            confidence_score = random.uniform(0.8, 0.95)
            
            result = {
                "insights": insights,
                "summary": f"Analysis of {document.title} completed successfully",
                "confidence_score": confidence_score,
                "processing_time": processing_time,
                "document_id": id(document)
            }
            
            logger.info(f"Successfully analyzed document: {document.title}")
            return result
            
        except Exception as e:
            logger.error(f"Document analysis failed for '{document.title}': {e}")
            raise
    
    def _simulate_memory_usage(self) -> None:
        """Simulate memory usage for testing aigie's memory monitoring."""
        # Simulate memory allocation
        memory_increase = random.uniform(5, 20)  # MB
        self.memory_usage += memory_increase
        
        # Simulate memory overflow
        if self.memory_usage > self.config.MAX_MEMORY_MB:
            self._simulate_error(
                ErrorType.MEMORY_ERROR,
                1.0,  # Always trigger when threshold is exceeded
                "Memory allocation error - insufficient resources"
            )
        
        # Simulate memory leak
        self._simulate_error(
            ErrorType.MEMORY_ERROR,
            self.config.MEMORY_LEAK_RATE,
            f"Memory leak detected - usage increased to {self.memory_usage:.1f}MB"
        )
    
    def reset_memory(self) -> None:
        """Reset memory usage (simulate garbage collection)."""
        self.memory_usage = 0.0
        logger.info("Memory usage reset")


class CodeGenerationTool(ResearchTool):
    """Tool for generating code snippets for data analysis."""
    
    def __init__(self, config: Config):
        super().__init__(config)
        self.generated_snippets: List[CodeSnippet] = []
        self.language_support = ["python", "r", "matlab", "julia"]
        self.template_count = 0
    
    def execute(self, task: str, language: Optional[str] = None) -> CodeSnippet:
        """Execute code generation."""
        return self.generate_code(task, language or self.config.DEFAULT_LANGUAGE)
    
    def generate_code(self, task: str, language: str) -> CodeSnippet:
        """Generate code snippet for a given research task."""
        try:
            self.operation_count += 1
            
            # Validate inputs
            if not task.strip():
                raise ValueError("Task description cannot be empty")
            if not language.strip():
                raise ValueError("Language cannot be empty")
            
            # Validate language support
            if language not in self.language_support:
                raise ValueError(f"Unsupported language: {language}. Supported: {self.language_support}")
            
            # Simulate code generation errors
            self._simulate_error(
                ErrorType.CODE_GENERATION_ERROR,
                self.config.CODE_GENERATION_ERROR_RATE,
                "Code generation failed - invalid task description"
            )
            
            # Generate code based on task (mock implementation)
            code = self._generate_code_by_task(task, language)
            
            snippet = CodeSnippet(
                language=language,
                code=code,
                description=f"Generated code for: {task}",
                dependencies=self._get_dependencies(language)
            )
            
            self.generated_snippets.append(snippet)
            self.template_count += 1
            
            logger.info(f"Successfully generated {language} code for task: {task}")
            return snippet
            
        except Exception as e:
            logger.error(f"Code generation failed for task '{task}' in {language}: {e}")
            raise
    
    def _generate_code_by_task(self, task: str, language: str) -> str:
        """Generate code based on the specific task type."""
        task_lower = task.lower()
        
        if "data analysis" in task_lower:
            return self._generate_data_analysis_code(language)
        elif "visualization" in task_lower:
            return self._generate_visualization_code(language)
        elif "statistical" in task_lower:
            return self._generate_statistical_code(language)
        else:
            return self._generate_generic_code(language, task)
    
    def _generate_data_analysis_code(self, language: str) -> str:
        """Generate data analysis code."""
        if language == "python":
            return '''
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load and preprocess data
df = pd.read_csv('data.csv')
df_clean = df.dropna()

# Basic statistics
print(df_clean.describe())

# Correlation analysis
correlation_matrix = df_clean.corr()
print(correlation_matrix)
'''
        else:
            return f"# {language} code for data analysis\n# Implementation would go here"
    
    def _generate_visualization_code(self, language: str) -> str:
        """Generate visualization code."""
        if language == "python":
            return '''
import matplotlib.pyplot as plt
import seaborn as sns

# Create visualizations
plt.figure(figsize=(12, 8))

# Histogram
plt.subplot(2, 2, 1)
plt.hist(df['value'], bins=30, alpha=0.7)
plt.title('Distribution of Values')

# Box plot
plt.subplot(2, 2, 2)
plt.boxplot(df['value'])
plt.title('Box Plot')

plt.tight_layout()
plt.show()
'''
        else:
            return f"# {language} code for visualization\n# Implementation would go here"
    
    def _generate_statistical_code(self, language: str) -> str:
        """Generate statistical analysis code."""
        if language == "python":
            return '''
from scipy import stats
import statsmodels.api as sm

# T-test
t_stat, p_value = stats.ttest_ind(group1, group2)
print(f"T-test: t={t_stat:.4f}, p={p_value:.4f}")

# Linear regression
X = sm.add_constant(x)
model = sm.OLS(y, X).fit()
print(model.summary())
'''
        else:
            return f"# {language} code for statistical analysis\n# Implementation would go here"
    
    def _generate_generic_code(self, language: str, task: str) -> str:
        """Generate generic code for any task."""
        return f"# {language} code for: {task}\n# Custom implementation required"
    
    def _get_dependencies(self, language: str) -> List[str]:
        """Get dependencies for a given language."""
        if language == "python":
            return ["pandas", "numpy", "matplotlib", "seaborn", "scipy", "statsmodels"]
        elif language == "r":
            return ["dplyr", "ggplot2", "stats", "car"]
        elif language == "matlab":
            return ["Statistics and Machine Learning Toolbox"]
        elif language == "julia":
            return ["DataFrames", "Plots", "Statistics", "GLM"]
        else:
            return []


# ============================================================================
# LangGraph Workflow
# ============================================================================

class ResearchWorkflowManager:
    """Manages the research workflow and tool coordination."""
    
    def __init__(self, config: Config):
        self.config = config
        self.web_search = WebSearchTool(config)
        self.doc_analyzer = DocumentAnalysisTool(config)
        self.code_generator = CodeGenerationTool(config)
    
    def get_tool_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all tools."""
        return {
            "web_search": self.web_search.get_stats(),
            "doc_analyzer": self.doc_analyzer.get_stats(),
            "code_generator": self.code_generator.get_stats()
        }


def create_research_workflow(config: Config):
    """Create the LangGraph workflow for research assistance."""
    try:
        from langgraph.graph import StateGraph, END
        
        # Initialize workflow manager and tools
        workflow_manager = ResearchWorkflowManager(config)
        
        # Create the workflow graph
        workflow = StateGraph(ResearchState)
        
        # Define workflow nodes
        def search_node(state: ResearchState) -> ResearchState:
            """Search for research information."""
            try:
                logger.info(f"Searching for: {state['research_topic']}")
                
                # Perform web search
                search_results = workflow_manager.web_search.search(
                    state['research_topic'], 
                    config.MAX_SEARCH_RESULTS
                )
                state['search_results'] = search_results
                state['current_step'] = WorkflowStep.SEARCH_COMPLETED.value
                state['last_update'] = time.time()
                
                logger.info(f"Found {len(search_results)} search results")
                return state
                
            except Exception as e:
                error_msg = f"Search failed: {str(e)}"
                state['errors'].append(error_msg)
                state['current_step'] = WorkflowStep.SEARCH_FAILED.value
                logger.error(error_msg)
                raise
        
        def analyze_node(state: ResearchState) -> ResearchState:
            """Analyze search results and documents."""
            try:
                logger.info("Analyzing search results and documents")
                
                # Create documents from search results
                documents = []
                for result in state['search_results']:
                    try:
                        doc = ResearchDocument(
                            title=result['title'],
                            content=f"Content analysis of {result['title']}",
                            source=result['source'],
                            timestamp=datetime.now(),
                            relevance_score=result['relevance']
                        )
                        documents.append(doc)
                    except ValueError as e:
                        error_msg = f"Invalid document data: {str(e)}"
                        state['errors'].append(error_msg)
                        logger.warning(error_msg)
                
                # Analyze documents
                insights = []
                for doc in documents:
                    try:
                        analysis = workflow_manager.doc_analyzer.analyze_document(doc)
                        insights.extend(analysis['insights'])
                    except Exception as e:
                        error_msg = f"Document analysis failed for '{doc.title}': {str(e)}"
                        state['errors'].append(error_msg)
                        logger.error(error_msg)
                
                state['documents'] = [vars(doc) for doc in documents]
                state['insights'] = insights
                state['current_step'] = WorkflowStep.ANALYSIS_COMPLETED.value
                state['last_update'] = time.time()
                
                logger.info(f"Generated {len(insights)} insights from {len(documents)} documents")
                return state
                
            except Exception as e:
                error_msg = f"Analysis failed: {str(e)}"
                state['errors'].append(error_msg)
                state['current_step'] = WorkflowStep.ANALYSIS_FAILED.value
                logger.error(error_msg)
                raise
        
        def code_generation_node(state: ResearchState) -> ResearchState:
            """Generate code snippets for research tasks."""
            try:
                logger.info("Generating code snippets")
                
                # Define code generation tasks
                code_tasks = [
                    "data analysis of research findings",
                    "visualization of results",
                    "statistical analysis"
                ]
                
                code_snippets = []
                for task in code_tasks:
                    try:
                        snippet = workflow_manager.code_generator.generate_code(
                            task, 
                            config.DEFAULT_LANGUAGE
                        )
                        code_snippets.append(vars(snippet))
                    except Exception as e:
                        error_msg = f"Code generation failed for '{task}': {str(e)}"
                        state['errors'].append(error_msg)
                        logger.error(error_msg)
                
                state['code_snippets'] = code_snippets
                state['current_step'] = WorkflowStep.CODE_GENERATION_COMPLETED.value
                state['last_update'] = time.time()
                
                logger.info(f"Generated {len(code_snippets)} code snippets")
                return state
                
            except Exception as e:
                error_msg = f"Code generation failed: {str(e)}"
                state['errors'].append(error_msg)
                state['current_step'] = WorkflowStep.CODE_GENERATION_FAILED.value
                logger.error(error_msg)
                raise
        
        def summary_node(state: ResearchState) -> ResearchState:
            """Generate research summary."""
            try:
                logger.info("Generating research summary")
                
                # Calculate execution metrics
                execution_time = time.time() - state['start_time']
                memory_usage = sum(len(str(item)) for item in state.values()) / 1024  # Rough estimate
                
                # Get tool statistics
                tool_stats = workflow_manager.get_tool_stats()
                
                state['execution_count'] += 1
                state['memory_usage'] = int(memory_usage)
                state['current_step'] = WorkflowStep.COMPLETED.value
                state['last_update'] = time.time()
                
                # Log summary information
                logger.info(f"Research workflow completed in {execution_time:.2f}s")
                logger.info(f"Tool statistics: {tool_stats}")
                
                return state
                
            except Exception as e:
                error_msg = f"Summary generation failed: {str(e)}"
                state['errors'].append(error_msg)
                state['current_step'] = WorkflowStep.SUMMARY_FAILED.value
                logger.error(error_msg)
                raise
        
        # Add nodes to the graph
        workflow.add_node("search", search_node)
        workflow.add_node("analyze", analyze_node)
        workflow.add_node("code_generation", code_generation_node)
        workflow.add_node("summary", summary_node)
        
        # Define the workflow edges
        workflow.set_entry_point("search")
        workflow.add_edge("search", "analyze")
        workflow.add_edge("analyze", "code_generation")
        workflow.add_edge("code_generation", "summary")
        workflow.add_edge("summary", END)
        
        # Compile the workflow
        compiled_workflow = workflow.compile()
        
        logger.info("Research workflow created successfully")
        return compiled_workflow, workflow_manager
        
    except ImportError as e:
        logger.error(f"Failed to import LangGraph: {e}")
        raise
    except Exception as e:
        logger.error(f"Failed to create workflow: {e}")
        raise


# ============================================================================
# Main Execution
# ============================================================================

async def main():
    """Main execution function."""
    print("🚀 Starting AI Research Assistant with Aigie Monitoring")
    print("=" * 60)
    
    try:
        # Initialize configuration
        config = Config()
        
        # Initialize aigie components
        print("\n📊 Initializing Aigie Error Detection System...")
        
        # Create error detector with Gemini integration
        error_detector = ErrorDetector(
            enable_performance_monitoring=True,
            enable_resource_monitoring=True,
            enable_gemini_analysis=True
        )
        
        # Create logger
        aigie_logger = AigieLogger()
        
        # Create interceptors
        langchain_interceptor = LangChainInterceptor(error_detector, aigie_logger)
        langgraph_interceptor = LangGraphInterceptor(error_detector, aigie_logger)
        
        # Start monitoring
        error_detector.start_monitoring()
        langchain_interceptor.start_intercepting()
        langgraph_interceptor.start_intercepting()
        
        print("✅ Aigie monitoring started successfully")
        
        # Create the research workflow
        print("\n🔧 Creating Research Workflow...")
        workflow, workflow_manager = create_research_workflow(config)
        print("✅ Workflow created successfully")
        
        # Initialize research state
        initial_state = ResearchState(
            current_step=WorkflowStep.INITIALIZED.value,
            research_topic=config.DEFAULT_RESEARCH_TOPIC,
            search_results=[],
            documents=[],
            code_snippets=[],
            insights=[],
            errors=[],
            execution_count=0,
            memory_usage=0,
            start_time=time.time(),
            last_update=time.time()
        )
        
        # Execute the workflow with aigie monitoring
        print(f"\n🔍 Executing research workflow for: {initial_state['research_topic']}")
        print("-" * 50)
        
        with error_detector.monitor_execution("langgraph", "research_workflow", "invoke"):
            try:
                # Execute the workflow
                result = workflow.invoke(initial_state)
                
                # Display results
                print("\n📋 Research Results:")
                print(f"  • Search Results: {len(result['search_results'])} found")
                print(f"  • Documents Analyzed: {len(result['documents'])}")
                print(f"  • Insights Generated: {len(result['insights'])}")
                print(f"  • Code Snippets: {len(result['code_snippets'])}")
                print(f"  • Execution Time: {time.time() - result['start_time']:.2f}s")
                print(f"  • Final Step: {result['current_step']}")
                
                # Display tool statistics
                tool_stats = workflow_manager.get_tool_stats()
                print(f"\n📊 Tool Performance:")
                for tool_name, stats in tool_stats.items():
                    success_rate = stats['success_rate'] * 100
                    print(f"  • {tool_name}: {stats['operation_count']} ops, "
                          f"{stats['error_count']} errors, {success_rate:.1f}% success")
                
                if result['errors']:
                    print(f"\n⚠️  Errors encountered: {len(result['errors'])}")
                    for error in result['errors']:
                        print(f"  • {error}")
                
                print("\n✅ Research workflow completed successfully!")
                
            except Exception as e:
                print(f"\n❌ Workflow execution failed: {e}")
                logger.error(f"Workflow execution failed: {e}")
        
        # Show comprehensive aigie monitoring results
        print("\n📊 Comprehensive Aigie Monitoring Analysis:")
        print("=" * 60)
        
        # LangChain interceptor detailed status
        lc_status = langchain_interceptor.get_interception_status()
        print(f"\n🔗 LangChain Interceptor Status:")
        print(f"   • Active: {lc_status['is_intercepting']}")
        print(f"   • Intercepted Classes: {len(lc_status['intercepted_classes'])}")
        print(f"   • Patched Methods: {len(lc_status['patched_methods'])}")
        print(f"   • Target Classes: {lc_status['target_classes']}")
        
        if lc_status['intercepted_classes']:
            print(f"   • Monitored Components: {list(lc_status['intercepted_classes'])}")
        
        # LangGraph interceptor comprehensive status
        lg_status = langgraph_interceptor.get_interception_status()
        print(f"\n🔀 LangGraph Interceptor Status:")
        print(f"   • Active: {lg_status['is_intercepting']}")
        print(f"   • Tracked Graphs: {lg_status['tracked_graphs']}")
        print(f"   • Streaming Sessions: {lg_status['streaming_sessions']}")
        print(f"   • Active Streams: {lg_status['active_streams']}")
        print(f"   • Event History: {lg_status['event_history_size']}")
        print(f"   • Human Interactions: {lg_status['human_interactions']}")
        print(f"   • Checkpoint Operations: {lg_status['checkpoint_operations']}")
        
        # Show streaming analysis if available
        if lg_status['streaming_sessions'] > 0:
            streaming_analysis = langgraph_interceptor.get_streaming_analysis()
            print(f"\n📡 Streaming Event Analysis:")
            print(f"   • Total Sessions: {streaming_analysis['total_sessions']}")
            print(f"   • Completed Sessions: {streaming_analysis['completed_sessions']}")
            print(f"   • Error Sessions: {streaming_analysis['error_sessions']}")
            print(f"   • Total Events Processed: {streaming_analysis['total_events']}")
            
            if streaming_analysis.get('recent_event_types'):
                print(f"   • Event Types Detected: {streaming_analysis['recent_event_types']}")
        
        # Show checkpoint analysis if available
        if lg_status['checkpoint_operations'] > 0:
            checkpoint_analysis = langgraph_interceptor.get_checkpoint_analysis()
            print(f"\n💾 Checkpoint Operation Analysis:")
            print(f"   • Total Operations: {checkpoint_analysis['total_operations']}")
            print(f"   • Success Rate: {checkpoint_analysis['success_rate']:.1f}%")
            print(f"   • Operation Types: {checkpoint_analysis['operation_types']}")
        
        # Show human interaction analysis if available  
        if lg_status['human_interactions'] > 0:
            human_analysis = langgraph_interceptor.get_human_interaction_analysis()
            print(f"\n👤 Human Interaction Analysis:")
            print(f"   • Total Interactions: {human_analysis['total_interactions']}")
            print(f"   • Interaction Types: {human_analysis['interaction_types']}")
        
        # Error detection and system health analysis
        error_summary = error_detector.get_error_summary(window_minutes=60)
        print(f"\n🚨 Error Detection Summary (Last Hour):")
        print(f"   • Total Errors: {error_summary['total_errors']}")
        
        if error_summary['total_errors'] > 0:
            print(f"   • Severity Distribution: {error_summary['severity_distribution']}")
            print(f"   • Component Distribution: {error_summary['component_distribution']}")
            print(f"   • Gemini AI Analyzed: {error_summary.get('gemini_analyzed', 0)}")
            print(f"   • Automatic Retries: {error_summary.get('retry_attempts', 0)}")
            
            if error_summary.get('most_recent_error'):
                recent = error_summary['most_recent_error']
                print(f"   • Most Recent Error: {recent.get('error_type', 'unknown')} in {recent.get('component', 'unknown')}")
        else:
            print(f"   ✅ No errors detected - system running smoothly!")
        
        # System health overview
        system_health = error_detector.get_system_health()
        print(f"\n💚 System Health Overview:")
        print(f"   • Monitoring Status: {'🟢 Active' if system_health['is_monitoring'] else '🔴 Inactive'}")
        print(f"   • Total Historical Errors: {system_health['total_errors']}")
        print(f"   • Recent Errors (5min): {system_health['recent_errors']}")
        
        # Performance monitoring results
        if 'performance_summary' in system_health:
            perf = system_health['performance_summary']
            print(f"\n⚡ Performance Monitoring:")
            print(f"   • Average Execution Time: {perf.get('avg_execution_time', 'N/A')}")
            print(f"   • Memory Usage Efficiency: {perf.get('avg_memory_usage', 'N/A')}")
            print(f"   • CPU Usage Patterns: {perf.get('avg_cpu_usage', 'N/A')}")
        
        # Gemini AI analysis status
        if error_detector.gemini_analyzer:
            gemini_status = error_detector.get_gemini_status()
            print(f"\n🤖 Gemini AI Analysis Status:")
            print(f"   • Available: {'✅ Yes' if gemini_status.get('enabled', False) else '❌ No'}")
            if gemini_status.get('enabled'):
                print(f"   • Analysis Count: {gemini_status.get('analysis_count', 0)}")
                print(f"   • Success Rate: {gemini_status.get('success_rate', 'N/A')}")
        
        # Show detailed error analysis if any errors occurred
        if error_detector.error_history:
            print(f"\n🔍 Detailed Error Analysis ({len(error_detector.error_history)} errors):")
            show_analysis()
        
        print(f"\n📈 Monitoring Capabilities Demonstrated:")
        print(f"   ✓ Real-time LangChain component interception")  
        print(f"   ✓ LangGraph workflow and state monitoring")
        print(f"   ✓ Streaming event analysis and tracking")
        print(f"   ✓ Error detection with AI-powered analysis")
        print(f"   ✓ Performance and resource monitoring")
        print(f"   ✓ System health and recovery tracking")
        print(f"   ✓ Human interaction and approval monitoring")
        print(f"   ✓ Checkpoint and state persistence monitoring")
        
        # Stop monitoring
        error_detector.stop_monitoring()
        langchain_interceptor.stop_intercepting()
        langgraph_interceptor.stop_intercepting()
        
        print("\n✅ Aigie monitoring stopped")
        
    except Exception as e:
        print(f"\n❌ Failed to initialize or execute: {e}")
        logger.error(f"Main execution failed: {e}")
        raise
    
    print("\n🎉 AI Research Assistant demo completed!")


if __name__ == "__main__":
    # Run the main function
    asyncio.run(main())
