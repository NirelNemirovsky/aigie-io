# AI Research Assistant Example

A comprehensive demonstration of aigie's error detection and monitoring capabilities with a real-world AI research assistant using multiple tools and LangGraph orchestration.

## 🚀 Quick Start

```bash
cd examples
python3 ai_research_assistant.py
```

## 📋 What This Example Demonstrates

### Core Capabilities
- **Multi-Tool Agent**: Web search, document analysis, and code generation
- **LangGraph Workflow**: Complex orchestration with state management
- **Error Simulation**: Intentional failures to test aigie's detection
- **Real-Time Monitoring**: Performance metrics and tool statistics

### Aigie Integration
- **Error Detection**: Automatic classification and severity assessment
- **Performance Monitoring**: Execution time, memory usage, CPU tracking
- **Framework Interceptors**: LangChain and LangGraph automatic patching
- **Intelligent Retry**: Context-aware retry with enhanced prompts

## 🔧 Requirements

- Python 3.8+
- Google Cloud project with Vertex AI enabled (set `GOOGLE_CLOUD_PROJECT`)
- Or Gemini API key (set `GEMINI_API_KEY`)
- Internet connection for web search functionality

## 🏗️ Architecture

The example implements a research workflow with three main tools:

1. **WebSearchTool**: Searches for research papers and information
2. **DocumentAnalysisTool**: Analyzes documents and extracts insights
3. **CodeGenerationTool**: Generates code snippets for data analysis

The workflow is orchestrated using LangGraph with state management and error handling.

## 🎯 Key Features

- **Real-time Error Detection**: Catches and classifies various error types
- **Performance Monitoring**: Tracks execution metrics across all tools
- **Intelligent Retry**: Automatically retries failed operations with context
- **Pattern Learning**: Learns from successful and failed operations
- **Prompt Injection**: Injects error context into AI agent prompts for remediation

## 📊 Expected Behavior

The example will:
1. Initialize aigie monitoring and interceptors
2. Create a LangGraph workflow with research tools
3. Execute the workflow with simulated errors
4. Demonstrate aigie's error detection and handling
5. Show tool performance statistics and error summaries

## 🚨 Troubleshooting

| Issue | Solution |
|-------|----------|
| Gemini not available | Set `GOOGLE_CLOUD_PROJECT` or `GEMINI_API_KEY` |
| Import errors | Install requirements: `pip install -r requirements.txt` |
| LangGraph errors | Update to latest version: `pip install -U langgraph` |
| Permission errors | Check Google Cloud authentication: `gcloud auth login` |

## 🔍 Customization

You can modify the example to:
- Change error simulation rates in the `Config` class
- Add new research tools by extending `ResearchTool`
- Modify the LangGraph workflow structure
- Adjust monitoring and retry parameters

## 📚 Related Documentation

- [Aigie Core Documentation](../README.md)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [LangChain Documentation](https://python.langchain.com/)
