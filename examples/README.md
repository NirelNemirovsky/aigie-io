# AI Research Assistant Examples

Comprehensive demonstrations of aigie's error detection and monitoring capabilities with modern LangChain and LangGraph patterns.

## 🚀 Quick Start

```bash
cd examples

# 🚀 Start Here
python3 ai_research_assistant.py          # Comprehensive research assistant with full monitoring demo

# 🌟 Advanced Features  
python3 advanced_langgraph_features.py    # Human-in-the-loop • SQLite checkpointing • Error recovery
```

## 📋 What These Examples Demonstrate

### Modern Framework Integration
- **LangChain v0.2/0.3 Patterns**: Using `init_chat_model()`, `@tool`, LCEL chains
- **LangGraph ReAct Agents**: Pre-built agents with `create_react_agent`  
- **Streaming & Events**: Real-time monitoring with `stream_events`/`astream_events`
- **Checkpointing**: Persistent state management with MemorySaver/SqliteSaver
- **Human-in-the-Loop**: Interactive workflows with approval checkpoints

### Complete Aigie Monitoring (All Shown In Context)
- **LangChain Interception**: ChatModels, LCEL Runnables, @tool functions, output parsers
- **LangGraph Monitoring**: State graphs, streaming events, checkpoints, human interactions
- **Real-Time Analytics**: Event stream analysis, checkpoint operations, performance metrics
- **Error Intelligence**: AI-powered detection, classification, and automatic remediation
- **System Health**: Resource monitoring, performance tracking, recovery statistics
- **Production Insights**: All monitoring capabilities demonstrated during realistic workflows

## 🔧 Requirements

- Python 3.9+
- **Modern Dependencies**: Latest LangChain, LangGraph, and LangChain community packages
- **API Access**: OpenAI, Anthropic, or Google AI for model providers
- **Gemini API Key (Recommended)**: Get from [Google AI Studio](https://makersuite.google.com/app/apikey)
  ```bash
  export GEMINI_API_KEY="your-api-key-here"
  export OPENAI_API_KEY="your-openai-key"  # For ChatGPT models
  export ANTHROPIC_API_KEY="your-anthropic-key"  # For Claude models
  ```

## 🏗️ Modern Architecture

### Updated Research Assistant (`ai_research_assistant.py`)
Uses modern LangGraph patterns with:
- **ReAct Agent**: Built with `create_react_agent` from `langgraph.prebuilt`
- **Function-based Tools**: Simple `@tool` decorated functions instead of classes
- **Modern Chat Models**: Using `init_chat_model()` with string model identifiers
- **Event Streaming**: Real-time monitoring with `astream_events()`
- **Checkpointing**: Persistent state with `MemorySaver`

### Research Tools (Modern Implementation)
1. **web_search_tool**: Simple function for finding research information
2. **document_analysis_tool**: Analyzes documents and extracts insights  
3. **code_generation_tool**: Generates code snippets for data analysis

### Human-in-the-Loop Features
- **Approval Checkpoints**: User can approve/reject agent actions
- **Interactive Corrections**: Modify agent plans before execution
- **Error Intervention**: Human oversight when errors are detected

## 🎯 Enhanced Features

- **Real-time Stream Monitoring**: Every event in the agent execution is tracked
- **Modern Component Detection**: Monitors ChatModels, LCEL chains, modern tools
- **Checkpoint Monitoring**: Tracks state persistence operations
- **Event-Driven Analytics**: Detailed analysis of agent execution patterns
- **AI-Powered Remediation**: Gemini analyzes errors and suggests fixes
- **Streaming Analytics**: Real-time performance metrics during execution

## 📊 What You'll See

The updated examples will:
1. Initialize modern LangChain/LangGraph components with aigie monitoring
2. Create ReAct agents with real-time event streaming
3. Execute workflows with comprehensive error detection
4. Show streaming event analysis and checkpoint monitoring
5. Demonstrate human-in-the-loop intervention capabilities
6. Display modern monitoring dashboards and analytics

## 📚 Example Details

### 🚀 ai_research_assistant.py
**Comprehensive Research Assistant & Monitoring Demo**
- **Complete workflow**: Web search, document analysis, code generation with realistic complexity
- **Comprehensive monitoring demo**: Shows ALL interceptor capabilities in realistic context
- **Error simulation**: Extensive testing of aigie's detection and recovery capabilities
- **Real-world patterns**: 900+ lines demonstrating production-ready workflows
- **Full analytics**: LangChain/LangGraph monitoring, streaming analysis, error analytics
- **Interactive monitoring**: See interceptor capabilities during actual agent execution

**Perfect for:** Understanding complete workflows AND seeing all monitoring capabilities in action

### 🌟 advanced_langgraph_features.py
**Advanced Production Features**
- **Human-in-the-loop**: Approval workflows with interrupt() and checkpoints
- **Advanced persistence**: SQLite checkpointing with thread management
- **Error recovery**: Conditional routing and intelligent retry patterns
- **Multi-agent coordination**: Complex workflow orchestration
- **Advanced state management**: Proper TypedDict schemas and validation
- **Production patterns**: Command objects and dynamic flow control

**Perfect for:** Production deployments, advanced LangGraph features, enterprise workflows

## 🚨 Troubleshooting

| Issue | Solution |
|-------|----------|
| Gemini not available | Set `GOOGLE_CLOUD_PROJECT` or `GEMINI_API_KEY` |
| Import errors | Install requirements: `pip install -r requirements.txt` |
| LangGraph errors | Update to latest version: `pip install -U langgraph` |
| Permission errors | Check Google Cloud authentication: `gcloud auth login` |

## 🔧 Gemini API Key Setup

For a simple demonstration of setting up Aigie with Gemini API key authentication:

```bash
python3 gemini_api_key_setup.py
```

This example shows:
- How to configure Gemini API key authentication
- Testing error analysis capabilities
- Generating remediation strategies
- Configuration options and best practices

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
