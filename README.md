# Aigie - AI Agent Runtime Error Detection & Remediation

Aigie is a real-time error detection and monitoring system for LangChain and LangGraph applications with **intelligent error remediation capabilities**. It provides seamless integration without requiring additional code from users, automatically detecting, analyzing, and fixing runtime errors as they occur.

## 🎯 Mission

Enable developers to build more reliable AI agents by providing real-time visibility into runtime errors, performance issues, and state problems in LangChain and LangGraph applications, with the ability to automatically remediate errors using AI-powered analysis.

## 🚀 Key Features

- **Zero-Code Integration**: Automatically detects and wraps LangChain/LangGraph applications
- **Real-time Error Detection**: Immediate error reporting with classification and severity assessment
- **🤖 Gemini-Powered Error Analysis**: AI-powered error classification and intelligent remediation
- **🔄 Intelligent Retry System**: Automatic retry with enhanced context from Gemini
- **💉 Prompt Injection Remediation**: Actually fixes errors by injecting guidance into AI agent prompts
- **Comprehensive Monitoring**: Covers execution, API, state, and memory errors
- **Performance Insights**: Track execution time, memory usage, and resource consumption
- **Pattern Learning**: Learns from successful and failed operations to improve over time

## 📦 Installation

```bash
pip install aigie
```

## 🤖 Gemini Integration

Aigie supports two ways to use Gemini:

### 1. Vertex AI (Recommended for production)
```bash
export GOOGLE_CLOUD_PROJECT=your-project-id
gcloud auth application-default login
gcloud services enable aiplatform.googleapis.com
```

### 2. Gemini API Key (Best for local/dev)
```bash
export GEMINI_API_KEY=your-gemini-api-key
# Get from https://aistudio.google.com/app/apikey
```

### Install Gemini dependencies
```bash
pip install google-cloud-aiplatform vertexai google-generativeai
```

## 🔧 Quick Start

### Basic Usage (Zero Code Changes)

```python
# Just import Aigie - it automatically starts monitoring
from aigie import auto_integrate

# Your existing LangChain code works unchanged
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence

# Aigie automatically intercepts and monitors
prompt = PromptTemplate.from_template("Tell me a joke about {topic}")
chain = prompt | llm

# Run normally - Aigie monitors in background
result = chain.invoke({"topic": "programming"})
```

### LangGraph Integration

```python
# Your existing LangGraph code works unchanged
from langgraph.graph import StateGraph, END

# Aigie automatically monitors state transitions and node execution
graph = StateGraph(StateType)
# ... your graph setup ...
app = graph.compile()

# Run normally - Aigie monitors in background
result = app.invoke({"input": "Hello"})
```

### Intelligent Retry with Enhanced Context

```python
from aigie.core.intelligent_retry import intelligent_retry

@intelligent_retry(max_retries=3)
def my_function(input_data):
    # If this fails, Aigie will:
    # 1. Analyze the error with Gemini
    # 2. Generate enhanced retry context
    # 3. Automatically retry with better parameters
    return process_data(input_data)
```

### CLI Usage

```bash
# Enable monitoring
aigie enable --config development

# Show status
aigie status

# Show detailed analysis
aigie analysis

# Gemini Integration
aigie gemini --setup your-project-id
aigie gemini --status
aigie gemini --test
```

## 🏗️ Project Structure

```
aigie/
├── core/                    # Core error detection engine
│   ├── error_detector.py   # Main error detection with real-time remediation
│   ├── error_types.py      # Error classification and severity
│   ├── monitoring.py       # Performance and resource monitoring
│   ├── gemini_analyzer.py  # 🤖 Gemini-powered error analysis
│   └── intelligent_retry.py # 🔄 Intelligent retry with prompt injection
├── interceptors/           # Framework-specific interceptors
│   ├── langchain.py        # LangChain interceptor
│   └── langgraph.py        # LangGraph interceptor
├── reporting/              # Error reporting and logging
├── utils/                  # Utility functions
├── cli.py                  # Command-line interface
└── auto_integration.py     # Automatic integration system
```

## 🔍 Error Types Detected

- **Execution Errors**: Runtime exceptions, timeouts, infinite loops
- **API Errors**: External service failures, rate limits, authentication issues
- **State Errors**: Invalid state transitions, data corruption, type mismatches
- **Memory Errors**: Overflow, corruption, persistence failures
- **Performance Issues**: Slow execution, resource exhaustion, memory leaks
- **Framework-specific**: LangChain chain/tool/agent errors, LangGraph node/state errors

## 📊 Monitoring Capabilities

- **Real-time Error Logging**: Immediate error reporting with classification
- **Performance Metrics**: Execution time, memory usage, API call latency
- **State Tracking**: Monitor agent state changes and transitions
- **Resource Monitoring**: CPU, memory, and disk usage with health indicators
- **AI-Powered Analysis**: Intelligent error classification and remediation strategies
- **Pattern Learning**: Learns from successful and failed operations

## 🛠️ Development

### Setup Development Environment

```bash
git clone <repository>
cd aigie
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

### Run Tests

```bash
pytest tests/ -v
```

### Run Examples

```bash
# Set up Gemini (one-time)
export GOOGLE_CLOUD_PROJECT=your-project-id

# Run comprehensive example
python examples/ai_research_assistant.py
```

## 📝 Configuration

### Environment Variables

```bash
export AIGIE_LOG_LEVEL=INFO
export AIGIE_ENABLE_METRICS=true
export AIGIE_ERROR_THRESHOLD=5
export AIGIE_ENABLE_ALERTS=true
```

### Configuration Files

```bash
# Generate configuration
aigie config --generate config.yml

# Use configuration
aigie enable --config config.yml
```

## 🎯 Current Status

✅ **Fully Implemented and Working**:
- Core error detection engine with Gemini integration
- Real-time error remediation with prompt injection
- LangChain and LangGraph interceptors
- Intelligent retry system with pattern learning
- CLI interface with Gemini setup
- Working examples with real AI integration

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details

## 🆘 Support

- GitHub Issues: [Report bugs and feature requests](https://github.com/your-org/aigie/issues)
- Documentation: [Full API reference](https://aigie.readthedocs.io)
