# Aigie - AI Agent Runtime Error Detection

Aigie is a real-time error detection and monitoring system for LangChain and LangGraph applications. It provides seamless integration without requiring additional code from users, automatically detecting and reporting runtime errors as they occur.

## 🎯 Mission

Enable developers to build more reliable AI agents by providing real-time visibility into runtime errors, performance issues, and state problems in LangChain and LangGraph applications.

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   LangChain    │    │    LangGraph    │    │     Aigie      │
│   Application  │    │   Application   │    │   Core Engine  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Aigie Wrapper Layer                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │
│  │LangChain    │  │LangGraph    │  │Error        │            │
│  │Interceptor │  │Interceptor  │  │Detection    │            │
│  └─────────────┘  └─────────────┘  └─────────────┘            │
└─────────────────────────────────────────────────────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Error Processing Pipeline                    │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │
│  │Error        │  │Error        │  │Real-time    │            │
│  │Capture      │  │Analysis     │  │Reporting    │            │
│  └─────────────┘  └─────────────┘  └─────────────┘            │
└─────────────────────────────────────────────────────────────────┘
```

## 🚀 Features

- **Zero-Code Integration**: Automatically detects and wraps LangChain/LangGraph applications
- **Real-time Error Detection**: Immediate error reporting as they occur
- **Comprehensive Monitoring**: Covers execution, API, state, and memory errors
- **Performance Insights**: Track execution time, memory usage, and resource consumption
- **Seamless Operation**: No changes required to existing code

## 📦 Installation

```bash
pip install aigie
```

## 🔧 Quick Start

### Basic Usage (Zero Code Changes)

```python
# Your existing LangChain code works unchanged
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# Aigie automatically intercepts and monitors
llm = OpenAI()
prompt = PromptTemplate.from_template("Tell me a joke about {topic}")
chain = LLMChain(llm=llm, prompt=prompt)

# Run normally - Aigie monitors in background
result = chain.run("programming")
```

### LangGraph Integration

```python
# Your existing LangGraph code works unchanged
from langgraph.graph import StateGraph

# Aigie automatically monitors state transitions and node execution
graph = StateGraph()
# ... your graph setup ...
app = graph.compile()

# Run normally - Aigie monitors in background
result = app.invoke({"input": "Hello"})
```

## 🏗️ Project Structure

```
aigie/
├── core/                    # Core error detection engine
│   ├── __init__.py
│   ├── error_detector.py   # Main error detection logic
│   ├── error_types.py      # Error classification
│   └── monitoring.py       # Performance monitoring
├── interceptors/           # Framework-specific interceptors
│   ├── __init__.py
│   ├── langchain.py        # LangChain interceptor
│   └── langgraph.py        # LangGraph interceptor
├── reporting/              # Error reporting and logging
│   ├── __init__.py
│   ├── logger.py           # Real-time logging
│   └── metrics.py          # Performance metrics
├── utils/                  # Utility functions
│   ├── __init__.py
│   ├── decorators.py       # Decorator utilities
│   └── config.py           # Configuration management
└── tests/                  # Test suite
    ├── __init__.py
    ├── test_langchain.py   # LangChain integration tests
    └── test_langgraph.py   # LangGraph integration tests
```

## 🔍 Error Types Detected

1. **Execution Errors**: Runtime exceptions, timeouts, infinite loops
2. **API Errors**: External service failures, rate limits, authentication issues
3. **State Errors**: Invalid state transitions, data corruption, type mismatches
4. **Memory Errors**: Overflow, corruption, persistence failures
5. **Performance Issues**: Slow execution, resource exhaustion, memory leaks

## 📊 Monitoring Capabilities

- **Real-time Error Logging**: Immediate error reporting with stack traces
- **Performance Metrics**: Execution time, memory usage, API call latency
- **State Tracking**: Monitor agent state changes and transitions
- **Resource Monitoring**: CPU, memory, and network usage
- **Custom Alerts**: Configurable error thresholds and notifications

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
pytest tests/
```

### Run Example

```bash
python examples/basic_langchain.py
python examples/basic_langgraph.py
```

## 📝 Configuration

Aigie can be configured through environment variables:

```bash
export AIGIE_LOG_LEVEL=INFO
export AIGIE_ENABLE_METRICS=true
export AIGIE_ERROR_THRESHOLD=5
export AIGIE_ENABLE_ALERTS=true
```

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
- Community: [Join our Discord](https://discord.gg/aigie)
