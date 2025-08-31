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
- **Real-time Error Detection**: Immediate error reporting with classification and severity assessment
- **Comprehensive Monitoring**: Covers execution, API, state, and memory errors
- **Performance Insights**: Track execution time, memory usage, and resource consumption
- **Rich Console Output**: Beautiful, informative displays with emojis and structured information
- **CLI Interface**: Full command-line tool for monitoring and configuration
- **Seamless Operation**: No changes required to existing code

## 📦 Installation

```bash
pip install aigie
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
chain = prompt | llm  # Modern RunnableSequence syntax

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

### Manual Monitoring with Decorators

```python
from aigie.utils.decorators import monitor_langchain, monitor_langgraph

@monitor_langchain(component="CustomChain", method="run")
def my_custom_chain(input_data):
    # Function automatically monitored
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

# Generate configuration
aigie config --generate config.yml
```

## 🏗️ Project Structure

```
aigie/
├── core/                    # Core error detection engine
│   ├── __init__.py
│   ├── error_detector.py   # Main error detection logic
│   ├── error_types.py      # Error classification and severity
│   └── monitoring.py       # Performance and resource monitoring
├── interceptors/           # Framework-specific interceptors
│   ├── __init__.py
│   ├── langchain.py        # LangChain interceptor (patches LLMChain, Agent, Tool, LLM)
│   └── langgraph.py        # LangGraph interceptor (patches StateGraph, CompiledStateGraph)
├── reporting/              # Error reporting and logging
│   ├── __init__.py
│   ├── logger.py           # Real-time logging with Rich console output
│   └── metrics.py          # Performance metrics collection
├── utils/                  # Utility functions
│   ├── __init__.py
│   ├── decorators.py       # Monitoring decorators and context managers
│   └── config.py           # Configuration management with presets
├── cli.py                  # Command-line interface
├── auto_integration.py     # Automatic integration system
├── examples/               # Working examples
│   ├── basic_langchain.py  # LangChain integration example
│   └── basic_langgraph.py  # LangGraph integration example
└── tests/                  # Test suite
    └── test_basic.py       # Core functionality tests
```

## 🔍 Error Types Detected

1. **Execution Errors**: Runtime exceptions, timeouts, infinite loops
2. **API Errors**: External service failures, rate limits, authentication issues
3. **State Errors**: Invalid state transitions, data corruption, type mismatches
4. **Memory Errors**: Overflow, corruption, persistence failures
5. **Performance Issues**: Slow execution, resource exhaustion, memory leaks
6. **Framework-specific**: LangChain chain/tool/agent errors, LangGraph node/state errors

## 📊 Monitoring Capabilities

- **Real-time Error Logging**: Immediate error reporting with classification and severity
- **Performance Metrics**: Execution time, memory usage, API call latency
- **State Tracking**: Monitor agent state changes and transitions
- **Resource Monitoring**: CPU, memory, and disk usage with health indicators
- **Rich Console Output**: Beautiful displays with emojis, tables, and structured information
- **Error Suggestions**: AI-powered recommendations for fixing detected issues

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
# Basic examples
python examples/basic_langchain.py
python examples/basic_langgraph.py

# Comprehensive demo
python demo.py
```

### CLI Testing

```bash
# Test CLI commands
aigie --help
aigie version
aigie status
aigie analysis
```

## 📝 Configuration

Aigie can be configured through environment variables or configuration files:

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

### Configuration Presets

- **development**: Verbose logging, detailed error reporting
- **production**: Optimized performance, essential monitoring only
- **testing**: Minimal overhead, focused on validation

## 🎯 Current Status

✅ **Fully Implemented and Working**:
- Core error detection engine
- LangChain and LangGraph interceptors
- Real-time logging with Rich console output
- Performance monitoring and metrics
- CLI interface with all commands
- Working examples for both frameworks
- Comprehensive test suite (20 tests passing)

✅ **Compatible with Current Versions**:
- LangChain 0.3.27+
- LangGraph 0.6.6+
- Uses modern import paths and patterns

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
