# Aigie

[![PyPI version](https://badge.fury.io/py/aigie.svg)](https://badge.fury.io/py/aigie)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Tests](https://img.shields.io/badge/tests-pytest-blue.svg)](https://pytest.org/)

> **AI Agent Runtime Error Detection & Remediation**

Aigie is a real-time error detection and monitoring system for LangChain and LangGraph applications with **intelligent error remediation capabilities**. It provides seamless integration without requiring additional code from users, automatically detecting, analyzing, and fixing runtime errors as they occur.

## ✨ Features

- **🚀 Zero-Code Integration** - Automatically detects and wraps LangChain/LangGraph applications
- **⚡ Real-time Error Detection** - Immediate error reporting with classification and severity assessment
- **🤖 Gemini-Powered Analysis** - AI-powered error classification and intelligent remediation
- **🔄 Intelligent Retry System** - Automatic retry with enhanced context from Gemini
- **💉 Prompt Injection Remediation** - Actually fixes errors by injecting guidance into AI agent prompts
- **📊 Comprehensive Monitoring** - Covers execution, API, state, and memory errors
- **📈 Performance Insights** - Track execution time, memory usage, and resource consumption
- **🧠 Pattern Learning** - Learns from successful and failed operations to improve over time

## 🚀 Quick Start

### Installation

```bash
pip install aigie
```

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

## 🔧 Advanced Usage

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

## 📋 Error Types Detected

| Category | Description |
|----------|-------------|
| **Execution Errors** | Runtime exceptions, timeouts, infinite loops |
| **API Errors** | External service failures, rate limits, authentication issues |
| **State Errors** | Invalid state transitions, data corruption, type mismatches |
| **Memory Errors** | Overflow, corruption, persistence failures |
| **Performance Issues** | Slow execution, resource exhaustion, memory leaks |
| **Framework-specific** | LangChain chain/tool/agent errors, LangGraph node/state errors |

## 📊 Monitoring Capabilities

- **Real-time Error Logging** - Immediate error reporting with classification
- **Performance Metrics** - Execution time, memory usage, API call latency
- **State Tracking** - Monitor agent state changes and transitions
- **Resource Monitoring** - CPU, memory, and disk usage with health indicators
- **AI-Powered Analysis** - Intelligent error classification and remediation strategies
- **Pattern Learning** - Learns from successful and failed operations

## 🏗️ Project Structure

```
aigie/
├── core/                    # Core functionality
│   ├── types/              # Type definitions and data structures
│   ├── validation/         # Runtime validation system
│   ├── error_handling/     # Error detection and handling
│   ├── monitoring/         # Performance and resource monitoring
│   ├── ai/                 # AI/LLM components
│   └── utils/              # Utility functions
├── interceptors/           # Framework-specific interceptors
├── reporting/              # Error reporting and logging
├── utils/                  # Utility functions
├── cli.py                  # Command-line interface
└── auto_integration.py     # Automatic integration system
```

## ⚙️ Configuration

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

## 🛠️ Development

### Prerequisites

- Python 3.9+
- pip
- git

### Setup Development Environment

```bash
# Clone the repository
git clone https://github.com/your-org/aigie.git
cd aigie

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test categories
pytest tests/unit/ -v
pytest tests/integration/ -v
pytest tests/e2e/ -v
```

### Running Examples

```bash
# Set up Gemini (one-time)
export GOOGLE_CLOUD_PROJECT=your-project-id

# Run comprehensive example
python examples/ai_research_assistant.py
```

### Code Quality

```bash
# Format code
black aigie/ tests/ examples/

# Lint code
flake8 aigie/ tests/ examples/

# Type checking
mypy aigie/
```

## 📈 Current Status

✅ **Fully Implemented and Working**:
- Core error detection engine with Gemini integration
- Real-time error remediation with prompt injection
- LangChain and LangGraph interceptors
- Intelligent retry system with pattern learning
- CLI interface with Gemini setup
- Working examples with real AI integration

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### How to Contribute

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add some amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### Development Guidelines

- Follow the existing code style
- Add tests for new features
- Update documentation as needed
- Ensure all tests pass
- Follow semantic commit messages

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- **Documentation**: [Full API reference](https://aigie.readthedocs.io)
- **Issues**: [Report bugs and feature requests](https://github.com/your-org/aigie/issues)
- **Discussions**: [Community discussions](https://github.com/your-org/aigie/discussions)

## 🙏 Acknowledgments

- [LangChain](https://github.com/langchain-ai/langchain) for the amazing AI framework
- [LangGraph](https://github.com/langchain-ai/langgraph) for the powerful graph-based AI workflows
- [Google Gemini](https://ai.google.dev/) for the AI analysis capabilities

---

<div align="center">
  <strong>Built with ❤️ for the AI community</strong>
</div>