# Aigie

[![PyPI version](https://badge.fury.io/py/aigie.svg)](https://badge.fury.io/py/aigie)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Tests](https://img.shields.io/badge/tests-pytest-blue.svg)](https://pytest.org/)

> **AI Agent Runtime Error Detection & Remediation**

Aigie is a real-time error detection and monitoring system for LangChain and LangGraph applications with **intelligent error remediation capabilities** and a revolutionary **LLM-as-Judge validation system**. It provides seamless integration without requiring additional code from users, automatically detecting, analyzing, validating, and fixing runtime errors as they occur.

## ✨ Features

- **🚀 Zero-Code Integration** - Automatically detects and wraps LangChain/LangGraph applications
- **⚡ Real-time Error Detection** - Immediate error reporting with classification and severity assessment
- **🧠 LLM-as-Judge Validation** - Revolutionary AI-powered step validation using 6 validation strategies
- **🤖 Gemini-Powered Analysis** - AI-powered error classification and intelligent remediation
- **🔄 Intelligent Retry System** - Automatic retry with enhanced context from Gemini
- **💉 Prompt Injection Remediation** - Actually fixes errors by injecting guidance into AI agent prompts
- **🔧 Auto-Correction System** - Automatically fixes invalid steps using multiple correction strategies
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

## 🧠 LLM-as-Judge Validation System

Aigie's revolutionary **LLM-as-Judge** validation system provides real-time validation and correction of AI agent execution steps. This system ensures agents execute correctly and efficiently by continuously monitoring, validating, and automatically correcting their behavior.

### 🎯 Validation Strategies

The system uses **6 comprehensive validation strategies** to judge each execution step:

| Strategy | Description |
|----------|-------------|
| **Goal Alignment** | Does this step advance the agent's stated goal? |
| **Logical Consistency** | Is the step logically sound given the context? |
| **Output Quality** | Will this likely produce appropriate output? |
| **State Coherence** | Does this maintain consistent agent state? |
| **Safety Compliance** | Does this follow safety guidelines? |
| **Performance Optimality** | Is this the most efficient approach? |

### 🔧 Auto-Correction System

When validation fails, Aigie automatically attempts correction using multiple strategies:

- **Parameter Adjustment** - Fix incorrect parameters
- **Prompt Refinement** - Improve prompts and input data
- **Tool Substitution** - Replace wrong tools with correct ones
- **Logic Repair** - Fix logical errors in reasoning
- **Goal Realignment** - Align steps with agent goals
- **State Restoration** - Fix corrupted agent state

### 📊 Advanced Features

- **Parallel Validation** - Multiple strategies run simultaneously for faster processing
- **Pattern Learning** - Learns from validation history to improve future validations
- **Intelligent Caching** - Caches validation results for similar steps
- **Adaptive Thresholds** - Dynamically adjusts validation criteria based on performance
- **Rich Context Capture** - Captures comprehensive execution context for intelligent validation

### Example Usage

```python
from aigie import auto_integrate
from aigie.core.validation import ValidationEngine

# Auto-integration enables LLM-as-Judge validation automatically
auto_integrate()

# Your existing code works unchanged - validation happens automatically
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence

prompt = PromptTemplate.from_template("Tell me a joke about {topic}")
chain = prompt | llm

# Each step is automatically validated and corrected if needed
result = chain.invoke({"topic": "programming"})
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
│   │   ├── error_types.py      # Error classification and severity
│   │   └── validation_types.py # Validation data structures
│   ├── validation/         # 🧠 LLM-as-Judge validation system
│   │   ├── runtime_validator.py     # LLM-as-Judge implementation
│   │   ├── step_corrector.py        # Auto-correction system
│   │   ├── validation_engine.py     # Main orchestrator
│   │   ├── validation_pipeline.py   # Multi-stage validation
│   │   ├── validation_monitor.py    # Performance monitoring
│   │   └── context_extractor.py     # Context inference
│   ├── error_handling/     # Error detection and handling
│   │   ├── error_detector.py        # Main error detection engine
│   │   └── intelligent_retry.py     # Smart retry system
│   ├── monitoring/         # Performance and resource monitoring
│   │   └── monitoring.py           # Resource monitoring
│   ├── ai/                 # AI/LLM components
│   │   └── gemini_analyzer.py      # Gemini-powered analysis
│   └── utils/              # Utility functions
├── interceptors/           # Framework-specific interceptors
│   ├── langchain.py        # LangChain interceptor
│   ├── langgraph.py        # LangGraph interceptor
│   └── validation_interceptor.py # Enhanced interceptor with validation
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
git clone https://github.com/NirelNemirovsky/aigie-io.git
cd aigie-io

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
- **🧠 LLM-as-Judge Validation System** - Revolutionary AI-powered step validation with 6 validation strategies
- **🔧 Auto-Correction System** - Automatic step correction using multiple correction strategies
- **⚡ Core Error Detection Engine** - Real-time error detection with Gemini integration
- **💉 Prompt Injection Remediation** - Real-time error remediation with prompt injection
- **🔗 LangChain and LangGraph Interceptors** - Seamless framework integration
- **🔄 Intelligent Retry System** - Smart retry with pattern learning
- **📊 Performance Monitoring** - Comprehensive metrics and reporting
- **🛠️ CLI Interface** - Complete command-line interface with Gemini setup
- **📚 Working Examples** - Real AI integration examples and demos

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
- **Issues**: [Report bugs and feature requests](https://github.com/NirelNemirovsky/aigie-io/issues)
- **Discussions**: [Community discussions](https://github.com/NirelNemirovsky/aigie-io/discussions)

## 🙏 Acknowledgments

- [LangChain](https://github.com/langchain-ai/langchain) for the amazing AI framework
- [LangGraph](https://github.com/langchain-ai/langgraph) for the powerful graph-based AI workflows
- [Google Gemini](https://ai.google.dev/) for the AI analysis capabilities

---

<div align="center">
  <strong>Built with ❤️ for the AI community</strong>
</div>