# Aigie Integration with Open Deep Research

This directory contains the integration of Aigie (AI Runtime Validation System) with the Open Deep Research agent, demonstrating how to enhance AI agents with comprehensive validation, error handling, and monitoring capabilities.

## 🚀 Quick Start

### Prerequisites

1. **Python 3.11+** - Required for the Open Deep Research agent
2. **Aigie Package** - Located in `/Users/nirelnemirovsky/Documents/dev/aigie/aigie-io`
3. **API Keys** - Gemini API key for Aigie's validation features

### Installation

1. **Set up the environment:**
   ```bash
   cd /Users/nirelnemirovsky/Documents/dev/aigie/aigie-io/open_deep_research
   python3.11 -m venv .venv
   source .venv/bin/activate
   pip install -e .
   ```

2. **Configure environment variables:**
   ```bash
   # Create .env file
   echo "GOOGLE_API_KEY=your_gemini_api_key_here" > .env
   echo "OPENAI_API_KEY=your_openai_api_key_here" >> .env
   ```

3. **Install Aigie dependencies:**
   ```bash
   # Aigie is already available in the parent directory
   # The integration scripts will automatically import it
   ```

## 📁 Files Overview

### Core Integration Files

- **`aigie_integrated_researcher.py`** - Main integration file with Aigie-enhanced research functions
- **`aigie_research_example.py`** - Comprehensive example demonstrating all Aigie features
- **`test_aigie_integration.py`** - Test script for the integration

### Original Files (for reference)

- **`deep_researcher.py`** - Original Open Deep Research implementation
- **`configuration.py`** - Configuration management
- **`test_agent.py`** - Basic test script

## 🔧 Usage Examples

### 1. Basic Aigie Integration

```python
from aigie_integrated_researcher import aigie_deep_researcher, get_monitoring_system

# Configure the agent
config = {
    "configurable": {
        "research_model": "openai:gpt-4o-mini",
        "search_api": "none",
        "max_researcher_iterations": 2,
        "allow_clarification": False
    }
}

# Run research with Aigie monitoring
input_data = {
    "messages": [
        {"role": "user", "content": "What are the benefits of runtime validation in AI systems?"}
    ]
}

result = await aigie_deep_researcher.ainvoke(input_data, config)

# Get monitoring data
monitoring = get_monitoring_system()
print(f"Steps monitored: {len(monitoring.step_history)}")
```

### 2. Comprehensive Research Agent

```python
from aigie_research_example import AigieResearchAgent

# Initialize the enhanced agent
agent = AigieResearchAgent()

# Execute research with full monitoring
result = await agent.execute_research_with_monitoring(
    "How does error handling improve agent reliability?",
    {"max_iterations": 3, "validation_strict_mode": True}
)

# Check results
if result["success"]:
    print("Research completed successfully!")
    print(f"Report: {result['final_report']}")
else:
    print(f"Research failed: {result['error']}")
```

### 3. Running the Demo

```bash
# Run the comprehensive example
python aigie_research_example.py

# Run the integration test
python test_aigie_integration.py
```

## 🛡️ Aigie Features Demonstrated

### 1. Runtime Validation
- **Model Configuration Validation** - Ensures API keys and model settings are correct
- **Prompt Validation** - Validates prompts before sending to models
- **Response Validation** - Validates structured outputs and content
- **Content Validation** - Ensures generated content meets quality standards

### 2. Error Handling
- **Error Detection** - Automatically detects and categorizes errors
- **Context Extraction** - Extracts relevant context for error analysis
- **Intelligent Recovery** - Attempts to recover from failures gracefully
- **Error Logging** - Comprehensive error logging with context

### 3. Monitoring System
- **Step-by-Step Monitoring** - Tracks each step of the research process
- **Performance Metrics** - Measures success rates and execution times
- **Real-time Logging** - Provides real-time feedback on agent execution
- **Session Management** - Manages complete research sessions

### 4. Gemini Integration
- **Query Quality Analysis** - Uses Gemini to analyze query quality
- **Content Analysis** - Validates content using AI-powered analysis
- **Advanced Validation** - Leverages Gemini's capabilities for complex validation

## 📊 Monitoring and Metrics

The Aigie integration provides comprehensive monitoring data:

```python
# Get monitoring summary
monitoring = get_monitoring_system()
summary = monitoring.get_summary()

print(f"Total Steps: {summary['total_steps']}")
print(f"Success Rate: {summary['success_rate']:.1%}")
print(f"Average Execution Time: {summary['avg_execution_time']:.2f}s")
print(f"Error Count: {summary['error_count']}")
print(f"Recovery Attempts: {summary['recovery_attempts']}")
```

## 🔍 Validation Features

### Query Validation
- Checks query length and complexity
- Validates research-worthiness
- Provides suggestions for improvement
- Uses Gemini for advanced analysis

### Model Configuration Validation
- Verifies API keys are present
- Validates model parameters
- Checks token limits and constraints
- Ensures compatibility

### Content Validation
- Validates generated reports
- Checks for proper structure
- Ensures content quality
- Detects potential issues

## 🚨 Error Handling

### Automatic Error Detection
- Detects API errors
- Identifies validation failures
- Catches unexpected exceptions
- Monitors performance issues

### Intelligent Recovery
- Attempts graceful recovery
- Uses fallback strategies
- Provides alternative approaches
- Maintains session continuity

### Error Context Extraction
- Extracts relevant state information
- Captures error details
- Provides debugging information
- Enables better error analysis

## 🎯 Benefits of Aigie Integration

1. **Reliability** - Enhanced error handling and recovery
2. **Quality** - Comprehensive validation at every step
3. **Monitoring** - Real-time visibility into agent execution
4. **Debugging** - Detailed error context and logging
5. **Performance** - Metrics and optimization insights
6. **Maintainability** - Better error tracking and resolution

## 🔧 Configuration Options

### Aigie Configuration
```python
# Enable strict validation
config["aigie_validation_strict_mode"] = True

# Set validation timeout
config["aigie_validation_timeout"] = 30

# Enable recovery attempts
config["aigie_enable_recovery"] = True

# Set monitoring level
config["aigie_monitoring_level"] = "detailed"
```

### Research Agent Configuration
```python
config = {
    "configurable": {
        "research_model": "openai:gpt-4o-mini",
        "compression_model": "openai:gpt-4o-mini",
        "final_report_model": "openai:gpt-4o-mini",
        "search_api": "tavily",  # or "none" for testing
        "max_researcher_iterations": 3,
        "max_react_tool_calls": 5,
        "allow_clarification": True
    }
}
```

## 🧪 Testing

### Run Integration Tests
```bash
# Test basic integration
python test_aigie_integration.py

# Test comprehensive features
python aigie_research_example.py

# Test original agent (for comparison)
python test_agent.py
```

### Test Coverage
- ✅ Basic agent functionality
- ✅ Aigie validation features
- ✅ Error handling and recovery
- ✅ Monitoring and logging
- ✅ Gemini integration
- ✅ Configuration validation

## 📈 Performance Monitoring

The integration provides detailed performance metrics:

- **Execution Time** - Time for each step and overall process
- **Success Rate** - Percentage of successful operations
- **Error Rate** - Frequency and types of errors
- **Recovery Rate** - Success rate of recovery attempts
- **Resource Usage** - API calls and token usage

## 🔮 Future Enhancements

1. **Advanced Analytics** - More sophisticated performance analysis
2. **Predictive Error Detection** - Anticipate potential issues
3. **Automated Optimization** - Self-tuning based on performance data
4. **Enhanced Recovery** - More sophisticated recovery strategies
5. **Real-time Dashboards** - Live monitoring interfaces

## 📚 Additional Resources

- [Aigie Documentation](../README.md)
- [Open Deep Research Documentation](https://github.com/langchain-ai/open_deep_research)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [LangChain Documentation](https://python.langchain.com/)

## 🤝 Contributing

To contribute to the Aigie integration:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📄 License

This integration follows the same license as the original Open Deep Research project (MIT License).

---

**Note**: This integration demonstrates how Aigie can enhance any AI agent with comprehensive validation, error handling, and monitoring capabilities. The patterns shown here can be applied to other agent frameworks and use cases.

