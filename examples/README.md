# Aigie Examples

This directory contains real-world examples demonstrating Aigie's capabilities with various AI agent frameworks and use cases.

## Open Deep Research Integration

The `open_deep_research/` subdirectory contains a complete integration of Aigie with the [Open Deep Research](https://github.com/langchain-ai/open_deep_research) agent from LangChain AI.

### What's Included

- **Original Open Deep Research Agent**: Full LangGraph-based research agent
- **Aigie Integration Examples**: Multiple approaches to integrating Aigie monitoring
- **Working Configuration**: Google AI (Gemini) + Tavily search setup
- **Comprehensive Documentation**: Integration guides and examples

### Key Files

- `aigie_monitored_researcher.py` - Main Aigie-integrated research agent
- `test_agent.py` - Basic Open Deep Research agent test
- `working_aigie_demo.py` - Standalone Aigie demonstration
- `AIGIE_INTEGRATION_README.md` - Detailed integration guide
- `INTEGRATION_SUMMARY.md` - Project completion summary

### Quick Start

1. **Navigate to the example**:
   ```bash
   cd examples/open_deep_research
   ```

2. **Set up environment**:
   ```bash
   # Copy environment template
   cp .env.example .env
   
   # Edit .env with your API keys
   # GOOGLE_API_KEY=your_google_api_key
   # TAVILY_API_KEY=your_tavily_api_key
   ```

3. **Install dependencies**:
   ```bash
   # Create virtual environment
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   
   # Install dependencies
   pip install -e .
   ```

4. **Run examples**:
   ```bash
   # Test basic Open Deep Research agent
   python test_agent.py
   
   # Test Aigie-monitored research agent
   python aigie_monitored_researcher.py
   ```

### Features Demonstrated

- **Real-time Performance Monitoring**: Track execution time, memory usage, CPU usage
- **Error Detection & Recovery**: Automatic error detection and intelligent retry
- **Comprehensive Logging**: Structured logging with Aigie's logging system
- **Quality Assurance**: Runtime validation and monitoring
- **Multi-Agent Support**: Monitoring complex multi-agent workflows

### Integration Approaches

1. **Wrapper Pattern**: `aigie_monitored_researcher.py` - Wraps the original agent
2. **Direct Integration**: `aigie_integrated_researcher.py` - Direct modification
3. **Simple Demo**: `working_aigie_demo.py` - Standalone demonstration

### API Keys Required

- **Google API Key**: For Gemini models (research, compression, final report, summarization)
- **Tavily API Key**: For web search functionality

### Benefits of Aigie Integration

- **Production Monitoring**: Real-time performance and error tracking
- **Quality Assurance**: Automatic validation and error detection
- **Debugging Support**: Comprehensive logging and error context
- **Scalability**: Monitor multiple agents and complex workflows
- **Reliability**: Intelligent retry and error recovery

## Other Examples

Additional examples will be added here demonstrating Aigie's integration with:
- LangChain agents
- LangGraph workflows
- Custom AI applications
- Multi-agent systems
- Production deployments

## Contributing

To add new examples:

1. Create a new subdirectory for your example
2. Include a README.md explaining the integration
3. Provide working code and configuration
4. Document API key requirements
5. Include usage instructions

## Support

For questions about these examples or Aigie integration:
- Check the main Aigie documentation
- Review the integration README files
- Open an issue in the main Aigie repository