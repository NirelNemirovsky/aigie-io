# Aigie Test Suite

This directory contains the organized test suite for the Aigie project.

## Test Structure

```
tests/
├── unit/                    # Unit tests for individual components
│   ├── test_core_components.py
│   └── test_validation_components.py
├── integration/             # Integration tests with real agents
│   └── test_auto_integration.py
├── functional/              # Functional tests with real API calls
│   └── test_validation_system.py
├── e2e/                     # End-to-end system tests
│   └── test_complete_system.py
├── fixtures/                # Test fixtures and data
├── run_tests.py            # Unified test runner
└── README.md               # This file
```

## Test Categories

### Unit Tests (`unit/`)
- **Purpose**: Test individual components in isolation
- **Scope**: Core components, validation types, error handling
- **Dependencies**: Minimal, mostly mocked
- **Speed**: Fast (< 1 second per test)

### Integration Tests (`integration/`)
- **Purpose**: Test Aigie integration with real LangChain/LangGraph agents
- **Scope**: Auto-integration, real agent workflows
- **Dependencies**: Real agent frameworks, optional API keys
- **Speed**: Medium (1-10 seconds per test)

### Functional Tests (`functional/`)
- **Purpose**: Test complete functionality with real API calls
- **Scope**: Validation system, context extraction, LLM integration
- **Dependencies**: Gemini API key required
- **Speed**: Medium (5-30 seconds per test)

### End-to-End Tests (`e2e/`)
- **Purpose**: Test complete system with comprehensive scenarios
- **Scope**: Full system validation, performance, monitoring
- **Dependencies**: Gemini API key, full system setup
- **Speed**: Slow (30+ seconds per test)

## Running Tests

### Run All Tests
```bash
python tests/run_tests.py
```

### Run Specific Test Types
```bash
# Unit tests only
python tests/run_tests.py --type unit

# Integration tests only
python tests/run_tests.py --type integration

# Functional tests only
python tests/run_tests.py --type functional

# End-to-end tests only
python tests/run_tests.py --type e2e
```

### Run Individual Test Files
```bash
# Unit tests
python -m unittest tests.unit.test_core_components
python -m unittest tests.unit.test_validation_components

# Integration tests
python tests/integration/test_auto_integration.py

# Functional tests
python tests/functional/test_validation_system.py

# E2E tests
python tests/e2e/test_complete_system.py
```

## Test Requirements

### Environment Setup
1. Install dependencies: `pip install -r requirements.txt`
2. Set up environment variables in `.env`:
   ```
   GEMINI_API_KEY=your_gemini_api_key_here
   ```

### API Key Requirements
- **Unit Tests**: No API key required (uses mocks)
- **Integration Tests**: Optional API key (fallback mode available)
- **Functional Tests**: Gemini API key required
- **E2E Tests**: Gemini API key required

## Test Data and Fixtures

Test fixtures and sample data are stored in the `fixtures/` directory:
- Mock agent configurations
- Sample execution steps
- Test data for validation scenarios

## Continuous Integration

The test suite is designed to work with CI/CD pipelines:

```yaml
# Example GitHub Actions workflow
- name: Run Unit Tests
  run: python tests/run_tests.py --type unit

- name: Run Integration Tests
  run: python tests/run_tests.py --type integration
  env:
    GEMINI_API_KEY: ${{ secrets.GEMINI_API_KEY }}
```

## Test Coverage

The test suite covers:
- ✅ Core component functionality
- ✅ Error detection and handling
- ✅ Validation system with real LLM calls
- ✅ Context extraction (automatic and LLM-based)
- ✅ Integration with LangChain and LangGraph
- ✅ Performance optimization features
- ✅ Monitoring and metrics collection
- ✅ Real-world agent scenarios

## Troubleshooting

### Common Issues

1. **API Key Not Found**
   - Ensure `GEMINI_API_KEY` is set in your environment
   - Check that the `.env` file is in the project root

2. **Import Errors**
   - Ensure you're running tests from the project root
   - Check that all dependencies are installed

3. **Test Failures**
   - Check the test output for specific error messages
   - Ensure your API key has sufficient quota
   - Verify network connectivity for API calls

### Debug Mode

Run tests with verbose output:
```bash
python tests/run_tests.py --verbose
```

## Contributing

When adding new tests:
1. Place tests in the appropriate category directory
2. Follow the naming convention: `test_*.py`
3. Include proper docstrings and comments
4. Ensure tests are independent and can run in any order
5. Add appropriate error handling and cleanup
