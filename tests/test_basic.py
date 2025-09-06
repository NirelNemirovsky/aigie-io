#!/usr/bin/env python3
"""
DEPRECATED: This file has been moved to tests/unit/test_core_components.py

Please use the new test structure:
- Unit tests: tests/unit/
- Integration tests: tests/integration/
- Functional tests: tests/functional/
- E2E tests: tests/e2e/

Run tests with: python tests/run_tests.py
"""

import sys
import os

# Add the parent directory to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("⚠️  DEPRECATED: This test file has been moved to tests/unit/test_core_components.py")
print("   Please use the new test structure:")
print("   - Unit tests: tests/unit/")
print("   - Integration tests: tests/integration/")
print("   - Functional tests: tests/functional/")
print("   - E2E tests: tests/e2e/")
print()
print("   Run tests with: python tests/run_tests.py")
print()
print("   Redirecting to new test structure...")

# Import and run the new tests
if __name__ == "__main__":
    import unittest
    from tests.unit.test_core_components import *
    
    # Run the new tests
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules['tests.unit.test_core_components'])
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)