#!/usr/bin/env python3
"""
Unified test runner for the Aigie test suite.
"""

import asyncio
import sys
import os
import argparse
import time
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def run_unit_tests():
    """Run unit tests."""
    print("🧪 Running Unit Tests...")
    print("=" * 50)
    
    import unittest
    
    # Discover and run unit tests
    loader = unittest.TestLoader()
    start_dir = Path(__file__).parent / "unit"
    suite = loader.discover(start_dir, pattern="test_*.py")
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


async def run_integration_tests():
    """Run integration tests."""
    print("\n🔗 Running Integration Tests...")
    print("=" * 50)
    
    try:
        from tests.integration.test_auto_integration import main as integration_main
        return await integration_main()
    except Exception as e:
        print(f"❌ Integration tests failed: {e}")
        return False


async def run_functional_tests():
    """Run functional tests."""
    print("\n⚙️ Running Functional Tests...")
    print("=" * 50)
    
    try:
        from tests.functional.test_validation_system import main as functional_main
        return await functional_main()
    except Exception as e:
        print(f"❌ Functional tests failed: {e}")
        return False


async def run_e2e_tests():
    """Run end-to-end tests."""
    print("\n🌐 Running End-to-End Tests...")
    print("=" * 50)
    
    try:
        from tests.e2e.test_complete_system import main as e2e_main
        return await e2e_main()
    except Exception as e:
        print(f"❌ E2E tests failed: {e}")
        return False


async def main():
    """Main test runner."""
    parser = argparse.ArgumentParser(description="Run Aigie test suite")
    parser.add_argument(
        "--type", 
        choices=["unit", "integration", "functional", "e2e", "all"],
        default="all",
        help="Type of tests to run"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    
    args = parser.parse_args()
    
    print("🚀 AIGIE TEST SUITE")
    print("=" * 60)
    print(f"Running {args.type} tests...")
    print()
    
    start_time = time.time()
    results = {}
    
    if args.type in ["unit", "all"]:
        results["unit"] = run_unit_tests()
    
    if args.type in ["integration", "all"]:
        results["integration"] = await run_integration_tests()
    
    if args.type in ["functional", "all"]:
        results["functional"] = await run_functional_tests()
    
    if args.type in ["e2e", "all"]:
        results["e2e"] = await run_e2e_tests()
    
    total_time = time.time() - start_time
    
    # Summary
    print(f"\n{'='*60}")
    print("TEST SUITE SUMMARY")
    print('='*60)
    
    passed = 0
    total = 0
    
    for test_type, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_type.capitalize():<15} Tests")
        if success:
            passed += 1
        total += 1
    
    print(f"\nOverall: {passed}/{total} test suites passed ({passed/total*100:.1f}%)")
    print(f"Total time: {total_time:.2f}s")
    
    if passed == total:
        print("\n🎉 ALL TEST SUITES PASSED!")
    else:
        print(f"\n⚠️  {total-passed} test suites failed.")
    
    return passed == total


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
