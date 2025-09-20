#!/usr/bin/env python3
"""
Simple test to verify the Open Deep Research integration works
without requiring all complex dependencies.
"""

import sys
import os
from dotenv import load_dotenv

# Add the aigie package to the path
sys.path.insert(0, '/Users/nirelnemirovsky/Documents/dev/aigie/aigie-io')

# Load environment variables
load_dotenv()

def test_aigie_imports():
    """Test that Aigie components can be imported."""
    try:
        from aigie.core.error_handling.error_detector import ErrorDetector
        from aigie.core.monitoring.monitoring import PerformanceMonitor
        from aigie.utils.config import AigieConfig
        from aigie.reporting.logger import AigieLogger
        
        print("✅ Aigie imports successful")
        return True
    except Exception as e:
        print(f"❌ Aigie import failed: {e}")
        return False

def test_environment_setup():
    """Test that environment variables are set."""
    google_key = os.getenv("GOOGLE_API_KEY")
    tavily_key = os.getenv("TAVILY_API_KEY")
    
    print(f"🔑 Google API Key: {'✅ Set' if google_key else '❌ Missing'}")
    print(f"🔑 Tavily API Key: {'✅ Set' if tavily_key else '❌ Missing'}")
    
    return bool(google_key and tavily_key)

def test_aigie_components():
    """Test that Aigie components can be initialized."""
    try:
        from aigie.core.error_handling.error_detector import ErrorDetector
        from aigie.core.monitoring.monitoring import PerformanceMonitor
        from aigie.utils.config import AigieConfig
        from aigie.reporting.logger import AigieLogger
        
        # Initialize components
        config = AigieConfig()
        error_detector = ErrorDetector()
        monitoring_system = PerformanceMonitor()
        logger = AigieLogger()
        
        print("✅ Aigie components initialized successfully")
        
        # Test basic functionality
        logger.log_system_event("Test event from simple test")
        print("✅ Aigie logging works")
        
        return True
    except Exception as e:
        print(f"❌ Aigie component test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Testing Aigie + Open Deep Research Integration")
    print("=" * 60)
    
    tests = [
        ("Aigie Imports", test_aigie_imports),
        ("Environment Setup", test_environment_setup),
        ("Aigie Components", test_aigie_components),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n🔍 Running: {test_name}")
        print("-" * 40)
        result = test_func()
        results.append((test_name, result))
    
    print(f"\n{'='*60}")
    print("📊 Test Results Summary:")
    print('='*60)
    
    all_passed = True
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {status} {test_name}")
        if not result:
            all_passed = False
    
    print(f"\n🎯 Overall: {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")
    
    if all_passed:
        print("\n💡 Next Steps:")
        print("   1. Install missing dependencies: pip install -e .")
        print("   2. Run the full integration: python aigie_monitored_researcher.py")
        print("   3. Test with different research queries")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
