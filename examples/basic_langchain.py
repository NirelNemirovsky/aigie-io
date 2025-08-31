#!/usr/bin/env python3
"""
Basic LangChain example with Aigie monitoring.

This example demonstrates how Aigie automatically detects and reports
errors in LangChain operations without requiring any code changes.
"""

import os
import sys
import time

# Add the parent directory to the path so we can import aigie
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aigie import auto_integrate, show_status, show_analysis


def main():
    """Main example function."""
    print("🚀 Starting Aigie LangChain Example")
    print("=" * 50)
    
    # Enable Aigie monitoring
    print("\n1. Enabling Aigie monitoring...")
    auto_integrate()
    
    # Show initial status
    print("\n2. Initial monitoring status:")
    show_status()
    
    try:
        # Import LangChain components
        print("\n3. Importing LangChain components...")
        from langchain.llms import OpenAI
        from langchain.chains import LLMChain
        from langchain.prompts import PromptTemplate
        
        # Create a simple chain
        print("\n4. Creating LangChain components...")
        llm = OpenAI(temperature=0.7)
        prompt = PromptTemplate.from_template("Tell me a joke about {topic}")
        chain = LLMChain(llm=llm, prompt=prompt)
        
        # Run the chain normally
        print("\n5. Running LangChain chain...")
        result = chain.run("programming")
        print(f"Result: {result}")
        
        # Show status after successful execution
        print("\n6. Status after successful execution:")
        show_status()
        
        # Demonstrate error detection with invalid input
        print("\n7. Demonstrating error detection...")
        try:
            # This will cause an error (invalid prompt template)
            bad_prompt = PromptTemplate.from_template("Invalid template with {missing} {variables}")
            bad_chain = LLMChain(llm=llm, prompt=bad_prompt)
            bad_chain.run("test")
        except Exception as e:
            print(f"Expected error caught: {e}")
        
        # Show status after error
        print("\n8. Status after error detection:")
        show_status()
        
        # Show detailed analysis
        print("\n9. Detailed analysis:")
        show_analysis()
        
    except ImportError as e:
        print(f"❌ LangChain not available: {e}")
        print("Please install LangChain: pip install langchain openai")
        return 1
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return 1
    
    print("\n✅ Example completed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
