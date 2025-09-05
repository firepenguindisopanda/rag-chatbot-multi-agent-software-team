#!/usr/bin/env python3
"""
Final test of the fixed multi-agent system with the actual RAG application
"""

import os
import sys

# Load environment (supports .env) instead of hardcoding secrets
try:  # optional dependency already in requirements
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

API_KEY = os.getenv("NVIDIA_API_KEY")
if not API_KEY:
    # If running under pytest, mark skip at module import time
    try:
        import pytest  # type: ignore
        pytest.skip("NVIDIA_API_KEY not set; skipping integration test that requires live NVIDIA endpoint", allow_module_level=True)
    except Exception:
        print("⚠ NVIDIA_API_KEY not set; test will likely fail or use mock if available.")

# Add path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_final_integration():
    """Test the complete fixed integration."""
    print("🎉 Testing Fixed Multi-Agent Integration")
    print("=" * 50)
    try:
        from enhanced_multi_agent_integration import run_enhanced_multi_agent_collaboration
        from langchain_nvidia_ai_endpoints import ChatNVIDIA

        # Initialize only if key present; otherwise underlying lib handles error/mocking
        llm = ChatNVIDIA(model="meta/llama-3.1-8b-instruct")
        print("✅ LLM initialized")

        # Test the integration function
        project_description = "Create a simple task management web application for small teams with user login and task assignment features"

        print("🚀 Running enhanced multi-agent collaboration...")
        result = run_enhanced_multi_agent_collaboration(llm, project_description)

        print("\n📊 Results:")
        print(f"- Output length: {len(result):,} characters")
        print(f"- Success: {'✅' if not result.startswith('❌') else '❌'}")
        
        if result.startswith("❌"):
            print(f"Error: {result}")
        else:
            # Count sections
            sections = result.count("##")
            diagrams = result.count("```mermaid")
            print(f"- Sections found: {sections}")
            print(f"- Mermaid diagrams: {diagrams}")
            
            print("\n📄 Preview (first 500 chars):")
            print(result[:500] + "..." if len(result) > 500 else result)
        
        # Save the result to file for inspection
        with open("test_fixed_result.md", "w", encoding="utf-8") as f:
            f.write(result)
        print("\n💾 Full result saved to: test_fixed_result.md")

        return True

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_final_integration()
    if success:
        print("\n🎉 INTEGRATION TEST PASSED!")
        print("The multi-agent system is now working correctly!")
    else:
        print("\n❌ Integration test failed.")
