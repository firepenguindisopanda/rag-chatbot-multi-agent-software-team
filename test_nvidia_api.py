#!/usr/bin/env python3
"""
Quick test to check NVIDIA API connectivity
"""

import os
import sys

# Load environment variables (e.g., from .env) instead of hardcoding secrets
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

API_KEY = os.getenv("NVIDIA_API_KEY")
if not API_KEY:
    # If executed via pytest, skip; otherwise warn
    try:
        import pytest  # type: ignore
        pytest.skip("NVIDIA_API_KEY not set; skipping live NVIDIA API connectivity test", allow_module_level=True)
    except Exception:
        print("⚠ NVIDIA_API_KEY not set; skipping live call. Set it in your environment or .env file.")
        # Exit early to avoid network errors
        sys.exit(0)

try:
    from langchain_nvidia_ai_endpoints import ChatNVIDIA
    print("✅ Import successful")
    
    llm = ChatNVIDIA(model="meta/llama-3.1-8b-instruct")
    print("✅ LLM initialized")
    
    # Test with a simple message
    response = llm.invoke("Hello, this is a test message. Please respond with 'API working correctly'.")
    print(f"✅ Response received: {response.content[:200]}")
    
    # Check if it has bind_tools method
    if hasattr(llm, 'bind_tools'):
        print("✅ LLM has bind_tools method (LangGraph compatible)")
    else:
        print("❌ LLM missing bind_tools method")
        
except Exception as e:
    print(f"❌ Error: {e}")
    print("The API key might be invalid or there's a network issue.")
