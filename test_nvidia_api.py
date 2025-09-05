#!/usr/bin/env python3
"""
Quick test to check NVIDIA API connectivity
"""

import os
import sys

# Set the API key
os.environ["NVIDIA_API_KEY"] = "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"

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
