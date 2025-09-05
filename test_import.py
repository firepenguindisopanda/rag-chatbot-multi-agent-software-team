#!/usr/bin/env python3
"""
Test script to verify all imports work correctly for the RAG chatbot application.
"""

import sys
import traceback

def test_imports():
    """Test all module imports"""
    try:
        print("Testing Python version...")
        print(f"Python version: {sys.version}")
        
        print("\nTesting basic imports...")
        import gradio as gr
        import fastapi
        import pandas as pd
        import numpy as np
        print("Basic imports successful")
        
        print("\nTesting LangChain imports...")
        from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
        from langchain_core.prompts import ChatPromptTemplate
        print("LangChain imports successful")
        
        print("\nTesting custom modules...")
        import multi_agent_software_team
        import chat_with_data
        print("Custom module imports successful")
        
        print("\nTesting specific classes...")
        from chat_with_data import DataAnalysisAgent, DataChatAgent, DataProcessor
        from multi_agent_software_team import SoftwareTeamOrchestrator
        print("Class imports successful")
        
        print("\n🎉 All imports completed successfully!")
        return True
        
    except Exception as e:
        print(f"\nImport error: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)
