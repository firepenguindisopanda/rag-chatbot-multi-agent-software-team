#!/usr/bin/env python3
"""
Test script for the enhanced data analysis system.
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_imports():
    """Test that all imports work correctly."""
    print("Testing imports...")
    
    try:
        from chat_with_data.data_agent_schemas import DataAgentRole, DataAnalysisRequest
        from chat_with_data.data_analysis_agents import create_data_analysis_agents
        from chat_with_data.langgraph_data_orchestrator import LangGraphDataAnalysisOrchestrator
        from chat_with_data.langgraph_chat_agent import LangGraphDataChatAgent
        print("✅ All imports successful!")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def create_test_data():
    """Create a small test dataset."""
    print("Creating test dataset...")
    
    # Create simple test data
    rng = np.random.default_rng(42)
    n = 100
    
    data = {
        'id': range(1, n + 1),
        'age': rng.normal(30, 10, n).astype(int),
        'income': rng.normal(50000, 20000, n),
        'score': rng.uniform(0, 100, n),
        'category': rng.choice(['A', 'B', 'C'], n),
        'active': rng.choice([True, False], n)
    }
    
    df = pd.DataFrame(data)
    df['age'] = df['age'].clip(18, 65)
    df['income'] = df['income'].clip(20000, 100000)
    
    # Save test data
    test_file = "test_data.csv"
    df.to_csv(test_file, index=False)
    
    print(f"✅ Test data created: {test_file}")
    print(f"   Shape: {df.shape}")
    return test_file

def test_data_analysis_agents():
    """Test the specialized data analysis agents."""
    print("\nTesting data analysis agents...")
    
    try:
        from chat_with_data.data_analysis_agents import create_data_analysis_agents, DataAnalysisAgent
        from chat_with_data.data_agent_schemas import DataAgentRole
        
        # Mock LLM for testing
        class MockLLM:
            def invoke(self, messages):
                class MockResponse:
                    content = "This is a test analysis response with recommendations: 1. Test recommendation 2. Another recommendation"
                return MockResponse()
        
        # Create agents
        mock_llm = MockLLM()
        agents = create_data_analysis_agents(mock_llm)
        
        # Test that all agents are created
        expected_roles = list(DataAgentRole)
        assert len(agents) == len(expected_roles), f"Expected {len(expected_roles)} agents, got {len(agents)}"
        
        # Test one agent
        profiler = agents[DataAgentRole.DATA_PROFILER]
        assert isinstance(profiler, DataAnalysisAgent)
        
        # Test agent processing
        test_summary = {
            'shape': (100, 5),
            'columns': ['id', 'age', 'income', 'score', 'category'],
            'data_types': {'id': 'int64', 'age': 'int64'}
        }
        
        response = profiler.process("Test context", test_summary)
        assert response.role == DataAgentRole.DATA_PROFILER
        assert len(response.content) > 0
        
        print("✅ Data analysis agents test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Data analysis agents test failed: {e}")
        return False

def test_orchestrator():
    """Test the data analysis orchestrator."""
    print("\nTesting orchestrator...")
    
    try:
        from chat_with_data.langgraph_data_orchestrator import LangGraphDataAnalysisOrchestrator
        from chat_with_data.data_agent_schemas import DataAnalysisRequest
        
        # Mock LLM and embedder
        class MockLLM:
            def invoke(self, messages):
                class MockResponse:
                    content = "Test analysis complete with insights and recommendations."
                return MockResponse()
        
        class MockEmbedder:
            pass
        
        # Create orchestrator
        mock_llm = MockLLM()
        mock_embedder = MockEmbedder()
        
        orchestrator = LangGraphDataAnalysisOrchestrator(mock_llm, mock_embedder)
        assert orchestrator is not None
        
        print("✅ Orchestrator creation test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Orchestrator test failed: {e}")
        return False

def test_chat_agent():
    """Test the chat agent."""
    print("\nTesting chat agent...")
    
    try:
        from chat_with_data.langgraph_chat_agent import LangGraphDataChatAgent
        from chat_with_data.data_processor import DataProcessor
        
        # Mock LLM and embedder
        class MockLLM:
            def invoke(self, messages):
                class MockResponse:
                    content = "This is a test response to your question about the data."
                return MockResponse()
        
        class MockEmbedder:
            pass
        
        # Create data processor and chat agent
        data_processor = DataProcessor()
        mock_llm = MockLLM()
        mock_embedder = MockEmbedder()
        
        chat_agent = LangGraphDataChatAgent(mock_llm, mock_embedder, data_processor)
        assert chat_agent is not None
        
        print("✅ Chat agent creation test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Chat agent test failed: {e}")
        return False

def test_system_integration():
    """Test basic system integration."""
    print("\nTesting system integration...")
    
    try:
        # Test the interface functions
        from chat_with_data import (
            create_data_analysis_orchestrator,
            create_data_chat_agent,
            DataProcessor
        )
        
        # Mock components
        class MockLLM:
            def invoke(self, messages):
                class MockResponse:
                    content = "Integration test response"
                return MockResponse()
        
        class MockEmbedder:
            pass
        
        mock_llm = MockLLM()
        mock_embedder = MockEmbedder()
        
        # Test orchestrator creation
        orchestrator = create_data_analysis_orchestrator(mock_llm, mock_embedder)
        assert orchestrator is not None
        
        # Test chat agent creation
        data_processor = DataProcessor()
        chat_agent = create_data_chat_agent(mock_llm, mock_embedder, data_processor)
        assert chat_agent is not None
        
        print("✅ System integration test passed!")
        return True
        
    except Exception as e:
        print(f"❌ System integration test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Running Enhanced Data Analysis System Tests")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_data_analysis_agents,
        test_orchestrator,
        test_chat_agent,
        test_system_integration
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The enhanced data analysis system is working correctly.")
        
        # Create test data
        test_file = create_test_data()
        
        print(f"\n🎯 Ready for use! Try the demo:")
        print(f"   python demo_enhanced_data_assistant.py")
        print(f"\n📁 Test data file created: {test_file}")
        print(f"   You can use this for testing the system.")
        
        # Cleanup
        if os.path.exists(test_file):
            os.remove(test_file)
            print(f"   Cleaned up test file: {test_file}")
    else:
        print("❌ Some tests failed. Please check the error messages above.")

if __name__ == "__main__":
    main()
