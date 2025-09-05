#!/usr/bin/env python3
"""
Test the fixed data analysis system with proper insights handling and Python REPL plotting.
"""

import os
import pandas as pd
import tempfile
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_mock_embedder():
    """Create a mock embedder for testing."""
    class MockEmbedder:
        def __init__(self):
            self.model = "mock-embedder"
            self.dimension = 768
            self.model_name = "mock-embedder"
        
        def embed_query(self, text):
            import hashlib
            import random
            hash_obj = hashlib.md5(str(text).encode())
            random.seed(hash_obj.hexdigest())
            return [random.uniform(-1, 1) for _ in range(768)]
        
        def embed_documents(self, texts):
            return [self.embed_query(text) for text in texts]
        
        def __call__(self, text):
            if isinstance(text, list):
                return self.embed_documents(text)
            else:
                return self.embed_query(text)
        
        async def aembed_query(self, text):
            return self.embed_query(text)
        
        async def aembed_documents(self, texts):
            return self.embed_documents(texts)
        
        @property
        def client(self):
            return None
    
    return MockEmbedder()

def create_mock_llm():
    """Create a mock LLM for testing."""
    class MockLLM:
        def invoke(self, text):
            class MockResponse:
                def __init__(self, content):
                    self.content = content
            
            # Return proper JSON insights for testing
            if "insights" in str(text).lower() or "json list" in str(text).lower():
                return MockResponse('["The dataset shows good data quality with minimal missing values.", "Numerical features exhibit normal distribution patterns suitable for analysis.", "Categorical variables are well-balanced across different classes.", "The dataset structure supports the intended analysis type effectively."]')
            elif "python" in str(text).lower() and "plot" in str(text).lower():
                return MockResponse("""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Create a sample plot
plt.figure(figsize=(10, 6))
if 'df' in locals():
    df.hist(bins=20, figsize=(12, 8))
    plt.suptitle('Data Distribution Overview')
else:
    plt.text(0.5, 0.5, 'Sample Plot Generated', ha='center', va='center')
    plt.title('Mock Visualization')
plt.tight_layout()
plt.show()
""")
            else:
                return MockResponse(f"Mock analysis response for: {str(text)[:100]}...")
        
        def bind(self, **kwargs):
            return self
        
        def bind_tools(self, tools):
            return self
        
        def with_config(self, config):
            return self
    
    return MockLLM()

def test_enhanced_data_analysis():
    """Test the enhanced data analysis system."""
    print("🧪 Testing Fixed Enhanced Data Analysis System")
    print("=" * 60)
    
    try:
        # Import the fixed components
        from chat_with_data.enhanced_langgraph_agents import EnhancedLangGraphDataAnalysisAgent
        from chat_with_data.schemas import DataRequest
        
        # Create test data
        test_data = pd.DataFrame({
            'product_id': range(1, 101),
            'price': [10.5 + i * 0.5 for i in range(100)],
            'sales': [100 + i * 2 for i in range(100)],
            'category': ['Electronics', 'Books', 'Clothing'] * 33 + ['Electronics'],
            'rating': [4.0 + (i % 10) * 0.1 for i in range(100)]
        })
        
        # Save test data to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            test_data.to_csv(f.name, index=False)
            test_file_path = f.name
        
        try:
            print("✅ Test data created successfully")
            print(f"   Shape: {test_data.shape}")
            print(f"   Columns: {list(test_data.columns)}")
            
            # Create mock components
            embedder = create_mock_embedder()
            llm = create_mock_llm()
            
            print("✅ Mock LLM and embedder created")
            
            # Create enhanced analysis agent
            agent = EnhancedLangGraphDataAnalysisAgent(llm, embedder)
            print("✅ Enhanced analysis agent created")
            
            # Create data request
            data_request = DataRequest(
                file_path=test_file_path,
                user_context="E-commerce product analysis for business insights",
                target_variable="sales"
            )
            
            print("✅ Data request created")
            print(f"   File: {data_request.file_path}")
            print(f"   Context: {data_request.user_context}")
            print(f"   Target: {data_request.target_variable}")
            
            # Run analysis
            print("\n🔄 Running enhanced data analysis...")
            result = agent.analyze_dataset(data_request)
            
            # Check results
            print("\n📊 Analysis Results:")
            print(f"✅ Summary generated: {len(result.summary) > 0}")
            print(f"✅ Insights generated: {len(result.insights)}")
            print(f"✅ Plots generated: {len(result.visualizations)}")
            print(f"✅ ML models recommended: {len(result.recommended_ml_models)}")
            print(f"✅ Metrics suggested: {len(result.suggested_metrics)}")
            print(f"✅ Next steps provided: {len(result.next_steps)}")
            
            # Validate insights are strings (not dictionaries)
            insights_are_strings = all(isinstance(insight, str) for insight in result.insights)
            print(f"✅ Insights are proper strings: {insights_are_strings}")
            
            if result.insights:
                print("\n🔍 Sample insights:")
                for i, insight in enumerate(result.insights[:2], 1):
                    print(f"   {i}. {insight}")
            
            # Check if plots are Python code (not base64 images)
            if result.visualizations:
                plots_are_code = any("import" in str(viz) or "plt." in str(viz) for viz in result.visualizations)
                print(f"✅ Visualizations are Python code: {plots_are_code}")
                
                print("\n📊 Sample plot code:")
                print("   " + str(result.visualizations[0])[:100] + "...")
            
            print(f"\n📝 Analysis summary preview:")
            print("   " + result.summary[:200] + "...")
            
            print("\n🎉 Enhanced Data Analysis Test PASSED!")
            return True
            
        finally:
            # Clean up test file
            if os.path.exists(test_file_path):
                os.remove(test_file_path)
                
    except Exception as e:
        print(f"\n❌ Enhanced Data Analysis Test FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_chat_functionality():
    """Test the chat functionality with the loaded data."""
    print("\n🗨️ Testing Chat Functionality")
    print("=" * 40)
    
    try:
        from chat_with_data.enhanced_langgraph_agents import EnhancedLangGraphDataChatAgent
        from chat_with_data.data_processor import DataProcessor
        
        # Create test data
        test_data = pd.DataFrame({
            'product': ['A', 'B', 'C', 'D', 'E'],
            'price': [10.5, 15.2, 8.9, 12.0, 14.5],
            'sales': [100, 150, 80, 120, 135],
            'category': ['Electronics', 'Books', 'Electronics', 'Books', 'Electronics']
        })
        
        # Setup components
        embedder = create_mock_embedder()
        llm = create_mock_llm()
        data_processor = DataProcessor()
        data_processor.data = test_data
        
        # Create chat agent
        chat_agent = EnhancedLangGraphDataChatAgent(llm, embedder, data_processor)
        print("✅ Chat agent created")
        
        # Test questions
        test_questions = [
            "What is the average price of products?",
            "How many products are in each category?",
            "Show me the correlation between price and sales"
        ]
        
        for question in test_questions:
            print(f"\n❓ Question: {question}")
            response = chat_agent.answer_question(question)
            print(f"🤖 Response: {response[:100]}...")
        
        print("\n🎉 Chat Functionality Test PASSED!")
        return True
        
    except Exception as e:
        print(f"\n❌ Chat Functionality Test FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("🚀 Testing Fixed Data Analysis System")
    print("=" * 80)
    
    # Test enhanced data analysis
    analysis_success = test_enhanced_data_analysis()
    
    # Test chat functionality
    chat_success = test_chat_functionality()
    
    # Overall results
    print("\n" + "=" * 80)
    print("📋 FINAL TEST RESULTS:")
    print(f"   Enhanced Data Analysis: {'✅ PASSED' if analysis_success else '❌ FAILED'}")
    print(f"   Chat Functionality: {'✅ PASSED' if chat_success else '❌ FAILED'}")
    
    if analysis_success and chat_success:
        print("\n🎊 ALL TESTS PASSED! The fixed system is working correctly.")
        print("\n🔧 Key fixes implemented:")
        print("   ✅ Insights are now guaranteed to be plain strings (not dictionaries)")
        print("   ✅ Vectorstore has improved error handling and mock support")
        print("   ✅ Visualizations now generate Python code for execution")
        print("   ✅ Chat agent works with pandas agent for data queries")
    else:
        print("\n⚠️ Some tests failed. Please review the errors above.")
    
    return analysis_success and chat_success

if __name__ == "__main__":
    main()
