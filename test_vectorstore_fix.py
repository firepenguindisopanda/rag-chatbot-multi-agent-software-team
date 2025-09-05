"""
Test script to verify the vectorstore fix works properly.
"""

import sys
import os
import pandas as pd
import tempfile
import logging

# Add current directory to path
sys.path.insert(0, '.')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_mock_embedder():
    """Test the improved MockEmbedder."""
    print("🧪 Testing improved MockEmbedder...")
    
    # Import the mock embedder from the main server file
    # We'll create it here to avoid importing the whole server
    class MockEmbedder:
        def __init__(self):
            # Set required properties for LangChain Embeddings compatibility
            self.model = "mock-embedder"
            self.dimension = 768
            # Add model name property for compatibility
            self.model_name = "mock-embedder"
        
        def embed_query(self, text):
            # Return a simple vector of random values for testing
            import hashlib
            import random
            # Use hash to make embeddings consistent for same text
            hash_obj = hashlib.md5(str(text).encode())
            random.seed(hash_obj.hexdigest())
            return [random.uniform(-1, 1) for _ in range(768)]
        
        def embed_documents(self, texts):
            return [self.embed_query(text) for text in texts]
        
        def __call__(self, text):
            # Make the embedder callable - this is what FAISS tries to use
            if isinstance(text, list):
                return self.embed_documents(text)
            else:
                return self.embed_query(text)
        
        # Add async methods for compatibility
        async def aembed_query(self, text):
            return self.embed_query(text)
        
        async def aembed_documents(self, texts):
            return self.embed_documents(texts)
        
        # Add client property for compatibility with some vectorstores
        @property
        def client(self):
            return None
        
        def embed_documents_async(self, texts):
            return self.embed_documents(texts)
        
        # Additional methods for full compatibility
        def _embed_documents(self, texts):
            return self.embed_documents(texts)
        
        def _embed_query(self, text):
            return self.embed_query(text)
        
        # For compatibility with different versions of LangChain
        def embed(self, text):
            if isinstance(text, list):
                return self.embed_documents(text)
            else:
                return self.embed_query(text)
        
        # Make sure the object is properly callable
        def __getattr__(self, name):
            # If someone tries to access a method we don't have, 
            # return a function that logs and returns appropriate values
            if name.startswith('embed'):
                def fallback_method(*args, **kwargs):
                    if len(args) > 0:
                        if isinstance(args[0], list):
                            return self.embed_documents(args[0])
                        else:
                            return self.embed_query(args[0])
                    return []
                return fallback_method
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
    
    # Test the embedder
    embedder = MockEmbedder()
    
    print("Testing embedder properties...")
    print(f"✓ Callable: {callable(embedder)}")
    print(f"✓ Has embed_query: {hasattr(embedder, 'embed_query')}")
    print(f"✓ Has embed_documents: {hasattr(embedder, 'embed_documents')}")
    print(f"✓ Has __call__: {hasattr(embedder, '__call__')}")
    
    # Test functionality
    query_result = embedder.embed_query("test query")
    print(f"✓ embed_query result length: {len(query_result)}")
    
    docs_result = embedder.embed_documents(["doc1", "doc2"])
    print(f"✓ embed_documents result: {len(docs_result)} docs, each with {len(docs_result[0])} dimensions")
    
    call_result = embedder("test call")
    print(f"✓ __call__ result length: {len(call_result)}")
    
    call_docs_result = embedder(["doc1", "doc2"])
    print(f"✓ __call__ with docs: {len(call_docs_result)} docs")
    
    return embedder

def test_vectorstore_manager():
    """Test the DataVectorStoreManager with improved MockEmbedder."""
    print("\n📊 Testing DataVectorStoreManager...")
    
    try:
        from chat_with_data.vectorstore_manager import DataVectorStoreManager
        
        # Create improved MockEmbedder
        embedder = test_mock_embedder()
        
        # Create test data
        test_data = pd.DataFrame({
            'product': ['A', 'B', 'C'],
            'price': [10.5, 15.2, 8.9],
            'sales': [100, 150, 80]
        })
        
        file_info = {
            "filename": "test_data.csv",
            "user_context": "Test data for product analysis",
            "target_variable": "sales"
        }
        
        # Create vectorstore manager
        vm = DataVectorStoreManager(embedder)
        
        # Test adding data to vectorstore
        print("Adding data to vectorstore...")
        result = vm.add_data_to_vectorstore(test_data, file_info)
        
        if result["status"] == "success":
            print("✅ Data added to vectorstore successfully!")
            print(f"✓ Documents created: {result['documents_count']}")
            
            # Test search
            print("Testing vectorstore search...")
            search_results = vm.search_data_context("product analysis", k=2)
            print(f"✓ Search returned {len(search_results)} results")
            
            return True
        else:
            print(f"❌ Failed to add data: {result}")
            return False
            
    except Exception as e:
        print(f"❌ VectorStore Manager test failed: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return False

def test_enhanced_data_analysis_agent():
    """Test the EnhancedLangGraphDataAnalysisAgent."""
    print("\n🤖 Testing EnhancedLangGraphDataAnalysisAgent...")
    
    try:
        from chat_with_data.enhanced_langgraph_agents import EnhancedLangGraphDataAnalysisAgent
        from chat_with_data.schemas import DataRequest
        
        # Create improved MockEmbedder
        embedder = test_mock_embedder()
        
        # Create mock LLM
        class MockLLM:
            def invoke(self, text):
                class MockResponse:
                    content = f"Mock analysis for: {text}"
                return MockResponse()
            
            def bind(self, **kwargs):
                return self
            
            def bind_tools(self, tools):
                return self
            
            def with_config(self, config):
                return self
        
        llm = MockLLM()
        
        # Create test CSV file
        test_data = pd.DataFrame({
            'product': ['A', 'B', 'C', 'D', 'E'],
            'price': [10.5, 15.2, 8.9, 12.0, 14.5],
            'sales': [100, 150, 80, 120, 135],
            'category': ['Electronics', 'Books', 'Electronics', 'Books', 'Electronics']
        })
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            test_data.to_csv(f.name, index=False)
            test_file_path = f.name
        
        try:
            # Create analysis agent
            agent = EnhancedLangGraphDataAnalysisAgent(llm, embedder)
            
            # Create data request
            data_request = DataRequest(
                file_path=test_file_path,
                user_context="Test product analysis",
                target_variable="sales"
            )
            
            # Run analysis
            print("Running data analysis...")
            result = agent.analyze_dataset(data_request)
            
            print("✅ Data analysis completed!")
            print(f"✓ Analysis summary length: {len(result.summary)}")
            print(f"✓ Insights generated: {len(result.insights)}")
            print(f"✓ Visualizations created: {len(result.visualizations)}")
            
            return True
            
        finally:
            # Clean up test file
            if os.path.exists(test_file_path):
                os.remove(test_file_path)
            
    except Exception as e:
        print(f"❌ Enhanced Data Analysis Agent test failed: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return False

def main():
    """Run all tests."""
    print("🔧 Testing Vectorstore Fix")
    print("=" * 50)
    
    # Test 1: MockEmbedder functionality
    embedder = test_mock_embedder()
    if not embedder:
        print("❌ MockEmbedder test failed, stopping")
        return
    
    # Test 2: VectorStore Manager
    vectorstore_success = test_vectorstore_manager()
    
    # Test 3: Enhanced Data Analysis Agent
    agent_success = test_enhanced_data_analysis_agent()
    
    # Summary
    print("\n" + "=" * 50)
    if vectorstore_success and agent_success:
        print("🎉 All tests passed! The vectorstore fix should resolve the errors.")
    else:
        print("❌ Some tests failed. Please check the error messages above.")

if __name__ == "__main__":
    main()
