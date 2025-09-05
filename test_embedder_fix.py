#!/usr/bin/env python3
"""
Test script to verify the MockEmbedder fix.
"""
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_mock_embedder():
    """Test the MockEmbedder implementation."""
    print("🧪 Testing MockEmbedder fix...")
    
    # Create MockEmbedder (same as in rag_pdf_server.py)
    class MockEmbedder:
        def __init__(self):
            # Set required properties for LangChain Embeddings compatibility
            self.model = "mock-embedder"
            self.dimension = 768
        
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
        def aembed_query(self, text):
            return self.embed_query(text)
        
        def aembed_documents(self, texts):
            return self.embed_documents(texts)
        
        # Add client property for compatibility with some vectorstores
        @property
        def client(self):
            return None
    
    # Test the embedder
    embedder = MockEmbedder()
    
    print("Testing embedder properties...")
    print(f"✓ Callable: {callable(embedder)}")
    print(f"✓ Has embed_query: {hasattr(embedder, 'embed_query')}")
    print(f"✓ Has embed_documents: {hasattr(embedder, 'embed_documents')}")
    print(f"✓ Has __call__: {hasattr(embedder, '__call__')}")
    
    # Test embedding generation
    test_embedding = embedder.embed_query("test text")
    print(f"✓ Test embedding length: {len(test_embedding)}")
    print(f"✓ Test embedding type: {type(test_embedding)}")
    
    # Test callable interface
    callable_embedding = embedder("test text")
    print(f"✓ Callable embedding length: {len(callable_embedding)}")
    
    # Test documents embedding
    docs_embeddings = embedder.embed_documents(["text1", "text2"])
    print(f"✓ Documents embeddings count: {len(docs_embeddings)}")
    
    return embedder

def test_faiss_integration(embedder):
    """Test FAISS integration with MockEmbedder."""
    print("\n🔗 Testing FAISS integration...")
    
    try:
        from langchain_community.vectorstores import FAISS
        from langchain_core.documents import Document
        
        # Create test documents
        documents = [
            Document(page_content="This is test document 1", metadata={"id": 1}),
            Document(page_content="This is test document 2", metadata={"id": 2}),
            Document(page_content="Another test document about data analysis", metadata={"id": 3})
        ]
        
        print(f"✓ Created {len(documents)} test documents")
        
        # Try to create FAISS vectorstore
        print("Creating FAISS vectorstore...")
        vectorstore = FAISS.from_documents(documents, embedder)
        print("✅ FAISS vectorstore created successfully!")
        
        # Test search
        print("Testing similarity search...")
        results = vectorstore.similarity_search("test document", k=2)
        print(f"✓ Search returned {len(results)} results")
        
        for i, doc in enumerate(results):
            print(f"  Result {i+1}: {doc.page_content[:50]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ FAISS integration failed: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return False

def test_vectorstore_manager():
    """Test the DataVectorStoreManager with MockEmbedder."""
    print("\n📊 Testing DataVectorStoreManager...")
    
    try:
        from chat_with_data.vectorstore_manager import DataVectorStoreManager
        import pandas as pd
        
        # Create MockEmbedder
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

def main():
    """Run all tests."""
    print("🔧 Testing MockEmbedder Fix")
    print("=" * 50)
    
    # Test 1: MockEmbedder functionality
    embedder = test_mock_embedder()
    if embedder is None:
        print("❌ MockEmbedder test failed, stopping")
        return
    
    # Test 2: FAISS integration
    faiss_success = test_faiss_integration(embedder)
    if not faiss_success:
        print("❌ FAISS integration failed, stopping")
        return
    
    # Test 3: VectorStore Manager
    vm_success = test_vectorstore_manager()
    
    if vm_success:
        print("\n🎉 All tests passed! The MockEmbedder fix should resolve the vectorstore errors.")
    else:
        print("\n⚠️ Some tests failed. The issue may require additional fixes.")

if __name__ == "__main__":
    main()
