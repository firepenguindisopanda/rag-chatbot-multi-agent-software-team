#!/usr/bin/env python3
"""
Test script for the new RAG evaluation feature
"""

import os
import sys
import tempfile
import logging

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_evaluation_feature():
    """Test the evaluation functionality"""
    try:
        print("🧪 Testing RAG Evaluation Feature")
        print("=" * 50)
        
        # Test imports
        print("✅ Testing imports...")
        from rag_pdf_server import get_demo
        print("✅ Successfully imported get_demo")
        
        # Test demo creation
        print("✅ Testing demo creation...")
        demo = get_demo()
        print("✅ Successfully created Gradio demo")
        
        # Check if evaluation tab exists
        print("✅ Checking if evaluation tab was added...")
        
        # Create the demo to check its contents
        try:
            from gradio.blocks import Blocks
            
            # Check for evaluation components by looking at the demo structure
            # The demo object should contain our evaluation tab
            has_eval_tab = False
            has_eval_description = False
            
            # Check the source code of the demo function for our evaluation content
            import inspect
            source = inspect.getsource(get_demo)
            
            if "📊 Evaluate RAG" in source:
                has_eval_tab = True
                print("✅ Evaluation tab found in demo source")
            else:
                print("❌ Evaluation tab not found in source")
                
            if "LLM-as-a-Judge" in source:
                has_eval_description = True
                print("✅ Evaluation description found in source")
            else:
                print("❌ Evaluation description not found in source")
            
            if not (has_eval_tab and has_eval_description):
                return False
                
        except Exception as e:
            print(f"⚠️ Could not check demo structure: {e}")
            # Just check if the source contains our evaluation code
            import inspect
            source = inspect.getsource(get_demo)
            if "📊 Evaluate RAG" in source and "run_rag_evaluation" in source:
                print("✅ Evaluation functionality found in source code")
            else:
                print("❌ Evaluation functionality not found")
                return False
        
        print("\n🎉 All tests passed!")
        print("📊 The evaluation feature has been successfully added to your RAG PDF server!")
        print("\nNext steps:")
        print("1. Run the server: python rag_pdf_server.py")
        print("2. Open http://localhost:8000/gradio")
        print("3. Upload a PDF document")
        print("4. Click on the '📊 Evaluate RAG' tab")
        print("5. Click 'Start Evaluation' to test your RAG system")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_evaluation_feature()
    if success:
        print("\n✅ Test completed successfully!")
    else:
        print("\n❌ Test failed!")
        sys.exit(1)
