#!/usr/bin/env python3
"""
Enhanced Data Analysis Assistant Demo

This script demonstrates the new multi-agent data analysis system that provides
comprehensive analysis of CSV/Excel files using specialized AI agents.

Features:
- Multi-agent analysis workflow (6 specialized agents)
- LangGraph orchestration for complex workflows  
- Intelligent data chat with RAG and pandas integration
- Comprehensive visualizations and insights
- Professional reporting and documentation

Usage:
    python demo_enhanced_data_assistant.py
"""

import os
import sys
import pandas as pd
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
    from chat_with_data import (
        LangGraphDataAnalysisOrchestrator, 
        LangGraphDataChatAgent,
        DataProcessor,
        DataAnalysisRequest,
        run_comprehensive_data_analysis,
        create_data_chat_agent
    )
    print("✅ All imports successful!")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure all dependencies are installed and the module structure is correct.")
    sys.exit(1)

def create_sample_dataset():
    """Create a sample dataset for demonstration."""
    print("\n📊 Creating sample dataset...")
    
    # Create a realistic sample dataset
    import numpy as np
    np.random.seed(42)
    
    n_samples = 1000
    
    # Generate realistic business data
    data = {
        'customer_id': range(1, n_samples + 1),
        'age': np.random.normal(35, 12, n_samples).astype(int),
        'income': np.random.normal(50000, 15000, n_samples),
        'purchase_amount': np.random.exponential(100, n_samples),
        'satisfaction_score': np.random.uniform(1, 10, n_samples),
        'region': np.random.choice(['North', 'South', 'East', 'West'], n_samples),
        'product_category': np.random.choice(['Electronics', 'Clothing', 'Books', 'Home'], n_samples),
        'is_premium_customer': np.random.choice([True, False], n_samples, p=[0.3, 0.7])
    }
    
    # Add some realistic correlations
    data['income'] = np.where(data['age'] > 40, data['income'] * 1.2, data['income'])
    data['purchase_amount'] = np.where(data['is_premium_customer'], 
                                     data['purchase_amount'] * 2, 
                                     data['purchase_amount'])
    
    # Add some missing values to make it realistic
    missing_indices = np.random.choice(n_samples, size=int(n_samples * 0.05), replace=False)
    for i in missing_indices:
        data['satisfaction_score'][i] = np.nan
    
    df = pd.DataFrame(data)
    
    # Ensure age is reasonable
    df['age'] = df['age'].clip(18, 80)
    df['income'] = df['income'].clip(20000, 200000)
    df['purchase_amount'] = df['purchase_amount'].clip(10, 1000)
    
    # Save to CSV
    csv_path = "sample_customer_data.csv"
    df.to_csv(csv_path, index=False)
    
    print(f"✅ Sample dataset created: {csv_path}")
    print(f"   - Shape: {df.shape}")
    print(f"   - Columns: {', '.join(df.columns)}")
    print(f"   - Sample preview:")
    print(df.head(3).to_string())
    
    return csv_path

def setup_llm_and_embeddings():
    """Setup LLM and embeddings."""
    print("\n🤖 Setting up AI models...")
    
    try:
        # Initialize NVIDIA models
        llm = ChatNVIDIA(
            model="meta/llama3-70b-instruct",
            api_key=os.getenv("NVIDIA_API_KEY"),
            temperature=0.1
        )
        
        embedder = NVIDIAEmbeddings(
            model="nvidia/nv-embedqa-e5-v5",
            api_key=os.getenv("NVIDIA_API_KEY")
        )
        
        print("✅ AI models initialized successfully!")
        return llm, embedder
        
    except Exception as e:
        print(f"❌ Error setting up AI models: {e}")
        print("Make sure your NVIDIA_API_KEY environment variable is set.")
        return None, None

def demonstrate_comprehensive_analysis(llm, embedder, csv_path):
    """Demonstrate the comprehensive multi-agent data analysis."""
    print("\n" + "="*60)
    print("🔬 COMPREHENSIVE MULTI-AGENT DATA ANALYSIS")
    print("="*60)
    
    # Create analysis request
    request = DataAnalysisRequest(
        file_path=csv_path,
        filename="sample_customer_data.csv",
        user_context="Customer behavior analysis for business intelligence",
        problem_statement="Understand customer segments and purchasing patterns to improve marketing strategy",
        analysis_goals=[
            "Identify customer segments",
            "Analyze purchasing behavior",
            "Find correlations between demographics and spending",
            "Recommend ML approaches for customer prediction"
        ]
    )
    
    print(f"📋 Analysis Request:")
    print(f"   - Dataset: {request.filename}")
    print(f"   - Context: {request.user_context}")
    print(f"   - Goals: {', '.join(request.analysis_goals)}")
    
    # Run comprehensive analysis
    print("\n🚀 Starting multi-agent analysis workflow...")
    print("   This will execute 6 specialized agents in sequence:")
    print("   1. 🔍 Data Profiler - Quality assessment and structure analysis")
    print("   2. 📊 Statistical Analyst - Descriptive and inferential statistics")
    print("   3. 📈 Visualization Specialist - Chart and plot recommendations")
    print("   4. 🤖 ML Advisor - Model selection and feature engineering")
    print("   5. 💡 Insights Generator - Business insights and recommendations")
    print("   6. 📝 Report Writer - Comprehensive documentation")
    
    try:
        # Use the orchestrator
        orchestrator = LangGraphDataAnalysisOrchestrator(llm, embedder)
        result = orchestrator.analyze_dataset(request)
        
        if result["success"]:
            print("\n✅ Analysis completed successfully!")
            
            # Display summary
            print(f"\n📊 Analysis Summary:")
            print(f"   - Report length: {len(result['report'])} characters")
            print(f"   - Visualizations created: {len(result['visualizations'])}")
            print(f"   - Insights generated: {len(result['insights'])}")
            print(f"   - Recommendations: {len(result['recommendations'])} categories")
            
            # Show key insights
            if result['insights']:
                print(f"\n💡 Key Insights:")
                for i, insight in enumerate(result['insights'][:3], 1):
                    print(f"   {i}. {insight}")
            
            # Show ML recommendations
            if 'ml_models' in result['recommendations']:
                print(f"\n🤖 ML Model Recommendations:")
                for model in result['recommendations']['ml_models'][:3]:
                    print(f"   - {model}")
            
            # Save full report
            report_file = "comprehensive_analysis_report.md"
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(result['report'])
            print(f"\n📄 Full report saved to: {report_file}")
            
            return True
            
        else:
            print(f"❌ Analysis failed: {result.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        return False

def demonstrate_intelligent_chat(llm, embedder, csv_path):
    """Demonstrate the intelligent data chat functionality."""
    print("\n" + "="*60)
    print("💬 INTELLIGENT DATA CHAT DEMONSTRATION")
    print("="*60)
    
    # Setup data processor and chat agent
    print("🔧 Setting up data chat agent...")
    data_processor = DataProcessor()
    data_processor.load_data(csv_path)
    data_processor.analyze_data_structure()
    
    chat_agent = create_data_chat_agent(llm, embedder, data_processor)
    
    # Demonstration questions
    demo_questions = [
        "What is the overall structure of this dataset?",
        "What's the average purchase amount by region?",
        "Are there any interesting correlations in the data?",
        "What can you tell me about premium vs regular customers?",
        "What visualizations would you recommend for this data?",
        "How many missing values are there and where?",
        "What machine learning approaches would work well here?"
    ]
    
    print("🎯 Running demonstration chat session...")
    print("   The agent will answer questions using:")
    print("   - 🧠 RAG (Retrieval Augmented Generation) from vectorstore")
    print("   - 🐼 Pandas computational analysis")
    print("   - 📊 Statistical insights")
    print("   - 💡 Domain expertise")
    
    for i, question in enumerate(demo_questions, 1):
        print(f"\n❓ Question {i}: {question}")
        
        try:
            answer = chat_agent.answer_question(question)
            print(f"🤖 Answer: {answer}")
            
            if i < len(demo_questions):
                print("   " + "-"*50)
                
        except Exception as e:
            print(f"❌ Error answering question: {e}")
    
    print("\n✅ Chat demonstration completed!")

def demonstrate_custom_analysis():
    """Show how to use the system with custom data."""
    print("\n" + "="*60)
    print("⚙️ CUSTOM ANALYSIS EXAMPLE")
    print("="*60)
    
    print("""
🎯 Quick Start Guide for Your Own Data:

1. **Prepare Your Data:**
   ```python
   # Your CSV/Excel file should be ready
   file_path = "your_data.csv"
   ```

2. **Basic Comprehensive Analysis:**
   ```python
   from chat_with_data import run_comprehensive_data_analysis
   
   result = run_comprehensive_data_analysis(
       llm=llm,
       embedder=embedder,
       file_path="your_data.csv",
       filename="your_data.csv",
       user_context="Your specific business context",
       problem_statement="What you want to discover"
   )
   
   print(result['report'])  # Full analysis report
   ```

3. **Interactive Chat:**
   ```python
   from chat_with_data import DataProcessor, create_data_chat_agent
   
   # Load your data
   processor = DataProcessor()
   processor.load_data("your_data.csv")
   
   # Create chat agent
   chat_agent = create_data_chat_agent(llm, embedder, processor)
   
   # Ask questions
   answer = chat_agent.answer_question("What patterns do you see?")
   ```

4. **Advanced Custom Orchestration:**
   ```python
   from chat_with_data import LangGraphDataAnalysisOrchestrator, DataAnalysisRequest
   
   orchestrator = LangGraphDataAnalysisOrchestrator(llm, embedder)
   
   request = DataAnalysisRequest(
       file_path="your_data.csv",
       filename="your_data.csv",
       user_context="Your context",
       problem_statement="Your problem",
       analysis_goals=["Goal 1", "Goal 2", "Goal 3"],
       target_variable="your_target_column"  # Optional
   )
   
   result = orchestrator.analyze_dataset(request)
   ```

💡 **Best Practices:**
- Provide clear context about your data and business problem
- Clean your data beforehand for best results
- Ask specific questions for better chat responses
- Review the generated visualizations and insights
- Use the ML recommendations as a starting point

🔧 **Supported File Formats:**
- CSV files (.csv)
- Excel files (.xlsx, .xls)
- Tab-separated files (.tsv)

📊 **Analysis Capabilities:**
- Data profiling and quality assessment
- Statistical analysis and correlation detection
- Visualization recommendations and generation
- Machine learning model suggestions
- Business insights and recommendations
- Professional report generation
""")

def main():
    """Main demonstration function."""
    print("🚀 Enhanced Data Analysis Assistant Demo")
    print("="*50)
    
    # Check environment
    if not os.getenv("NVIDIA_API_KEY"):
        print("❌ Please set your NVIDIA_API_KEY environment variable")
        print("   export NVIDIA_API_KEY='your_api_key_here'")
        return
    
    # Setup
    llm, embedder = setup_llm_and_embeddings()
    if not llm or not embedder:
        return
    
    # Create sample data
    csv_path = create_sample_dataset()
    
    # Run demonstrations
    print("\n🎯 Starting demonstrations...")
    
    # 1. Comprehensive Analysis
    success = demonstrate_comprehensive_analysis(llm, embedder, csv_path)
    
    if success:
        # 2. Intelligent Chat
        demonstrate_intelligent_chat(llm, embedder, csv_path)
    
    # 3. Custom Analysis Guide
    demonstrate_custom_analysis()
    
    # Cleanup
    print(f"\n🧹 Cleaning up...")
    if os.path.exists(csv_path):
        os.remove(csv_path)
        print(f"   Removed: {csv_path}")
    
    print("\n✅ Demo completed successfully!")
    print("\n💡 Next Steps:")
    print("   1. Try with your own CSV/Excel files")
    print("   2. Customize the analysis context for your domain")
    print("   3. Experiment with different types of questions")
    print("   4. Integrate into your data workflow")

if __name__ == "__main__":
    main()
