#!/usr/bin/env python3
"""
Integration Example: Enhanced Data Analysis Assistant

This script demonstrates how the enhanced chat_with_data module now follows
the same LangGraph-based multi-agent pattern as the multi_agent_software_team
module, providing specialized agents working in orchestrated workflows.

Comparison with multi_agent_software_team:
- Similar LangGraph StateGraph workflow orchestration
- Specialized agents with defined roles and deliverables
- Sequential execution with context passing between agents
- Professional report generation and structured outputs
- Error handling and fallback mechanisms
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path

def create_sample_data():
    """Create sample data for demonstration."""
    rng = np.random.default_rng(42)
    
    data = {
        'customer_id': range(1, 201),
        'age': rng.normal(35, 12, 200).astype(int).clip(18, 70),
        'annual_income': rng.normal(55000, 20000, 200).clip(25000, 150000),
        'spending_score': rng.uniform(1, 100, 200),
        'membership_years': rng.exponential(2, 200).clip(0, 15),
        'region': rng.choice(['North', 'South', 'East', 'West'], 200),
        'product_category': rng.choice(['Electronics', 'Fashion', 'Home', 'Sports'], 200),
        'is_premium': rng.choice([True, False], 200, p=[0.25, 0.75])
    }
    
    df = pd.DataFrame(data)
    
    # Add realistic correlations
    df.loc[df['is_premium'], 'spending_score'] *= 1.4
    df.loc[df['age'] > 50, 'annual_income'] *= 1.2
    
    return df

def demonstrate_enhanced_data_analysis():
    """Demonstrate the enhanced data analysis system."""
    print("🔬 Enhanced Data Analysis Assistant Demo")
    print("=" * 50)
    
    # Create sample data
    print("📊 Creating sample customer dataset...")
    df = create_sample_data()
    csv_file = "sample_customer_analysis.csv"
    df.to_csv(csv_file, index=False)
    
    print(f"✅ Dataset created: {csv_file}")
    print(f"   - Shape: {df.shape}")
    print(f"   - Columns: {', '.join(df.columns)}")
    print("\n" + "="*60)
    
    # Show how to use the system (mock example since we need API keys)
    print("🤖 ENHANCED MULTI-AGENT DATA ANALYSIS WORKFLOW")
    print("="*60)
    
    print("""
The enhanced chat_with_data module now provides:

🎯 SPECIALIZED AGENT ROLES (similar to software team agents):
┌─────────────────────┬──────────────────────────────────────┐
│ DATA PROFILER       │ Quality assessment & structure       │
│ STATISTICAL_ANALYST │ Descriptive & inferential stats     │
│ VISUALIZATION_SPEC  │ Chart recommendations & generation   │
│ ML_ADVISOR          │ Model selection & feature engineering│
│ INSIGHTS_GENERATOR  │ Business insights & recommendations  │
│ REPORT_WRITER       │ Professional documentation          │
└─────────────────────┴──────────────────────────────────────┘

🔄 LANGGRAPH ORCHESTRATION (like software team workflow):
    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
    │ Initialize  │ -> │ Profile     │ -> │ Statistics  │
    └─────────────┘    └─────────────┘    └─────────────┘
           |                   |                   |
    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
    │ Visualize   │ <- │ ML Advice   │ <- │ Insights    │
    └─────────────┘    └─────────────┘    └─────────────┘
           |
    ┌─────────────┐
    │ Final Report│
    └─────────────┘

📋 USAGE EXAMPLE:
""")

    # Show code example
    example_code = '''
# 1. COMPREHENSIVE ANALYSIS (like software team collaboration)
from chat_with_data import run_comprehensive_data_analysis

result = run_comprehensive_data_analysis(
    llm=llm,
    embedder=embedder,
    file_path="customer_data.csv",
    filename="customer_data.csv",
    user_context="E-commerce customer behavior analysis",
    problem_statement="Identify customer segments and predict high-value customers"
)

# Get professional report (like software team deliverables)
print(result['report'])  # Comprehensive markdown report
print(result['insights'])  # Business insights
print(result['recommendations'])  # ML and action recommendations

# 2. INTERACTIVE DATA CHAT (enhanced with RAG + pandas)
from chat_with_data import create_data_chat_agent, DataProcessor

processor = DataProcessor()
processor.load_data("customer_data.csv")

chat_agent = create_data_chat_agent(llm, embedder, processor)

# Ask intelligent questions about your data
questions = [
    "What's the correlation between age and spending?",
    "Calculate average income by region",
    "What visualizations would work best?",
    "Which ML models would you recommend?",
    "Are there any data quality issues?"
]

for question in questions:
    answer = chat_agent.answer_question(question)
    print(f"Q: {question}")
    print(f"A: {answer}")

# 3. CUSTOM ORCHESTRATOR (like custom software team workflow)
from chat_with_data import LangGraphDataAnalysisOrchestrator, DataAnalysisRequest

orchestrator = LangGraphDataAnalysisOrchestrator(llm, embedder)

request = DataAnalysisRequest(
    file_path="customer_data.csv",
    filename="customer_data.csv",
    user_context="Customer analytics for marketing optimization",
    analysis_goals=[
        "Segment customers by behavior patterns",
        "Identify factors driving high spending",
        "Recommend personalization strategies"
    ],
    target_variable="spending_score"
)

result = orchestrator.analyze_dataset(request)
'''

    print(example_code)
    
    print("\n" + "="*60)
    print("📊 COMPARISON: Software Team vs Data Analysis")
    print("="*60)
    
    comparison = """
┌─────────────────────┬───────────────────────┬─────────────────────────┐
│ ASPECT              │ SOFTWARE TEAM         │ DATA ANALYSIS           │
├─────────────────────┼───────────────────────┼─────────────────────────┤
│ Agent Roles         │ Product Owner         │ Data Profiler           │
│                     │ Analyst               │ Statistical Analyst     │
│                     │ Architect             │ Visualization Specialist│
│                     │ Developer             │ ML Advisor              │
│                     │ Reviewer              │ Insights Generator      │
│                     │ Tester                │ Report Writer           │
│                     │ Tech Writer           │                         │
├─────────────────────┼───────────────────────┼─────────────────────────┤
│ Orchestration       │ LangGraph StateGraph  │ LangGraph StateGraph    │
│ Workflow            │ Sequential execution  │ Sequential execution    │
│ State Management    │ TeamState TypedDict   │ DataAnalysisState       │
│ Error Handling      │ Robust fallbacks     │ Robust fallbacks        │
├─────────────────────┼───────────────────────┼─────────────────────────┤
│ Input               │ Project description   │ CSV/Excel data files    │
│ Context             │ Business requirements │ Data analysis context   │
│ Goals               │ Software deliverables │ Analysis objectives     │
├─────────────────────┼───────────────────────┼─────────────────────────┤
│ Outputs             │ Code, architecture    │ Statistics, insights    │
│                     │ Documentation         │ Visualizations          │
│                     │ Test plans            │ ML recommendations      │
│                     │ Deployment guides     │ Professional reports    │
├─────────────────────┼───────────────────────┼─────────────────────────┤
│ Additional Features │ Handoff mechanisms    │ RAG-enhanced chat       │
│                     │ Human-in-the-loop     │ Pandas integration      │
│                     │ Mermaid diagrams      │ Vectorstore knowledge   │
│                     │ Clean output mode     │ Auto-analysis detection │
└─────────────────────┴───────────────────────┴─────────────────────────┘
"""
    
    print(comparison)
    
    print("\n🎯 KEY IMPROVEMENTS IN DATA ANALYSIS MODULE:")
    improvements = """
1. 🧠 INTELLIGENT ANALYSIS
   - Automatic analysis type detection (regression, classification, clustering)
   - Smart feature engineering suggestions
   - Domain-specific insights generation

2. 💬 ENHANCED CHAT CAPABILITIES  
   - RAG (Retrieval Augmented Generation) with vectorstore
   - Pandas agent integration for computational queries
   - Context-aware conversation memory
   - Multi-modal query handling (statistical, computational, visualization)

3. 📊 COMPREHENSIVE VISUALIZATIONS
   - Automatic chart type selection based on data characteristics
   - Base64 encoded plots for web integration
   - Interactive dashboard recommendations
   - Professional-quality matplotlib/seaborn/plotly integration

4. 🤖 ML INTEGRATION
   - Scikit-learn model recommendations
   - Feature engineering pipeline suggestions  
   - Evaluation metrics selection
   - Preprocessing requirement analysis

5. 📈 BUSINESS VALUE FOCUS
   - Actionable business insights extraction
   - ROI-focused recommendations
   - Risk assessment and opportunity identification
   - Executive-level reporting
"""
    
    print(improvements)
    
    print("\n✅ READY TO USE!")
    print("The enhanced data analysis system is now available with:")
    print("- Multi-agent workflow orchestration (like software team)")
    print("- Specialized data analysis agents") 
    print("- LangGraph integration for complex workflows")
    print("- RAG-enhanced intelligent chat")
    print("- Professional reporting and insights")
    
    # Cleanup
    if os.path.exists(csv_file):
        os.remove(csv_file)
        print(f"\n🧹 Cleaned up: {csv_file}")

if __name__ == "__main__":
    demonstrate_enhanced_data_analysis()
