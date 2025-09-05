# 🚀 Enhanced Data Analysis Assistant

A sophisticated multi-agent data analysis system powered by NVIDIA AI and LangGraph that provides comprehensive analysis of CSV/Excel files through specialized AI agents.

## 🌟 Features

### 🤖 Multi-Agent Analysis System
- **6 Specialized AI Agents** working in orchestrated workflow
- **LangGraph Integration** for complex, stateful workflows
- **Sequential & Parallel Processing** for optimal performance
- **Error Handling & Recovery** with robust fallback mechanisms

### 📊 Specialized Agent Roles

1. **🔍 Data Profiler**
   - Data quality assessment and structural analysis
   - Missing value detection and data type optimization
   - Completeness reporting and data reliability metrics

2. **📈 Statistical Analyst**
   - Descriptive and inferential statistics
   - Correlation analysis and distribution assessment
   - Hypothesis generation and statistical test recommendations

3. **📊 Visualization Specialist**
   - Optimal chart type selection and visualization strategy
   - Interactive dashboard suggestions
   - Plot generation with meaningful insights

4. **🤖 ML Advisor**
   - Machine learning model recommendations
   - Feature engineering suggestions
   - Preprocessing requirements and evaluation metrics

5. **💡 Insights Generator**
   - Business-focused insight extraction
   - Actionable recommendations and opportunity identification
   - Risk assessment and success metrics

6. **📝 Report Writer**
   - Professional comprehensive documentation
   - Technical and business audience targeting
   - Executive summaries and detailed findings

### 💬 Intelligent Data Chat
- **RAG-Enhanced Conversations** using vectorstore knowledge
- **Pandas Integration** for computational queries
- **Context-Aware Responses** with conversation memory
- **Multiple Query Types** (statistical, computational, visualization)

### 🎯 Advanced Capabilities
- **Automatic Analysis Type Detection** (regression, classification, clustering, time series)
- **Comprehensive Visualizations** with base64 encoding for web integration
- **Professional Report Generation** in Markdown format
- **Vectorstore Integration** for persistent data knowledge
- **Fallback Mechanisms** for environments without LangGraph

## 🛠️ Installation

### Prerequisites
```bash
# Core dependencies
pip install pandas numpy matplotlib seaborn plotly
pip install langchain langchain-core langchain-nvidia-ai-endpoints
pip install pydantic faiss-cpu

# Optional: For full LangGraph support
pip install langgraph langchain-experimental
```

### Environment Setup
```bash
# Set your NVIDIA API key
export NVIDIA_API_KEY="your_nvidia_api_key_here"
```

## 🚀 Quick Start

### 1. Comprehensive Data Analysis

```python
from chat_with_data import run_comprehensive_data_analysis
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings

# Setup AI models
llm = ChatNVIDIA(model="meta/llama3-70b-instruct")
embedder = NVIDIAEmbeddings(model="nvidia/nv-embedqa-e5-v5")

# Run comprehensive analysis
result = run_comprehensive_data_analysis(
    llm=llm,
    embedder=embedder,
    file_path="your_data.csv",
    filename="your_data.csv",
    user_context="Customer behavior analysis for e-commerce",
    problem_statement="Identify customer segments and predict churn"
)

# Access results
print(result['report'])  # Full professional report
print(f"Insights: {result['insights']}")  # Key business insights
print(f"ML Recommendations: {result['recommendations']['ml_models']}")
```

### 2. Interactive Data Chat

```python
from chat_with_data import DataProcessor, create_data_chat_agent

# Load and process data
processor = DataProcessor()
processor.load_data("your_data.csv")

# Create intelligent chat agent
chat_agent = create_data_chat_agent(llm, embedder, processor)

# Ask questions about your data
questions = [
    "What's the distribution of my target variable?",
    "Calculate the correlation between age and income",
    "What visualizations would work best for this data?",
    "Are there any outliers I should be concerned about?",
    "What machine learning approach would you recommend?"
]

for question in questions:
    answer = chat_agent.answer_question(question)
    print(f"Q: {question}")
    print(f"A: {answer}\n")
```

### 3. Custom Analysis Workflow

```python
from chat_with_data import (
    LangGraphDataAnalysisOrchestrator, 
    DataAnalysisRequest
)

# Create custom orchestrator
orchestrator = LangGraphDataAnalysisOrchestrator(llm, embedder)

# Define analysis request
request = DataAnalysisRequest(
    file_path="customer_data.csv",
    filename="customer_data.csv",
    user_context="Retail customer analytics",
    problem_statement="Optimize marketing campaigns",
    analysis_goals=[
        "Segment customers by behavior",
        "Identify high-value customer characteristics", 
        "Predict customer lifetime value",
        "Recommend personalization strategies"
    ],
    target_variable="total_spend"  # Optional
)

# Execute analysis
result = orchestrator.analyze_dataset(request)

if result["success"]:
    # Save professional report
    with open("analysis_report.md", "w") as f:
        f.write(result["report"])
    
    # Process visualizations
    for i, viz in enumerate(result["visualizations"]):
        # viz is base64 encoded image
        with open(f"chart_{i}.png", "wb") as f:
            f.write(base64.b64decode(viz))
```

## 📋 Supported Data Types

### File Formats
- **CSV** (.csv) - Comma-separated values
- **Excel** (.xlsx, .xls) - Microsoft Excel files
- **TSV** (.tsv) - Tab-separated values

### Data Types
- **Numerical** - Integers, floats, decimals
- **Categorical** - String categories, factors
- **Boolean** - True/false values
- **DateTime** - Dates and timestamps
- **Mixed Types** - Automatic detection and handling

## 🎯 Analysis Types

The system automatically detects and optimizes for:

- **📊 Exploratory Data Analysis** - General data exploration
- **🔮 Predictive Analytics** - Forecasting and prediction
- **🏷️ Classification** - Category prediction and labeling  
- **📈 Regression** - Continuous value prediction
- **🔍 Clustering** - Pattern discovery and segmentation
- **⏰ Time Series** - Temporal pattern analysis

## 📊 Generated Outputs

### Professional Reports
- Executive summary with key findings
- Technical methodology documentation
- Detailed statistical analysis
- Visualization recommendations
- ML model suggestions
- Business insights and actions

### Visualizations
- Data distribution plots
- Correlation heatmaps
- Statistical summary charts
- Missing value analysis
- Categorical comparisons
- Time series plots (when applicable)

### Machine Learning Recommendations
- Algorithm suggestions based on data characteristics
- Feature engineering recommendations
- Preprocessing requirements
- Evaluation metrics
- Implementation roadmap

## 🔧 Configuration

### Agent Customization
```python
from chat_with_data.data_analysis_agents import DataAnalysisAgent
from chat_with_data.data_agent_schemas import DataAgentRole

# Create custom agent with specialized prompt
custom_agent = DataAnalysisAgent(DataAgentRole.STATISTICAL_ANALYST, llm)

# Modify system prompt for domain-specific analysis
custom_agent.system_prompt = """
You are a financial data analyst specializing in risk assessment...
[custom prompt here]
"""
```

### Workflow Customization
```python
from chat_with_data import LangGraphDataAnalysisOrchestrator

class CustomOrchestrator(LangGraphDataAnalysisOrchestrator):
    def __init__(self, llm, embedder):
        super().__init__(llm, embedder)
        # Customize workflow order
        self.analysis_workflow = [
            DataAgentRole.DATA_PROFILER,
            DataAgentRole.STATISTICAL_ANALYST,
            # Add custom order here
        ]
```

## 🧪 Testing

Run the test suite to verify installation:

```bash
python test_enhanced_data_system.py
```

Run the comprehensive demo:

```bash
python demo_enhanced_data_assistant.py
```

## 📚 Architecture

### System Design
```
┌─────────────────────────────────────────────────────────┐
│                    User Interface                       │
├─────────────────────────────────────────────────────────┤
│              LangGraph Orchestrator                     │
├─────────────────────────────────────────────────────────┤
│  Agent 1   │  Agent 2   │  Agent 3   │  Agent 4       │
│ Profiler   │Statistical │Visualization│  ML Advisor    │
├─────────────────────────────────────────────────────────┤
│  Agent 5   │  Agent 6   │            │                │
│ Insights   │ Reporter   │   Chat     │   Vectorstore  │
├─────────────────────────────────────────────────────────┤
│         Data Processing & Visualization Layer          │
├─────────────────────────────────────────────────────────┤
│                 Data Storage Layer                      │
└─────────────────────────────────────────────────────────┘
```

### Agent Workflow
1. **Initialization** - Load data and setup vectorstore
2. **Data Profiling** - Assess quality and structure
3. **Statistical Analysis** - Descriptive and inferential stats
4. **Visualization** - Create charts and visual recommendations
5. **ML Advisory** - Model selection and feature engineering
6. **Insights Generation** - Business value extraction
7. **Report Writing** - Professional documentation
8. **Finalization** - Compile results and save outputs

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines:

1. **Fork the repository**
2. **Create feature branch** (`git checkout -b feature/amazing-feature`)
3. **Commit changes** (`git commit -m 'Add amazing feature'`)
4. **Push to branch** (`git push origin feature/amazing-feature`)
5. **Open Pull Request**

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For support and questions:

1. **Check the documentation** in this README
2. **Run the test suite** to verify installation
3. **Try the demo script** for examples
4. **Review error logs** for debugging information

## 🔮 Future Enhancements

- **Real-time streaming data support**
- **Integration with cloud data sources**
- **Advanced statistical testing suite**
- **Custom visualization templates**
- **Multi-language report generation**
- **API endpoint for web integration**
- **Scheduled analysis automation**

---

**Built with ❤️ using NVIDIA AI, LangGraph, and specialized AI agents for comprehensive data analysis.**
