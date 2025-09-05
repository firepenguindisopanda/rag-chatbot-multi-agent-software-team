# Enhanced RAG Chatbot with LangGraph Agents and Separate Vectorstores

## 🚀 Overview

This enhanced RAG chatbot system now features sophisticated LangGraph-based agents with separate vectorstore management for different data types. The system intelligently separates PDF documents from CSV/Excel data, providing specialized analysis capabilities for each.

## 🏗️ Architecture Enhancements

### 1. **Separate Vectorstore Management**
- **PDF Documents**: Stored in `docstore_index` (original functionality)
- **CSV/Excel Data**: Stored in `data_vectorstore_index` (new dedicated vectorstore)
- **Intelligent Routing**: System automatically routes queries to appropriate vectorstore

### 2. **Enhanced LangGraph Agents**

#### Data Analysis Agent (`EnhancedLangGraphDataAnalysisAgent`)
- **State-driven workflow** with multiple phases:
  - Data loading and cleaning
  - Vectorstore integration
  - Analysis type detection
  - Insight generation
  - Visualization creation
  - ML model recommendations
- **Comprehensive Analysis**: Uses vectorstore context for enhanced insights
- **Advanced Visualizations**: Creates multiple plot types with proper error handling
- **ML Recommendations**: Suggests appropriate models and metrics based on data characteristics

#### Data Chat Agent (`EnhancedLangGraphDataChatAgent`)
- **Multi-source RAG**: Combines vectorstore search with pandas agent capabilities
- **Context-aware responses**: Uses both statistical context and semantic search
- **Calculation support**: Handles computational queries via integrated pandas agent
- **Conversation memory**: Maintains context across chat sessions

#### Software Team Orchestrator (`EnhancedLangGraphSoftwareTeamOrchestrator`)
- **Multi-phase collaboration**:
  - Requirements analysis
  - Architecture design
  - Implementation planning
  - Code review
  - Testing strategy
  - Deployment planning
- **Stateful workflows**: Maintains collaboration state across phases
- **Human-in-the-loop**: Supports feedback integration and iterative refinement
- **Role-based specialization**: Each agent role provides specialized deliverables

## 🔧 Key Features

### **Vectorstore Separation**
```python
# PDF documents go to docstore_index
pdf_vectorstore = FAISS.load_local("docstore_index", embedder)

# CSV/Excel data goes to data_vectorstore_index  
data_vectorstore = FAISS.load_local("data_vectorstore_index", embedder)
```

### **LangGraph Workflows**
- **State Management**: Each workflow maintains comprehensive state
- **Error Handling**: Robust error recovery and reporting
- **Conditional Logic**: Smart routing based on data characteristics
- **Progress Tracking**: Real-time workflow status updates

### **Enhanced Data Analysis**
- **Automatic Analysis Type Detection**: Regression, Classification, Clustering, Time Series, Exploratory
- **Vectorstore-enhanced Insights**: Uses semantic search for context-aware analysis
- **Comprehensive Visualizations**: Overview plots, target analysis, correlation matrices
- **ML Model Recommendations**: Tailored suggestions based on data size and type

### **Advanced Chat Capabilities**
- **Multi-modal responses**: Combines statistical analysis with semantic understanding
- **Pandas Agent Integration**: Handles complex calculations and data queries
- **Context preservation**: Maintains conversation history and data context
- **Error resilience**: Graceful fallbacks for various query types

## 📊 Data Flow

### CSV/Excel Upload Flow
```
1. File Upload → 2. Data Processing → 3. Vectorstore Creation → 4. Agent Initialization
    ↓                    ↓                    ↓                      ↓
   Validation        Cleaning            Document Creation      Chat Agent Ready
   File Type         Type Detection      Embedding             Analysis Complete
```

### Chat Query Flow
```
User Query → Vectorstore Search → Pandas Agent (if needed) → LLM Response
     ↓              ↓                        ↓                     ↓
  Query Analysis   Context Retrieval    Calculation Results   Final Response
```

### Software Team Flow
```
Project Description → Team Validation → Multi-phase Collaboration → Deliverables
        ↓                    ↓                      ↓                     ↓
   Requirements         Role Assignment      State Management       Final Output
```

## 🛠️ Technical Implementation

### **Enhanced State Management**
```python
class DataAnalysisState(TypedDict):
    messages: Annotated[List, add_messages]
    data_summary: Optional[DataSummary] 
    analysis_type: Optional[DataAnalysisType]
    insights: List[str]
    plots: List[str]
    recommendations: Dict[str, List[str]]
    vectorstore_status: Optional[str]
    # ... additional state fields
```

### **Workflow Orchestration**
```python
def _build_analysis_graph(self) -> StateGraph:
    workflow = StateGraph(DataAnalysisState)
    
    # Add sequential workflow nodes
    workflow.add_node("start", self._start_analysis)
    workflow.add_node("load_data", self._load_data) 
    workflow.add_node("add_to_vectorstore", self._add_to_vectorstore)
    workflow.add_node("detect_analysis", self._detect_analysis_type)
    # ... more nodes
    
    return workflow.compile()
```

### **Vectorstore Integration**
```python
class DataVectorStoreManager:
    def __init__(self, embedder):
        self.embedder = embedder
        self.vectorstore = None
        self.data_vectorstore_path = "data_vectorstore_index"  # Separate from PDF
        self.data_vectorstore_archive = "data_vectorstore_index.tgz"
```

## 📈 Performance Improvements

### **Memory Efficiency**
- Separate vectorstores prevent cross-contamination
- Efficient document chunking and embedding
- State-based processing reduces redundant operations

### **Response Quality**
- Context-aware insights using vectorstore semantic search
- Multi-source information synthesis
- Domain-specific agent specialization

### **Scalability**
- Modular agent architecture
- Configurable workflow phases
- Extensible team composition

## 🔄 Usage Examples

### **Data Analysis**
```python
# Upload CSV/Excel file
data_request = DataRequest(
    file_path="data.csv",
    user_context="Sales forecasting analysis", 
    target_variable="revenue"
)

# Analyze with LangGraph workflow
result = enhanced_agent.analyze_dataset(data_request)
# Result includes insights, plots, ML recommendations
```

### **Data Chat**
```python
# Chat about uploaded data
response = chat_agent.answer_question("What's the correlation between price and sales?")
# Uses both vectorstore context and pandas calculations
```

### **Software Team Collaboration**
```python
# Multi-agent software development
project_request = ProjectRequest(
    description="Build a web-based inventory management system",
    selected_roles=[TeamRole.PRODUCT_OWNER, TeamRole.ARCHITECT, TeamRole.DEVELOPER]
)

result = orchestrator.collaborate_on_project(project_request)
# Returns comprehensive deliverables from each agent
```

## 🚀 Benefits

1. **Intelligent Data Separation**: PDF and tabular data don't interfere with each other
2. **Enhanced Analysis**: LangGraph workflows provide sophisticated, stateful processing
3. **Better Chat Experience**: Multi-source responses with computational capabilities  
4. **Professional Team Output**: Structured, phase-based software development process
5. **Scalable Architecture**: Modular design supports easy extension and customization

## 📝 Next Steps

The system is now ready for production use with:
- ✅ Separate vectorstores for different data types
- ✅ LangGraph-powered sophisticated agent workflows
- ✅ Enhanced chat capabilities with pandas integration
- ✅ Professional multi-agent software team collaboration
- ✅ Comprehensive error handling and state management

The enhanced implementation provides a robust foundation for advanced RAG applications with specialized data handling and intelligent agent collaboration.
