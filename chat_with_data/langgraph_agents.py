"""
Enhanced LangGraph-based agents for data analysis and chat functionality.
Integrates CSV/Excel vectorstore with advanced agent capabilities.
"""

import json
import logging
import pandas as pd
from typing import Dict, List, Any, Optional, TypedDict, Annotated, Union
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_experimental.tools import PythonREPLTool
from langchain.agents.agent_types import AgentType

from .data_processor import DataProcessor, DataVisualizer, MLModelRecommender
from .vectorstore_manager import DataVectorStoreManager
from .schemas import DataRequest, DataSummary, AnalysisResult, DataAnalysisType

logger = logging.getLogger(__name__)

# Constants
TARGET_VARIABLE_KEY = "target_variable:"
CONTEXT_KEY = "context:"
FILE_PATH_KEY = "file_path:"

class DataAnalysisState(TypedDict):
    """State for data analysis graph."""
    messages: Annotated[List, add_messages]
    data_summary: Optional[DataSummary]
    analysis_type: Optional[DataAnalysisType]
    insights: List[str]
    plots: List[str]
    recommendations: Dict[str, List[str]]
    current_step: str
    error: Optional[str]
    file_path: Optional[str]
    user_context: Optional[str]
    target_variable: Optional[str]

class DataChatState(TypedDict):
    """State for data chat graph."""
    messages: Annotated[List, add_messages]
    query: str
    context: str
    response: str
    vectorstore_results: List[str]
    data_context: Optional[str]

class LangGraphDataAnalysisAgent:
    """LangGraph-based agent for comprehensive data analysis."""
    
    def __init__(self, llm, embedder):
        self.llm = llm
        self.embedder = embedder
        self.data_processor = DataProcessor()
        self.vectorstore_manager = DataVectorStoreManager(embedder)
        self.vectorstore_manager.load_existing_vectorstore()
        self.graph = self._build_analysis_graph()
    
    def _build_analysis_graph(self) -> StateGraph:
        """Build the data analysis workflow graph."""
        
        # Define analysis tools
        @tool
        def load_and_process_data(file_path: str) -> Dict[str, Any]:
            """Load and process the uploaded data file."""
            try:
                self.data_processor.load_data(file_path)
                self.data_processor.clean_data()
                summary = self.data_processor.analyze_data_structure()
                
                # Add to vectorstore
                file_info = {"filename": file_path.split("/")[-1]}
                vectorstore_result = self.vectorstore_manager.add_data_to_vectorstore(
                    self.data_processor.data, file_info
                )
                
                return {
                    "success": True,
                    "summary": summary.dict(),
                    "vectorstore_status": vectorstore_result["status"]
                }
            except Exception as e:
                return {"success": False, "error": str(e)}
        
        @tool
        def detect_analysis_type(user_context: str, target_variable: Optional[str] = None) -> str:
            """Detect the most appropriate analysis type for the data."""
            try:
                analysis_type = self.data_processor.detect_analysis_type(user_context, target_variable)
                return analysis_type.value
            except Exception as e:
                return f"error: {str(e)}"
        
        @tool
        def generate_insights(data_summary_dict: Dict, analysis_type: str) -> List[str]:
            """Generate AI-powered insights about the data."""
            try:
                # Convert dict back to DataSummary object
                data_summary = DataSummary(**data_summary_dict)
                
                prompt = ChatPromptTemplate.from_template(
                    """You are a data scientist analyzing a dataset. Based on the following information, provide 3-4 key insights.

DATASET INFO:
- Shape: {shape[0]} rows, {shape[1]} columns  
- Numerical columns: {numerical_count}
- Categorical columns: {categorical_count}
- Total missing values: {missing_count}
- Analysis type: {analysis_type}

Provide insights as a JSON list of 3-4 strings. Each insight should be:
1. Specific and actionable
2. Based on the data characteristics
3. Relevant to the analysis type
4. Keep each insight to 1-2 sentences maximum

Return only the JSON list, no other text.
"""
                )
                
                formatted_prompt = prompt.format(
                    shape=data_summary.shape,
                    numerical_count=len(data_summary.numerical_columns),
                    categorical_count=len(data_summary.categorical_columns),
                    missing_count=sum(data_summary.missing_values.values()),
                    analysis_type=analysis_type
                )
                
                result = self.llm.invoke(formatted_prompt)
                insights_data = json.loads(result.content)
                
                # Process insights to ensure they are strings
                processed_insights = []
                if isinstance(insights_data, list):
                    for insight in insights_data:
                        if isinstance(insight, dict) and 'text' in insight:
                            # Extract text from dictionary format
                            processed_insights.append(insight['text'])
                        elif isinstance(insight, str):
                            processed_insights.append(insight)
                        else:
                            # Convert any other format to string
                            processed_insights.append(str(insight))
                else:
                    # Fallback for unexpected format
                    processed_insights = [str(insights_data)]
                    
                return processed_insights
                
            except Exception as e:
                logger.error(f"Error generating insights: {str(e)}")
                return [f"Error generating insights: {str(e)}"]
        
        @tool
        def create_visualizations(target_variable: Optional[str] = None) -> List[str]:
            """Create data visualizations."""
            try:
                if self.data_processor.data is None:
                    return ["Error: No data loaded"]
                
                visualizer = DataVisualizer(self.data_processor.data)
                plots = visualizer.create_overview_plots()
                
                if target_variable:
                    target_plots = visualizer.create_target_analysis(target_variable)
                    plots.extend(target_plots)
                
                return plots
            except Exception as e:
                return [f"Error creating visualizations: {str(e)}"]
        
        @tool
        def get_ml_recommendations(data_summary_dict: Dict, analysis_type: str) -> Dict[str, List[str]]:
            """Get ML model and metrics recommendations."""
            try:
                data_summary = DataSummary(**data_summary_dict)
                analysis_type_enum = DataAnalysisType(analysis_type)
                
                recommender = MLModelRecommender(data_summary, analysis_type_enum)
                models = recommender.recommend_models()
                metrics = recommender.recommend_metrics()
                
                return {
                    "models": models,
                    "metrics": metrics
                }
            except Exception as e:
                return {
                    "models": [f"Error: {str(e)}"],
                    "metrics": [f"Error: {str(e)}"]
                }
        
        # Create tool node
        tools = [load_and_process_data, detect_analysis_type, generate_insights, 
                create_visualizations, get_ml_recommendations]
        tool_node = ToolNode(tools)
        
        # Define workflow nodes
        def start_analysis(state: DataAnalysisState) -> DataAnalysisState:
            """Start the data analysis workflow."""
            state["current_step"] = "loading_data"
            state["messages"].append(SystemMessage(content="Starting data analysis workflow..."))
            return state
        
        def process_data_step(state: DataAnalysisState) -> DataAnalysisState:
            """Process the uploaded data."""
            try:
                # Extract file path from messages
                file_path = None
                for msg in state["messages"]:
                    if isinstance(msg, HumanMessage) and "file_path:" in msg.content:
                        file_path = msg.content.split("file_path:")[-1].strip()
                        break
                
                if not file_path:
                    state["error"] = "No file path provided"
                    return state
                
                # Load and process data
                result = load_and_process_data.invoke({"file_path": file_path})
                
                if result["success"]:
                    state["data_summary"] = DataSummary(**result["summary"])
                    state["current_step"] = "detecting_analysis_type"
                else:
                    state["error"] = result["error"]
                
            except Exception as e:
                state["error"] = str(e)
            
            return state
        
        def detect_analysis_step(state: DataAnalysisState) -> DataAnalysisState:
            """Detect the appropriate analysis type."""
            try:
                # Extract context from messages
                user_context = ""
                target_variable = None
                
                for msg in state["messages"]:
                    if isinstance(msg, HumanMessage):
                        if "context:" in msg.content:
                            user_context = msg.content.split("context:")[-1].split("target_variable:")[0].strip()
                        if "target_variable:" in msg.content:
                            target_variable = msg.content.split("target_variable:")[-1].strip()
                
                analysis_type_str = detect_analysis_type.invoke({
                    "user_context": user_context,
                    "target_variable": target_variable
                })
                
                if not analysis_type_str.startswith("error"):
                    state["analysis_type"] = DataAnalysisType(analysis_type_str)
                    state["current_step"] = "generating_insights"
                else:
                    state["error"] = analysis_type_str
                
            except Exception as e:
                state["error"] = str(e)
            
            return state
        
        def generate_insights_step(state: DataAnalysisState) -> DataAnalysisState:
            """Generate insights about the data."""
            try:
                if state["data_summary"] and state["analysis_type"]:
                    insights = generate_insights.invoke({
                        "data_summary_dict": state["data_summary"].dict(),
                        "analysis_type": state["analysis_type"].value
                    })
                    # Ensure insights are strings
                    processed_insights = []
                    if isinstance(insights, list):
                        for insight in insights:
                            if isinstance(insight, dict) and 'text' in insight:
                                processed_insights.append(insight['text'])
                            elif isinstance(insight, str):
                                processed_insights.append(insight)
                            else:
                                processed_insights.append(str(insight))
                        state["insights"] = processed_insights
                    else:
                        state["insights"] = [str(insights)]
                    state["current_step"] = "creating_visualizations"
                else:
                    state["error"] = "Missing data summary or analysis type"
                    
            except Exception as e:
                state["error"] = str(e)
            
            return state
        
        def create_visualizations_step(state: DataAnalysisState) -> DataAnalysisState:
            """Create data visualizations."""
            try:
                # Extract target variable from messages if provided
                target_variable = None
                for msg in state["messages"]:
                    if isinstance(msg, HumanMessage) and "target_variable:" in msg.content:
                        target_variable = msg.content.split("target_variable:")[-1].strip()
                        break
                
                plots = create_visualizations.invoke({"target_variable": target_variable})
                state["plots"] = plots
                state["current_step"] = "getting_recommendations"
                
            except Exception as e:
                state["error"] = str(e)
            
            return state
        
        def get_recommendations_step(state: DataAnalysisState) -> DataAnalysisState:
            """Get ML model and metrics recommendations."""
            try:
                if state["data_summary"] and state["analysis_type"]:
                    recommendations = get_ml_recommendations.invoke({
                        "data_summary_dict": state["data_summary"].dict(),
                        "analysis_type": state["analysis_type"].value
                    })
                    state["recommendations"] = recommendations
                    state["current_step"] = "completed"
                else:
                    state["error"] = "Missing data summary or analysis type"
                    
            except Exception as e:
                state["error"] = str(e)
            
            return state
        
        def should_continue(state: DataAnalysisState) -> str:
            """Determine next step in the workflow."""
            if state.get("error"):
                return "error"
            
            current_step = state.get("current_step", "")
            
            if current_step == "loading_data":
                return "process_data"
            elif current_step == "detecting_analysis_type":
                return "detect_analysis"
            elif current_step == "generating_insights":
                return "generate_insights"
            elif current_step == "creating_visualizations":
                return "create_visualizations"
            elif current_step == "getting_recommendations":
                return "get_recommendations"
            elif current_step == "completed":
                return END
            else:
                return "error"
        
        def handle_error(state: DataAnalysisState) -> DataAnalysisState:
            """Handle errors in the workflow."""
            error_msg = state.get("error", "Unknown error occurred")
            state["messages"].append(AIMessage(content=f"Error in analysis: {error_msg}"))
            return state
        
        # Build the graph
        workflow = StateGraph(DataAnalysisState)
        
        # Add nodes
        workflow.add_node("start", start_analysis)
        workflow.add_node("process_data", process_data_step)
        workflow.add_node("detect_analysis", detect_analysis_step)
        workflow.add_node("generate_insights", generate_insights_step)
        workflow.add_node("create_visualizations", create_visualizations_step)
        workflow.add_node("get_recommendations", get_recommendations_step)
        workflow.add_node("error", handle_error)
        
        # Add edges
        workflow.set_entry_point("start")
        workflow.add_conditional_edges("start", should_continue)
        workflow.add_conditional_edges("process_data", should_continue)
        workflow.add_conditional_edges("detect_analysis", should_continue)
        workflow.add_conditional_edges("generate_insights", should_continue)
        workflow.add_conditional_edges("create_visualizations", should_continue)
        workflow.add_conditional_edges("get_recommendations", should_continue)
        workflow.add_edge("error", END)
        
        return workflow.compile()
    
    def analyze_dataset(self, data_request: DataRequest) -> AnalysisResult:
        """Perform comprehensive data analysis using LangGraph workflow."""
        try:
            # Prepare initial state
            initial_state = DataAnalysisState(
                messages=[
                    HumanMessage(
                        content=f"file_path: {data_request.file_path}\n"
                               f"context: {data_request.user_context or ''}\n"
                               f"target_variable: {data_request.target_variable or ''}"
                    )
                ],
                data_summary=None,
                analysis_type=None,
                insights=[],
                plots=[],
                recommendations={},
                current_step="start",
                error=None
            )
            
            # Run the workflow
            final_state = self.graph.invoke(initial_state)
            
            if final_state.get("error"):
                raise Exception(final_state["error"])
            
            # Generate summary and next steps
            summary = self._generate_summary(
                final_state["data_summary"], 
                final_state["analysis_type"], 
                data_request
            )
            
            next_steps = self._generate_next_steps(
                final_state["analysis_type"], 
                final_state["data_summary"]
            )
            
            return AnalysisResult(
                summary=summary,
                visualizations=final_state["plots"],
                insights=final_state["insights"],
                recommended_ml_models=final_state["recommendations"].get("models", []),
                suggested_metrics=final_state["recommendations"].get("metrics", []),
                next_steps=next_steps
            )
            
        except Exception as e:
            logger.error(f"Error in LangGraph data analysis: {str(e)}")
            raise
    
    def _generate_summary(self, data_summary: DataSummary, analysis_type: DataAnalysisType, data_request: DataRequest) -> str:
        """Generate a concise summary of the analysis."""
        if not data_summary:
            return "Analysis failed to complete."
        
        missing_percent = sum(data_summary.missing_values.values()) / (data_summary.shape[0] * data_summary.shape[1]) * 100
        
        return (f"Analyzed dataset with {data_summary.shape[0]} rows and {data_summary.shape[1]} columns. "
                f"Detected {analysis_type.value} analysis type. "
                f"Data completeness: {100-missing_percent:.1f}%. Ready for {analysis_type.value} modeling.")
    
    def _generate_next_steps(self, analysis_type: DataAnalysisType, data_summary: DataSummary) -> List[str]:
        """Generate actionable next steps."""
        steps = []
        
        if not data_summary:
            return ["Complete data analysis first"]
        
        # Common preprocessing steps
        if sum(data_summary.missing_values.values()) > 0:
            steps.append("Handle missing values using appropriate imputation strategies")
        
        if len(data_summary.categorical_columns) > 0:
            steps.append("Encode categorical variables for machine learning")
        
        # Analysis-specific steps
        if analysis_type == DataAnalysisType.REGRESSION:
            steps.append("Check for multicollinearity between features")
            steps.append("Consider feature scaling for better model performance")
        elif analysis_type == DataAnalysisType.CLASSIFICATION:
            steps.append("Check class balance and consider resampling if needed")
            steps.append("Perform feature selection to identify most important variables")
        elif analysis_type == DataAnalysisType.CLUSTERING:
            steps.append("Standardize features before clustering")
            steps.append("Determine optimal number of clusters using elbow method")
        elif analysis_type == DataAnalysisType.TIME_SERIES:
            steps.append("Check for seasonality and trend patterns")
            steps.append("Test for stationarity and apply differencing if needed")
        else:
            steps.append("Explore data relationships and patterns")
        
        return steps[:5]  # Limit to 5 steps


class LangGraphDataChatAgent:
    """LangGraph-based agent for chatting about data using RAG."""
    
    def __init__(self, llm, data_processor: DataProcessor, vectorstore_manager: DataVectorStoreManager):
        self.llm = llm
        self.data_processor = data_processor
        self.vectorstore_manager = vectorstore_manager
        self.conversation_history = []
        self.graph = self._build_chat_graph()
    
    def _build_chat_graph(self) -> StateGraph:
        """Build the data chat workflow graph."""
        
        def retrieve_context(state: DataChatState) -> DataChatState:
            """Retrieve relevant context from vectorstore."""
            try:
                # Search vectorstore for relevant context
                docs = self.vectorstore_manager.search_data_context(state["query"], k=5)
                
                # Extract context from documents
                context_parts = []
                for doc in docs:
                    context_parts.append(f"{doc.metadata.get('type', 'data')}: {doc.page_content[:500]}")
                
                state["context"] = "\n\n".join(context_parts)
                state["vectorstore_results"] = [doc.page_content for doc in docs]
                
            except Exception as e:
                logger.error(f"Error retrieving context: {str(e)}")
                state["context"] = f"Error retrieving context: {str(e)}"
                state["vectorstore_results"] = []
            
            return state
        
        def generate_response(state: DataChatState) -> DataChatState:
            """Generate response based on query and context."""
            try:
                # Get additional data context
                data_context = self._get_data_context()
                
                prompt = ChatPromptTemplate.from_template(
                    """You are a data analyst assistant. Answer the user's question about their dataset based on the following information:

VECTORSTORE CONTEXT:
{vectorstore_context}

CURRENT DATASET CONTEXT:
{data_context}

CONVERSATION HISTORY:
{conversation_history}

USER QUESTION: {question}

Provide a clear, specific answer based on the data. If you need to perform calculations, show your reasoning. Keep responses concise but informative (max 3-4 sentences).
"""
                )
                
                conversation_history = "\n".join([
                    f"Human: {msg['question']}\nAssistant: {msg['response']}" 
                    for msg in self.conversation_history[-3:]  # Last 3 exchanges
                ])
                
                formatted_prompt = prompt.format(
                    vectorstore_context=state["context"],
                    data_context=data_context,
                    conversation_history=conversation_history,
                    question=state["query"]
                )
                
                result = self.llm.invoke(formatted_prompt)
                state["response"] = result.content
                
                # Add to conversation history
                self.conversation_history.append({
                    "question": state["query"],
                    "response": result.content
                })
                
            except Exception as e:
                logger.error(f"Error generating response: {str(e)}")
                state["response"] = f"I apologize, but I encountered an error while processing your question: {str(e)}"
            
            return state
        
        # Build the graph
        workflow = StateGraph(DataChatState)
        
        # Add nodes
        workflow.add_node("retrieve_context", retrieve_context)
        workflow.add_node("generate_response", generate_response)
        
        # Add edges
        workflow.set_entry_point("retrieve_context")
        workflow.add_edge("retrieve_context", "generate_response")
        workflow.add_edge("generate_response", END)
        
        return workflow.compile()
    
    def answer_question(self, question: str) -> str:
        """Answer questions about the loaded dataset using LangGraph workflow."""
        if self.data_processor.data is None:
            return "No dataset is currently loaded. Please upload a CSV or Excel file first."
        
        try:
            # Prepare initial state
            initial_state = DataChatState(
                messages=[HumanMessage(content=question)],
                query=question,
                context="",
                response="",
                vectorstore_results=[]
            )
            
            # Run the workflow
            final_state = self.graph.invoke(initial_state)
            
            return final_state["response"]
            
        except Exception as e:
            logger.error(f"Error in LangGraph data chat: {str(e)}")
            return f"I apologize, but I encountered an error: {str(e)}"
    
    def _get_data_context(self) -> str:
        """Get relevant context about the current dataset."""
        if self.data_processor.data is None:
            return "No dataset loaded"
        
        data = self.data_processor.data
        
        # Basic statistics
        context = f"Dataset shape: {data.shape[0]} rows, {data.shape[1]} columns\n"
        context += f"Columns: {', '.join(data.columns.tolist()[:10])}{'...' if len(data.columns) > 10 else ''}\n"
        
        # Numerical columns summary
        numeric_cols = data.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            context += f"Numerical columns: {', '.join(numeric_cols[:5])}{'...' if len(numeric_cols) > 5 else ''}\n"
        
        # Categorical columns
        cat_cols = data.select_dtypes(include=['object']).columns
        if len(cat_cols) > 0:
            context += f"Categorical columns: {', '.join(cat_cols[:5])}{'...' if len(cat_cols) > 5 else ''}\n"
        
        # Missing values
        missing = data.isnull().sum()
        if missing.sum() > 0:
            missing_cols = missing[missing > 0].head(3)
            context += f"Missing values in: {', '.join([f'{col} ({count})' for col, count in missing_cols.items()])}\n"
        
        return context
