"""
Enhanced LangGraph-based Data Chat Agent.
Provides intelligent chat functionality for data analysis with RAG and pandas integration.
"""
import logging
from typing import Dict, List, Any, Optional
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

try:
    from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
    from langchain_experimental.tools import PythonREPLTool
    PANDAS_AGENT_AVAILABLE = True
except ImportError:
    create_pandas_dataframe_agent = None
    PythonREPLTool = None
    PANDAS_AGENT_AVAILABLE = False

try:
    from langgraph.graph import StateGraph, END
    LANGGRAPH_AVAILABLE = True
except ImportError:
    StateGraph = None
    END = None
    LANGGRAPH_AVAILABLE = False

from .data_agent_schemas import DataChatState
from .data_processor import DataProcessor
from .vectorstore_manager import DataVectorStoreManager

logger = logging.getLogger(__name__)

class LangGraphDataChatAgent:
    """Enhanced LangGraph-based agent for intelligent data conversations."""
    
    def __init__(self, llm, embedder, data_processor: DataProcessor):
        self.llm = llm
        self.embedder = embedder
        self.data_processor = data_processor
        self.vectorstore_manager = DataVectorStoreManager(embedder)
        self.vectorstore_manager.load_existing_vectorstore()
        self.pandas_agent = None
        self.conversation_history = []
        
        # Build chat workflow if LangGraph is available
        self.chat_graph = self._build_chat_graph() if LANGGRAPH_AVAILABLE else None
    
    def answer_question(self, question: str) -> str:
        """Answer questions about the loaded dataset."""
        try:
            if self.data_processor.data is None:
                return "No dataset is currently loaded. Please upload a CSV or Excel file first."
            
            # Initialize pandas agent if not done
            if self.pandas_agent is None:
                self._initialize_pandas_agent()
            
            if self.chat_graph:
                # Use LangGraph workflow
                return self._run_langgraph_chat(question)
            else:
                # Use direct approach
                return self._run_direct_chat(question)
                
        except Exception as e:
            logger.error(f"Error in data chat: {str(e)}")
            return f"I encountered an error while processing your question: {str(e)}"
    
    def _initialize_pandas_agent(self):
        """Initialize the pandas DataFrame agent."""
        try:
            if self.data_processor.data is not None and PANDAS_AGENT_AVAILABLE:
                self.pandas_agent = create_pandas_dataframe_agent(
                    self.llm,
                    self.data_processor.data,
                    verbose=False,
                    allow_dangerous_code=True,
                    return_intermediate_steps=True
                )
        except Exception as e:
            logger.error(f"Error initializing pandas agent: {str(e)}")
            self.pandas_agent = None
    
    def _run_langgraph_chat(self, question: str) -> str:
        """Run chat using LangGraph workflow."""
        try:
            initial_state = DataChatState(
                messages=[HumanMessage(content=question)],
                query=question,
                context="",
                response="",
                vectorstore_results=[],
                data_context=None,
                pandas_results=None,
                chart_suggestions=[]
            )
            
            final_state = self.chat_graph.invoke(initial_state)
            return final_state["response"]
            
        except Exception as e:
            logger.error(f"Error in LangGraph chat: {str(e)}")
            return f"Error processing your question: {str(e)}"
    
    def _run_direct_chat(self, question: str) -> str:
        """Run chat using direct approach (fallback)."""
        try:
            # Search vectorstore for relevant context
            vectorstore_results = self.vectorstore_manager.search_data_context(question, k=3)
            
            # Get data context
            data_context = self._get_data_context()
            
            # Try pandas agent for computational queries
            pandas_result = None
            if self._is_computational_query(question):
                pandas_result = self._query_pandas_agent(question)
            
            # Generate comprehensive response
            response = self._generate_enhanced_response(
                question, vectorstore_results, data_context, pandas_result
            )
            
            # Update conversation history
            self.conversation_history.append({"question": question, "answer": response})
            
            return response
            
        except Exception as e:
            logger.error(f"Error in direct chat: {str(e)}")
            return f"Error processing your question: {str(e)}"
    
    def _build_chat_graph(self) -> Optional[Any]:
        """Build the LangGraph workflow for data chat."""
        if not LANGGRAPH_AVAILABLE:
            return None
        
        try:
            workflow = StateGraph(DataChatState)
            
            # Add nodes
            workflow.add_node("analyze_query", self._analyze_query)
            workflow.add_node("search_vectorstore", self._search_vectorstore)
            workflow.add_node("get_data_context", self._get_data_context_node)
            workflow.add_node("query_pandas", self._query_pandas_node)
            workflow.add_node("generate_response", self._generate_response_node)
            
            # Set up edges
            workflow.set_entry_point("analyze_query")
            workflow.add_edge("analyze_query", "search_vectorstore")
            workflow.add_edge("search_vectorstore", "get_data_context")
            workflow.add_edge("get_data_context", "query_pandas")
            workflow.add_edge("query_pandas", "generate_response")
            workflow.add_edge("generate_response", END)
            
            return workflow.compile()
            
        except Exception as e:
            logger.error(f"Error building chat graph: {str(e)}")
            return None
    
    # LangGraph node functions
    def _analyze_query(self, state: DataChatState) -> DataChatState:
        """Analyze the user query to understand intent."""
        query = state["query"]
        
        # Determine query type and add to context
        if self._is_computational_query(query):
            state["context"] += "Computational query detected. "
        if self._is_visualization_query(query):
            state["context"] += "Visualization request detected. "
        if self._is_statistical_query(query):
            state["context"] += "Statistical analysis requested. "
        
        return state
    
    def _search_vectorstore(self, state: DataChatState) -> DataChatState:
        """Search vectorstore for relevant context."""
        try:
            docs = self.vectorstore_manager.search_data_context(state["query"], k=3)
            state["vectorstore_results"] = [doc.page_content for doc in docs]
        except Exception as e:
            logger.error(f"Error searching vectorstore: {str(e)}")
            state["vectorstore_results"] = []
        
        return state
    
    def _get_data_context_node(self, state: DataChatState) -> DataChatState:
        """Get current data context."""
        state["data_context"] = self._get_data_context()
        return state
    
    def _query_pandas_node(self, state: DataChatState) -> DataChatState:
        """Query pandas agent if appropriate."""
        if self._is_computational_query(state["query"]) and self.pandas_agent:
            try:
                state["pandas_results"] = self._query_pandas_agent(state["query"])
            except Exception as e:
                logger.error(f"Error in pandas query: {str(e)}")
                state["pandas_results"] = f"Error in computation: {str(e)}"
        
        return state
    
    def _generate_response_node(self, state: DataChatState) -> DataChatState:
        """Generate the final response."""
        state["response"] = self._generate_enhanced_response(
            state["query"],
            [{"page_content": content} for content in state["vectorstore_results"]],
            state["data_context"],
            state["pandas_results"]
        )
        return state
    
    def _get_data_context(self) -> str:
        """Get relevant context about the current dataset."""
        if self.data_processor.data is None:
            return "No dataset loaded."
        
        data = self.data_processor.data
        
        context = "Dataset Overview:\n"
        context += f"- Shape: {data.shape[0]} rows, {data.shape[1]} columns\n"
        context += f"- Columns: {', '.join(data.columns.tolist()[:8])}{'...' if len(data.columns) > 8 else ''}\n"
        
        # Numerical columns summary
        numeric_cols = data.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            context += f"- Numerical columns ({len(numeric_cols)}): {', '.join(numeric_cols[:5])}{'...' if len(numeric_cols) > 5 else ''}\n"
        
        # Categorical columns
        cat_cols = data.select_dtypes(include=['object']).columns
        if len(cat_cols) > 0:
            context += f"- Categorical columns ({len(cat_cols)}): {', '.join(cat_cols[:5])}{'...' if len(cat_cols) > 5 else ''}\n"
        
        # Missing values
        missing = data.isnull().sum()
        if missing.sum() > 0:
            missing_cols = missing[missing > 0].head(3)
            context += f"- Missing values in: {', '.join([f'{col} ({count})' for col, count in missing_cols.items()])}\n"
        
        return context
    
    def _is_computational_query(self, query: str) -> bool:
        """Check if query requires computational analysis."""
        computational_keywords = [
            'calculate', 'compute', 'sum', 'average', 'mean', 'median', 'count',
            'maximum', 'minimum', 'correlation', 'percentage', 'ratio', 'total',
            'statistics', 'std', 'variance', 'quantile', 'percentile'
        ]
        return any(keyword in query.lower() for keyword in computational_keywords)
    
    def _is_visualization_query(self, query: str) -> bool:
        """Check if query is asking for visualizations."""
        viz_keywords = [
            'plot', 'chart', 'graph', 'visualize', 'show', 'display',
            'histogram', 'scatter', 'bar chart', 'line plot', 'boxplot'
        ]
        return any(keyword in query.lower() for keyword in viz_keywords)
    
    def _is_statistical_query(self, query: str) -> bool:
        """Check if query requires statistical analysis."""
        stats_keywords = [
            'distribution', 'normal', 'skewed', 'outliers', 'correlation',
            'relationship', 'pattern', 'trend', 'significant', 'test'
        ]
        return any(keyword in query.lower() for keyword in stats_keywords)
    
    def _query_pandas_agent(self, question: str) -> str:
        """Query the pandas agent for computational results."""
        if not self.pandas_agent:
            return "Pandas agent not available for computation."
        
        try:
            result = self.pandas_agent.invoke({"input": question})
            if isinstance(result, dict) and "output" in result:
                return result["output"]
            else:
                return str(result)
        except Exception as e:
            logger.error(f"Error in pandas agent query: {str(e)}")
            return f"Computation error: {str(e)}"
    
    def _generate_enhanced_response(self, question: str, vectorstore_results: List[Dict], 
                                  data_context: str, pandas_result: str = None) -> str:
        """Generate comprehensive response using all available information."""
        
        # Build context from vectorstore results
        vectorstore_context = ""
        if vectorstore_results:
            vectorstore_context = "\n".join([
                doc.get("page_content", "") for doc in vectorstore_results[:3]
            ])
        
        # Build conversation context
        recent_history = ""
        if self.conversation_history:
            recent_history = "\n".join([
                f"Q: {item['question'][:100]}...\nA: {item['answer'][:200]}..."
                for item in self.conversation_history[-2:]
            ])
        
        prompt = ChatPromptTemplate.from_template(
            """You are an expert data analyst assistant. Answer the user's question about their dataset using all available information.

CURRENT DATASET CONTEXT:
{data_context}

RELEVANT DATA INSIGHTS:
{vectorstore_context}

COMPUTATIONAL RESULTS:
{pandas_result}

RECENT CONVERSATION:
{recent_history}

USER QUESTION: {question}

Guidelines:
1. Provide specific, data-driven answers
2. Reference actual data values when available
3. Suggest follow-up questions or analyses
4. If computational results are available, explain them clearly
5. Keep responses concise but informative (max 4-5 sentences)
6. If visualizations would help, suggest specific chart types

Answer:"""
        )
        
        try:
            messages = prompt.format_messages(
                data_context=data_context,
                vectorstore_context=vectorstore_context or "No specific insights found in knowledge base.",
                pandas_result=pandas_result or "No computational analysis performed.",
                recent_history=recent_history or "No previous conversation.",
                question=question
            )
            
            response = self.llm.invoke(messages)
            return response.content
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return f"I encountered an error generating the response: {str(e)}"

# Interface functions
def create_data_chat_agent(llm, embedder, data_processor: DataProcessor):
    """Create an enhanced data chat agent."""
    return LangGraphDataChatAgent(llm, embedder, data_processor)

def chat_with_data(llm, embedder, data_processor: DataProcessor, question: str) -> str:
    """Chat with loaded data."""
    agent = LangGraphDataChatAgent(llm, embedder, data_processor)
    return agent.answer_question(question)
