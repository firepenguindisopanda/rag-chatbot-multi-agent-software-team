"""
Integration module for enhanced LangGraph agents with CSV/Excel vectorstore.
This module provides a unified interface for both data analysis and software team collaboration.
"""

import logging
from typing import Dict, Any, Optional
from .enhanced_langgraph_agents import EnhancedLangGraphDataAnalysisAgent, EnhancedLangGraphDataChatAgent
from multi_agent_software_team.enhanced_langgraph_orchestrator import EnhancedLangGraphSoftwareTeamOrchestrator
from .vectorstore_manager import DataVectorStoreManager
from .data_processor import DataProcessor
from .schemas import DataRequest, AnalysisResult

logger = logging.getLogger(__name__)

class IntegratedLangGraphSystem:
    """
    Integrated system that combines enhanced data analysis and software team collaboration
    using LangGraph workflows with proper CSV/Excel vectorstore separation.
    """
    
    def __init__(self, llm, embedder):
        self.llm = llm
        self.embedder = embedder
        
        # Initialize data analysis components
        self.data_analysis_agent = EnhancedLangGraphDataAnalysisAgent(llm, embedder)
        self.data_processor = DataProcessor()
        self.data_vectorstore_manager = DataVectorStoreManager(embedder)
        self.data_vectorstore_manager.load_existing_vectorstore()
        
        # Initialize software team components
        self.software_team_orchestrator = EnhancedLangGraphSoftwareTeamOrchestrator(llm)
        
        # Chat agent will be initialized after data is loaded
        self.data_chat_agent = None
        
        logger.info("Integrated LangGraph system initialized with separate CSV/Excel vectorstore")
    
    def analyze_data(self, data_request: DataRequest) -> AnalysisResult:
        """
        Analyze CSV/Excel data using enhanced LangGraph agents.
        Data is stored in a separate vectorstore from PDF documents.
        """
        try:
            # Perform analysis using enhanced LangGraph workflow
            result = self.data_analysis_agent.analyze_dataset(data_request)
            
            # Initialize chat agent with the loaded data
            if self.data_analysis_agent.data_processor.data is not None:
                self.data_chat_agent = EnhancedLangGraphDataChatAgent(
                    self.llm, 
                    self.embedder, 
                    self.data_analysis_agent.data_processor
                )
                logger.info("Data chat agent initialized with loaded dataset")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in integrated data analysis: {str(e)}")
            raise
    
    def chat_with_data(self, question: str) -> str:
        """
        Chat about the loaded data using enhanced LangGraph chat agent.
        Uses both the CSV/Excel vectorstore and pandas agent for comprehensive responses.
        """
        if self.data_chat_agent is None:
            return "Please upload and analyze a dataset first before asking questions."
        
        try:
            return self.data_chat_agent.answer_question(question)
        except Exception as e:
            logger.error(f"Error in data chat: {str(e)}")
            return f"Sorry, I encountered an error while processing your question: {str(e)}"
    
    def collaborate_on_software_project(self, project_request) -> Dict[str, Any]:
        """
        Execute software team collaboration using enhanced LangGraph orchestrator.
        Provides stateful, multi-phase collaboration with human-in-the-loop capabilities.
        """
        try:
            return self.software_team_orchestrator.collaborate_on_project(project_request)
        except Exception as e:
            logger.error(f"Error in software team collaboration: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "deliverables": {},
                "agent_outputs": {}
            }
    
    def get_data_vectorstore_status(self) -> Dict[str, Any]:
        """Get status of the CSV/Excel vectorstore (separate from PDF docstore)."""
        try:
            if self.data_vectorstore_manager.vectorstore is not None:
                # Try to get some basic stats about the vectorstore
                return {
                    "status": "active",
                    "type": "csv_excel_data",
                    "message": "CSV/Excel vectorstore is loaded and ready"
                }
            else:
                return {
                    "status": "empty",
                    "type": "csv_excel_data", 
                    "message": "No CSV/Excel data has been uploaded yet"
                }
        except Exception as e:
            return {
                "status": "error",
                "type": "csv_excel_data",
                "message": f"Error checking vectorstore status: {str(e)}"
            }
    
    def get_system_summary(self) -> str:
        """Get a summary of the integrated system status."""
        data_status = self.get_data_vectorstore_status()
        
        summary = f"""
🔧 **Integrated LangGraph System Status**

**Data Analysis Components:**
- ✅ Enhanced LangGraph Data Analysis Agent: Ready
- ✅ CSV/Excel Vectorstore: {data_status['status'].title()}
- {'✅' if self.data_chat_agent else '⏳'} Data Chat Agent: {'Ready' if self.data_chat_agent else 'Waiting for data upload'}

**Software Team Components:**
- ✅ Enhanced LangGraph Software Team Orchestrator: Ready
- ✅ Multi-Phase Collaboration Workflow: Active
- ✅ Human-in-the-Loop Capabilities: Enabled

**Key Features:**
- 🗂️ **Separate Vectorstores**: CSV/Excel data stored separately from PDF documents
- 🤖 **LangGraph Workflows**: State-driven, sophisticated agent collaboration
- 💬 **Enhanced Chat**: Combines vectorstore RAG with pandas agent capabilities
- 🔄 **Stateful Collaboration**: Multi-phase software team workflow
- 📊 **Comprehensive Analysis**: Advanced data analysis with ML recommendations

**Data Vectorstore Status:** {data_status['message']}
"""
        return summary.strip()

# Factory function for easy initialization
def create_integrated_system(llm, embedder) -> IntegratedLangGraphSystem:
    """Factory function to create an integrated LangGraph system."""
    return IntegratedLangGraphSystem(llm, embedder)
