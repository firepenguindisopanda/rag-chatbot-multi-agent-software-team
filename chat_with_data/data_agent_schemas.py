from enum import Enum
from pydantic import BaseModel
from typing import Dict, List, Optional, Any, TypedDict, Annotated
from langchain_core.messages import BaseMessage

class DataAgentRole(Enum):
    """Specialized data analysis agent roles."""
    DATA_PROFILER = "data_profiler"
    STATISTICAL_ANALYST = "statistical_analyst"
    VISUALIZATION_SPECIALIST = "visualization_specialist"
    ML_ADVISOR = "ml_advisor"
    INSIGHTS_GENERATOR = "insights_generator"
    REPORT_WRITER = "report_writer"

class DataAnalysisPhase(Enum):
    """Phases of data analysis workflow."""
    INITIALIZATION = "initialization"
    DATA_PROFILING = "data_profiling"
    STATISTICAL_ANALYSIS = "statistical_analysis"
    VISUALIZATION = "visualization"
    ML_RECOMMENDATIONS = "ml_recommendations"
    INSIGHTS_GENERATION = "insights_generation"
    REPORT_GENERATION = "report_generation"
    FINALIZATION = "finalization"

class DataAgentResponse(BaseModel):
    """Response from a data analysis agent."""
    role: DataAgentRole
    phase: DataAnalysisPhase
    content: str
    charts: List[str] = []  # Base64 encoded charts
    recommendations: List[str] = []
    next_steps: List[str] = []
    confidence: float = 0.8
    metadata: Dict[str, Any] = {}

class DataAnalysisRequest(BaseModel):
    """Request for comprehensive data analysis."""
    file_path: str
    filename: str
    user_context: Optional[str] = None
    problem_statement: Optional[str] = None
    target_variable: Optional[str] = None
    analysis_goals: List[str] = []
    preferred_visualizations: List[str] = []

class DataAnalysisState(TypedDict):
    """State for data analysis workflow."""
    messages: List[BaseMessage]
    request: DataAnalysisRequest
    current_phase: DataAnalysisPhase
    active_agent: Optional[DataAgentRole]
    agent_outputs: Dict[str, DataAgentResponse]
    completed_agents: List[DataAgentRole]
    data_summary: Optional[Dict[str, Any]]
    analysis_context: Dict[str, Any]
    visualizations: List[str]
    recommendations: Dict[str, List[str]]
    insights: List[str]
    final_report: str
    error: Optional[str]
    iteration_count: int

class DataChatState(TypedDict):
    """State for data chat interactions."""
    messages: List[BaseMessage]
    query: str
    context: str
    response: str
    vectorstore_results: List[str]
    data_context: Optional[str]
    pandas_results: Optional[str]
    chart_suggestions: List[str]

class AgentCapabilities(BaseModel):
    """Capabilities of each data analysis agent."""
    role: DataAgentRole
    name: str
    description: str
    specializations: List[str]
    deliverables: List[str]
    dependencies: List[DataAgentRole]
