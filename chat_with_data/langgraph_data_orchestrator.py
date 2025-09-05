"""
LangGraph-based Data Analysis Orchestrator.
Coordinates specialized data analysis agents in a structured workflow.
"""
import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from langchain_core.messages import SystemMessage, HumanMessage

try:
    from langgraph.graph import StateGraph, END
    LANGGRAPH_AVAILABLE = True
except ImportError:
    # Fallback for environments without langgraph
    StateGraph = None
    END = None
    LANGGRAPH_AVAILABLE = False

from .data_agent_schemas import (
    DataAgentRole, DataAnalysisPhase, DataAnalysisState, 
    DataAnalysisRequest, DataAgentResponse
)
from .data_analysis_agents import create_data_analysis_agents
from .data_processor import DataProcessor, DataVisualizer, MLModelRecommender
from .vectorstore_manager import DataVectorStoreManager

logger = logging.getLogger(__name__)

class LangGraphDataAnalysisOrchestrator:
    """LangGraph-based orchestrator for comprehensive data analysis."""
    
    def __init__(self, llm, embedder):
        self.llm = llm
        self.embedder = embedder
        self.agents = create_data_analysis_agents(llm)
        self.data_processor = DataProcessor()
        self.vectorstore_manager = DataVectorStoreManager(embedder)
        self.vectorstore_manager.load_existing_vectorstore()
        
        # Agent execution order
        self.analysis_workflow = [
            DataAgentRole.DATA_PROFILER,
            DataAgentRole.STATISTICAL_ANALYST,
            DataAgentRole.VISUALIZATION_SPECIALIST,
            DataAgentRole.ML_ADVISOR,
            DataAgentRole.INSIGHTS_GENERATOR,
            DataAgentRole.REPORT_WRITER
        ]
        
        # Build workflow graph if LangGraph is available
        self.workflow_graph = self._build_analysis_graph() if LANGGRAPH_AVAILABLE else None
    
    def analyze_dataset(self, request: DataAnalysisRequest) -> Dict[str, Any]:
        """Perform comprehensive data analysis using the agent workflow."""
        try:
            # Load and process data
            data = self.data_processor.load_data(request.file_path)
            data_summary = self.data_processor.analyze_data_structure()
            
            # Add data to vectorstore
            file_info = {"filename": request.filename}
            self.vectorstore_manager.add_data_to_vectorstore(data, file_info)
            
            if self.workflow_graph:
                # Use LangGraph workflow
                return self._run_langgraph_analysis(request, data_summary)
            else:
                # Use sequential workflow
                return self._run_sequential_analysis(request, data_summary)
                
        except Exception as e:
            logger.error(f"Error in data analysis: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "report": f"Data analysis failed: {str(e)}"
            }
    
    def _run_langgraph_analysis(self, request: DataAnalysisRequest, data_summary: Dict[str, Any]) -> Dict[str, Any]:
        """Run analysis using LangGraph workflow."""
        try:
            # Initialize state
            initial_state = DataAnalysisState(
                messages=[HumanMessage(content=f"Analyzing dataset: {request.filename}")],
                request=request,
                current_phase=DataAnalysisPhase.INITIALIZATION,
                active_agent=None,
                agent_outputs={},
                completed_agents=[],
                data_summary=data_summary.__dict__ if hasattr(data_summary, '__dict__') else data_summary,
                analysis_context={
                    "user_context": request.user_context or "",
                    "problem_statement": request.problem_statement or "",
                    "analysis_goals": request.analysis_goals or []
                },
                visualizations=[],
                recommendations={},
                insights=[],
                final_report="",
                error=None,
                iteration_count=0
            )
            
            # Execute workflow
            final_state = self.workflow_graph.invoke(initial_state)
            
            if final_state.get("error"):
                raise RuntimeError(final_state["error"])
            
            return {
                "success": True,
                "report": final_state["final_report"],
                "visualizations": final_state["visualizations"],
                "recommendations": final_state["recommendations"],
                "insights": final_state["insights"],
                "agent_outputs": {
                    role: output.content 
                    for role, output in final_state["agent_outputs"].items()
                },
                "analysis_summary": self._generate_analysis_summary(final_state)
            }
            
        except Exception as e:
            logger.error(f"Error in LangGraph analysis: {str(e)}")
            raise
    
    def _run_sequential_analysis(self, request: DataAnalysisRequest, data_summary: Dict[str, Any]) -> Dict[str, Any]:
        """Run analysis using sequential workflow (fallback)."""
        try:
            agent_outputs = {}
            previous_outputs = []
            visualizations = []
            recommendations = {}
            insights = []
            
            # Build initial context
            context = self._build_analysis_context(request, data_summary)
            
            # Execute agents sequentially
            for agent_role in self.analysis_workflow:
                logger.info(f"Executing {agent_role.value} agent...")
                
                agent = self.agents[agent_role]
                response = agent.process(context, data_summary, previous_outputs)
                
                agent_outputs[agent_role.value] = response
                previous_outputs.append(response.content)
                
                # Collect outputs
                if response.charts:
                    visualizations.extend(response.charts)
                if response.recommendations:
                    recommendations[agent_role.value] = response.recommendations
                if "insight" in response.content.lower():
                    insights.extend(self._extract_insights(response.content))
            
            # Generate final report
            final_report = self._compile_final_report(agent_outputs, request, data_summary)
            
            return {
                "success": True,
                "report": final_report,
                "visualizations": visualizations,
                "recommendations": recommendations,
                "insights": insights,
                "agent_outputs": {
                    role: output.content 
                    for role, output in agent_outputs.items()
                },
                "analysis_summary": self._generate_sequential_summary(agent_outputs)
            }
            
        except Exception as e:
            logger.error(f"Error in sequential analysis: {str(e)}")
            raise
    
    def _build_analysis_graph(self) -> Optional[Any]:
        """Build the LangGraph workflow for data analysis."""
        if not LANGGRAPH_AVAILABLE:
            return None
            
        try:
            workflow = StateGraph(DataAnalysisState)
            
            # Add nodes for each phase
            workflow.add_node("initialize", self._initialize_analysis)
            workflow.add_node("profile_data", self._profile_data)
            workflow.add_node("statistical_analysis", self._statistical_analysis)
            workflow.add_node("create_visualizations", self._create_visualizations)
            workflow.add_node("ml_recommendations", self._ml_recommendations)
            workflow.add_node("generate_insights", self._generate_insights)
            workflow.add_node("write_report", self._write_report)
            workflow.add_node("finalize", self._finalize_analysis)
            
            # Set up workflow edges
            workflow.set_entry_point("initialize")
            workflow.add_edge("initialize", "profile_data")
            workflow.add_edge("profile_data", "statistical_analysis")
            workflow.add_edge("statistical_analysis", "create_visualizations")
            workflow.add_edge("create_visualizations", "ml_recommendations")
            workflow.add_edge("ml_recommendations", "generate_insights")
            workflow.add_edge("generate_insights", "write_report")
            workflow.add_edge("write_report", "finalize")
            workflow.add_edge("finalize", END)
            
            return workflow.compile()
            
        except Exception as e:
            logger.error(f"Error building analysis graph: {str(e)}")
            return None
    
    # LangGraph node functions
    def _initialize_analysis(self, state: DataAnalysisState) -> DataAnalysisState:
        """Initialize the data analysis workflow."""
        state["current_phase"] = DataAnalysisPhase.INITIALIZATION
        state["messages"].append(SystemMessage(content="Starting comprehensive data analysis..."))
        state["analysis_context"]["start_time"] = datetime.now().isoformat()
        return state
    
    def _profile_data(self, state: DataAnalysisState) -> DataAnalysisState:
        """Execute data profiling phase."""
        try:
            state["current_phase"] = DataAnalysisPhase.DATA_PROFILING
            state["active_agent"] = DataAgentRole.DATA_PROFILER
            
            context = self._build_analysis_context(state["request"], state["data_summary"])
            agent = self.agents[DataAgentRole.DATA_PROFILER]
            response = agent.process(context, state["data_summary"])
            
            state["agent_outputs"][DataAgentRole.DATA_PROFILER.value] = response
            state["completed_agents"].append(DataAgentRole.DATA_PROFILER)
            
        except Exception as e:
            state["error"] = f"Error in data profiling: {str(e)}"
        
        return state
    
    def _statistical_analysis(self, state: DataAnalysisState) -> DataAnalysisState:
        """Execute statistical analysis phase."""
        try:
            state["current_phase"] = DataAnalysisPhase.STATISTICAL_ANALYSIS
            state["active_agent"] = DataAgentRole.STATISTICAL_ANALYST
            
            context = self._build_analysis_context(state["request"], state["data_summary"])
            previous_outputs = [
                output.content for output in state["agent_outputs"].values()
            ]
            
            agent = self.agents[DataAgentRole.STATISTICAL_ANALYST]
            response = agent.process(context, state["data_summary"], previous_outputs)
            
            state["agent_outputs"][DataAgentRole.STATISTICAL_ANALYST.value] = response
            state["completed_agents"].append(DataAgentRole.STATISTICAL_ANALYST)
            
        except Exception as e:
            state["error"] = f"Error in statistical analysis: {str(e)}"
        
        return state
    
    def _create_visualizations(self, state: DataAnalysisState) -> DataAnalysisState:
        """Execute visualization creation phase."""
        try:
            state["current_phase"] = DataAnalysisPhase.VISUALIZATION
            state["active_agent"] = DataAgentRole.VISUALIZATION_SPECIALIST
            
            context = self._build_analysis_context(state["request"], state["data_summary"])
            previous_outputs = [
                output.content for output in state["agent_outputs"].values()
            ]
            
            agent = self.agents[DataAgentRole.VISUALIZATION_SPECIALIST]
            response = agent.process(context, state["data_summary"], previous_outputs)
            
            # Create actual visualizations
            visualizer = DataVisualizer(self.data_processor.data)
            plots = visualizer.create_overview_plots()
            state["visualizations"].extend(plots)
            
            state["agent_outputs"][DataAgentRole.VISUALIZATION_SPECIALIST.value] = response
            state["completed_agents"].append(DataAgentRole.VISUALIZATION_SPECIALIST)
            
        except Exception as e:
            state["error"] = f"Error in visualization: {str(e)}"
        
        return state
    
    def _ml_recommendations(self, state: DataAnalysisState) -> DataAnalysisState:
        """Execute ML recommendations phase."""
        try:
            state["current_phase"] = DataAnalysisPhase.ML_RECOMMENDATIONS
            state["active_agent"] = DataAgentRole.ML_ADVISOR
            
            context = self._build_analysis_context(state["request"], state["data_summary"])
            previous_outputs = [
                output.content for output in state["agent_outputs"].values()
            ]
            
            agent = self.agents[DataAgentRole.ML_ADVISOR]
            response = agent.process(context, state["data_summary"], previous_outputs)
            
            # Get ML recommendations
            analysis_type = self.data_processor.detect_analysis_type(
                state["request"].user_context,
                state["request"].target_variable
            )
            
            recommender = MLModelRecommender(state["data_summary"], analysis_type)
            ml_models = recommender.recommend_models()
            ml_metrics = recommender.recommend_metrics()
            
            state["recommendations"]["ml_models"] = ml_models
            state["recommendations"]["ml_metrics"] = ml_metrics
            
            state["agent_outputs"][DataAgentRole.ML_ADVISOR.value] = response
            state["completed_agents"].append(DataAgentRole.ML_ADVISOR)
            
        except Exception as e:
            state["error"] = f"Error in ML recommendations: {str(e)}"
        
        return state
    
    def _generate_insights(self, state: DataAnalysisState) -> DataAnalysisState:
        """Execute insights generation phase."""
        try:
            state["current_phase"] = DataAnalysisPhase.INSIGHTS_GENERATION
            state["active_agent"] = DataAgentRole.INSIGHTS_GENERATOR
            
            context = self._build_analysis_context(state["request"], state["data_summary"])
            previous_outputs = [
                output.content for output in state["agent_outputs"].values()
            ]
            
            agent = self.agents[DataAgentRole.INSIGHTS_GENERATOR]
            response = agent.process(context, state["data_summary"], previous_outputs)
            
            # Extract insights
            insights = self._extract_insights(response.content)
            state["insights"].extend(insights)
            
            state["agent_outputs"][DataAgentRole.INSIGHTS_GENERATOR.value] = response
            state["completed_agents"].append(DataAgentRole.INSIGHTS_GENERATOR)
            
        except Exception as e:
            state["error"] = f"Error in insights generation: {str(e)}"
        
        return state
    
    def _write_report(self, state: DataAnalysisState) -> DataAnalysisState:
        """Execute report writing phase."""
        try:
            state["current_phase"] = DataAnalysisPhase.REPORT_GENERATION
            state["active_agent"] = DataAgentRole.REPORT_WRITER
            
            context = self._build_analysis_context(state["request"], state["data_summary"])
            previous_outputs = [
                output.content for output in state["agent_outputs"].values()
            ]
            
            agent = self.agents[DataAgentRole.REPORT_WRITER]
            response = agent.process(context, state["data_summary"], previous_outputs)
            
            state["final_report"] = response.content
            state["agent_outputs"][DataAgentRole.REPORT_WRITER.value] = response
            state["completed_agents"].append(DataAgentRole.REPORT_WRITER)
            
        except Exception as e:
            state["error"] = f"Error in report writing: {str(e)}"
        
        return state
    
    def _finalize_analysis(self, state: DataAnalysisState) -> DataAnalysisState:
        """Finalize the analysis workflow."""
        state["current_phase"] = DataAnalysisPhase.FINALIZATION
        state["analysis_context"]["end_time"] = datetime.now().isoformat()
        state["messages"].append(SystemMessage(content="Data analysis completed successfully"))
        return state
    
    def _build_analysis_context(self, request: DataAnalysisRequest, data_summary: Dict[str, Any]) -> str:
        """Build comprehensive context for analysis."""
        context = f"""
Dataset Analysis Request:
- Filename: {request.filename}
- User Context: {request.user_context or 'Not provided'}
- Problem Statement: {request.problem_statement or 'General analysis requested'}
- Target Variable: {request.target_variable or 'Not specified'}
- Analysis Goals: {', '.join(request.analysis_goals) if request.analysis_goals else 'Comprehensive analysis'}

Dataset Overview:
- Shape: {data_summary.get('shape', 'Unknown')}
- Columns: {len(data_summary.get('columns', []))} total columns
- Data Types: Numerical: {len(data_summary.get('numerical_columns', []))}, Categorical: {len(data_summary.get('categorical_columns', []))}
- Missing Values: {sum(data_summary.get('missing_values', {}).values())} total missing
"""
        return context.strip()
    
    def _extract_insights(self, content: str) -> List[str]:
        """Extract insights from agent content."""
        insights = []
        lines = content.split('\n')
        
        for line in lines:
            line = line.strip()
            if any(keyword in line.lower() for keyword in ['insight:', 'finding:', 'discovery:', 'key point:']):
                insight = line.split(':', 1)[-1].strip()
                if insight:
                    insights.append(insight)
            elif line.startswith(('-', '•', '*')) and len(line) > 10:
                insights.append(line[1:].strip())
        
        return insights[:10]  # Limit to top 10 insights
    
    def _compile_final_report(self, agent_outputs: Dict[str, DataAgentResponse], request: DataAnalysisRequest, data_summary: Dict[str, Any]) -> str:
        """Compile final analysis report from all agent outputs."""
        report = f"""# 📊 Comprehensive Data Analysis Report

**Dataset:** {request.filename}
**Analysis Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Generated by:** Multi-Agent Data Analysis System

## 📋 Executive Summary

This comprehensive analysis was conducted by a team of specialized AI agents, each contributing their expertise to provide thorough insights into your dataset.

**Dataset Overview:**
- Shape: {data_summary.get('shape', 'Unknown')}
- Columns: {len(data_summary.get('columns', []))}
- Analysis Scope: {'User-defined: ' + request.user_context if request.user_context else 'Comprehensive exploratory analysis'}

---

"""
        
        # Add each agent's analysis
        agent_sections = {
            DataAgentRole.DATA_PROFILER.value: "## 🔍 Data Profiling & Quality Assessment",
            DataAgentRole.STATISTICAL_ANALYST.value: "## 📈 Statistical Analysis",
            DataAgentRole.VISUALIZATION_SPECIALIST.value: "## 📊 Visualization Strategy", 
            DataAgentRole.ML_ADVISOR.value: "## 🤖 Machine Learning Recommendations",
            DataAgentRole.INSIGHTS_GENERATOR.value: "## 💡 Key Insights & Business Value",
            DataAgentRole.REPORT_WRITER.value: "## 📝 Technical Documentation"
        }
        
        for role, response in agent_outputs.items():
            section_title = agent_sections.get(role, f"## {role.replace('_', ' ').title()}")
            report += f"{section_title}\n\n"
            report += f"{response.content}\n\n"
            report += "---\n\n"
        
        report += """## 🎯 Summary & Next Steps

This analysis provides a comprehensive foundation for data-driven decision making. The specialized agents have identified key patterns, quality issues, and opportunities within your dataset.

**Recommended Actions:**
1. Address data quality issues identified in the profiling phase
2. Implement suggested visualizations for ongoing monitoring
3. Consider the ML approaches recommended for your specific use case
4. Leverage the key insights for strategic planning

For questions or follow-up analysis, please provide specific queries about any aspect of this report.
"""
        
        return report
    
    def _generate_analysis_summary(self, state: DataAnalysisState) -> str:
        """Generate analysis summary from final state."""
        summary = f"Analysis completed with {len(state['completed_agents'])} specialized agents. "
        summary += f"Generated {len(state['visualizations'])} visualizations and {len(state['insights'])} key insights. "
        summary += f"Analysis phases: {[phase.value for phase in [agent._get_phase_for_role() for agent in self.agents.values()]]}"
        return summary
    
    def _generate_sequential_summary(self, agent_outputs: Dict[str, DataAgentResponse]) -> str:
        """Generate summary for sequential analysis."""
        return f"Sequential analysis completed with {len(agent_outputs)} agents: {', '.join(agent_outputs.keys())}"

# Interface functions
def create_data_analysis_orchestrator(llm, embedder):
    """Create a data analysis orchestrator."""
    return LangGraphDataAnalysisOrchestrator(llm, embedder)

def run_comprehensive_data_analysis(llm, embedder, file_path: str, filename: str, 
                                   user_context: str = None, problem_statement: str = None) -> Dict[str, Any]:
    """Run comprehensive data analysis."""
    orchestrator = LangGraphDataAnalysisOrchestrator(llm, embedder)
    
    request = DataAnalysisRequest(
        file_path=file_path,
        filename=filename,
        user_context=user_context,
        problem_statement=problem_statement
    )
    
    return orchestrator.analyze_dataset(request)
