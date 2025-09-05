"""
Enhanced LangGraph-based Multi-Agent Software Team.
Provides sophisticated team collaboration with state management and human-in-the-loop capabilities.
"""

import logging
import json
from typing import Dict, List, Any, Optional, TypedDict, Annotated, Union
from enum import Enum
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

from .schemas import TeamRole, AgentResponse, AgentMessage, ProjectRequest
from .agents import Agent
from .utils import format_agent_responses, validate_team_composition

logger = logging.getLogger(__name__)

class CollaborationPhase(Enum):
    """Phases of team collaboration."""
    INITIALIZATION = "initialization"
    REQUIREMENTS_ANALYSIS = "requirements_analysis"
    ARCHITECTURE_DESIGN = "architecture_design"
    IMPLEMENTATION_PLANNING = "implementation_planning"
    CODE_REVIEW = "code_review"
    TESTING_STRATEGY = "testing_strategy"
    DEPLOYMENT_PLANNING = "deployment_planning"
    FINALIZATION = "finalization"

class TeamState(TypedDict):
    """Enhanced state for software team collaboration."""
    messages: Annotated[List, add_messages]
    project_description: str
    selected_roles: List[TeamRole]
    current_phase: CollaborationPhase
    agent_outputs: Dict[str, AgentResponse]
    completed_agents: List[TeamRole]
    active_agent: Optional[TeamRole]
    collaboration_context: Dict[str, Any]
    deliverables: Dict[str, str]
    feedback_required: bool
    human_feedback: Optional[str]
    error: Optional[str]
    iteration_count: int

class EnhancedLangGraphSoftwareTeamOrchestrator:
    """Enhanced LangGraph-based orchestrator for multi-agent software team collaboration."""
    
    def __init__(self, llm):
        self.llm = llm
        self.agents = {role: Agent(role, llm) for role in TeamRole}
        self.team_graph = self._build_enhanced_team_graph()
        self.max_iterations = 3
    
    def collaborate_on_project(self, project_request: ProjectRequest) -> Dict[str, Any]:
        """Execute enhanced team collaboration with state management."""
        try:
            initial_state = TeamState(
                messages=[HumanMessage(content=f"Project: {project_request.description}")],
                project_description=project_request.description,
                selected_roles=project_request.selected_roles,
                current_phase=CollaborationPhase.INITIALIZATION,
                agent_outputs={},
                completed_agents=[],
                active_agent=None,
                collaboration_context={
                    "requirements": {},
                    "architecture": {},
                    "implementation": {},
                    "testing": {},
                    "deployment": {}
                },
                deliverables={},
                feedback_required=False,
                human_feedback=None,
                error=None,
                iteration_count=0
            )
            
            # Execute the workflow
            final_state = self.team_graph.invoke(initial_state)
            
            if final_state.get("error"):
                raise Exception(final_state["error"])
            
            return {
                "success": True,
                "status": "success",
                "deliverables": final_state.get("deliverables", {}),
                "agent_outputs": final_state.get("agent_outputs", {}),
                "collaboration_summary": self._generate_collaboration_summary(final_state),
                "phases_completed": [phase.value for phase in self._get_completed_phases(final_state)]
            }
            
        except Exception as e:
            logger.error(f"Error in team collaboration: {str(e)}")
            return {
                "success": False,
                "status": "error",
                "error": str(e),
                "deliverables": {},
                "agent_outputs": {}
            }
    
    def _build_enhanced_team_graph(self) -> StateGraph:
        """Build the enhanced team collaboration workflow graph."""
        workflow = StateGraph(TeamState)
        
        # Add workflow nodes
        workflow.add_node("initialize", self._initialize_collaboration)
        workflow.add_node("validate_team", self._validate_team_composition)
        workflow.add_node("requirements_phase", self._requirements_analysis_phase)
        workflow.add_node("architecture_phase", self._architecture_design_phase)
        workflow.add_node("implementation_phase", self._implementation_planning_phase)
        workflow.add_node("review_phase", self._code_review_phase)
        workflow.add_node("testing_phase", self._testing_strategy_phase)
        workflow.add_node("deployment_phase", self._deployment_planning_phase)
        workflow.add_node("synthesize", self._synthesize_deliverables)
        workflow.add_node("finalize", self._finalize_collaboration)
        
        # Add conditional routing
        workflow.add_node("check_feedback", self._check_feedback_required)
        workflow.add_node("handle_feedback", self._handle_human_feedback)
        
        # Set up workflow edges
        workflow.set_entry_point("initialize")
        workflow.add_edge("initialize", "validate_team")
        workflow.add_edge("validate_team", "requirements_phase")
        workflow.add_edge("requirements_phase", "architecture_phase")
        workflow.add_edge("architecture_phase", "implementation_phase")
        workflow.add_edge("implementation_phase", "review_phase")
        workflow.add_edge("review_phase", "testing_phase")
        workflow.add_edge("testing_phase", "deployment_phase")
        workflow.add_edge("deployment_phase", "check_feedback")
        
        # Conditional edges for feedback loop
        workflow.add_conditional_edges(
            "check_feedback",
            self._should_request_feedback,
            {
                "request_feedback": "handle_feedback",
                "continue": "synthesize"
            }
        )
        workflow.add_edge("handle_feedback", "synthesize")
        workflow.add_edge("synthesize", "finalize")
        workflow.add_edge("finalize", END)
        
        return workflow.compile()
    
    def _initialize_collaboration(self, state: TeamState) -> TeamState:
        """Initialize the enhanced collaboration workflow."""
        state["current_phase"] = CollaborationPhase.INITIALIZATION
        state["messages"].append(SystemMessage(content="Initializing enhanced team collaboration..."))
        
        # Set up collaboration context
        state["collaboration_context"] = {
            "project_scope": state["project_description"],
            "team_size": len(state["selected_roles"]),
            "start_time": "now",
            "requirements": {},
            "architecture": {},
            "implementation": {},
            "testing": {},
            "deployment": {}
        }
        
        return state
    
    def _validate_team_composition(self, state: TeamState) -> TeamState:
        """Validate and optimize team composition."""
        try:
            is_valid, validation_message = validate_team_composition(state["selected_roles"])
            if not is_valid:
                state["error"] = f"Team composition invalid: {validation_message}"
                return state
            
            # Add team optimization note
            state["collaboration_context"]["team_optimization"] = validation_message
            
        except Exception as e:
            state["error"] = f"Error validating team: {str(e)}"
        
        return state
    
    def _requirements_analysis_phase(self, state: TeamState) -> TeamState:
        """Execute requirements analysis phase with multiple agents."""
        state["current_phase"] = CollaborationPhase.REQUIREMENTS_ANALYSIS
        
        # Get requirements from product owner and analyst
        requirements_agents = [role for role in state["selected_roles"] 
                             if role in [TeamRole.PRODUCT_OWNER, TeamRole.ANALYST]]
        
        requirements_context = {}
        
        for agent_role in requirements_agents:
            try:
                agent = self.agents[agent_role]
                
                # Convert context to string format that the process method expects
                context_str = f"Project: {state['project_description']}\nPhase: requirements_analysis"
                previous_outputs = []
                for completed_role in state.get("completed_agents", []):
                    if completed_role.value in state.get("agent_outputs", {}):
                        agent_output = state["agent_outputs"][completed_role.value]
                        previous_outputs.append(AgentMessage(role=completed_role, content=agent_output.output))
                
                response = agent.process(context_str, previous_outputs)
                state["agent_outputs"][agent_role.value] = response
                requirements_context[agent_role.value] = response.output
                
            except Exception as e:
                logger.error(f"Error in requirements phase for {agent_role.value}: {str(e)}")
        
        state["collaboration_context"]["requirements"] = requirements_context
        state["completed_agents"].extend(requirements_agents)
        
        return state
    
    def _architecture_design_phase(self, state: TeamState) -> TeamState:
        """Execute architecture design phase."""
        state["current_phase"] = CollaborationPhase.ARCHITECTURE_DESIGN
        
        # Get architecture design from solution architect and senior developer
        architecture_agents = [role for role in state["selected_roles"] 
                             if role in [TeamRole.SOLUTION_ARCHITECT, TeamRole.SENIOR_DEVELOPER]]
        
        architecture_context = {}
        
        for agent_role in architecture_agents:
            try:
                agent = self.agents[agent_role]
                
                # Convert context to string format that the process method expects
                context_str = f"Project: {state['project_description']}\nPhase: architecture_design"
                if state["collaboration_context"]["requirements"]:
                    context_str += f"\nRequirements: {str(state['collaboration_context']['requirements'])}"
                
                previous_outputs = []
                for completed_role in state.get("completed_agents", []):
                    if completed_role.value in state.get("agent_outputs", {}):
                        agent_output = state["agent_outputs"][completed_role.value]
                        previous_outputs.append(AgentMessage(role=completed_role, content=agent_output.output))
                
                response = agent.process(context_str, previous_outputs)
                state["agent_outputs"][agent_role.value] = response
                architecture_context[agent_role.value] = response.output
                
            except Exception as e:
                logger.error(f"Error in architecture phase for {agent_role.value}: {str(e)}")
        
        state["collaboration_context"]["architecture"] = architecture_context
        state["completed_agents"].extend(architecture_agents)
        return state
    
    def _implementation_planning_phase(self, state: TeamState) -> TeamState:
        """Execute implementation planning phase."""
        state["current_phase"] = CollaborationPhase.IMPLEMENTATION_PLANNING
        
        # Get implementation plans from developers
        implementation_agents = [role for role in state["selected_roles"] 
                               if role in [TeamRole.DEVELOPER]]
        
        implementation_context = {}
        
        for agent_role in implementation_agents:
            try:
                agent = self.agents[agent_role]
                
                # Convert context to string format that the process method expects
                context_str = f"Project: {state['project_description']}\nPhase: implementation_planning"
                if state["collaboration_context"]["requirements"]:
                    context_str += f"\nRequirements: {str(state['collaboration_context']['requirements'])}"
                if state["collaboration_context"]["architecture"]:
                    context_str += f"\nArchitecture: {str(state['collaboration_context']['architecture'])}"
                
                previous_outputs = []
                for completed_role in state.get("completed_agents", []):
                    if completed_role.value in state.get("agent_outputs", {}):
                        agent_output = state["agent_outputs"][completed_role.value]
                        previous_outputs.append(AgentMessage(role=completed_role, content=agent_output.output))
                
                response = agent.process(context_str, previous_outputs)
                state["agent_outputs"][agent_role.value] = response
                implementation_context[agent_role.value] = response.output
                
            except Exception as e:
                logger.error(f"Error in implementation phase for {agent_role.value}: {str(e)}")
        
        state["collaboration_context"]["implementation"] = implementation_context
        state["completed_agents"].extend(implementation_agents)
        return state
    
    def _code_review_phase(self, state: TeamState) -> TeamState:
        """Execute code review phase."""
        state["current_phase"] = CollaborationPhase.CODE_REVIEW
        
        # Get code review from senior developers and architects
        review_agents = [role for role in state["selected_roles"] 
                        if role in [TeamRole.SENIOR_DEVELOPER, TeamRole.SOLUTION_ARCHITECT]]
        
        for agent_role in review_agents:
            try:
                agent = self.agents[agent_role]
                
                # Convert context to string format that the process method expects
                context_str = f"Project: {state['project_description']}\nPhase: code_review"
                if state["collaboration_context"]["implementation"]:
                    context_str += f"\nImplementation Plans: {str(state['collaboration_context']['implementation'])}"
                
                previous_outputs = []
                for completed_role in state.get("completed_agents", []):
                    if completed_role.value in state.get("agent_outputs", {}):
                        agent_output = state["agent_outputs"][completed_role.value]
                        previous_outputs.append(AgentMessage(role=completed_role, content=agent_output.output))
                
                response = agent.process(context_str, previous_outputs)
                state["agent_outputs"][agent_role.value] = response
                
            except Exception as e:
                logger.error(f"Error in review phase for {agent_role.value}: {str(e)}")
        
        return state
    
    def _testing_strategy_phase(self, state: TeamState) -> TeamState:
        """Execute testing strategy phase."""
        state["current_phase"] = CollaborationPhase.TESTING_STRATEGY
        
        # Get testing strategy from tester and developers
        testing_agents = [role for role in state["selected_roles"] 
                         if role in [TeamRole.TESTER, TeamRole.SENIOR_DEVELOPER]]
        
        testing_context = {}
        
        for agent_role in testing_agents:
            try:
                agent = self.agents[agent_role]
                
                # Convert context to string format that the process method expects
                context_str = f"Project: {state['project_description']}\nPhase: testing_strategy"
                if state["collaboration_context"]["implementation"]:
                    context_str += f"\nImplementation Plans: {str(state['collaboration_context']['implementation'])}"
                
                previous_outputs = []
                for completed_role in state.get("completed_agents", []):
                    if completed_role.value in state.get("agent_outputs", {}):
                        agent_output = state["agent_outputs"][completed_role.value]
                        previous_outputs.append(AgentMessage(role=completed_role, content=agent_output.output))
                
                response = agent.process(context_str, previous_outputs)
                state["agent_outputs"][agent_role.value] = response
                testing_context[agent_role.value] = response.output
                
            except Exception as e:
                logger.error(f"Error in testing phase for {agent_role.value}: {str(e)}")
        
        state["collaboration_context"]["testing"] = testing_context
        state["completed_agents"].extend(testing_agents)
        return state
    
    def _deployment_planning_phase(self, state: TeamState) -> TeamState:
        """Execute deployment planning phase."""
        state["current_phase"] = CollaborationPhase.DEPLOYMENT_PLANNING
        
        # Get deployment strategy from DevOps engineer and solution architect
        deployment_agents = [role for role in state["selected_roles"] 
                           if role in [TeamRole.DEVOPS_ENGINEER, TeamRole.SOLUTION_ARCHITECT]]
        
        deployment_context = {}
        
        for agent_role in deployment_agents:
            try:
                agent = self.agents[agent_role]
                
                # Convert context to string format that the process method expects
                context_str = f"Project: {state['project_description']}\nPhase: deployment_planning"
                if state["collaboration_context"]["architecture"]:
                    context_str += f"\nArchitecture: {str(state['collaboration_context']['architecture'])}"
                if state["collaboration_context"]["testing"]:
                    context_str += f"\nTesting: {str(state['collaboration_context']['testing'])}"
                
                previous_outputs = []
                for completed_role in state.get("completed_agents", []):
                    if completed_role.value in state.get("agent_outputs", {}):
                        agent_output = state["agent_outputs"][completed_role.value]
                        previous_outputs.append(AgentMessage(role=completed_role, content=agent_output.output))
                
                response = agent.process(context_str, previous_outputs)
                state["agent_outputs"][agent_role.value] = response
                deployment_context[agent_role.value] = response.output
                
            except Exception as e:
                logger.error(f"Error in deployment phase for {agent_role.value}: {str(e)}")
        
        state["collaboration_context"]["deployment"] = deployment_context
        state["completed_agents"].extend(deployment_agents)
        return state
    
    def _check_feedback_required(self, state: TeamState) -> TeamState:
        """Check if human feedback is required."""
        # Simple heuristic: request feedback if complex project or many agents involved
        complex_project = len(state["selected_roles"]) > 4 or "complex" in state["project_description"].lower()
        state["feedback_required"] = complex_project and state["iteration_count"] < self.max_iterations
        return state
    
    def _should_request_feedback(self, state: TeamState) -> str:
        """Determine if feedback should be requested."""
        return "request_feedback" if state.get("feedback_required", False) else "continue"
    
    def _handle_human_feedback(self, state: TeamState) -> TeamState:
        """Handle human feedback and adjust collaboration."""
        # In a real implementation, this would integrate with a UI for human input
        # For now, we'll simulate feedback processing
        state["iteration_count"] += 1
        state["messages"].append(SystemMessage(content="Human feedback incorporated, refining deliverables..."))
        return state
    
    def _synthesize_deliverables(self, state: TeamState) -> TeamState:
        """Synthesize all agent outputs into final deliverables."""
        try:
            # Synthesize requirements
            requirements_content = []
            for agent_role, response in state["agent_outputs"].items():
                if agent_role in ["product_owner", "analyst"]:
                    role_name = agent_role.replace('_', ' ').title()
                    requirements_content.append(f"**{role_name}:**\n{response.output}\n")
            
            state["deliverables"]["requirements"] = "\n".join(requirements_content)
            
            # Synthesize architecture
            architecture_content = []
            for agent_role, response in state["agent_outputs"].items():
                if "architect" in agent_role.lower():
                    role_name = agent_role.replace('_', ' ').title()
                    architecture_content.append(f"**{role_name}:**\n{response.output}\n")
            
            state["deliverables"]["architecture"] = "\n".join(architecture_content)
            
            # Synthesize implementation plan
            implementation_content = []
            for agent_role, response in state["agent_outputs"].items():
                if "developer" in agent_role.lower():
                    role_name = agent_role.replace('_', ' ').title()
                    implementation_content.append(f"**{role_name}:**\n{response.output}\n")
            
            state["deliverables"]["implementation"] = "\n".join(implementation_content)
            
            # Synthesize testing strategy
            testing_content = []
            for agent_role, response in state["agent_outputs"].items():
                if agent_role in ["tester"]:
                    role_name = agent_role.replace('_', ' ').title()
                    testing_content.append(f"**{role_name}:**\n{response.output}\n")
            
            state["deliverables"]["testing"] = "\n".join(testing_content)
            
            # Synthesize deployment plan
            deployment_content = []
            for agent_role, response in state["agent_outputs"].items():
                if agent_role in ["devops_engineer"]:
                    role_name = agent_role.replace('_', ' ').title()
                    deployment_content.append(f"**{role_name}:**\n{response.output}\n")
            
            state["deliverables"]["deployment"] = "\n".join(deployment_content)
            
            # Synthesize all other roles
            other_content = []
            handled_roles = ["product_owner", "analyst", "tester", "devops_engineer"]
            for agent_role, response in state["agent_outputs"].items():
                if agent_role not in handled_roles and "architect" not in agent_role.lower() and "developer" not in agent_role.lower():
                    role_name = agent_role.replace('_', ' ').title()
                    other_content.append(f"**{role_name}:**\n{response.output}\n")
            
            if other_content:
                state["deliverables"]["additional"] = "\n".join(other_content)

        except Exception as e:
            state["error"] = f"Error synthesizing deliverables: {str(e)}"
        
        return state
    
    def _finalize_collaboration(self, state: TeamState) -> TeamState:
        """Finalize the collaboration workflow."""
        state["current_phase"] = CollaborationPhase.FINALIZATION
        state["messages"].append(SystemMessage(content="Team collaboration completed successfully"))
        return state
    
    def _generate_collaboration_summary(self, state: TeamState) -> str:
        """Generate a summary of the collaboration process."""
        roles_list = "\n".join([f"• {role.value.replace('_', ' ').title()}" for role in state['selected_roles']])
        phases_list = "\n".join([f"✅ {phase.value.replace('_', ' ').title()}" for phase in self._get_completed_phases(state)])
        deliverables_list = "\n".join([f"• {deliverable.title()}" for deliverable in state.get('deliverables', {}).keys()])
        
        summary = f"""
🚀 **Software Team Collaboration Summary**

**Project:** {state['project_description']}

**Team Composition:** {len(state['selected_roles'])} agents
{roles_list}

**Phases Completed:**
{phases_list}

**Deliverables Generated:**
{deliverables_list}

**Collaboration Iterations:** {state.get('iteration_count', 0)}

The team has successfully collaborated to provide comprehensive project deliverables including requirements analysis, architecture design, implementation planning, testing strategy, and deployment planning.
"""
        return summary.strip()
    
    def _get_completed_phases(self, state: TeamState) -> List[CollaborationPhase]:
        """Get list of completed collaboration phases."""
        phases = []
        if state.get("collaboration_context", {}).get("requirements"):
            phases.append(CollaborationPhase.REQUIREMENTS_ANALYSIS)
        if state.get("collaboration_context", {}).get("architecture"):
            phases.append(CollaborationPhase.ARCHITECTURE_DESIGN)
        if state.get("collaboration_context", {}).get("implementation"):
            phases.append(CollaborationPhase.IMPLEMENTATION_PLANNING)
        if any("review" in key for key in state.get("agent_outputs", {}).keys()):
            phases.append(CollaborationPhase.CODE_REVIEW)
        if state.get("collaboration_context", {}).get("testing"):
            phases.append(CollaborationPhase.TESTING_STRATEGY)
        if state.get("collaboration_context", {}).get("deployment"):
            phases.append(CollaborationPhase.DEPLOYMENT_PLANNING)
        
        return phases

class EnhancedAgentCollaborationManager:
    """Manager for enhanced agent collaboration with stateful workflows."""
    
    def __init__(self, orchestrator: EnhancedLangGraphSoftwareTeamOrchestrator):
        self.orchestrator = orchestrator
        self.active_collaborations = {}
    
    def start_collaboration(self, project_id: str, project_request: ProjectRequest) -> Dict[str, Any]:
        """Start a new collaboration session."""
        try:
            result = self.orchestrator.collaborate_on_project(project_request)
            self.active_collaborations[project_id] = result
            return result
        except Exception as e:
            logger.error(f"Error starting collaboration: {str(e)}")
            return {"status": "error", "error": str(e)}
    
    def get_collaboration_status(self, project_id: str) -> Dict[str, Any]:
        """Get the status of an ongoing collaboration."""
        return self.active_collaborations.get(project_id, {"status": "not_found"})
    
    def provide_feedback(self, project_id: str, feedback: str) -> Dict[str, Any]:
        """Provide human feedback to an ongoing collaboration."""
        # In a real implementation, this would update the workflow state
        if project_id in self.active_collaborations:
            collaboration = self.active_collaborations[project_id]
            collaboration["human_feedback"] = feedback
            return {"status": "feedback_received"}
        return {"status": "collaboration_not_found"}
