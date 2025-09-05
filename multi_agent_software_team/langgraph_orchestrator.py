"""
Enhanced LangGraph-based Multi-Agent Software Team with Handoff Mechanism.
"""

import logging
from typing import Dict, List, Any, Optional, TypedDict, Annotated
from datetime import datetime

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

from .schemas import TeamRole, AgentResponse, AgentMessage, ProjectRequest
from .agents import Agent
from .utils import format_agent_responses, validate_team_composition

logger = logging.getLogger(__name__)

class SoftwareTeamState(TypedDict):
    """State for software team collaboration graph."""
    messages: Annotated[List, add_messages]
    project_description: str
    selected_roles: List[TeamRole]
    agent_outputs: Dict[str, AgentResponse]
    current_agent: Optional[TeamRole]
    completed_agents: List[TeamRole]
    final_deliverables: Dict[str, str]
    error: Optional[str]
    workflow_complete: bool

class LangGraphSoftwareTeamOrchestrator:
    """Enhanced LangGraph-based orchestrator with handoff mechanisms."""
    
    def __init__(self, llm):
        self.llm = llm
        self.agents = {role: Agent(role, llm) for role in TeamRole}
        self.graph = self._build_team_graph()
        
        # Define standard workflow sequence
        self.workflow_sequence = [
            TeamRole.PRODUCT_OWNER,
            TeamRole.ANALYST, 
            TeamRole.ARCHITECT,
            TeamRole.DEVELOPER,
            TeamRole.REVIEWER,
            TeamRole.TESTER,
            TeamRole.TECH_WRITER
        ]
    
    def _extract_handoff_signal(self, content: str) -> Optional[TeamRole]:
        """Extract handoff signal from agent output."""
        content_upper = content.upper()
        
        handoff_mappings = {
            "HANDOFF TO ANALYST": TeamRole.ANALYST,
            "HANDOFF TO ARCHITECT": TeamRole.ARCHITECT, 
            "HANDOFF TO DEVELOPER": TeamRole.DEVELOPER,
            "HANDOFF TO REVIEWER": TeamRole.REVIEWER,
            "HANDOFF TO TESTER": TeamRole.TESTER,
            "HANDOFF TO TECH_WRITER": TeamRole.TECH_WRITER,
        }
        
        for signal, role in handoff_mappings.items():
            if signal in content_upper:
                return role
        
        if "FINAL ANSWER" in content_upper:
            return None  # Indicates completion
            
        return None
    
    def __init__(self, llm):
        self.llm = llm
        self.agents = {role: Agent(role, llm) for role in TeamRole}
        self.graph = self._build_team_graph()
    
    def _build_team_graph(self) -> StateGraph:
        """Build the software team collaboration workflow graph."""
        
        def initialize_project(state: SoftwareTeamState) -> SoftwareTeamState:
            """Initialize the project and validate team composition."""
            try:
                # Validate team composition
                is_valid, validation_message = validate_team_composition(state["selected_roles"])
                if not is_valid:
                    state["error"] = validation_message
                    return state
                
                state["agent_outputs"] = {}
                state["completed_agents"] = []
                state["current_agent"] = None
                state["final_deliverables"] = {}
                
                # Add initialization message
                state["messages"].append(
                    SystemMessage(content=f"Initializing software team with {len(state['selected_roles'])} agents: {', '.join([role.value for role in state['selected_roles']])}")
                )
                
            except Exception as e:
                state["error"] = str(e)
            
            return state
        
        def select_next_agent(state: SoftwareTeamState) -> SoftwareTeamState:
            """Select the next agent to work based on dependencies."""
            try:
                # Define agent execution order based on dependencies
                execution_order = [
                    TeamRole.PRODUCT_OWNER,  # Always first
                    TeamRole.BUSINESS_ANALYST,
                    TeamRole.SYSTEM_ARCHITECT,
                    TeamRole.DEVELOPER,
                    TeamRole.QA_TESTER,
                    TeamRole.UI_UX_DESIGNER,
                    TeamRole.CODE_REVIEWER,
                    TeamRole.TECHNICAL_WRITER  # Always last
                ]
                
                # Find next agent to execute
                for role in execution_order:
                    if role in state["selected_roles"] and role not in state["completed_agents"]:
                        state["current_agent"] = role
                        return state
                
                # All agents completed
                state["current_agent"] = None
                
            except Exception as e:
                state["error"] = str(e)
            
            return state
        
        def execute_agent_work(state: SoftwareTeamState) -> SoftwareTeamState:
            """Execute work for the current agent."""
            try:
                if not state["current_agent"]:
                    return state
                
                current_role = state["current_agent"]
                agent = self.agents[current_role]
                
                # Prepare previous outputs for context
                previous_outputs = [
                    AgentMessage(role=role, content=response.output)
                    for role, response in state["agent_outputs"].items()
                ]
                
                # Execute agent work
                agent_response = agent.process(
                    project_description=state["project_description"],
                    previous_outputs=previous_outputs
                )
                
                # Store the response
                state["agent_outputs"][current_role.value] = agent_response
                state["completed_agents"].append(current_role)
                
                # Add to messages
                state["messages"].append(
                    AIMessage(content=f"Completed {current_role.value}: {agent_response.output[:200]}...")
                )
                
            except Exception as e:
                state["error"] = f"Error executing {state['current_agent'].value if state['current_agent'] else 'unknown'} agent: {str(e)}"
            
            return state
        
        def compile_deliverables(state: SoftwareTeamState) -> SoftwareTeamState:
            """Compile final deliverables from all agent outputs."""
            try:
                # Format all agent responses
                final_output = format_agent_responses([
                    AgentMessage(role=role, content=response.output)
                    for role, response in state["agent_outputs"].items()
                ])
                
                # Create structured deliverables
                deliverables = {}
                for role_str, response in state["agent_outputs"].items():
                    role_name = role_str.replace('_', ' ').title()
                    deliverables[role_name] = response.output
                
                state["final_deliverables"] = deliverables
                
                # Add completion message
                state["messages"].append(
                    SystemMessage(content=f"Software team collaboration completed successfully with {len(deliverables)} deliverables")
                )
                
            except Exception as e:
                state["error"] = f"Error compiling deliverables: {str(e)}"
            
            return state
        
        def should_continue(state: SoftwareTeamState) -> str:
            """Determine next step in the workflow."""
            if state.get("error"):
                return "error"
            
            if state["current_agent"] is None:
                # Check if we have more agents to process
                remaining_agents = [
                    role for role in state["selected_roles"] 
                    if role not in state["completed_agents"]
                ]
                
                if remaining_agents:
                    return "select_next_agent"
                else:
                    return "compile_deliverables"
            else:
                return "execute_agent"
        
        def handle_error(state: SoftwareTeamState) -> SoftwareTeamState:
            """Handle errors in the workflow."""
            error_msg = state.get("error", "Unknown error occurred")
            state["messages"].append(AIMessage(content=f"Error in software team workflow: {error_msg}"))
            return state
        
        # Build the graph
        workflow = StateGraph(SoftwareTeamState)
        
        # Add nodes
        workflow.add_node("initialize", initialize_project)
        workflow.add_node("select_next_agent", select_next_agent)
        workflow.add_node("execute_agent", execute_agent_work)
        workflow.add_node("compile_deliverables", compile_deliverables)
        workflow.add_node("error", handle_error)
        
        # Add edges
        workflow.set_entry_point("initialize")
        workflow.add_conditional_edges("initialize", should_continue)
        workflow.add_conditional_edges("select_next_agent", should_continue)
        workflow.add_conditional_edges("execute_agent", should_continue)
        workflow.add_edge("compile_deliverables", END)
        workflow.add_edge("error", END)
        
        return workflow.compile()
    
    def create_software_solution(self, project_request: ProjectRequest) -> Dict[str, Any]:
        """Create a comprehensive software solution using the multi-agent team."""
        try:
            # Prepare initial state
            initial_state = SoftwareTeamState(
                messages=[HumanMessage(content=f"Project: {project_request.description}")],
                project_description=project_request.description,
                selected_roles=project_request.selected_roles,
                agent_outputs={},
                current_agent=None,
                completed_agents=[],
                final_deliverables={},
                error=None
            )
            
            # Run the workflow
            final_state = self.graph.invoke(initial_state)
            
            if final_state.get("error"):
                return {
                    "success": False,
                    "error": final_state["error"],
                    "deliverables": {}
                }
            
            return {
                "success": True,
                "deliverables": final_state["final_deliverables"],
                "agent_outputs": {
                    role.value: response.output 
                    for role, response in final_state["agent_outputs"].items()
                },
                "messages": [msg.content for msg in final_state["messages"]]
            }
            
        except Exception as e:
            logger.error(f"Error in LangGraph software team: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "deliverables": {}
            }
    
    def get_agent_summary(self, role: TeamRole) -> Dict[str, str]:
        """Get summary information about a specific agent."""
        agent = self.agents[role]
        return {
            "role": role.value,
            "name": role.value.replace('_', ' ').title(),
            "description": f"AI agent specialized in {role.value.replace('_', ' ').lower()} tasks",
            "capabilities": self._get_agent_capabilities(role)
        }
    
    def _get_agent_capabilities(self, role: TeamRole) -> str:
        """Get capabilities description for a specific role."""
        capabilities = {
            TeamRole.PRODUCT_OWNER: "Requirements gathering, user stories, acceptance criteria, business analysis",
            TeamRole.BUSINESS_ANALYST: "Functional requirements, technical specifications, process analysis",
            TeamRole.SYSTEM_ARCHITECT: "System design, architecture patterns, technology stack, scalability planning",
            TeamRole.DEVELOPER: "Code implementation, database design, API development, technical implementation",
            TeamRole.QA_TESTER: "Test planning, quality assurance, automation scripts, testing strategies",
            TeamRole.UI_UX_DESIGNER: "User interface design, user experience, system diagrams, visual design",
            TeamRole.CODE_REVIEWER: "Code review, security analysis, best practices, quality assessment",
            TeamRole.TECHNICAL_WRITER: "Documentation, user guides, deployment instructions, technical writing"
        }
        return capabilities.get(role, "General software development tasks")
