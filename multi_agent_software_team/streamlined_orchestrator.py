"""
Streamlined Multi-Agent Software Development Team
A simplified LangGraph implementation compatible with existing infrastructure.
"""

import logging
from typing import Dict, List, Any, Optional, TypedDict, Annotated
from datetime import datetime

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

from .schemas import TeamRole, AgentResponse, AgentMessage, ProjectRequest
from .agents import Agent
from .utils import format_agent_responses

logger = logging.getLogger(__name__)

class StreamlinedTeamState(TypedDict):
    """Simplified state for multi-agent collaboration."""
    messages: Annotated[List, add_messages]
    project_description: str
    current_agent: Optional[TeamRole]
    agent_outputs: Dict[str, str]
    completed_agents: List[TeamRole]
    workflow_complete: bool
    error: Optional[str]

class StreamlinedSoftwareTeamOrchestrator:
    """Streamlined orchestrator for multi-agent software team with handoff mechanisms."""
    
    def __init__(self, llm):
        self.llm = llm
        self.agents = {role: Agent(role, llm) for role in TeamRole}
        self.team_graph = self._build_workflow()
        
        # Define the standard workflow sequence
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
    
    def _build_workflow(self) -> StateGraph:
        """Build the streamlined workflow graph."""
        
        def initialize_collaboration(state: StreamlinedTeamState) -> StreamlinedTeamState:
            """Initialize the collaboration."""
            state["current_agent"] = TeamRole.PRODUCT_OWNER
            state["agent_outputs"] = {}
            state["completed_agents"] = []
            state["workflow_complete"] = False
            state["error"] = None
            
            initial_message = SystemMessage(content="🚀 Starting multi-agent software development collaboration...")
            state["messages"].append(initial_message)
            
            return state
        
        def execute_current_agent(state: StreamlinedTeamState) -> StreamlinedTeamState:
            """Execute the current agent's work."""
            try:
                current_role = state["current_agent"]
                if not current_role:
                    state["workflow_complete"] = True
                    return state
                
                logger.info(f"🔄 {current_role.value.upper()} starting work...")
                
                # Get previous outputs for context
                previous_outputs = []
                for completed_role in state["completed_agents"]:
                    if completed_role.value in state["agent_outputs"]:
                        previous_outputs.append(AgentMessage(
                            role=completed_role,
                            content=state["agent_outputs"][completed_role.value]
                        ))
                
                # Execute agent
                agent = self.agents[current_role]
                response = agent.process(state["project_description"], previous_outputs)
                
                # Store output
                state["agent_outputs"][current_role.value] = response.output
                state["completed_agents"].append(current_role)
                
                # Add to messages
                role_message = HumanMessage(
                    content=response.output,
                    name=current_role.value
                )
                state["messages"].append(role_message)
                
                logger.info(f"✅ {current_role.value.upper()} completed work")
                
                return state
                
            except Exception as e:
                logger.error(f"❌ Error in {state['current_agent'].value}: {str(e)}")
                state["error"] = str(e)
                return state
        
        def determine_next_agent(state: StreamlinedTeamState) -> StreamlinedTeamState:
            """Determine the next agent based on handoff signals."""
            if state.get("error") or state.get("workflow_complete"):
                return state
            
            current_role = state["current_agent"]
            if not current_role or current_role.value not in state["agent_outputs"]:
                state["workflow_complete"] = True
                return state
            
            # Check for handoff signal in the latest output
            latest_output = state["agent_outputs"][current_role.value]
            next_role = self._extract_handoff_signal(latest_output)
            
            if next_role:
                state["current_agent"] = next_role
                logger.info(f"🔄 Handoff detected: {current_role.value} → {next_role.value}")
            else:
                # Check for FINAL ANSWER or default to completion
                if "FINAL ANSWER" in latest_output.upper():
                    state["workflow_complete"] = True
                    state["current_agent"] = None
                    logger.info("✅ Workflow completed with FINAL ANSWER")
                else:
                    # Fallback: continue to next in sequence
                    try:
                        current_index = self.workflow_sequence.index(current_role)
                        if current_index < len(self.workflow_sequence) - 1:
                            state["current_agent"] = self.workflow_sequence[current_index + 1]
                            logger.info(f"🔄 Sequential handoff: {current_role.value} → {state['current_agent'].value}")
                        else:
                            state["workflow_complete"] = True
                            state["current_agent"] = None
                            logger.info("✅ Workflow completed - end of sequence")
                    except ValueError:
                        state["workflow_complete"] = True
                        state["current_agent"] = None
                        logger.info("✅ Workflow completed - role not in sequence")
            
            return state
        
        def finalize_collaboration(state: StreamlinedTeamState) -> StreamlinedTeamState:
            """Finalize the collaboration."""
            completion_message = SystemMessage(
                content="✅ Multi-agent software development collaboration completed successfully!"
            )
            state["messages"].append(completion_message)
            
            logger.info(f"🎉 Collaboration completed with {len(state['completed_agents'])} agents")
            return state
        
        def should_continue(state: StreamlinedTeamState) -> str:
            """Determine if workflow should continue."""
            if state.get("error"):
                return "finalize"
            elif state.get("workflow_complete"):
                return "finalize"
            elif state.get("current_agent"):
                return "execute_agent"
            else:
                return "finalize"
        
        # Build the workflow
        workflow = StateGraph(StreamlinedTeamState)
        
        # Add nodes
        workflow.add_node("initialize", initialize_collaboration)
        workflow.add_node("execute_agent", execute_current_agent)
        workflow.add_node("determine_next", determine_next_agent)
        workflow.add_node("finalize", finalize_collaboration)
        
        # Add edges
        workflow.set_entry_point("initialize")
        workflow.add_edge("initialize", "execute_agent")
        workflow.add_edge("execute_agent", "determine_next")
        workflow.add_conditional_edges("determine_next", should_continue)
        workflow.add_edge("finalize", END)
        
        return workflow.compile()
    
    def collaborate_on_project(self, project_request: ProjectRequest) -> Dict[str, Any]:
        """Execute team collaboration on a project."""
        try:
            logger.info("🚀 Starting streamlined software development team collaboration...")
            
            # Prepare initial state
            initial_state = {
                "messages": [HumanMessage(content=project_request.description)],
                "project_description": project_request.description,
                "current_agent": None,
                "agent_outputs": {},
                "completed_agents": [],
                "workflow_complete": False,
                "error": None
            }
            
            # Execute workflow
            final_state = self.team_graph.invoke(initial_state)
            
            # Format results
            if final_state.get("error"):
                return {
                    "success": False,
                    "status": "error",
                    "error": final_state["error"],
                    "output": f"❌ Error: {final_state['error']}"
                }
            
            # Create agent responses for formatting
            agent_responses = []
            for role in final_state["completed_agents"]:
                if role.value in final_state["agent_outputs"]:
                    agent_responses.append(AgentResponse(
                        role=role,
                        output=final_state["agent_outputs"][role.value]
                    ))
            
            formatted_output = format_agent_responses(agent_responses)
            
            return {
                "success": True,
                "status": "completed",
                "output": formatted_output,
                "agent_outputs": final_state["agent_outputs"],
                "completed_agents": [role.value for role in final_state["completed_agents"]],
                "total_agents": len(final_state["completed_agents"])
            }
            
        except Exception as e:
            logger.error(f"Error in streamlined collaboration: {str(e)}")
            return {
                "success": False,
                "status": "error", 
                "error": str(e),
                "output": f"❌ Error in collaboration: {str(e)}"
            }
    
    def run_example(self, project_description: str = None) -> Dict[str, Any]:
        """Run an example project."""
        if not project_description:
            project_description = """
            Create a task management system for small development teams. 
            The system should allow:
            - User registration and authentication
            - Project creation and management
            - Task assignment and tracking
            - Team collaboration features
            - Progress reporting and analytics
            
            The system should be web-based, scalable, and secure.
            """
        
        project_request = ProjectRequest(
            description=project_description,
            selected_roles=None
        )
        
        return self.collaborate_on_project(project_request)

# Factory function for easy integration
def create_streamlined_software_team(llm):
    """Create a streamlined software team orchestrator."""
    return StreamlinedSoftwareTeamOrchestrator(llm)

def run_streamlined_collaboration(llm, project_description: str, file_content: str = None) -> str:
    """Run streamlined team collaboration."""
    orchestrator = StreamlinedSoftwareTeamOrchestrator(llm)
    
    project_request = ProjectRequest(
        description=project_description,
        file_content=file_content,
        selected_roles=None
    )
    
    result = orchestrator.collaborate_on_project(project_request)
    
    if result["success"]:
        return result["output"]
    else:
        return f"❌ Error: {result.get('error', 'Unknown error occurred')}"
