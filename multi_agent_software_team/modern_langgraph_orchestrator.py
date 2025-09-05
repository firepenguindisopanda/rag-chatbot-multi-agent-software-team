# -*- coding: utf-8 -*-
"""
Modern Multi-Agent Software Development Team

A streamlined LangGraph implementation of a collaborative software development team
with specialized roles working together to deliver complete solutions.
Based on the latest LangGraph patterns with proper handoff mechanisms.
"""

import logging
from typing import Annotated, Dict, List, Literal, Optional, TypedDict
from enum import Enum

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.types import Command

from .schemas import TeamRole, ProjectRequest
from .prompts import create_system_prompts

logger = logging.getLogger(__name__)

# Define specialized tools for different roles
@tool
def create_documentation(
    content: Annotated[str, "The documentation content to create"],
    doc_type: Annotated[str, "Type of documentation (requirements, architecture, API, etc.)"]
):
    """Create and store documentation for the project."""
    return f"Documentation created: {doc_type}\nContent: {content}\n\nDocument saved successfully."

@tool
def create_diagram(
    diagram_code: Annotated[str, "Mermaid diagram code"],
    diagram_type: Annotated[str, "Type of diagram (architecture, ERD, sequence, etc.)"]
):
    """Create technical diagrams using Mermaid syntax."""
    return f"Diagram created: {diagram_type}\nMermaid Code:\n```mermaid\n{diagram_code}\n```\n\nDiagram generated successfully."

@tool
def code_review_tool(
    code: Annotated[str, "Code to review"],
    review_type: Annotated[str, "Type of review (security, performance, quality, etc.)"]
):
    """Perform code review and provide feedback."""
    return f"Code review completed: {review_type}\nCode analyzed:\n{code}\n\nReview feedback generated."

@tool
def test_generation_tool(
    component: Annotated[str, "Component or feature to test"],
    test_type: Annotated[str, "Type of test (unit, integration, e2e, etc.)"]
):
    """Generate test cases and test scripts."""
    return f"Tests generated for: {component}\nTest type: {test_type}\n\nTest suite created successfully."

@tool
def architecture_analysis_tool(
    system_description: Annotated[str, "Description of the system to analyze"],
    analysis_type: Annotated[str, "Type of analysis (scalability, security, performance, etc.)"]
):
    """Analyze system architecture and provide recommendations."""
    return f"Architecture analysis completed: {analysis_type}\nSystem: {system_description}\n\nAnalysis and recommendations generated."

class ModernSoftwareTeamOrchestrator:
    """Modern streamlined orchestrator for multi-agent software team."""
    
    def __init__(self, llm):
        self.llm = llm
        self.system_prompts = create_system_prompts()
        self.team_graph = self._create_team_workflow()
    
    def _create_enhanced_prompts(self) -> Dict[TeamRole, str]:
        """Create enhanced prompts with handoff mechanisms."""
        # Use the base prompts directly since they already have handoff instructions
        # Just add tool usage reminders
        enhanced_prompts = self.system_prompts.copy()
        
        # Add tool usage instructions where needed
        enhanced_prompts[TeamRole.ARCHITECT] = enhanced_prompts[TeamRole.ARCHITECT] + """

TOOL USAGE: Use the create_diagram and architecture_analysis_tool for technical diagrams and analysis."""
        
        enhanced_prompts[TeamRole.DEVELOPER] = enhanced_prompts[TeamRole.DEVELOPER] + """

TOOL USAGE: Use create_diagram tool for ERD and create_documentation for code documentation."""
        
        enhanced_prompts[TeamRole.REVIEWER] = enhanced_prompts[TeamRole.REVIEWER] + """

TOOL USAGE: Use code_review_tool for reviews and create_documentation for review reports."""
        
        enhanced_prompts[TeamRole.TESTER] = enhanced_prompts[TeamRole.TESTER] + """

TOOL USAGE: Use test_generation_tool and create_diagram for test flows."""
        
        enhanced_prompts[TeamRole.TECH_WRITER] = enhanced_prompts[TeamRole.TECH_WRITER] + """

TOOL USAGE: Use create_documentation tool for all documentation."""
        
        return enhanced_prompts
    
    def _create_team_agents(self):
        """Create agents for each role with appropriate tools."""
        prompts = self._create_enhanced_prompts()
        agents = {}
        
        # Product Owner - starts the process
        agents[TeamRole.PRODUCT_OWNER] = create_react_agent(
            self.llm,
            tools=[create_documentation],
            prompt=prompts[TeamRole.PRODUCT_OWNER]
        )
        
        # Analyst - analyzes requirements
        agents[TeamRole.ANALYST] = create_react_agent(
            self.llm,
            tools=[create_documentation],
            prompt=prompts[TeamRole.ANALYST]
        )
        
        # Architect - designs system
        agents[TeamRole.ARCHITECT] = create_react_agent(
            self.llm,
            tools=[create_diagram, create_documentation, architecture_analysis_tool],
            prompt=prompts[TeamRole.ARCHITECT]
        )
        
        # Developer - implements solution
        agents[TeamRole.DEVELOPER] = create_react_agent(
            self.llm,
            tools=[create_diagram, create_documentation],
            prompt=prompts[TeamRole.DEVELOPER]
        )
        
        # Reviewer - reviews code
        agents[TeamRole.REVIEWER] = create_react_agent(
            self.llm,
            tools=[code_review_tool, create_documentation],
            prompt=prompts[TeamRole.REVIEWER]
        )
        
        # Tester - creates tests
        agents[TeamRole.TESTER] = create_react_agent(
            self.llm,
            tools=[test_generation_tool, create_diagram, create_documentation],
            prompt=prompts[TeamRole.TESTER]
        )
        
        # Tech Writer - final documentation
        agents[TeamRole.TECH_WRITER] = create_react_agent(
            self.llm,
            tools=[create_documentation],
            prompt=prompts[TeamRole.TECH_WRITER]
        )
        
        return agents
    
    def _check_handoff_in_content_without_final(self, content: str) -> str:
        """Check for handoff signals in message content, excluding FINAL ANSWER."""
        content_upper = content.upper()
        
        if "HANDOFF TO ANALYST" in content_upper:
            return "analyst"
        elif "HANDOFF TO ARCHITECT" in content_upper:
            return "architect"
        elif "HANDOFF TO DEVELOPER" in content_upper:
            return "developer"
        elif "HANDOFF TO REVIEWER" in content_upper:
            return "reviewer"
        elif "HANDOFF TO TESTER" in content_upper:
            return "tester"
        elif "HANDOFF TO TECH_WRITER" in content_upper:
            return "tech_writer"
        else:
            return END
    
    def _check_handoff_in_content(self, content: str) -> str:
        """Check for handoff signals in message content."""
        content_upper = content.upper()
        
        if "FINAL ANSWER" in content_upper:
            return END
        elif "HANDOFF TO ANALYST" in content_upper:
            return "analyst"
        elif "HANDOFF TO ARCHITECT" in content_upper:
            return "architect"
        elif "HANDOFF TO DEVELOPER" in content_upper:
            return "developer"
        elif "HANDOFF TO REVIEWER" in content_upper:
            return "reviewer"
        elif "HANDOFF TO TESTER" in content_upper:
            return "tester"
        elif "HANDOFF TO TECH_WRITER" in content_upper:
            return "tech_writer"
        else:
            return END
    
    def _get_next_node(self, last_message: BaseMessage) -> str:
        """Determine the next node based on handoff signals."""
        content = last_message.content.upper()
        
        if "FINAL ANSWER" in content:
            return END
        elif "HANDOFF TO ANALYST" in content:
            return "analyst"
        elif "HANDOFF TO ARCHITECT" in content:
            return "architect"
        elif "HANDOFF TO DEVELOPER" in content:
            return "developer"
        elif "HANDOFF TO REVIEWER" in content:
            return "reviewer"
        elif "HANDOFF TO TESTER" in content:
            return "tester"
        elif "HANDOFF TO TECH_WRITER" in content:
            return "tech_writer"
        else:
            return END
    
    def _create_node_function(self, role: TeamRole, agent, next_roles: List[str]):
        """Create a node function for a specific role."""
        def node_function(state: MessagesState) -> Command:
            try:
                # Log the current agent working
                logger.info(f"🔄 {role.value.upper()} starting work...")
                
                result = agent.invoke(state)
                
                # Look for handoff signals in any AI message from this agent
                goto = END  # Default to ending
                
                # First, check if there's a FINAL ANSWER in any message (highest priority)
                for msg in reversed(result["messages"]):
                    if hasattr(msg, 'content') and msg.content and "FINAL ANSWER" in msg.content.upper():
                        goto = END
                        logger.info("🎯 Found FINAL ANSWER signal - ending workflow")
                        break
                
                # If no FINAL ANSWER found, look for handoff signals
                if goto == END and role.value != "tech_writer":  # Don't look for handoffs if we already found FINAL ANSWER or if tech_writer
                    for msg in reversed(result["messages"]):
                        if hasattr(msg, 'content') and msg.content:
                            handoff_result = self._check_handoff_in_content_without_final(msg.content)
                            if handoff_result != END:
                                goto = handoff_result
                                logger.info(f"🔄 Found handoff signal: {handoff_result}")
                                break
                
                # Convert to human message with role name for better tracking
                result["messages"][-1] = HumanMessage(
                    content=result["messages"][-1].content,
                    name=role.value
                )
                
                logger.info(f"✅ {role.value.upper()} completed. Next: {goto}")
                
                return Command(
                    update={"messages": result["messages"]},
                    goto=goto
                )
            except Exception as e:
                logger.error(f"❌ Error in {role.value}: {str(e)}")
                error_message = HumanMessage(
                    content=f"Error in {role.value}: {str(e)}. Proceeding to next agent.",
                    name=role.value
                )
                # Try to continue to next logical step
                next_step = next_roles[0] if next_roles else END
                return Command(
                    update={"messages": [error_message]},
                    goto=next_step
                )
        
        return node_function
    
    def _create_team_workflow(self):
        """Create the software development team workflow."""
        agents = self._create_team_agents()
        
        workflow = StateGraph(MessagesState)
        
        # Add nodes for each role with their next possible steps
        workflow.add_node("product_owner", self._create_node_function(
            TeamRole.PRODUCT_OWNER, agents[TeamRole.PRODUCT_OWNER], ["analyst"]
        ))
        workflow.add_node("analyst", self._create_node_function(
            TeamRole.ANALYST, agents[TeamRole.ANALYST], ["architect"]
        ))
        workflow.add_node("architect", self._create_node_function(
            TeamRole.ARCHITECT, agents[TeamRole.ARCHITECT], ["developer"]
        ))
        workflow.add_node("developer", self._create_node_function(
            TeamRole.DEVELOPER, agents[TeamRole.DEVELOPER], ["reviewer"]
        ))
        workflow.add_node("reviewer", self._create_node_function(
            TeamRole.REVIEWER, agents[TeamRole.REVIEWER], ["tester"]
        ))
        workflow.add_node("tester", self._create_node_function(
            TeamRole.TESTER, agents[TeamRole.TESTER], ["tech_writer"]
        ))
        workflow.add_node("tech_writer", self._create_node_function(
            TeamRole.TECH_WRITER, agents[TeamRole.TECH_WRITER], []
        ))
        
        # Define the workflow
        workflow.add_edge(START, "product_owner")
        
        return workflow.compile()
    
    def collaborate_on_project(self, project_request: ProjectRequest) -> Dict:
        """Execute team collaboration on a project."""
        try:
            logger.info("🚀 Starting Software Development Team Collaboration...")
            logger.info(f"Project: {project_request.description[:100]}...")
            
            # Prepare the input message
            input_message = f"""
Project Description: {project_request.description}

{f"Additional Context: {project_request.file_content}" if project_request.file_content else ""}

Please work together as a software development team to deliver a complete solution.
            """.strip()
            
            # Execute the workflow
            events = self.team_graph.stream(
                {
                    "messages": [("user", input_message)],
                },
                {"recursion_limit": 50},
            )
            
            # Collect all messages
            all_messages = []
            agent_outputs = {}
            
            for event in events:
                for node_name, node_data in event.items():
                    if node_name != "__end__" and "messages" in node_data:
                        messages = node_data["messages"]
                        all_messages.extend(messages)
                        
                        # Track outputs by agent
                        for msg in messages:
                            if hasattr(msg, 'name') and msg.name:
                                if msg.name not in agent_outputs:
                                    agent_outputs[msg.name] = []
                                agent_outputs[msg.name].append(msg.content)
            
            # Format the final result
            formatted_output = self._format_team_output(agent_outputs)
            
            return {
                "success": True,
                "status": "completed",
                "output": formatted_output,
                "agent_outputs": agent_outputs,
                "message_count": len(all_messages)
            }
            
        except Exception as e:
            logger.error(f"Error in team collaboration: {str(e)}")
            return {
                "success": False,
                "status": "error",
                "error": str(e),
                "output": f"Error in team collaboration: {str(e)}"
            }
    
    def _format_team_output(self, agent_outputs: Dict[str, List[str]]) -> str:
        """Format the team output into a comprehensive report."""
        result = "# 🚀 Software Development Team Collaboration Results\n\n"
        result += f"**Generated:** {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        result += f"**Team Size:** {len(agent_outputs)} agents\n\n"
        result += "---\n\n"
        
        # Define the order of agents
        agent_order = [
            TeamRole.PRODUCT_OWNER,
            TeamRole.ANALYST,
            TeamRole.ARCHITECT,
            TeamRole.DEVELOPER,
            TeamRole.REVIEWER,
            TeamRole.TESTER,
            TeamRole.TECH_WRITER
        ]
        
        # Role icons
        role_icons = {
            "product_owner": "📋",
            "analyst": "🔍",
            "architect": "🏗️",
            "developer": "💻",
            "reviewer": "👀",
            "tester": "🧪",
            "tech_writer": "📝"
        }
        
        # Process agents in order
        for role in agent_order:
            role_key = role.value
            if role_key in agent_outputs:
                role_name = role_key.replace('_', ' ').title()
                icon = role_icons.get(role_key, "🤖")
                result += f"## {icon} {role_name}\n\n"
                
                # Combine all outputs for this agent
                combined_output = "\n\n".join(agent_outputs[role_key])
                result += f"{combined_output}\n\n"
                result += "---\n\n"
        
        result += "*✅ Software solution delivered through collaborative multi-agent development*"
        return result
    
    def run_example(self, project_description: str = None):
        """Run an example project through the team."""
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
            selected_roles=None  # Will use all roles by default
        )
        
        return self.collaborate_on_project(project_request)

# Create a simple interface function for Gradio integration
def create_software_team(llm):
    """Create a software team orchestrator for use in Gradio interfaces."""
    return ModernSoftwareTeamOrchestrator(llm)

def run_team_collaboration(llm, project_description: str, file_content: str = None) -> str:
    """Run team collaboration and return formatted results."""
    orchestrator = ModernSoftwareTeamOrchestrator(llm)
    
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

if __name__ == "__main__":
    # Example usage
    print("Modern Software Development Team workflow created successfully!")
    print("Use ModernSoftwareTeamOrchestrator(llm).run_example() to test the workflow.")
