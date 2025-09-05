# -*- coding: utf-8 -*-
"""
Improved Multi-Agent Software Development Team

A refined implementation that produces clean, professional output without debug statements.
Focuses on delivering comprehensive software solutions with proper documentation and diagrams.
"""

import logging
import re
from typing import Annotated, Dict, List, Optional, TypedDict
from datetime import datetime

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.types import Command

from .schemas import TeamRole, ProjectRequest
from .improved_prompts import create_improved_system_prompts

logger = logging.getLogger(__name__)

# Enhanced tools for better diagram and documentation generation
@tool
def create_technical_diagram(
    diagram_code: Annotated[str, "Mermaid diagram code"],
    diagram_type: Annotated[str, "Type of diagram (architecture, ERD, sequence, flowchart, etc.)"],
    description: Annotated[str, "Description of what the diagram shows"]
):
    """Create technical diagrams using Mermaid syntax with proper formatting."""
    return f"""
## {diagram_type.title()} Diagram

{description}

```mermaid
{diagram_code}
```

*Diagram generated using Mermaid syntax*
"""

@tool
def create_code_example(
    code: Annotated[str, "Code implementation example"],
    language: Annotated[str, "Programming language (python, javascript, sql, etc.)"],
    description: Annotated[str, "Description of what the code does"]
):
    """Create well-formatted code examples with explanations."""
    return f"""
### {description}

```{language}
{code}
```
"""

@tool
def create_documentation_section(
    content: Annotated[str, "Documentation content"],
    section_title: Annotated[str, "Title of the documentation section"],
    section_type: Annotated[str, "Type of documentation (requirements, api, user-guide, etc.)"]
):
    """Create well-structured documentation sections."""
    return f"""
## {section_title}

{content}
"""

class ImprovedSoftwareTeamOrchestrator:
    """Improved orchestrator that produces clean, professional output."""
    
    def __init__(self, llm):
        self.llm = llm
        self.system_prompts = create_improved_system_prompts()
        self.team_graph = self._create_team_workflow()
    
    def _create_team_agents(self):
        """Create agents with improved tools and prompts."""
        agents = {}
        
        # Product Owner - business analysis
        agents[TeamRole.PRODUCT_OWNER] = create_react_agent(
            self.llm,
            tools=[create_documentation_section],
            prompt=self.system_prompts[TeamRole.PRODUCT_OWNER]
        )
        
        # Analyst - technical requirements
        agents[TeamRole.ANALYST] = create_react_agent(
            self.llm,
            tools=[create_documentation_section],
            prompt=self.system_prompts[TeamRole.ANALYST]
        )
        
        # Architect - system design with diagrams
        agents[TeamRole.ARCHITECT] = create_react_agent(
            self.llm,
            tools=[create_technical_diagram, create_documentation_section],
            prompt=self.system_prompts[TeamRole.ARCHITECT]
        )
        
        # Developer - implementation with code examples
        agents[TeamRole.DEVELOPER] = create_react_agent(
            self.llm,
            tools=[create_technical_diagram, create_code_example, create_documentation_section],
            prompt=self.system_prompts[TeamRole.DEVELOPER]
        )
        
        # Reviewer - quality analysis
        agents[TeamRole.REVIEWER] = create_react_agent(
            self.llm,
            tools=[create_documentation_section],
            prompt=self.system_prompts[TeamRole.REVIEWER]
        )
        
        # Tester - testing strategy
        agents[TeamRole.TESTER] = create_react_agent(
            self.llm,
            tools=[create_technical_diagram, create_documentation_section],
            prompt=self.system_prompts[TeamRole.TESTER]
        )
        
        # Tech Writer - comprehensive documentation
        agents[TeamRole.TECH_WRITER] = create_react_agent(
            self.llm,
            tools=[create_documentation_section],
            prompt=self.system_prompts[TeamRole.TECH_WRITER]
        )
        
        return agents
    
    def _clean_agent_output(self, content: str) -> str:
        """Remove handoff statements and debug information from agent output."""
        # Remove handoff statements
        handoff_patterns = [
            r"HANDOFF TO [A-Z_]+",
            r"Ready for [a-z\s]+$",
            r"Technical specifications ready for [a-z\s]+$",
            r"Architecture design complete, ready for [a-z\s]+$",
            r"Implementation guide complete, ready for [a-z\s]+$",
            r"Code review complete, ready for [a-z\s]+$",
            r"Testing plan complete, ready for [a-z\s]+$",
            r"Complete documentation delivered$",
            r"FINAL ANSWER"
        ]
        
        cleaned_content = content
        for pattern in handoff_patterns:
            cleaned_content = re.sub(pattern, "", cleaned_content, flags=re.IGNORECASE | re.MULTILINE)
        
        # Clean up extra whitespace
        cleaned_content = re.sub(r'\n\s*\n\s*\n', '\n\n', cleaned_content)
        cleaned_content = cleaned_content.strip()
        
        return cleaned_content
    
    def _check_handoff_in_content(self, content: str) -> str:
        """Determine next node based on handoff signals in content."""
        content_upper = content.upper()
        
        if any(phrase in content_upper for phrase in ["READY FOR TECHNICAL ANALYSIS", "HANDOFF TO ANALYST"]):
            return "analyst"
        elif any(phrase in content_upper for phrase in ["READY FOR ARCHITECTURE DESIGN", "HANDOFF TO ARCHITECT"]):
            return "architect"
        elif any(phrase in content_upper for phrase in ["READY FOR DEVELOPMENT", "HANDOFF TO DEVELOPER"]):
            return "developer"
        elif any(phrase in content_upper for phrase in ["READY FOR REVIEW", "HANDOFF TO REVIEWER"]):
            return "reviewer"
        elif any(phrase in content_upper for phrase in ["READY FOR TESTING", "HANDOFF TO TESTER"]):
            return "tester"
        elif any(phrase in content_upper for phrase in ["READY FOR DOCUMENTATION", "HANDOFF TO TECH_WRITER"]):
            return "tech_writer"
        elif any(phrase in content_upper for phrase in ["COMPLETE DOCUMENTATION DELIVERED", "FINAL ANSWER"]):
            return END
        else:
            return END
    
    def _create_node_function(self, role: TeamRole, agent, next_roles: List[str]):
        """Create a node function that processes agent responses and manages handoffs."""
        
        def node_function(state: MessagesState):
            try:
                logger.info(f"🤖 {role.value.upper()} starting work...")
                
                # Run the agent
                result = agent.invoke(state)
                
                # Get the content and clean it
                if result["messages"]:
                    content = result["messages"][-1].content
                    cleaned_content = self._clean_agent_output(content)
                    
                    # Create cleaned message
                    cleaned_message = HumanMessage(
                        content=cleaned_content,
                        name=role.value
                    )
                    
                    # Determine next step
                    goto = self._check_handoff_in_content(content)
                    if goto == END and next_roles:
                        goto = next_roles[0]  # Fallback to next role if no handoff detected
                    
                    logger.info(f"✅ {role.value.upper()} completed. Next: {goto}")
                    
                    return Command(
                        update={"messages": [cleaned_message]},
                        goto=goto
                    )
                else:
                    # No messages generated, move to next role
                    next_step = next_roles[0] if next_roles else END
                    return Command(goto=next_step)
                    
            except Exception as e:
                logger.error(f"❌ Error in {role.value}: {str(e)}")
                error_message = HumanMessage(
                    content=f"An error occurred in {role.value}. Moving to next step.",
                    name=role.value
                )
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
        
        # Add nodes with proper handoff sequences
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
        
        # Start with product owner
        workflow.add_edge(START, "product_owner")
        
        return workflow.compile()
    
    def collaborate_on_project(self, project_request: ProjectRequest) -> Dict:
        """Execute team collaboration with clean output."""
        try:
            logger.info("🚀 Starting Improved Software Development Team Collaboration...")
            logger.info(f"Project: {project_request.description[:100]}...")
            
            # Prepare the input message
            input_message = f"""
Project Requirements: {project_request.description}

{f"Additional Context: {project_request.file_content}" if project_request.file_content else ""}

Please deliver a complete software solution with comprehensive documentation, diagrams, and implementation guidance.
            """.strip()
            
            # Execute the workflow
            events = self.team_graph.stream(
                {"messages": [("user", input_message)]},
                {"recursion_limit": 50}
            )
            
            # Collect clean agent outputs
            agent_outputs = {}
            
            for event in events:
                for node_name, node_data in event.items():
                    if node_name != "__end__" and "messages" in node_data:
                        messages = node_data["messages"]
                        
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
                "message_count": sum(len(outputs) for outputs in agent_outputs.values())
            }
            
        except Exception as e:
            logger.error(f"Error in team collaboration: {str(e)}")
            return {
                "success": False,
                "status": "error",
                "error": str(e),
                "output": f"❌ Error in team collaboration: {str(e)}"
            }
    
    def _format_team_output(self, agent_outputs: Dict[str, List[str]]) -> str:
        """Format the team output into a comprehensive, clean report."""
        result = "# 🚀 Complete Software Solution\n\n"
        result += f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        result += f"**Collaboration Team:** {len(agent_outputs)} specialized AI agents\n\n"
        result += "---\n\n"
        
        # Define agent order and metadata
        agent_order = [
            (TeamRole.PRODUCT_OWNER, "📋", "Business Requirements & User Stories"),
            (TeamRole.ANALYST, "🔍", "Technical Specifications & Requirements"),
            (TeamRole.ARCHITECT, "🏗️", "System Architecture & Design"),
            (TeamRole.DEVELOPER, "💻", "Implementation Guide & Code Examples"),
            (TeamRole.REVIEWER, "👀", "Code Review & Quality Analysis"),
            (TeamRole.TESTER, "🧪", "Testing Strategy & Quality Assurance"),
            (TeamRole.TECH_WRITER, "📝", "Complete Documentation")
        ]
        
        # Process agents in order
        for role, icon, description in agent_order:
            role_key = role.value
            if role_key in agent_outputs and agent_outputs[role_key]:
                result += f"## {icon} {description}\n\n"
                
                # Combine all outputs for this agent
                combined_output = "\n\n".join(agent_outputs[role_key])
                if combined_output.strip():
                    result += f"{combined_output}\n\n"
                
                result += "---\n\n"
        
        result += "## 🎯 Solution Summary\n\n"
        result += "This comprehensive software solution includes:\n\n"
        result += "- ✅ **Business Requirements**: Complete user stories and acceptance criteria\n"
        result += "- ✅ **Technical Specifications**: Detailed functional and non-functional requirements\n"
        result += "- ✅ **System Architecture**: Visual diagrams and component design\n"
        result += "- ✅ **Implementation Guide**: Code examples and database design\n"
        result += "- ✅ **Quality Assurance**: Code review and testing strategy\n"
        result += "- ✅ **Documentation**: Complete technical and user documentation\n\n"
        result += "*🤖 Delivered by collaborative AI software development team*"
        
        return result

# Interface functions for easy integration
def create_improved_software_team(llm):
    """Create an improved software team orchestrator."""
    return ImprovedSoftwareTeamOrchestrator(llm)

def run_improved_team_collaboration(llm, project_description: str, file_content: str = None) -> str:
    """Run improved team collaboration with clean output."""
    orchestrator = ImprovedSoftwareTeamOrchestrator(llm)
    
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
    print("Improved Multi-Agent Software Development Team")
    print("=" * 50)
    print("✅ Clean output without debug statements")
    print("✅ Comprehensive Mermaid diagrams")
    print("✅ Professional documentation")
    print("✅ Code examples and implementation guides")
