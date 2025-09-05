# -*- coding: utf-8 -*-
"""
Clean Multi-Agent Software Development Team

A simplified implementation that produces clean, professional output without debug statements.
Uses sequential execution instead of complex graph workflows for better reliability.
"""

import logging
import re
from typing import Dict, List, Optional
from datetime import datetime

from .schemas import TeamRole, ProjectRequest
from .improved_prompts import create_improved_system_prompts

logger = logging.getLogger(__name__)

class CleanSoftwareTeamOrchestrator:
    """Clean orchestrator that produces professional output without handoff statements."""
    
    def __init__(self, llm):
        self.llm = llm
        self.system_prompts = create_improved_system_prompts()
        
        # Define the execution order
        self.execution_order = [
            TeamRole.PRODUCT_OWNER,
            TeamRole.ANALYST,
            TeamRole.ARCHITECT,
            TeamRole.DEVELOPER,
            TeamRole.REVIEWER,
            TeamRole.TESTER,
            TeamRole.TECH_WRITER
        ]
    
    def _clean_agent_output(self, content: str) -> str:
        """Remove handoff statements and debug information from agent output."""
        # Remove handoff statements and similar debug content
        patterns_to_remove = [
            r"HANDOFF TO [A-Z_]+",
            r"Ready for [a-z\s]+\.?$",
            r"Technical specifications ready for [a-z\s]+\.?$",
            r"Architecture design complete,? ready for [a-z\s]+\.?$",
            r"Implementation guide complete,? ready for [a-z\s]+\.?$",
            r"Code review complete,? ready for [a-z\s]+\.?$",
            r"Testing plan complete,? ready for [a-z\s]+\.?$",
            r"Complete documentation delivered\.?$",
            r"FINAL ANSWER",
            r"^\s*---\s*$",  # Remove standalone separator lines
        ]
        
        cleaned_content = content
        for pattern in patterns_to_remove:
            cleaned_content = re.sub(pattern, "", cleaned_content, flags=re.IGNORECASE | re.MULTILINE)
        
        # Clean up extra whitespace and newlines
        cleaned_content = re.sub(r'\n\s*\n\s*\n+', '\n\n', cleaned_content)
        cleaned_content = re.sub(r'(^\s+)|(\s+$)', '', cleaned_content, flags=re.MULTILINE)
        cleaned_content = cleaned_content.strip()
        
        return cleaned_content
    
    def _get_agent_context(self, previous_outputs: Dict[str, str], current_role: TeamRole) -> str:
        """Build context from previous agent outputs."""
        if not previous_outputs:
            return ""
        
        context = "\n\n## Previous Team Work:\n\n"
        
        role_order = [
            (TeamRole.PRODUCT_OWNER, "Business Requirements"),
            (TeamRole.ANALYST, "Technical Specifications"),
            (TeamRole.ARCHITECT, "System Architecture"),
            (TeamRole.DEVELOPER, "Implementation Guide"),
            (TeamRole.REVIEWER, "Code Review"),
            (TeamRole.TESTER, "Testing Strategy"),
        ]
        
        for role, description in role_order:
            if role == current_role:
                break
            if role.value in previous_outputs:
                context += f"### {description}:\n{previous_outputs[role.value][:500]}...\n\n"
        
        return context
    
    def _execute_agent(self, role: TeamRole, project_description: str, context: str) -> str:
        """Execute a single agent with clean output."""
        try:
            logger.info(f"🤖 Executing {role.value.replace('_', ' ').title()}...")
            
            # Build the complete prompt
            system_prompt = self.system_prompts[role]
            user_message = f"""
Project Description:
{project_description}

{context}

Please provide comprehensive {role.value.replace('_', ' ')} deliverables as specified in your role instructions.
Focus on creating professional, actionable output without debug statements or handoff instructions.
            """.strip()
            
            # Create messages for the LLM
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ]
            
            # Get response from LLM
            response = self.llm.invoke(messages)
            
            # Extract content and clean it
            if hasattr(response, 'content'):
                content = response.content
            else:
                content = str(response)
            
            cleaned_content = self._clean_agent_output(content)
            
            logger.info(f"✅ {role.value.replace('_', ' ').title()} completed successfully")
            return cleaned_content
            
        except Exception as e:
            logger.error(f"❌ Error in {role.value}: {str(e)}")
            return f"Error occurred in {role.value.replace('_', ' ')}: {str(e)}"
    
    def collaborate_on_project(self, project_request: ProjectRequest) -> Dict:
        """Execute team collaboration with clean, sequential processing."""
        try:
            logger.info("🚀 Starting Clean Software Development Team Collaboration...")
            logger.info(f"Project: {project_request.description[:100]}...")
            
            # Prepare the project description
            full_description = project_request.description
            if project_request.file_content:
                full_description += f"\n\nAdditional Context:\n{project_request.file_content}"
            
            # Execute agents sequentially
            agent_outputs = {}
            
            for role in self.execution_order:
                # Build context from previous outputs
                context = self._get_agent_context(agent_outputs, role)
                
                # Execute the agent
                output = self._execute_agent(role, full_description, context)
                
                if output.strip():
                    agent_outputs[role.value] = output
            
            # Format the final result
            formatted_output = self._format_team_output(agent_outputs)
            
            return {
                "success": True,
                "status": "completed",
                "output": formatted_output,
                "agent_outputs": agent_outputs,
                "agent_count": len(agent_outputs)
            }
            
        except Exception as e:
            logger.error(f"Error in team collaboration: {str(e)}")
            return {
                "success": False,
                "status": "error",
                "error": str(e),
                "output": f"❌ Error in team collaboration: {str(e)}"
            }
    
    def _format_team_output(self, agent_outputs: Dict[str, str]) -> str:
        """Format the team output into a comprehensive, clean report."""
        result = "# 🚀 Complete Software Solution\n\n"
        result += f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        result += f"**Development Team:** {len(agent_outputs)} specialized AI agents\n\n"
        
        # Add project overview
        result += "## 📋 Project Overview\n\n"
        result += "This comprehensive software solution was developed through collaboration between specialized AI agents, "
        result += "each contributing their expertise to deliver a complete, production-ready solution.\n\n"
        result += "---\n\n"
        
        # Define agent order and metadata
        agent_sections = [
            (TeamRole.PRODUCT_OWNER, "📋", "Business Requirements & User Stories"),
            (TeamRole.ANALYST, "🔍", "Technical Specifications & Analysis"),
            (TeamRole.ARCHITECT, "🏗️", "System Architecture & Design"),
            (TeamRole.DEVELOPER, "💻", "Implementation Guide & Code"),
            (TeamRole.REVIEWER, "👀", "Code Review & Quality Assurance"),
            (TeamRole.TESTER, "🧪", "Testing Strategy & QA Plan"),
            (TeamRole.TECH_WRITER, "📝", "Documentation & User Guides")
        ]
        
        # Process agents in order
        for role, icon, description in agent_sections:
            role_key = role.value
            if role_key in agent_outputs and agent_outputs[role_key].strip():
                result += f"## {icon} {description}\n\n"
                result += f"{agent_outputs[role_key]}\n\n"
                result += "---\n\n"
        
        # Add solution summary
        result += "## 🎯 Solution Deliverables\n\n"
        result += "This software solution includes:\n\n"
        
        deliverables = [
            "✅ **Business Analysis**: User stories, acceptance criteria, and business requirements",
            "✅ **Technical Specifications**: Detailed functional and non-functional requirements",
            "✅ **System Architecture**: Visual diagrams, component design, and technical infrastructure",
            "✅ **Implementation Guide**: Code examples, database design, and development instructions",
            "✅ **Quality Assurance**: Code review guidelines and security analysis",
            "✅ **Testing Strategy**: Comprehensive test plans and quality validation",
            "✅ **Documentation**: Complete technical documentation and user guides"
        ]
        
        for deliverable in deliverables:
            result += f"- {deliverable}\n"
        
        result += f"\n*🤖 Delivered by {len(agent_outputs)} collaborative AI specialists*\n"
        result += f"*📅 Generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}*"
        
        return result

# Interface functions for integration
def create_clean_software_team(llm):
    """Create a clean software team orchestrator."""
    return CleanSoftwareTeamOrchestrator(llm)

def run_clean_team_collaboration(llm, project_description: str, file_content: str = None) -> str:
    """Run clean team collaboration with professional output."""
    orchestrator = CleanSoftwareTeamOrchestrator(llm)
    
    project_request = ProjectRequest(
        description=project_description,
        file_content=file_content,
        selected_roles=None
    )
    
    result = orchestrator.collaborate_on_project(project_request)
    
    if result["success"]:
        return result["output"]
    else:
        return f"❌ **Error**: {result.get('error', 'Unknown error occurred')}\n\nPlease try again with a clearer project description."

if __name__ == "__main__":
    print("Clean Multi-Agent Software Development Team")
    print("=" * 45)
    print("✅ Professional output without debug statements")
    print("✅ Sequential execution for reliability")
    print("✅ Comprehensive documentation and diagrams")
    print("✅ Ready for production use")
