"""
Fixed Enhanced Multi-Agent Software Team - Gradio Integration
Clean, professional output without any debug statements or repetitive handoffs
"""

import logging
from typing import Dict, List, Any
from multi_agent_software_team.fixed_clean_orchestrator import FixedCleanSoftwareTeamOrchestrator
from multi_agent_software_team.schemas import ProjectRequest, TeamRole

logger = logging.getLogger(__name__)

class FixedEnhancedMultiAgentTeam:
    """Fixed interface for the enhanced multi-agent software team with completely clean output."""
    
    def __init__(self, llm):
        """Initialize with LLM instance."""
        self.llm = llm
        self.orchestrator = FixedCleanSoftwareTeamOrchestrator(llm)
    
    def create_software_solution(self, project_description: str, file_content: str = None) -> str:
        """
        Create a software solution using the multi-agent team.
        
        Args:
            project_description: Description of the software project
            file_content: Optional additional context from uploaded file
            
        Returns:
            Formatted string with the complete team collaboration results
        """
        try:
            logger.info("🚀 Starting fixed enhanced multi-agent software development collaboration...")
            
            # Prepare project request
            full_description = project_description
            if file_content and file_content.strip():
                full_description += f"\n\nAdditional Context:\n{file_content}"
            
            project_request = ProjectRequest(
                description=full_description,
                selected_roles=[
                    TeamRole.PRODUCT_OWNER,
                    TeamRole.ANALYST,
                    TeamRole.ARCHITECT,
                    TeamRole.DEVELOPER,
                    TeamRole.REVIEWER,
                    TeamRole.TESTER,
                    TeamRole.TECH_WRITER
                ]
            )
            
            # Execute the collaboration using fixed clean sequential workflow
            result = self.orchestrator.collaborate_on_project(project_request)
            
            if result.get("success", False):
                # Use the clean formatted output from the orchestrator
                formatted_output = result.get("output", "No output generated")
                
                return formatted_output
            else:
                error_msg = result.get("error", "Unknown error occurred")
                logger.error(f"Multi-agent collaboration failed: {error_msg}")
                return f"❌ **Error in Multi-Agent Collaboration**\n\n{error_msg}\n\nPlease try again with a clearer project description."
                
        except Exception as e:
            logger.error(f"Exception in multi-agent team: {str(e)}")
            return f"❌ **System Error**\n\nAn unexpected error occurred: {str(e)}\n\nPlease try again."
    
    def get_team_info(self) -> str:
        """Get information about the multi-agent team."""
        team_info = """
# 🚀 Fixed Enhanced Multi-Agent Software Development Team

## Team Composition
Our collaborative AI team consists of specialized agents working sequentially:

### 👥 **Core Team Members**
1. **📋 Product Owner** - Defines requirements and user stories
2. **🔍 Analyst** - Creates detailed technical specifications  
3. **🏗️ Architect** - Designs system architecture with Mermaid diagrams
4. **💻 Developer** - Implements code examples and database design
5. **👀 Reviewer** - Conducts code quality and security review
6. **🧪 Tester** - Creates comprehensive testing strategy
7. **📝 Tech Writer** - Produces complete technical documentation

## 🔄 **Fixed Workflow Features**
- **Sequential Execution**: Reliable step-by-step collaboration
- **Clean Output**: Professional deliverables without any debug statements
- **No Handoff Messages**: Eliminated all repetitive handoff statements
- **Context Preservation**: Each agent builds upon previous work
- **Mermaid Diagrams**: Automatic generation of technical diagrams
- **Complete Documentation**: Full solution from concept to deployment

## 🎯 **What You Get**
- Business requirements and user stories
- Technical specifications and constraints
- System architecture with visual diagrams
- Implementation guidance with code examples
- Security and performance analysis
- Comprehensive testing strategy
- Complete technical documentation

## ✅ **Fixed Issues**
- ❌ Removed repetitive "HANDOFF" statements
- ❌ Eliminated debug output
- ✅ Tech Writer now produces full documentation
- ✅ Clean, professional output only
- ✅ Proper Mermaid diagram generation
"""
        return team_info

# Simple function interface for Gradio
def run_fixed_enhanced_multi_agent_collaboration(llm, project_description: str, file_content: str = None) -> str:
    """
    Fixed function interface for Gradio integration with clean output.
    
    Args:
        llm: Language model instance
        project_description: Description of the software project
        file_content: Optional file content for additional context
        
    Returns:
        Clean, professional multi-agent collaboration output
    """
    team = FixedEnhancedMultiAgentTeam(llm)
    return team.create_software_solution(project_description, file_content)

if __name__ == "__main__":
    print("Fixed Enhanced Multi-Agent Software Team - Gradio Integration")
    print("=" * 70)
    print("This module provides a completely fixed interface for clean output.")
    print("Use run_fixed_enhanced_multi_agent_collaboration(llm, description)")
    print("\nFixed Features:")
    print("• ✅ No handoff statements or debug output")
    print("• ✅ Sequential workflow between specialized agents")
    print("• ✅ Automatic diagram generation with Mermaid")
    print("• ✅ Complete technical documentation")
    print("• ✅ Professional software solution delivery")
    print("• ✅ Ready for production Gradio UI integration")
