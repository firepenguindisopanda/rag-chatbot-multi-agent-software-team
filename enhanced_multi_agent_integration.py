"""
Enhanced Multi-Agent Software Team - Gradio Integration
Ready-to-use interface with clean, professional output (no debug statements)
"""

import logging
from typing import Dict, List, Any
from multi_agent_software_team.clean_orchestrator import CleanSoftwareTeamOrchestrator
from multi_agent_software_team.schemas import ProjectRequest, TeamRole

logger = logging.getLogger(__name__)

class EnhancedMultiAgentTeam:
    """Simplified interface for the enhanced multi-agent software team with clean output."""
    
    def __init__(self, llm):
        """Initialize with LLM instance."""
        self.llm = llm
        self.orchestrator = CleanSoftwareTeamOrchestrator(llm)
    
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
            logger.info("🚀 Starting enhanced multi-agent software development collaboration...")
            
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
            
            # Execute the collaboration using clean sequential workflow
            result = self.orchestrator.collaborate_on_project(project_request)
            
            if result.get("success", False):
                # Use the clean formatted output from the orchestrator
                formatted_output = result.get("output", "No output generated")
                
                # Add completion context
                if result.get("agent_count"):
                    agent_count = result["agent_count"]
                    formatted_output += f"\n\n---\n\n*✅ Professional solution delivered by {agent_count} specialized AI agents*"
                
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
# 🚀 Enhanced Multi-Agent Software Development Team

## Team Composition
Our collaborative AI team consists of specialized agents:

### 👥 **Core Team Members**
1. **📋 Product Owner** - Defines requirements and user stories
2. **🔍 Analyst** - Creates detailed technical specifications  
3. **🏗️ Architect** - Designs system architecture and diagrams
4. **💻 Developer** - Implements code and database design
5. **👀 Reviewer** - Conducts code quality and security review
6. **🧪 Tester** - Creates comprehensive testing strategy
7. **📝 Tech Writer** - Produces final documentation

## 🔄 **Workflow Features**
- **Sequential Execution**: Reliable step-by-step collaboration
- **Clean Output**: Professional deliverables without debug statements
- **Context Preservation**: Each agent builds upon previous work
- **Mermaid Diagrams**: Automatic generation of technical diagrams
- **Comprehensive Documentation**: Complete solution from concept to deployment

## 🎯 **What You Get**
- Business requirements and user stories
- Technical specifications and constraints
- System architecture with visual diagrams
- Database design and implementation code
- Code review and security analysis
- Testing strategy and test cases
- Complete technical documentation

## 💡 **Best Results Tips**
- Provide clear, detailed project descriptions
- Mention specific technologies if you have preferences
- Include any constraints or special requirements
- Upload relevant files for additional context

Ready to transform your idea into a complete software solution!
"""
        return team_info
    
    def run_example(self) -> str:
        """Run an example to demonstrate the system."""
        example_project = """
        Create a task management system for small development teams. 
        The system should allow:
        - User registration and authentication
        - Project creation and management
        - Task assignment and tracking
        - Team collaboration features
        - Progress reporting and analytics
        
        The system should be web-based, scalable, and secure.
        Target users: 10-50 person development teams
        Technology preference: Modern web stack (React, Node.js)
        """
        
        return self.create_software_solution(example_project)

# Factory function for easy Gradio integration
def create_enhanced_multi_agent_team(llm):
    """Create an enhanced multi-agent team instance for Gradio integration."""
    return EnhancedMultiAgentTeam(llm)

# Simple function interface for Gradio
def run_enhanced_multi_agent_collaboration(llm, project_description: str, file_content: str = None) -> str:
    """
    Simple function interface for Gradio integration with modern LangGraph implementation.
    
    Usage in Gradio:
    ```python
    import gradio as gr
    from your_llm_module import your_llm  # Replace with your LLM
    from enhanced_multi_agent_integration import run_enhanced_multi_agent_collaboration
    
    def gradio_interface(project_description, file_upload):
        file_content = file_upload.read().decode('utf-8') if file_upload else None
        return run_enhanced_multi_agent_collaboration(your_llm, project_description, file_content)
    
    iface = gr.Interface(
        fn=gradio_interface,
        inputs=[
            gr.Textbox(label="Project Description", lines=5, placeholder="Describe your software project..."),
            gr.File(label="Upload Context File (optional)", file_types=[".txt", ".md"])
        ],
        outputs=gr.Markdown(label="Multi-Agent Team Results"),
        title="🚀 Enhanced Multi-Agent Software Development Team",
        description="Collaborative AI team that transforms your ideas into complete software solutions"
    )
    
    iface.launch()
    ```
    """
    team = EnhancedMultiAgentTeam(llm)
    return team.create_software_solution(project_description, file_content)

if __name__ == "__main__":
    print("Enhanced Multi-Agent Software Team - Gradio Integration")
    print("=" * 60)
    print("This module provides a simplified interface for Gradio integration.")
    print("Use create_enhanced_multi_agent_team(llm) or run_enhanced_multi_agent_collaboration(llm, description)")
    print("\nFeatures:")
    print("• Handoff-based workflow between specialized agents")
    print("• Automatic diagram generation with Mermaid")
    print("• Comprehensive software solution delivery")
    print("• Error handling and graceful degradation")
    print("• Ready for Gradio UI integration")
