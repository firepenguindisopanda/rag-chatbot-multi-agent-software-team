#!/usr/bin/env python3
"""
Demo script showing how to use the Multi-Agent Software Team programmatically.
This demonstrates the API usage outside of the Gradio interface.
"""

import os
import sys

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def demo_multi_agent_team():
    """Demonstrate the multi-agent software team functionality."""
    
    # Import the multi-agent components
    from multi_agent_software_team import (
        TeamRole, 
        ProjectRequest, 
        Orchestrator,
        format_agent_responses,
        validate_team_composition
    )
    
    print("🤖 Multi-Agent Software Team API Demo")
    print("=" * 50)
    
    # Example project description
    project_description = """
    Create a modern task management web application for small businesses with the following features:
    
    Core Features:
    - User registration and authentication system
    - Project creation and management
    - Task assignment with priorities and due dates
    - Team collaboration tools
    - Progress tracking and reporting
    - Email notifications for important updates
    
    Technical Requirements:
    - Responsive web design (mobile-friendly)
    - RESTful API backend
    - Database for data persistence
    - User role management (admin, manager, employee)
    - Real-time updates for collaborative features
    - Export functionality for reports
    
    Business Requirements:
    - Support for up to 100 users per organization
    - Multi-tenant architecture for different companies
    - Integration with popular email providers
    - Basic analytics and reporting dashboard
    """
    
    print(f"📋 Project Description:")
    print(project_description[:200] + "..." if len(project_description) > 200 else project_description)
    print()
    
    # Create project request
    project_request = ProjectRequest(
        description=project_description,
        file_content=None  # Could include file content here
    )
    
    # Define the team composition
    selected_roles = [
        TeamRole.PRODUCT_OWNER,
        TeamRole.ANALYST,
        TeamRole.ARCHITECT, 
        TeamRole.DEVELOPER,
        TeamRole.TESTER
    ]
    
    print(f"👥 Selected Team Roles:")
    for role in selected_roles:
        print(f"   • {role.value.replace('_', ' ').title()}")
    print()
    
    # Validate team composition
    is_valid, validation_message = validate_team_composition(selected_roles)
    print(f"✅ Team Validation: {validation_message}")
    print()
    
    if not is_valid:
        print("❌ Invalid team composition. Please fix the team and try again.")
        return
    
    # Note: For this demo, we're not actually running the LLM
    # because it requires API keys and actual LLM setup
    print("🔄 In a real scenario, this would:")
    print("   1. Initialize the LLM (ChatNVIDIA or other)")
    print("   2. Create the Orchestrator with the LLM")
    print("   3. Run the selected roles sequentially")
    print("   4. Each agent would build upon previous outputs")
    print("   5. Return comprehensive results from all agents")
    print()
    
    # Show what the orchestrator would do
    print("📋 Execution Plan:")
    from multi_agent_software_team.orchestrator import Orchestrator
    
    # Create a mock orchestrator (without LLM for demo)
    class MockLLM:
        def invoke(self, text):
            class MockResponse:
                content = f"Mock response for {text[:50]}..."
            return MockResponse()
    
    mock_orchestrator = Orchestrator(MockLLM())
    execution_plan = mock_orchestrator.get_execution_plan(selected_roles)
    print(execution_plan)
    
    print("🎯 Example Real Usage:")
    print("""
    # With a real LLM setup:
    from langchain_nvidia_ai_endpoints import ChatNVIDIA
    
    llm = ChatNVIDIA(model="meta/llama-3.1-405b-instruct")
    orchestrator = Orchestrator(llm)
    responses = orchestrator.run_selected_roles(project_request, selected_roles)
    
    # Format and display results
    formatted_output = format_agent_responses(responses)
    print(formatted_output)
    """)

if __name__ == "__main__":
    demo_multi_agent_team()
