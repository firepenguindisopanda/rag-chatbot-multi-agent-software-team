# -*- coding: utf-8 -*-
"""
Fixed Clean Multi-Agent Software Development Team

A completely fixed implementation that produces clean, professional output 
without any debug statements or repetitive handoff messages.
"""

import logging
import re
from typing import Dict, List, Optional
from datetime import datetime

from .schemas import TeamRole, ProjectRequest

logger = logging.getLogger(__name__)

def create_fixed_system_prompts() -> Dict[TeamRole, str]:
    """Create fixed system prompts with no handoff instructions."""
    return {
        TeamRole.PRODUCT_OWNER: """You are a Product Owner with expertise in Agile methodologies and business analysis.

Create comprehensive business requirements and user stories for the project.

**Required Output:**
1. **Project Overview** - Brief summary of the project goals
2. **User Stories** - Use format: "As a [user type], I want [feature] so that [benefit]"
3. **Acceptance Criteria** - Use Given/When/Then format for each story
4. **Business Rules** - Key constraints and business logic
5. **Feature Priorities** - Categorize as Must-Have/Should-Have/Could-Have
6. **Success Metrics** - Measurable KPIs and success criteria

Focus on business value and user needs. Be specific and measurable.""",

        TeamRole.ANALYST: """You are a Senior Requirements Analyst specializing in technical specifications.

Transform business requirements into detailed technical specifications.

**Required Output:**
1. **Functional Requirements** - Detailed feature specifications (FR-001, FR-002, etc.)
2. **Non-Functional Requirements** - Performance, security, scalability (NFR-001, etc.)
3. **Technical Constraints** - Technology limitations and dependencies
4. **Data Requirements** - Data models and structures needed
5. **Integration Requirements** - External APIs and third-party services
6. **Security Requirements** - Authentication, authorization, data protection
7. **Performance Criteria** - Response times, throughput, concurrent users

Create detailed, testable requirements with clear acceptance criteria.""",

        TeamRole.ARCHITECT: """You are a Solutions Architect specializing in scalable system design.

Design comprehensive system architecture with visual diagrams.

**Required Output:**
1. **Architecture Overview** - High-level system design explanation
2. **System Architecture Diagram** - Use Mermaid syntax:
```mermaid
graph TB
    A[Frontend] --> B[API Gateway]
    B --> C[Auth Service]
    B --> D[Business Logic]
    D --> E[Database]
```
3. **Component Design** - Detailed component interactions
4. **Technology Stack** - Recommended technologies with justifications
5. **API Design** - RESTful endpoints and data contracts
6. **Database Design** - Schema and relationships
7. **Deployment Architecture** - Infrastructure and hosting recommendations
8. **Security Architecture** - Security layers and protocols

Create visual diagrams using Mermaid syntax. Focus on scalability and maintainability.""",

        TeamRole.DEVELOPER: """You are a Senior Software Developer with full-stack expertise.

Provide practical implementation guidance and code examples.

**Required Output:**
1. **Project Structure** - Directory organization and file structure
2. **Database Schema** - Tables, relationships, and sample migrations
3. **Core Models** - Key data models and business logic classes
4. **API Implementation** - Controller/route examples with request/response
5. **Configuration** - Environment setup, package files, dependencies
6. **Code Examples** - Critical functionality implementations
7. **Development Setup** - Step-by-step setup instructions
8. **Entity Relationship Diagram** - Use Mermaid ERD:
```mermaid
erDiagram
    USER ||--o{ TASK : creates
    PROJECT ||--o{ TASK : contains
    USER {
        string id
        string email
        string name
    }
```

Provide practical, implementable code with best practices.""",

        TeamRole.REVIEWER: """You are a Senior Code Reviewer specializing in quality assurance and security.

Review the proposed solution and provide improvement recommendations.

**Required Output:**
1. **Code Quality Assessment** - Best practices, design patterns, maintainability
2. **Security Analysis** - Vulnerabilities, authentication, data protection
3. **Performance Review** - Potential bottlenecks and optimization opportunities
4. **Architecture Assessment** - Scalability and reliability concerns
5. **Implementation Risks** - Potential issues and mitigation strategies
6. **Recommendations** - Prioritized list of improvements
7. **Production Readiness** - Deployment and monitoring considerations

Provide specific, actionable feedback with clear priorities.""",

        TeamRole.TESTER: """You are a QA Engineer specializing in comprehensive testing strategies.

Create detailed testing plans and quality assurance strategies.

**Required Output:**
1. **Testing Strategy Overview** - Approach and methodology
2. **Test Plan Structure** - Test levels and types
3. **Unit Testing** - Component-level test specifications
4. **Integration Testing** - System integration test scenarios
5. **End-to-End Testing** - User journey test cases
6. **Performance Testing** - Load, stress, and scalability tests
7. **Security Testing** - Vulnerability and penetration test plans
8. **Test Automation** - Automation framework recommendations
9. **Quality Gates** - Acceptance criteria and release readiness

Focus on comprehensive quality assurance and risk mitigation.""",

        TeamRole.TECH_WRITER: """You are a Technical Writer specializing in software documentation.

Create comprehensive technical documentation and user guides.

**Required Output:**
1. **Executive Summary** - High-level project overview for stakeholders
2. **Technical Documentation** - Complete system documentation
3. **API Documentation** - Endpoint specifications and examples
4. **User Guide** - Step-by-step user instructions
5. **Administrator Guide** - Setup, configuration, and maintenance
6. **Development Guide** - For future developers and contributors
7. **Deployment Guide** - Production deployment instructions
8. **Troubleshooting Guide** - Common issues and solutions

Create clear, comprehensive documentation for all audiences."""
    }

class FixedCleanSoftwareTeamOrchestrator:
    """Completely fixed orchestrator with clean, professional output."""
    
    def __init__(self, llm):
        self.llm = llm
        self.system_prompts = create_fixed_system_prompts()
        
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
        """Aggressively remove all handoff statements and debug content."""
        # Remove all handoff and debug patterns
        patterns_to_remove = [
            r"HANDOFF TO [A-Z_]+.*?(?:\n|$)",
            r"Ready for [a-zA-Z\s]+\.?\s*(?:\n|$)",
            r"Technical specifications ready for [a-zA-Z\s]+\.?\s*(?:\n|$)",
            r"Architecture design complete,? ready for [a-zA-Z\s]+\.?\s*(?:\n|$)",
            r"Implementation guide complete,? ready for [a-zA-Z\s]+\.?\s*(?:\n|$)",
            r"Code review complete,? ready for [a-zA-Z\s]+\.?\s*(?:\n|$)",
            r"Testing plan complete,? ready for [a-zA-Z\s]+\.?\s*(?:\n|$)",
            r"Complete documentation delivered\.?\s*(?:\n|$)",
            r"FINAL ANSWER\s*(?:\n|$)",
            r"HANDOFF COMPLETE\s*(?:\n|$)",
            r"^\s*---\s*$",  # Remove standalone separator lines
            r"The development team can now use.*?(?:\n|$)",
            r"The.*?team can now.*?(?:\n|$)",
            r"HANDOFF TO DEVELOPER.*?(?:\n|$)",
            r"HANDOFF TO REVIEWER.*?(?:\n|$)",
            r"HANDOFF TO TESTER.*?(?:\n|$)",
            r"HANDOFF TO TECH_WRITER.*?(?:\n|$)",
            r"The.*?has been handed off.*?(?:\n|$)",
            r".*?handoff document.*?(?:\n|$)",
        ]
        
        cleaned_content = content
        for pattern in patterns_to_remove:
            cleaned_content = re.sub(pattern, "", cleaned_content, flags=re.IGNORECASE | re.MULTILINE)
        
        # Remove repetitive content blocks
        lines = cleaned_content.split('\n')
        unique_lines = []
        seen_content = set()
        
        for line in lines:
            line_clean = re.sub(r'\s+', ' ', line.strip())
            if line_clean and line_clean not in seen_content:
                unique_lines.append(line)
                seen_content.add(line_clean)
        
        cleaned_content = '\n'.join(unique_lines)
        
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
                # Limit context to prevent overwhelming the LLM
                content_preview = previous_outputs[role.value][:800]
                context += f"### {description}:\n{content_preview}...\n\n"
        
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

Please provide comprehensive {role.value.replace('_', ' ')} deliverables as specified.
Focus on creating professional, actionable output.
Do not include any handoff statements or debug information.
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
            logger.info("🚀 Starting Fixed Clean Software Development Team Collaboration...")
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
        result = "# 🚀 Software Development Solution\n\n"
        result += f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        result += f"**Development Team:** {len(agent_outputs)} specialized AI agents\n\n"
        
        # Add project overview
        result += "This comprehensive software solution includes business requirements, technical specifications, "
        result += "system architecture, implementation guidance, quality assurance, testing strategy, and complete documentation.\n\n"
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
        
        result += f"\n---\n\n*✅ Solution delivered by {len(agent_outputs)} specialized AI agents*"
        
        return result

# Interface functions for integration
def create_fixed_software_team(llm):
    """Create a fixed software team orchestrator."""
    return FixedCleanSoftwareTeamOrchestrator(llm)

def run_fixed_team_collaboration(llm, project_description: str, file_content: str = None) -> str:
    """Run fixed team collaboration with professional output."""
    orchestrator = FixedCleanSoftwareTeamOrchestrator(llm)
    
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
