"""
Demo script for the Enhanced Multi-Agent Software Team
Demonstrates the improved handoff-based workflow
"""

import sys
import os
import logging
from typing import Dict, Any

# Add the parent directory to the path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def demo_enhanced_multi_agent_workflow():
    """Demonstrate the enhanced multi-agent workflow with handoff mechanisms."""
    
    try:
        # Mock LLM for demonstration (replace with actual LLM)
        class MockLLM:
            def invoke(self, prompt):
                class MockResponse:
                    def __init__(self, content):
                        self.content = content
                
                # Extract role from prompt to provide appropriate response
                if "Product Owner" in prompt:
                    return MockResponse("""
# Product Owner Analysis

## User Stories
1. As a developer, I want to create and manage projects so that I can organize my tasks effectively
2. As a team member, I want to be assigned tasks so that I can contribute to project completion
3. As a project manager, I want to track progress so that I can ensure timely delivery

## Acceptance Criteria
- Users can register and authenticate securely
- Projects can be created with title, description, and team assignments
- Tasks can be assigned to specific team members
- Progress tracking with visual indicators

## Business Rules
- Only authenticated users can access the system
- Project owners can assign tasks to team members
- Tasks must have clear descriptions and deadlines

## Priority Level
Must-Have: Authentication, Project Creation, Task Assignment
Should-Have: Progress Tracking, Team Collaboration
Could-Have: Advanced Analytics, Reporting

HANDOFF TO ANALYST
""")
                
                elif "Analyst" in prompt:
                    return MockResponse("""
# Requirements Analysis

## Functional Requirements
FR-001: User authentication and authorization system
FR-002: Project creation and management interface
FR-003: Task assignment and tracking functionality
FR-004: Team collaboration features
FR-005: Progress reporting and analytics

## Non-Functional Requirements
NFR-001: System must support 100+ concurrent users
NFR-002: Response time under 2 seconds for all operations
NFR-003: 99.9% uptime availability
NFR-004: Data encryption in transit and at rest

## Technical Constraints
- Web-based application using modern frameworks
- Database must handle relational data efficiently
- Mobile-responsive design required

## Performance Criteria
- Load time: < 3 seconds
- API response: < 500ms
- Database queries: < 100ms

HANDOFF TO ARCHITECT
""")
                
                elif "Architect" in prompt:
                    return MockResponse("""
# System Architecture Design

## System Overview
The task management system will use a microservices architecture with:
- Frontend: React.js with TypeScript
- Backend: Node.js with Express
- Database: PostgreSQL with Redis cache
- Authentication: JWT with refresh tokens

## Component Architecture
```mermaid
graph TD
    Client[Web Client] --> Gateway[API Gateway]
    Gateway --> Auth[Auth Service]
    Gateway --> Projects[Project Service]
    Gateway --> Tasks[Task Service]
    Gateway --> Users[User Service]
    
    Auth --> AuthDB[(Auth DB)]
    Projects --> ProjectDB[(Project DB)]
    Tasks --> TaskDB[(Task DB)]
    Users --> UserDB[(User DB)]
    
    Gateway --> Cache[(Redis Cache)]
```

## API Specifications
- REST API with OpenAPI 3.0 documentation
- GraphQL endpoint for complex queries
- WebSocket for real-time updates

## Security Architecture
- OAuth 2.0 + JWT authentication
- Role-based access control (RBAC)
- API rate limiting and input validation

## Technology Stack
- Frontend: React 18, TypeScript, Tailwind CSS
- Backend: Node.js 18, Express, TypeORM
- Database: PostgreSQL 14, Redis 6
- Infrastructure: Docker, Kubernetes, AWS

HANDOFF TO DEVELOPER
""")
                
                elif "Developer" in prompt:
                    return MockResponse("""
# Implementation Plan

## Database Schema
```mermaid
erDiagram
    USER {
        uuid id PK
        string email UK
        string name
        string password_hash
        timestamp created_at
        timestamp updated_at
    }
    PROJECT {
        uuid id PK
        string title
        text description
        uuid owner_id FK
        timestamp created_at
        timestamp updated_at
    }
    TASK {
        uuid id PK
        string title
        text description
        uuid project_id FK
        uuid assignee_id FK
        enum status
        date due_date
        timestamp created_at
        timestamp updated_at
    }
    
    USER ||--o{ PROJECT : owns
    USER ||--o{ TASK : assigned_to
    PROJECT ||--o{ TASK : contains
```

## Core Implementation
```javascript
// User Model (TypeORM)
@Entity()
export class User {
  @PrimaryGeneratedColumn('uuid')
  id: string;
  
  @Column({ unique: true })
  email: string;
  
  @Column()
  name: string;
  
  @Column()
  passwordHash: string;
  
  @OneToMany(() => Project, project => project.owner)
  projects: Project[];
  
  @OneToMany(() => Task, task => task.assignee)
  tasks: Task[];
}

// Project Controller
@Controller('projects')
export class ProjectController {
  @Post()
  async createProject(@Body() data: CreateProjectDto) {
    return this.projectService.create(data);
  }
  
  @Get()
  async getProjects(@Query() filters: ProjectFilters) {
    return this.projectService.findAll(filters);
  }
}
```

## API Documentation
- POST /auth/login - User authentication
- GET /projects - List user projects
- POST /projects - Create new project
- POST /tasks - Create task
- PUT /tasks/:id - Update task status

HANDOFF TO REVIEWER
""")
                
                elif "Reviewer" in prompt:
                    return MockResponse("""
# Code Review Analysis

## Code Quality Assessment
✅ **Strengths:**
- Clean separation of concerns with proper layered architecture
- TypeScript usage for type safety
- Proper use of decorators and dependency injection
- RESTful API design principles followed

⚠️ **Areas for Improvement:**
- Add input validation middleware
- Implement proper error handling with custom exceptions
- Add request logging and monitoring
- Include API rate limiting

## Security Review
✅ **Security Measures:**
- Password hashing implemented
- JWT token authentication
- SQL injection prevention with ORM

🔒 **Security Recommendations:**
- Implement CSRF protection
- Add request rate limiting
- Use HTTPS in production
- Implement session management

## Performance Review
📈 **Performance Considerations:**
- Database indexing on frequently queried fields
- Redis caching for session data
- Lazy loading for related entities
- API response compression

## Recommendations
1. Add comprehensive unit and integration tests
2. Implement proper logging with structured logs
3. Add health check endpoints
4. Use environment-specific configurations

HANDOFF TO TESTER
""")
                
                elif "Tester" in prompt:
                    return MockResponse("""
# Testing Strategy

## Test Plan Overview
Comprehensive testing strategy covering unit, integration, and end-to-end testing.

## Test Cases

### Authentication Tests
**TC-001: User Login**
- Given: Valid user credentials
- When: User submits login form
- Then: User receives JWT token and is redirected to dashboard

**TC-002: Invalid Login**
- Given: Invalid credentials
- When: User submits login form
- Then: Error message displayed, no token issued

### Project Management Tests
**TC-003: Create Project**
- Given: Authenticated user
- When: User creates new project
- Then: Project is saved and appears in project list

## Test Flow Diagram
```mermaid
flowchart TD
    Start([Start Testing]) --> Unit[Unit Tests]
    Unit --> |Pass| Integration[Integration Tests]
    Unit --> |Fail| Fix[Fix Issues]
    Fix --> Unit
    Integration --> |Pass| E2E[E2E Tests]
    Integration --> |Fail| Fix
    E2E --> |Pass| Security[Security Tests]
    E2E --> |Fail| Fix
    Security --> |Pass| Performance[Performance Tests]
    Security --> |Fail| Fix
    Performance --> |Pass| Deploy[Ready for Deployment]
    Performance --> |Fail| Fix
```

## Automated Test Scripts
```javascript
// Jest unit test example
describe('ProjectService', () => {
  it('should create project successfully', async () => {
    const projectData = { title: 'Test Project', description: 'Test' };
    const result = await projectService.create(projectData);
    expect(result.id).toBeDefined();
    expect(result.title).toBe('Test Project');
  });
});

// Cypress E2E test example
describe('Project Management', () => {
  it('should allow user to create and manage projects', () => {
    cy.login('user@example.com', 'password');
    cy.visit('/projects');
    cy.get('[data-cy=create-project]').click();
    cy.get('[data-cy=project-title]').type('New Project');
    cy.get('[data-cy=submit]').click();
    cy.contains('New Project').should('be.visible');
  });
});
```

## Test Coverage Report
- Target: 90% code coverage
- Unit tests: 95% coverage
- Integration tests: 85% coverage
- E2E tests: Critical user journeys

HANDOFF TO TECH_WRITER
""")
                
                elif "Tech Writer" in prompt:
                    return MockResponse("""
# Technical Documentation

## Executive Summary
The Task Management System is a comprehensive web-based solution designed for small development teams. It provides user authentication, project management, task tracking, and team collaboration features with a focus on scalability and security.

## Technical Overview
- **Architecture**: Microservices-based with React frontend and Node.js backend
- **Database**: PostgreSQL with Redis caching
- **Authentication**: JWT-based with OAuth 2.0
- **Deployment**: Containerized with Docker and Kubernetes

## Implementation Details

### Frontend Architecture
- React 18 with TypeScript for type safety
- Tailwind CSS for responsive design
- Redux Toolkit for state management
- React Query for API data fetching

### Backend Architecture
- Node.js 18 with Express framework
- TypeORM for database operations
- JWT for authentication
- Redis for session storage and caching

### Database Design
- PostgreSQL as primary database
- Optimized schema with proper indexing
- Foreign key relationships for data integrity
- Audit trails for tracking changes

## API Documentation

### Authentication Endpoints
- `POST /auth/register` - User registration
- `POST /auth/login` - User login
- `POST /auth/refresh` - Token refresh
- `POST /auth/logout` - User logout

### Project Management Endpoints
- `GET /projects` - List user projects
- `POST /projects` - Create new project
- `PUT /projects/:id` - Update project
- `DELETE /projects/:id` - Delete project

### Task Management Endpoints
- `GET /tasks` - List tasks with filters
- `POST /tasks` - Create new task
- `PUT /tasks/:id` - Update task
- `DELETE /tasks/:id` - Delete task

## Deployment Guide

### Prerequisites
- Docker and Docker Compose
- Node.js 18+ for local development
- PostgreSQL 14+ database
- Redis 6+ for caching

### Environment Setup
```bash
# Clone repository
git clone <repository-url>
cd task-management-system

# Install dependencies
npm install

# Setup environment variables
cp .env.example .env

# Start development server
npm run dev
```

### Production Deployment
```bash
# Build production images
docker-compose -f docker-compose.prod.yml build

# Deploy with Kubernetes
kubectl apply -f k8s/

# Monitor deployment
kubectl get pods -w
```

## Maintenance Instructions

### Regular Maintenance Tasks
1. **Database Backup**: Daily automated backups to AWS S3
2. **Log Rotation**: Weekly log cleanup and archival
3. **Security Updates**: Monthly dependency updates
4. **Performance Monitoring**: Continuous monitoring with alerts

### Troubleshooting Common Issues
- **Database Connection Issues**: Check connection strings and firewall settings
- **Authentication Failures**: Verify JWT secret and token expiration
- **Performance Issues**: Monitor database queries and API response times

## Source Code Documentation
Complete API documentation available at `/docs` endpoint when running the application. All code includes inline comments and follows TypeScript/JavaScript best practices.

FINAL ANSWER
""")
                
                return MockResponse("Mock response for unknown role")
        
        # Import required modules
        from multi_agent_software_team.langgraph_orchestrator import LangGraphSoftwareTeamOrchestrator
        from multi_agent_software_team.schemas import ProjectRequest, TeamRole
        
        print("🚀 Starting Enhanced Multi-Agent Software Team Demo")
        print("=" * 60)
        
        # Initialize the orchestrator with mock LLM
        orchestrator = LangGraphSoftwareTeamOrchestrator(MockLLM())
        
        # Create project request
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
        
        print(f"Project Description: {project_description.strip()}")
        print("\nSelected Team Roles:")
        for role in project_request.selected_roles:
            print(f"  • {role.value.replace('_', ' ').title()}")
        
        print("\n" + "=" * 60)
        print("🔄 Executing Multi-Agent Workflow...")
        print("=" * 60)
        
        # Execute the workflow
        result = orchestrator.create_software_solution(project_request)
        
        if result["success"]:
            print("\n✅ SUCCESS: Multi-Agent Collaboration Completed!")
            print(f"Generated {len(result['deliverables'])} deliverables")
            
            print("\n📋 Deliverables Summary:")
            for role_name, content in result["deliverables"].items():
                print(f"\n🔹 {role_name}:")
                # Show first 200 characters of each deliverable
                preview = content[:200] + "..." if len(content) > 200 else content
                print(f"   {preview}")
            
            print(f"\n📊 Final Output Preview:")
            output_preview = result["formatted_output"][:500] + "..." if len(result["formatted_output"]) > 500 else result["formatted_output"]
            print(output_preview)
            
        else:
            print(f"\n❌ ERROR: {result['error']}")
        
        print("\n" + "=" * 60)
        print("Demo completed!")
        
        return result
        
    except Exception as e:
        logger.error(f"Demo failed: {str(e)}")
        print(f"\n❌ Demo failed: {str(e)}")
        return {"success": False, "error": str(e)}

def show_handoff_mechanism():
    """Demonstrate the handoff mechanism in the multi-agent system."""
    
    print("\n🔄 Multi-Agent Handoff Mechanism")
    print("=" * 50)
    
    handoff_flow = [
        ("Product Owner", "HANDOFF TO ANALYST", "Analyst"),
        ("Analyst", "HANDOFF TO ARCHITECT", "Architect"),
        ("Architect", "HANDOFF TO DEVELOPER", "Developer"),
        ("Developer", "HANDOFF TO REVIEWER", "Reviewer"),
        ("Reviewer", "HANDOFF TO TESTER", "Tester"),
        ("Tester", "HANDOFF TO TECH_WRITER", "Tech Writer"),
        ("Tech Writer", "FINAL ANSWER", "END")
    ]
    
    print("Agent Flow with Handoff Signals:")
    for i, (current, signal, next_agent) in enumerate(handoff_flow, 1):
        arrow = "🔄" if next_agent != "END" else "✅"
        print(f"{i}. {current} → {signal} → {arrow} {next_agent}")
    
    print("\nKey Features:")
    print("• Each agent ends output with handoff signal")
    print("• Signals determine next agent in workflow")
    print("• FINAL ANSWER signals workflow completion")
    print("• Fallback to sequential order if no signal")
    print("• Error handling with graceful degradation")

if __name__ == "__main__":
    print("Enhanced Multi-Agent Software Team Demo")
    print("=" * 50)
    
    # Show handoff mechanism
    show_handoff_mechanism()
    
    # Run the demo
    result = demo_enhanced_multi_agent_workflow()
    
    if result.get("success"):
        print("\n🎉 Demo completed successfully!")
        print("The enhanced multi-agent system is ready for Gradio integration!")
    else:
        print(f"\n❌ Demo failed: {result.get('error')}")
