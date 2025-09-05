#!/usr/bin/env python3
"""
Test script for the clean multi-agent software team implementation.
Verifies that the output is clean and professional without debug statements.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from enhanced_multi_agent_integration import run_enhanced_multi_agent_collaboration

# Mock LLM for testing
class MockLLM:
    """Mock LLM that returns predefined responses for testing."""
    
    def __init__(self):
        self.responses = {
            "product_owner": """
# Business Requirements & User Stories

## User Stories
1. As a team member, I want to create and assign tasks so that work can be distributed effectively
2. As a project manager, I want to track progress so that I can monitor team productivity
3. As a user, I want to collaborate in real-time so that communication is seamless

## Acceptance Criteria
- Users must be able to create tasks with titles, descriptions, and due dates
- Project managers can view dashboard with progress metrics
- Real-time notifications for task updates

## Business Rules
- Only authenticated users can access the system
- Task assignments require valid team member selection
- Progress tracking updates automatically

## Priority Levels
Must-Have: User authentication, task creation, basic progress tracking
Should-Have: Real-time collaboration, advanced analytics
Could-Have: Mobile app, third-party integrations
            """,
            
            "analyst": """
# Technical Specifications & Analysis

## Functional Requirements
FR-001: User authentication with email/password
FR-002: CRUD operations for tasks and projects
FR-003: Real-time collaboration using WebSockets
FR-004: Dashboard with progress visualization

## Non-Functional Requirements
NFR-001: System must support 100 concurrent users
NFR-002: Response time < 2 seconds for all operations
NFR-003: 99.9% uptime availability
NFR-004: Data encryption at rest and in transit

## Technical Constraints
- Web-based application using modern frameworks
- RESTful API design for scalability
- Database support for relational data

## Security Requirements
- JWT-based authentication
- Role-based access control
- Input validation and sanitization
            """,
            
            "architect": """
# System Architecture & Design

## System Architecture Overview
The application follows a microservices architecture with clear separation of concerns.

## Component Architecture

```mermaid
graph TB
    A[React Frontend] --> B[API Gateway]
    B --> C[Authentication Service]
    B --> D[Task Service]
    B --> E[Notification Service]
    C --> F[User Database]
    D --> G[Task Database]
    E --> H[Redis Cache]
    
    I[WebSocket Server] --> D
    I --> E
```

## Technology Stack
- Frontend: React.js with TypeScript
- Backend: Node.js with Express
- Database: PostgreSQL for main data, Redis for caching
- Authentication: JWT with bcrypt
- Real-time: Socket.io for WebSockets

## API Design
REST endpoints following OpenAPI 3.0 specification with proper HTTP status codes and error handling.
            """,
            
            "developer": """
# Implementation Guide & Code

## Project Structure
```
task-manager/
├── frontend/           # React application
├── backend/           # Node.js API
├── database/          # SQL migrations
└── docker-compose.yml # Development setup
```

## Database Schema

```mermaid
erDiagram
    USER ||--o{ TASK : creates
    PROJECT ||--o{ TASK : contains
    USER ||--o{ PROJECT : owns
    
    USER {
        uuid id
        string email
        string password_hash
        timestamp created_at
    }
    
    TASK {
        uuid id
        string title
        text description
        enum status
        date due_date
        uuid assigned_to
        uuid project_id
    }
```

## Key Implementation Examples

### Task Model (Node.js)
```javascript
class Task {
  constructor(data) {
    this.id = data.id;
    this.title = data.title;
    this.description = data.description;
    this.status = data.status || 'pending';
    this.dueDate = data.dueDate;
    this.assignedTo = data.assignedTo;
    this.projectId = data.projectId;
  }
  
  async save() {
    // Database save logic
    return await db.tasks.create(this);
  }
}
```

## Environment Setup
```bash
npm install
cp .env.example .env
npm run migrate
npm run dev
```
            """,
            
            "reviewer": """
# Code Review & Quality Analysis

## Code Quality Review
✅ **Strengths:**
- Clean separation of concerns with microservices
- Proper error handling and validation
- Modern technology stack choices
- Comprehensive database design

⚠️ **Areas for Improvement:**
- Add input validation middleware
- Implement rate limiting
- Add comprehensive logging
- Consider database connection pooling

## Security Analysis
🔒 **Security Recommendations:**
1. Implement CORS properly for production
2. Add SQL injection protection with parameterized queries
3. Use HTTPS in production with proper SSL certificates
4. Implement session timeout and refresh tokens

## Performance Review
🚀 **Performance Optimizations:**
- Add database indexing for frequently queried fields
- Implement caching strategy for user sessions
- Consider CDN for static assets
- Add database query optimization

## Risk Assessment
- **Low Risk**: Basic CRUD operations
- **Medium Risk**: Real-time features complexity
- **High Risk**: Scalability with increased user load

Recommended mitigation: Load testing and gradual scaling approach.
            """,
            
            "tester": """
# Testing Strategy & QA Plan

## Test Strategy
Comprehensive testing approach covering unit, integration, and end-to-end scenarios.

## Unit Test Examples
```javascript
describe('Task Model', () => {
  test('should create task with valid data', () => {
    const task = new Task({
      title: 'Test Task',
      description: 'Test Description'
    });
    expect(task.title).toBe('Test Task');
    expect(task.status).toBe('pending');
  });
});
```

## Integration Test Scenarios
- API endpoint authentication flow
- Database operations with rollback
- WebSocket connection handling
- External service integrations

## End-to-End Test Cases
TC-001: User Login Flow
- Given: Valid credentials
- When: User submits login form
- Then: Redirected to dashboard

TC-002: Task Creation
- Given: Authenticated user
- When: Creates new task
- Then: Task appears in task list

## Performance Testing
- Load testing for 100 concurrent users
- Stress testing for database connections
- WebSocket connection limits

## Test Automation Strategy
- Jest for unit tests
- Supertest for API testing
- Cypress for E2E testing
- GitHub Actions for CI/CD pipeline
            """,
            
            "tech_writer": """
# Complete Documentation & User Guides

## Executive Summary
Task Management System is a modern web application designed for small to medium development teams. It provides intuitive task creation, assignment, and tracking capabilities with real-time collaboration features.

## Technical Overview
Built using React frontend with Node.js backend, PostgreSQL database, and WebSocket support for real-time updates. The system follows microservices architecture for scalability and maintainability.

## Installation Guide
### Prerequisites
- Node.js 18+ and npm
- PostgreSQL 13+
- Redis (for caching)

### Setup Steps
1. Clone the repository
2. Install dependencies: `npm install`
3. Configure environment variables
4. Run database migrations: `npm run migrate`
5. Start the application: `npm run dev`

## User Manual
### Getting Started
1. Register account with valid email
2. Create your first project
3. Add team members to project
4. Start creating and assigning tasks

### Key Features
- **Dashboard**: Overview of tasks and progress
- **Task Management**: Create, edit, and track tasks
- **Team Collaboration**: Real-time updates and notifications
- **Progress Tracking**: Visual progress indicators

## API Documentation
### Authentication Endpoints
- POST /api/auth/login - User login
- POST /api/auth/register - User registration
- POST /api/auth/refresh - Token refresh

### Task Endpoints
- GET /api/tasks - List all tasks
- POST /api/tasks - Create new task
- PUT /api/tasks/:id - Update task
- DELETE /api/tasks/:id - Delete task

## Deployment Guide
### Production Setup
1. Configure production environment variables
2. Set up SSL certificates
3. Configure database for production
4. Deploy using Docker containers
5. Set up monitoring and logging

## Troubleshooting Guide
**Common Issues:**
- Login failures: Check email/password
- Tasks not loading: Verify database connection
- Real-time updates not working: Check WebSocket connection

**Performance Issues:**
- Slow dashboard loading: Clear browser cache
- Database timeout: Increase connection pool size
            """
        }
    
    def invoke(self, messages):
        """Mock invoke method that returns appropriate response based on system prompt."""
        # Extract the role from the system prompt
        system_content = messages[0]["content"] if messages else ""
        
        if "Product Owner" in system_content:
            return MockResponse(self.responses["product_owner"])
        elif "Requirements Analyst" in system_content:
            return MockResponse(self.responses["analyst"])
        elif "Solutions Architect" in system_content:
            return MockResponse(self.responses["architect"])
        elif "Software Developer" in system_content:
            return MockResponse(self.responses["developer"])
        elif "Code Reviewer" in system_content:
            return MockResponse(self.responses["reviewer"])
        elif "QA Engineer" in system_content:
            return MockResponse(self.responses["tester"])
        elif "Technical Writer" in system_content:
            return MockResponse(self.responses["tech_writer"])
        else:
            return MockResponse("Mock response for unknown role")

class MockResponse:
    """Mock response object that mimics LLM response."""
    def __init__(self, content):
        self.content = content

def test_clean_output():
    """Test that the output is clean and professional."""
    print("🧪 Testing Clean Multi-Agent Software Team")
    print("=" * 50)
    
    # Create mock LLM
    mock_llm = MockLLM()
    
    # Test project description
    project_description = """
    Create a task management system for development teams with the following features:
    - User authentication and role management
    - Task creation, assignment, and tracking
    - Real-time collaboration and notifications
    - Progress dashboards and reporting
    - Team management and project organization
    
    Technology preferences: React, Node.js, PostgreSQL
    Target users: 10-50 person development teams
    """
    
    # Run the collaboration
    print("🚀 Running multi-agent collaboration...")
    result = run_enhanced_multi_agent_collaboration(
        mock_llm, 
        project_description
    )
    
    # Check for debug statements
    debug_indicators = [
        "HANDOFF TO",
        "FINAL ANSWER",
        "Ready for",
        "complete, ready for"
    ]
    
    has_debug = any(indicator in result for indicator in debug_indicators)
    
    print("\n📊 Test Results:")
    print(f"✅ Output generated: {len(result) > 0}")
    print(f"{'❌' if has_debug else '✅'} Clean output (no debug statements): {not has_debug}")
    print(f"✅ Contains Mermaid diagrams: {'```mermaid' in result}")
    print(f"✅ Professional formatting: {'#' in result and '##' in result}")
    print(f"📏 Total output length: {len(result)} characters")
    
    if has_debug:
        print("\n⚠️ Found debug statements:")
        for indicator in debug_indicators:
            if indicator in result:
                print(f"   - Found: '{indicator}'")
    
    # Save the result for inspection
    with open("test_clean_output.md", "w", encoding="utf-8") as f:
        f.write(result)
    
    print("\n📄 Full output saved to: test_clean_output.md")
    print(f"\n{'🎉 TEST PASSED' if not has_debug else '❌ TEST FAILED'}")
    
    return not has_debug

if __name__ == "__main__":
    success = test_clean_output()
    sys.exit(0 if success else 1)
