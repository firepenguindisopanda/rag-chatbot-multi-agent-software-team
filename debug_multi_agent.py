#!/usr/bin/env python3
"""
Debug script to test the multi-agent system and identify the issue.
"""

import os
import sys
import logging
from datetime import datetime

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_multi_agent_system():
    """Test the multi-agent system with detailed debugging."""
    print("🔍 Multi-Agent System Debug Test")
    print("=" * 50)
    
    try:
        # Import required modules
        from multi_agent_software_team.modern_langgraph_orchestrator import ModernSoftwareTeamOrchestrator
        from multi_agent_software_team.schemas import ProjectRequest
        from enhanced_multi_agent_integration import run_enhanced_multi_agent_collaboration
        print("✅ Imports successful")
        
        # Create a mock LLM for testing
        class MockLLM:
            def invoke(self, messages):
                # Check the last message
                if isinstance(messages, dict) and "messages" in messages:
                    last_msg = messages["messages"][-1]
                    if isinstance(last_msg, tuple):
                        content = last_msg[1]
                    else:
                        content = str(last_msg)
                else:
                    content = str(messages)
                
                print(f"🤖 MockLLM received: {content[:100]}...")
                
                # Mock response based on content
                if "product owner" in content.lower() or "business requirements" in content.lower():
                    response = """# Product Owner Analysis

## User Stories
1. **As a team member**, I want to create and manage tasks so that I can track my work efficiently
2. **As a project manager**, I want to assign tasks to team members so that work is distributed properly
3. **As a user**, I want to receive notifications about task updates so that I stay informed

## Acceptance Criteria
- Users can create, edit, and delete tasks
- Tasks have priorities (high, medium, low) and due dates
- Email notifications are sent for important updates
- Dashboard shows task progress and analytics

## Business Rules
- Only authenticated users can access the system
- Task assignments require proper permissions
- Data must be backed up regularly

HANDOFF TO ANALYST"""
                
                elif "analyst" in content.lower() or "technical specifications" in content.lower():
                    response = """# Requirements Analysis

## Functional Requirements
- FR-001: User Authentication System
- FR-002: Task CRUD Operations
- FR-003: Real-time Notifications
- FR-004: Progress Tracking Dashboard

## Non-Functional Requirements
- NFR-001: System must support 100 concurrent users
- NFR-002: Response time < 2 seconds
- NFR-003: 99.9% uptime availability

## Technical Constraints
- Web-based application
- Mobile responsive design
- Database for persistence

HANDOFF TO ARCHITECT"""
                
                elif "architect" in content.lower() or "system design" in content.lower():
                    response = """# System Architecture Design

## Architecture Overview
```mermaid
graph TB
    A[Frontend React App] --> B[API Gateway]
    B --> C[Authentication Service]
    B --> D[Task Service]
    B --> E[Notification Service]
    D --> F[Database]
    E --> G[Email Service]
```

## Component Design
- Frontend: React with Material-UI
- Backend: Node.js with Express
- Database: PostgreSQL
- Authentication: JWT tokens

HANDOFF TO DEVELOPER"""
                
                elif "developer" in content.lower() or "implementation" in content.lower():
                    response = """# Implementation Details

## Backend Structure
```javascript
// Task Model
const Task = {
  id: UUID,
  title: String,
  description: String,
  assignee: String,
  priority: Enum['high', 'medium', 'low'],
  status: Enum['todo', 'in-progress', 'done'],
  dueDate: Date,
  createdAt: Date
}
```

## API Endpoints
- POST /api/tasks - Create task
- GET /api/tasks - List tasks
- PUT /api/tasks/:id - Update task
- DELETE /api/tasks/:id - Delete task

HANDOFF TO REVIEWER"""
                
                elif "reviewer" in content.lower() or "code review" in content.lower():
                    response = """# Code Review Results

## Security Assessment
✅ Input validation implemented
✅ SQL injection prevention
✅ Authentication checks on all endpoints
✅ HTTPS enforced

## Performance Review
✅ Database queries optimized
✅ Caching strategy implemented
✅ API response times acceptable

## Code Quality
✅ ESLint rules followed
✅ Unit tests coverage > 80%
✅ Documentation complete

HANDOFF TO TESTER"""
                
                elif "tester" in content.lower() or "testing strategy" in content.lower():
                    response = """# Testing Strategy

## Test Plan
1. **Unit Tests**: Individual component testing
2. **Integration Tests**: API endpoint testing
3. **E2E Tests**: Full user workflow testing
4. **Performance Tests**: Load testing with 100 users

## Test Cases
- User registration and login
- Task creation and management
- Notification delivery
- Dashboard functionality

## Automation
- CI/CD pipeline with automated testing
- Daily smoke tests
- Performance monitoring

HANDOFF TO TECH_WRITER"""
                
                elif "tech_writer" in content.lower() or "documentation" in content.lower():
                    response = """# Technical Documentation

## User Guide
- Getting Started Tutorial
- Task Management Features
- Dashboard Analytics
- Troubleshooting Guide

## Developer Documentation
- API Reference
- Database Schema
- Deployment Instructions
- Architecture Overview

## Administrative Guide
- System Configuration
- User Management
- Backup Procedures
- Monitoring Setup

FINAL ANSWER"""
                
                else:
                    response = f"Mock response for: {content[:50]}... \nFINAL ANSWER"
                
                # Create a mock AI message
                class MockMessage:
                    def __init__(self, content):
                        self.content = content
                        self.name = None
                
                return {"messages": [MockMessage(response)]}
        
        print("✅ Mock LLM created")
        
        # Test the orchestrator directly
        print("\n🧪 Testing ModernSoftwareTeamOrchestrator directly...")
        orchestrator = ModernSoftwareTeamOrchestrator(MockLLM())
        
        project_request = ProjectRequest(
            description="Create a simple task management web application for small teams",
            file_content=None
        )
        
        print("📝 Executing collaboration...")
        result = orchestrator.collaborate_on_project(project_request)
        
        print(f"\n📊 Result Summary:")
        print(f"Success: {result.get('success')}")
        print(f"Status: {result.get('status')}")
        print(f"Message Count: {result.get('message_count', 0)}")
        print(f"Agent Count: {len(result.get('agent_outputs', {}))}")
        print(f"Output Length: {len(result.get('output', ''))}")
        
        if result.get('agent_outputs'):
            print(f"\n👥 Agents that contributed:")
            for agent_name, outputs in result.get('agent_outputs', {}).items():
                print(f"  - {agent_name}: {len(outputs)} messages")
        
        print(f"\n📄 Output Preview:")
        output = result.get('output', '')
        print(output[:500] + ('...' if len(output) > 500 else ''))
        
        # Test the enhanced integration
        print("\n🧪 Testing Enhanced Integration...")
        enhanced_result = run_enhanced_multi_agent_collaboration(
            MockLLM(), 
            "Create a simple task management web application for small teams"
        )
        
        print(f"\n📊 Enhanced Integration Result:")
        print(f"Length: {len(enhanced_result)}")
        print(f"Preview: {enhanced_result[:300]}...")
        
        return result
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    test_multi_agent_system()
