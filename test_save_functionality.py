#!/usr/bin/env python3
"""
Test script specifically for the MD file save functionality.
"""

import sys
import os
import tempfile

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_validation_function():
    """Test that the validation function works correctly."""
    try:
        from multi_agent_software_team.utils import validate_team_composition
        from multi_agent_software_team.schemas import TeamRole
        
        # Test valid team
        valid_roles = [TeamRole.PRODUCT_OWNER, TeamRole.DEVELOPER]
        is_valid, message = validate_team_composition(valid_roles)
        
        if is_valid and isinstance(message, str):
            print("✅ Validation function works correctly")
            print(f"   Result: {is_valid}, Message: {message}")
            return True
        else:
            print(f"❌ Validation function issue: {is_valid}, {message}")
            return False
            
    except Exception as e:
        print(f"❌ Validation function error: {e}")
        return False

def test_save_functionality():
    """Test the save to MD file functionality."""
    try:
        from multi_agent_software_team.utils import save_response_with_auto_filename
        
        # Test content
        test_content = """# 🚀 Test Software Solution

## 📋 Product Owner Deliverables
- User stories created
- Acceptance criteria defined
- Business requirements documented

## 💻 Developer Deliverables
- Database schema designed
- API endpoints implemented
- Frontend components created

## 🏗️ Architect Deliverables
```mermaid
graph TD
    A[Client] --> B[API Gateway]
    B --> C[Auth Service]
    B --> D[Business Logic]
    D --> E[Database]
```

This solution provides a comprehensive approach to building the application.
"""
        
        # Test with descriptive project name
        result = save_response_with_auto_filename(test_content, "E-commerce Web Application with Authentication")
        
        if result and not result.startswith("Error"):
            print("✅ Save functionality works correctly")
            print(f"   File saved to: {result}")
            
            # Check if file exists and has content
            if os.path.exists(result):
                with open(result, 'r', encoding='utf-8') as f:
                    content = f.read()
                if len(content) > 100:  # Should have substantial content
                    print("✅ File created successfully with content")
                    return True
                else:
                    print("❌ File created but appears empty")
                    return False
            else:
                print("❌ File was not actually created")
                return False
        else:
            print(f"❌ Save function returned error: {result}")
            return False
            
    except Exception as e:
        print(f"❌ Save functionality error: {e}")
        return False

def test_solutions_directory():
    """Test that the solutions directory is created and accessible."""
    try:
        # Check if solutions directory exists or can be created
        solutions_dir = os.path.join(os.getcwd(), "solutions")
        
        if not os.path.exists(solutions_dir):
            os.makedirs(solutions_dir, exist_ok=True)
        
        # Test write permissions
        test_file = os.path.join(solutions_dir, "test_write_permissions.txt")
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write("Test content")
        
        # Clean up test file
        if os.path.exists(test_file):
            os.remove(test_file)
        
        print("✅ Solutions directory is accessible and writable")
        print(f"   Directory: {solutions_dir}")
        return True
        
    except Exception as e:
        print(f"❌ Solutions directory issue: {e}")
        return False

def test_filename_generation():
    """Test automatic filename generation from project descriptions."""
    try:
        from multi_agent_software_team.utils import create_safe_filename
        
        test_cases = [
            ("E-commerce Web Application", "e-commerce_web_application"),
            ("Task Management System with Real-time Updates", "task_management_system_with_real-time_updates"),
            ("Simple Chat Bot", "simple_chat_bot"),
            ("", ""),  # Empty description
            ("   ", ""),  # Whitespace only
            ("Special!@#$%^&*()Characters", "specialcharacters"),
        ]
        
        all_passed = True
        for description, expected in test_cases:
            result = create_safe_filename(description)
            if result == expected or (not result and not expected):
                print(f"✅ Filename generation for '{description}': '{result}'")
            else:
                print(f"❌ Filename generation failed for '{description}': got '{result}', expected '{expected}'")
                all_passed = False
        
        return all_passed
        
    except Exception as e:
        print(f"❌ Filename generation error: {e}")
        return False

def test_import_compatibility():
    """Test that all necessary imports work correctly."""
    try:
        # Test core imports
        from multi_agent_software_team.utils import (
            save_response_to_md,
            save_response_with_auto_filename,
            validate_team_composition,
            create_safe_filename
        )
        from multi_agent_software_team.schemas import TeamRole, ProjectRequest
        
        print("✅ All imports successful")
        return True
        
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def demo_save_feature():
    """Demonstrate the save feature with a realistic example."""
    print("\n🎨 Demonstrating Save Feature")
    print("=" * 50)
    
    try:
        from multi_agent_software_team.utils import save_response_with_auto_filename
        from datetime import datetime
        
        # Create a realistic multi-agent response
        sample_response = f"""# 🚀 Multi-Agent Software Team Solution

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 📋 Product Owner - User Stories & Requirements

### Epic: User Management System
- **US-001**: As a user, I want to register an account so that I can access the platform
- **US-002**: As a user, I want to log in securely so that my data is protected
- **US-003**: As an admin, I want to manage user roles so that I can control access

### Acceptance Criteria
- Registration requires email verification
- Password must meet security standards (8+ chars, mixed case, numbers)
- Failed login attempts are tracked and limited

## 🏗️ Architect - System Design

### System Architecture
```mermaid
graph TD
    A[Web Client] --> B[Load Balancer]
    B --> C[API Gateway]
    C --> D[Auth Service]
    C --> E[User Service]
    D --> F[Redis Cache]
    E --> G[PostgreSQL]
```

### Technology Stack
- **Frontend**: React.js with TypeScript
- **Backend**: Node.js with Express
- **Database**: PostgreSQL with Redis cache
- **Authentication**: JWT with refresh tokens

## 💻 Developer - Implementation Details

### Database Schema
```sql
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    role VARCHAR(50) DEFAULT 'user',
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);
```

### API Endpoints
- `POST /auth/register` - User registration
- `POST /auth/login` - User authentication
- `GET /users/profile` - Get user profile
- `PUT /users/profile` - Update user profile

## 🧪 Tester - Quality Assurance

### Test Plan
1. **Unit Tests**: Service layer functions
2. **Integration Tests**: API endpoint testing
3. **Security Tests**: Authentication and authorization
4. **Performance Tests**: Load testing with 1000+ concurrent users

### Test Cases
- Valid/invalid registration scenarios
- Password security validation
- JWT token expiration handling
- SQL injection prevention

## 📝 Tech Writer - Documentation

### Setup Instructions
1. Clone the repository
2. Install dependencies: `npm install`
3. Configure environment variables
4. Run database migrations
5. Start the development server

### Deployment Guide
- Docker containerization included
- CI/CD pipeline with GitHub Actions
- Production environment configuration
- Monitoring and logging setup

---

**Summary**: Complete user management system with secure authentication, scalable architecture, and comprehensive testing strategy.
"""
        
        # Save the sample response
        project_name = "User Management System with Authentication"
        filepath = save_response_with_auto_filename(sample_response, project_name)
        
        if filepath and not filepath.startswith("Error"):
            print("✅ Demo file created successfully!")
            print(f"📁 Location: {filepath}")
            print(f"📊 Content size: {len(sample_response):,} characters")
            
            # Show file contents preview
            if os.path.exists(filepath):
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                    print(f"📖 File size on disk: {len(content):,} characters")
                    print("\n📄 Preview (first 200 characters):")
                    print(content[:200] + "...")
            
            return True
        else:
            print(f"❌ Demo file creation failed: {filepath}")
            return False
            
    except Exception as e:
        print(f"❌ Demo error: {e}")
        return False

if __name__ == "__main__":
    print("💾 Testing MD File Save Functionality")
    print("=" * 55)
    
    tests = [
        test_import_compatibility,
        test_validation_function,
        test_solutions_directory,
        test_filename_generation,
        test_save_functionality
    ]
    
    passed = 0
    for test in tests:
        print(f"\n🔍 Running {test.__name__.replace('_', ' ').title()}...")
        if test():
            passed += 1
        else:
            print(f"💥 {test.__name__} failed")
    
    # Run demo
    demo_save_feature()
    
    print(f"\n📊 Test Results: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("\n🎉 All save functionality tests passed!")
        print("\n✨ Features Available:")
        print("   • Save team responses to markdown files")
        print("   • Automatic filename generation from project description")
        print("   • Solutions saved to dedicated 'solutions' directory")
        print("   • Comprehensive markdown formatting with metadata")
        print("   • Error handling and fallback mechanisms")
        print("   • Summary files generated alongside main solution")
        
        print("\n📝 How to Use:")
        print("   1. Generate a solution using the multi-agent team")
        print("   2. Click 'Save Solution to MD File' button")
        print("   3. Files will be saved to the 'solutions' directory")
        print("   4. Filenames are auto-generated from your project description")
        
        print("\n📁 File Structure:")
        print("   solutions/")
        print("   ├── project_name_YYYYMMDD_HHMMSS.md")
        print("   └── summary_project_name_YYYYMMDD_HHMMSS.md")
        
    else:
        print("\n⚠️ Some tests failed. Please check the errors above.")
        print("   The save functionality may still work, but with reduced features.")
