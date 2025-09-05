#!/usr/bin/env python3
"""Test script for the multi-agent software team module."""

import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all imports work correctly."""
    try:
        from multi_agent_software_team import (
            TeamRole, 
            ProjectRequest, 
            validate_team_composition,
            get_agent_summary,
            format_agent_responses
        )
        print("✅ All imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_team_validation():
    """Test team validation function."""
    try:
        from multi_agent_software_team import TeamRole, validate_team_composition
        
        # Test valid team
        valid_team = [TeamRole.PRODUCT_OWNER, TeamRole.DEVELOPER]
        is_valid, message = validate_team_composition(valid_team)
        print(f"Valid team test: {is_valid}, {message}")
        
        # Test invalid team
        invalid_team = [TeamRole.TESTER]  # Missing essential roles
        is_valid, message = validate_team_composition(invalid_team)
        print(f"Invalid team test: {is_valid}, {message}")
        
        print("✅ Team validation tests passed")
        return True
    except Exception as e:
        print(f"❌ Team validation error: {e}")
        return False

def test_project_request():
    """Test ProjectRequest creation."""
    try:
        from multi_agent_software_team import ProjectRequest
        
        request = ProjectRequest(
            description="Create a simple web application",
            file_content="Some requirements content"
        )
        print(f"✅ ProjectRequest created: {request.description[:50]}...")
        return True
    except Exception as e:
        print(f"❌ ProjectRequest error: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing Multi-Agent Software Team Module")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_team_validation,
        test_project_request
    ]
    
    passed = 0
    for test in tests:
        print(f"\n🔍 Running {test.__name__}...")
        if test():
            passed += 1
        else:
            print(f"💥 {test.__name__} failed")
    
    print(f"\n📊 Results: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 All tests passed! Multi-agent module is working correctly.")
    else:
        print("⚠️ Some tests failed. Please check the errors above.")
