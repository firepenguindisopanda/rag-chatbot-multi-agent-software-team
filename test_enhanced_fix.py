#!/usr/bin/env python3
"""
Test script to verify the enhanced orchestrator fixes.
"""

def test_enhanced_orchestrator_fixes():
    """Test the fixes applied to the enhanced orchestrator."""
    print("🧪 Testing Enhanced Orchestrator Fixes")
    print("=" * 50)
    
    try:
        # Test 1: Import enhanced orchestrator
        print("🔍 Test 1: Import enhanced orchestrator...")
        from multi_agent_software_team.enhanced_langgraph_orchestrator import EnhancedLangGraphSoftwareTeamOrchestrator
        from multi_agent_software_team.schemas import TeamRole, ProjectRequest
        print("✅ Import successful")
        
        # Test 2: Create orchestrator (with None LLM for testing)
        print("🔍 Test 2: Create orchestrator...")
        orchestrator = EnhancedLangGraphSoftwareTeamOrchestrator(None)
        print("✅ Orchestrator creation successful")
        
        # Test 3: Check TeamRole has FULL_STACK_DEVELOPER
        print("🔍 Test 3: Check FULL_STACK_DEVELOPER role...")
        assert hasattr(TeamRole, 'FULL_STACK_DEVELOPER'), "FULL_STACK_DEVELOPER role missing"
        print("✅ FULL_STACK_DEVELOPER role exists")
        
        # Test 4: Create project request
        print("🔍 Test 4: Create project request...")
        project_request = ProjectRequest(
            description="Test project for validation",
            selected_roles=[TeamRole.PRODUCT_OWNER, TeamRole.DEVELOPER]
        )
        print("✅ Project request creation successful")
        
        # Test 5: Check Agent has process method (not process_request)
        print("🔍 Test 5: Check Agent methods...")
        from multi_agent_software_team.agents import Agent
        agent = Agent(TeamRole.DEVELOPER, None)
        assert hasattr(agent, 'process'), "Agent missing process method"
        assert not hasattr(agent, 'process_request'), "Agent should not have process_request method"
        print("✅ Agent has correct methods")
        
        print("\n🎉 All tests passed! The enhanced orchestrator fixes are working correctly.")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_enhanced_orchestrator_fixes()
    exit(0 if success else 1)
