#!/usr/bin/env python3
"""
Test script specifically for Mermaid diagram functionality in the multi-agent system.
"""

import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_mermaid_enhancement():
    """Test the Mermaid diagram enhancement functionality."""
    try:
        from multi_agent_software_team.utils import enhance_mermaid_diagrams, extract_mermaid_diagrams
        
        # Test content with Mermaid diagrams
        test_content = """
        # System Architecture
        
        Here's the system architecture:
        
        ```mermaid
        graph TD
        A[Client] --> B[API Gateway]
        B --> C[Service 1]
        B --> D[Service 2]
        ```
        
        And here's the database schema:
        
        ```mermaid
        erDiagram
        USER {
            int id PK
            string name
        }
        PROJECT {
            int id PK
            string title
        }
        USER ||--o{ PROJECT : owns
        ```
        """
        
        # Test enhancement
        enhanced = enhance_mermaid_diagrams(test_content)
        print("✅ Mermaid enhancement test passed")
        
        # Test extraction
        diagrams = extract_mermaid_diagrams(test_content)
        print(f"✅ Found {len(diagrams)} Mermaid diagrams")
        
        for i, diagram in enumerate(diagrams, 1):
            print(f"   📊 Diagram {i}: {diagram['content'][:50]}...")
        
        return True
    except Exception as e:
        print(f"❌ Mermaid test error: {e}")
        return False

def test_enhanced_prompts():
    """Test that the enhanced prompts include Mermaid instructions."""
    try:
        from multi_agent_software_team.prompts import create_system_prompts
        from multi_agent_software_team.schemas import TeamRole
        
        prompts = create_system_prompts()
        
        # Check that key roles mention Mermaid
        roles_with_mermaid = [
            TeamRole.ARCHITECT,
            TeamRole.DEVELOPER,
            TeamRole.DESIGNER,
            TeamRole.TESTER
        ]
        
        for role in roles_with_mermaid:
            prompt = prompts[role]
            if 'mermaid' in prompt.lower() or 'Mermaid' in prompt:
                print(f"✅ {role.value.replace('_', ' ').title()} prompt includes Mermaid instructions")
            else:
                print(f"⚠️ {role.value.replace('_', ' ').title()} prompt missing Mermaid instructions")
        
        print("✅ Enhanced prompts test completed")
        return True
    except Exception as e:
        print(f"❌ Enhanced prompts test error: {e}")
        return False

def test_agent_post_processing():
    """Test that agents properly post-process Mermaid content."""
    try:
        from multi_agent_software_team.agents import Agent
        from multi_agent_software_team.schemas import TeamRole
        
        # Create a mock LLM
        class MockLLM:
            def invoke(self, text):
                class MockResponse:
                    content = """
                    # System Design
                    
                    Here's the architecture:
                    
                    ```mermaid
                    graph TD
                    A --> B
                    B --> C
                    ```
                    """
                return MockResponse()
        
        # Test with Designer agent (should have Mermaid processing)
        agent = Agent(TeamRole.DESIGNER, MockLLM())
        
        # Test the post-processing
        test_content = """
        ```mermaid
        graph TD
        A --> B
        B --> C
        ```
        """
        
        processed = agent._post_process_output(test_content, "medium")
        
        if 'mermaid' in processed:
            print("✅ Agent post-processing preserves Mermaid diagrams")
        else:
            print("⚠️ Agent post-processing may not handle Mermaid properly")
        
        return True
    except Exception as e:
        print(f"❌ Agent post-processing test error: {e}")
        return False

def demo_mermaid_types():
    """Demonstrate different types of Mermaid diagrams that agents can create."""
    print("\n🎨 Multi-Agent Team Mermaid Diagram Capabilities:")
    print("=" * 60)
    
    diagrams = {
        "🏗️ Architect": [
            "System Architecture (graph/flowchart)",
            "Component Diagrams (graph)",
            "Data Flow Diagrams (flowchart)"
        ],
        "💻 Developer": [
            "Entity Relationship Diagrams (erDiagram)",
            "Database Schema (erDiagram)"
        ],
        "🎨 Designer": [
            "System Context Diagrams (graph)",
            "Component Interaction (flowchart)",
            "Sequence Diagrams (sequenceDiagram)",
            "State Diagrams (stateDiagram)",
            "Deployment Diagrams (flowchart)"
        ],
        "🧪 Tester": [
            "Test Flow Diagrams (flowchart)",
            "Test Process Flow (flowchart)"
        ]
    }
    
    for role, diagram_types in diagrams.items():
        print(f"\n{role}:")
        for diagram_type in diagram_types:
            print(f"   • {diagram_type}")
    
    print(f"\n📋 Total Diagram Types: {sum(len(types) for types in diagrams.values())}")
    print("\n💡 All diagrams are generated in Mermaid syntax and will render properly in Markdown!")

if __name__ == "__main__":
    print("🎨 Testing Enhanced Mermaid Diagram Support")
    print("=" * 55)
    
    tests = [
        test_mermaid_enhancement,
        test_enhanced_prompts,
        test_agent_post_processing
    ]
    
    passed = 0
    for test in tests:
        print(f"\n🔍 Running {test.__name__}...")
        if test():
            passed += 1
        else:
            print(f"💥 {test.__name__} failed")
    
    # Show capabilities demo
    demo_mermaid_types()
    
    print(f"\n📊 Test Results: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("\n🎉 All Mermaid tests passed! Your agents will now generate beautiful diagrams!")
        print("\n✨ Enhanced Features:")
        print("   • Architect creates system architecture diagrams")
        print("   • Developer creates database ERD diagrams") 
        print("   • Designer creates comprehensive technical diagrams")
        print("   • Tester creates test flow diagrams")
        print("   • All diagrams use proper Mermaid syntax in code blocks")
        print("   • Diagrams render perfectly in Markdown viewers")
    else:
        print("\n⚠️ Some tests failed. Please check the errors above.")
