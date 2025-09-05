#!/usr/bin/env python3
"""
Focused debug test for the multi-agent handoff issue
"""

import os
import sys
import logging
from datetime import datetime

# Load environment variables instead of hardcoding secrets
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

if not os.getenv("NVIDIA_API_KEY"):
    print("⚠ NVIDIA_API_KEY not set. Handoff test may fail due to missing LLM access.")

# Add path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Configure detailed logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_handoff_mechanism():
    """Test the handoff mechanism specifically."""
    print("🔍 Testing Multi-Agent Handoff Mechanism")
    print("=" * 50)
    
    try:
        # Import required modules
        from langchain_nvidia_ai_endpoints import ChatNVIDIA
        from multi_agent_software_team.modern_langgraph_orchestrator import ModernSoftwareTeamOrchestrator
        from multi_agent_software_team.schemas import ProjectRequest
        
        print("✅ Imports successful")
        
        # Initialize real LLM
        llm = ChatNVIDIA(model="meta/llama-3.1-8b-instruct")
        print("✅ NVIDIA LLM initialized")
        
        # Create orchestrator
        orchestrator = ModernSoftwareTeamOrchestrator(llm)
        print("✅ Orchestrator created")
        
        # Test the handoff detection function
        from langchain_core.messages import HumanMessage
        test_messages = [
            HumanMessage(content="Some work done. HANDOFF TO ANALYST"),
            HumanMessage(content="Analysis complete. HANDOFF TO ARCHITECT"),
            HumanMessage(content="Architecture ready. HANDOFF TO DEVELOPER"),
            HumanMessage(content="Code implemented. HANDOFF TO REVIEWER"),
            HumanMessage(content="Review done. HANDOFF TO TESTER"),
            HumanMessage(content="Testing complete. HANDOFF TO TECH_WRITER"),
            HumanMessage(content="Documentation ready. FINAL ANSWER"),
        ]
        
        print("\n🧪 Testing handoff detection:")
        for msg in test_messages:
            next_node = orchestrator._get_next_node(msg)
            print(f"  Message: '{msg.content}' -> Next: {next_node}")
        
        # Test simple project
        print("\n🚀 Testing simple project execution...")
        project_request = ProjectRequest(
            description="Create a simple hello world web application with user login",
            file_content=None
        )
        
        print("📝 Starting collaboration...")
    result = orchestrator.collaborate_on_project(project_request)

    print("\n📊 Execution Results:")
        print(f"Success: {result.get('success')}")
        print(f"Status: {result.get('status')}")
        print(f"Error: {result.get('error', 'None')}")
        print(f"Message Count: {result.get('message_count', 0)}")
        print(f"Agent Outputs: {len(result.get('agent_outputs', {}))}")
        
        if result.get('agent_outputs'):
            print("\n👥 Agent Contributions:")
            for agent_name, outputs in result.get('agent_outputs', {}).items():
                total_length = sum(len(output) for output in outputs)
                print(f"  - {agent_name}: {len(outputs)} messages, {total_length} chars")
                # Show first few words of first message
                if outputs:
                    first_msg = outputs[0][:100] + ('...' if len(outputs[0]) > 100 else '')
                    print(f"    First: {first_msg}")
        
    output = result.get('output', '')
    print("\n📄 Output Summary:")
        print(f"Total Length: {len(output)} characters")
        if len(output) > 0:
            print(f"Preview:\n{output[:500]}...")
        else:
            print("❌ No output generated!")
            
        return result
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    test_handoff_mechanism()
