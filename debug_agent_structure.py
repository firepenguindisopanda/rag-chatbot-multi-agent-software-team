#!/usr/bin/env python3
"""
Debug the agent result structure to understand the handoff issue
"""

import os
import sys

# Set environment
os.environ["NVIDIA_API_KEY"] = "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"

# Add path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_agent_result_structure():
    """Test the actual structure returned by LangGraph agents."""
    print("🔍 Testing Agent Result Structure")
    print("=" * 50)
    
    try:
        from langchain_nvidia_ai_endpoints import ChatNVIDIA
        from langgraph.prebuilt import create_react_agent
        from multi_agent_software_team.prompts import create_system_prompts
        from multi_agent_software_team.schemas import TeamRole
        from multi_agent_software_team.modern_langgraph_orchestrator import create_documentation
        
        # Initialize LLM
        llm = ChatNVIDIA(model="meta/llama-3.1-8b-instruct")
        prompts = create_system_prompts()
        
        # Create a Product Owner agent like in the orchestrator
        po_agent = create_react_agent(
            llm,
            tools=[create_documentation],
            prompt=prompts[TeamRole.PRODUCT_OWNER]
        )
        
        print("✅ Product Owner agent created")
        
        # Test input state like LangGraph expects
        input_state = {
            "messages": [("user", "Create a simple hello world web application with user login")]
        }
        
        print("🧪 Invoking Product Owner agent...")
        result = po_agent.invoke(input_state)
        
        print(f"📊 Result structure:")
        print(f"Type: {type(result)}")
        print(f"Keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")
        
        if "messages" in result:
            messages = result["messages"]
            print(f"Messages count: {len(messages)}")
            print(f"Messages types: {[type(msg) for msg in messages]}")
            
            # Check the last message
            if messages:
                last_msg = messages[-1]
                print(f"Last message type: {type(last_msg)}")
                print(f"Last message attributes: {dir(last_msg) if hasattr(last_msg, '__dict__') else 'No attributes'}")
                
                if hasattr(last_msg, 'content'):
                    content = last_msg.content
                    print(f"Content length: {len(content)}")
                    print(f"Content ends with: ...{content[-200:]}")
                    
                    # Test handoff detection
                    if "HANDOFF TO ANALYST" in content.upper():
                        print("✅ Handoff signal found in content")
                    else:
                        print("❌ No handoff signal found")
                        # Look for handoff patterns
                        content_upper = content.upper()
                        if "HANDOFF" in content_upper:
                            idx = content_upper.find("HANDOFF")
                            print(f"Found HANDOFF at: {content[idx-20:idx+50]}")
                else:
                    print("❌ Last message has no content attribute")
                    print(f"Last message: {last_msg}")
        else:
            print("❌ No messages key in result")
            
        return result
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    test_agent_result_structure()
