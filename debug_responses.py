#!/usr/bin/env python3
"""
Debug the actual LLM responses to see why handoffs are failing
"""

import os
import sys

# Load environment variables (from .env if available) instead of hardcoding secrets
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

if not os.getenv("NVIDIA_API_KEY"):
    print("⚠ NVIDIA_API_KEY not set. Individual agent response debug may not function correctly.")

# Add path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_individual_agents():
    """Test individual agent responses to see what they're actually generating."""
    print("🔍 Testing Individual Agent Responses")
    print("=" * 50)
    
    try:
        from langchain_nvidia_ai_endpoints import ChatNVIDIA
        from multi_agent_software_team.prompts import create_system_prompts
        from multi_agent_software_team.schemas import TeamRole
        
        # Initialize LLM
        llm = ChatNVIDIA(model="meta/llama-3.1-8b-instruct")
        prompts = create_system_prompts()
        
        # Test Product Owner
        print("\n🧪 Testing Product Owner:")
        po_prompt = prompts[TeamRole.PRODUCT_OWNER]
        print(f"Prompt ends with: ...{po_prompt[-100:]}")
        
        input_message = """
Project Description: Create a simple hello world web application with user login

Please work together as a software development team to deliver a complete solution.
"""
        
        # Create messages for Product Owner
        po_messages = [
            {"role": "system", "content": po_prompt},
            {"role": "user", "content": input_message}
        ]
        
        po_response = llm.invoke(po_messages)
        print(f"PO Response length: {len(po_response.content)}")
        print(f"PO Response ends with: ...{po_response.content[-200:]}")
        
        # Check if it contains handoff
        if "HANDOFF TO ANALYST" in po_response.content:
            print("✅ Product Owner correctly handed off to Analyst")
        else:
            print("❌ Product Owner did NOT hand off to Analyst")
            print("Looking for handoff patterns in response...")
            content_upper = po_response.content.upper()
            if "HANDOFF" in content_upper:
                print(f"Found HANDOFF: {po_response.content[content_upper.find('HANDOFF')-20:content_upper.find('HANDOFF')+50]}")
            else:
                print("No HANDOFF found at all")
        
        # Test Analyst with Product Owner's output
        print("\n🧪 Testing Analyst:")
        analyst_prompt = prompts[TeamRole.ANALYST]
        
        # Simulate context with PO's work
        analyst_input = f"""
Previous work from Product Owner:
{po_response.content}

Continue the software development process by conducting requirements analysis.
"""
        
        analyst_messages = [
            {"role": "system", "content": analyst_prompt},
            {"role": "user", "content": analyst_input}
        ]
        
        analyst_response = llm.invoke(analyst_messages)
        print(f"Analyst Response length: {len(analyst_response.content)}")
        print(f"Analyst Response ends with: ...{analyst_response.content[-200:]}")
        
        # Check if it contains handoff
        if "HANDOFF TO ARCHITECT" in analyst_response.content:
            print("✅ Analyst correctly handed off to Architect")
        else:
            print("❌ Analyst did NOT hand off to Architect")
            content_upper = analyst_response.content.upper()
            if "HANDOFF" in content_upper:
                print(f"Found HANDOFF: {analyst_response.content[content_upper.find('HANDOFF')-20:content_upper.find('HANDOFF')+50]}")
            else:
                print("No HANDOFF found at all")
        
        return {
            "po_response": po_response.content,
            "analyst_response": analyst_response.content
        }
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    test_individual_agents()
