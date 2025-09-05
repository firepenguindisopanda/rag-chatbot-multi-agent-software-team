import tempfile
import os
import json
import re
from datetime import datetime
from typing import Dict, List, Any, Tuple
from .schemas import AgentResponse, TeamRole

def format_agent_responses(responses: List[AgentResponse]) -> str:
    """Format the agent responses into a readable string."""
    if not responses:
        return "*No responses generated. Please check your inputs and try again.*"
    
    # Add header with timestamp and summary
    result = "# 🚀 Multi-Agent Software Team Results\n\n"
    result += f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    result += f"**Team Size:** {len(responses)} agents\n"
    result += f"**Agents:** {', '.join([r.role.value.replace('_', ' ').title() for r in responses])}\n\n"
    result += "---\n\n"
    
    # Add role icons
    role_icons = {
        TeamRole.PRODUCT_OWNER: "📋",
        TeamRole.ANALYST: "🔍", 
        TeamRole.ARCHITECT: "🏗️",
        TeamRole.DEVELOPER: "💻",
        TeamRole.REVIEWER: "👀",
        TeamRole.TESTER: "🧪",
        TeamRole.DESIGNER: "🎨", 
        TeamRole.TECH_WRITER: "📝"
    }
    
    for i, response in enumerate(responses, 1):
        role_name = response.role.value.replace('_', ' ').title()
        icon = role_icons.get(response.role, "🤖")
        result += f"## {icon} {role_name} ({i}/{len(responses)})\n\n"
        result += f"{response.output}\n\n"
        result += "---\n\n"
    
    result += f"*✅ Software solution generated with {len(responses)} agent contributions*"
    return result

def save_to_file(responses: List[AgentResponse], directory: str = None) -> str:
    """Save agent responses to files in a directory."""
    if not directory:
        directory = tempfile.mkdtemp()
    
    os.makedirs(directory, exist_ok=True)
    
    for response in responses:
        filename = f"{response.role.value}.md"
        filepath = os.path.join(directory, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(response.output)
    
    return directory

def read_file_content(file):
    """Read content from uploaded file."""
    if file is None:
        return ""
    
    try:
        if isinstance(file, str):
            # file is already a path
            with open(file, "r", encoding="utf-8") as f:
                return f.read()
        elif hasattr(file, 'name'):
            # file is a file-like object with a name attribute
            with open(file.name, "r", encoding="utf-8") as f:
                return f.read()
        else:
            return ""
    except Exception as e:
        print(f"Error reading file: {str(e)}")
        return ""

def export_to_json(responses: List[AgentResponse], filepath: str = None) -> str:
    """Export agent responses to JSON format."""
    if not filepath:
        filepath = f"software_team_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    data = {
        "timestamp": datetime.now().isoformat(),
        "team_size": len(responses),
        "agents": []
    }
    
    for response in responses:
        data["agents"].append({
            "role": response.role.value,
            "role_display": response.role.value.replace('_', ' ').title(),
            "output": response.output
        })
    
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return f"Results exported to {filepath}"
    except Exception as e:
        return f"Error exporting to JSON: {str(e)}"

def get_agent_summary(responses: List[AgentResponse]) -> str:
    """Generate a summary of the multi-agent team output."""
    if not responses:
        return "No agent responses to summarize."
    
    summary = "## 📊 Multi-Agent Team Summary\n\n"
    summary += f"**Total Agents:** {len(responses)}\n"
    summary += "**Execution Order:** " + " → ".join([
        r.role.value.replace('_', ' ').title() for r in responses
    ]) + "\n\n"
    
    # Count output lengths
    total_chars = sum(len(r.output) for r in responses)
    avg_chars = total_chars // len(responses) if responses else 0
    
    summary += f"**Total Output:** {total_chars:,} characters\n"
    summary += f"**Average per Agent:** {avg_chars:,} characters\n\n"
    
    # List each agent's contribution
    summary += "### Agent Contributions:\n"
    for i, response in enumerate(responses, 1):
        role_name = response.role.value.replace('_', ' ').title()
        char_count = len(response.output)
        summary += f"{i}. **{role_name}**: {char_count:,} characters\n"
    
    return summary

def validate_team_composition(selected_roles: List[TeamRole]) -> Tuple[bool, str]:
    """Validate if the selected team composition makes sense."""
    if not selected_roles:
        return False, "No roles selected. Please select at least one role."
    
    # Essential roles for any project
    essential_roles = [TeamRole.PRODUCT_OWNER, TeamRole.DEVELOPER]
    missing_essential = [role for role in essential_roles if role not in selected_roles]
    
    if missing_essential:
        missing_names = [role.value.replace('_', ' ').title() for role in missing_essential]
        return False, f"Missing essential roles: {', '.join(missing_names)}. These are required for any software project."
    
    # Recommended combinations
    recommendations = []
    
    if TeamRole.ARCHITECT in selected_roles and TeamRole.ANALYST not in selected_roles:
        recommendations.append("Consider adding Analyst for better requirements analysis")
    
    if TeamRole.DEVELOPER in selected_roles and TeamRole.TESTER not in selected_roles:
        recommendations.append("Consider adding Tester for quality assurance")
    
    if TeamRole.REVIEWER not in selected_roles and len(selected_roles) > 3:
        recommendations.append("Consider adding Reviewer for code quality assessment")
    
    if recommendations:
        return True, f"Valid team composition. Suggestions: {'; '.join(recommendations)}"
    else:
        return True, "Excellent team composition!"

def enhance_mermaid_diagrams(content: str) -> str:
    """Enhance and validate Mermaid diagrams in agent content."""
    import re
    
    # Find all mermaid code blocks
    mermaid_pattern = r'```mermaid\n(.*?)\n```'
    
    def fix_mermaid_block(match):
        diagram_content = match.group(1)
        
        # Basic validation and enhancement
        lines = diagram_content.strip().split('\n')
        
        # Ensure proper indentation
        cleaned_lines = []
        for line in lines:
            # Remove excessive whitespace but preserve structure
            cleaned_line = line.strip()
            if cleaned_line:
                cleaned_lines.append(cleaned_line)
        
        # Join with proper newlines
        enhanced_content = '\n    '.join(cleaned_lines)
        
        return f"```mermaid\n    {enhanced_content}\n```"
    
    # Apply enhancements
    enhanced_content = re.sub(mermaid_pattern, fix_mermaid_block, content, flags=re.DOTALL)
    
    return enhanced_content

def extract_mermaid_diagrams(content: str) -> List[dict]:
    """Extract all Mermaid diagrams from content for separate processing."""
    import re
    
    diagrams = []
    mermaid_pattern = r'```mermaid\n(.*?)\n```'
    
    for i, match in enumerate(re.finditer(mermaid_pattern, content, re.DOTALL)):
        diagrams.append({
            'index': i + 1,
            'content': match.group(1),
            'full_match': match.group(0),
            'start': match.start(),
            'end': match.end()
        })
    
    return diagrams

def save_response_to_md(content: str, project_description: str = "", filename: str = None) -> str:
    """Save the complete multi-agent team response to a single markdown file."""
    if not filename:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"software_team_solution_{timestamp}.md"
    
    # Create the full markdown content
    md_content = f"""# 🚀 Multi-Agent Software Team Solution

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Filename:** {filename}

## 📋 Project Description

{project_description}

---

{content}

---

*Generated by Multi-Agent Software Team*
"""
    
    try:
        # Create solutions directory if it doesn't exist
        solutions_dir = "solutions"
        os.makedirs(solutions_dir, exist_ok=True)
        filepath = os.path.join(solutions_dir, filename)
        
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(md_content)
        
        return os.path.abspath(filepath)
    except Exception:
        # Fallback: save to current directory
        try:
            with open(filename, "w", encoding="utf-8") as f:
                f.write(md_content)
            return os.path.abspath(filename)
        except Exception as e2:
            return f"Error saving file: {str(e2)}"

def create_safe_filename(description: str, max_length: int = 50) -> str:
    """Create a safe filename from project description."""
    if not description or not description.strip():
        return ""
    
    # Clean the description for filename
    clean_desc = "".join(c for c in description[:max_length] if c.isalnum() or c in (' ', '-', '_')).strip()
    clean_desc = clean_desc.replace(' ', '_').lower()
    
    # Remove multiple underscores
    while '__' in clean_desc:
        clean_desc = clean_desc.replace('__', '_')
    
    # Remove leading/trailing underscores
    clean_desc = clean_desc.strip('_')
    
    return clean_desc if clean_desc else ""

def save_response_with_auto_filename(content: str, project_description: str = "") -> str:
    """Save response with automatically generated filename based on project description."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    safe_desc = create_safe_filename(project_description)
    if safe_desc:
        filename = f"{safe_desc}_{timestamp}.md"
    else:
        filename = f"software_team_solution_{timestamp}.md"
    
    return save_response_to_md(content, project_description, filename)

def create_project_summary(responses: List[AgentResponse], project_description: str = "") -> str:
    """Create a comprehensive project summary with download options."""
    if not responses:
        return "No responses to summarize."
    
    summary = f"""# 📊 Project Summary

## Overview
**Project:** {project_description or "Software Development Project"}
**Team Size:** {len(responses)} agents
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Team Deliverables

"""
    
    for response in responses:
        role_name = response.role.value.replace('_', ' ').title()
        char_count = len(response.output)
        word_count = len(response.output.split())
        
        summary += f"### {role_name}\n"
        summary += f"- **Length:** {char_count:,} characters / {word_count:,} words\n"
        summary += f"- **Key Topics:** {extract_key_topics(response.output)}\n\n"
    
    # Add statistics
    total_chars = sum(len(r.output) for r in responses)
    total_words = sum(len(r.output.split()) for r in responses)
    
    summary += f"""## Statistics
- **Total Content:** {total_chars:,} characters / {total_words:,} words
- **Average per Agent:** {total_chars // len(responses):,} characters
- **Roles Utilized:** {', '.join([r.role.value.replace('_', ' ').title() for r in responses])}

## Next Steps
1. Review each agent's deliverables
2. Implement the technical specifications
3. Use the provided documentation as reference
4. Follow the deployment guidelines

*This summary was generated automatically by the Multi-Agent Software Team system.*
"""
    
    return summary

def extract_key_topics(text: str) -> str:
    """Extract key topics from agent output."""
    # Simple keyword extraction based on common software development terms
    keywords = [
        'API', 'database', 'authentication', 'security', 'testing', 
        'deployment', 'architecture', 'frontend', 'backend', 'user',
        'system', 'component', 'service', 'interface', 'design'
    ]
    
    found_keywords = []
    text_lower = text.lower()
    
    for keyword in keywords:
        if keyword.lower() in text_lower:
            found_keywords.append(keyword)
    
    return ', '.join(found_keywords[:5]) if found_keywords else "General software development"
