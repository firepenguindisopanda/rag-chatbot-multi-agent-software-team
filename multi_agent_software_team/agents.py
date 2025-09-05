from typing import List, Dict, Optional
from langchain_core.prompts import ChatPromptTemplate
from .schemas import TeamRole, AgentResponse, AgentMessage
from .prompts import create_system_prompts
import logging
import re

logger = logging.getLogger(__name__)

class Agent:
    def __init__(self, role: TeamRole, llm):
        self.role = role
        self.llm = llm
        self.system_prompts = create_system_prompts()
        # Attempt to attach knowledge base lazily; llm expected to maybe have an embedder attribute
        self._kb = None
        try:
            from .knowledge_base_manager import get_kb  # local import to avoid hard dependency if modules missing
            embedder = getattr(llm, 'embedder', None) or getattr(llm, 'embeddings', None) or None
            if embedder:
                self._kb = get_kb(embedder)
        except Exception as e:
            logger.debug(f"Knowledge base not initialized for {self.role.value}: {e}")

    def _extract_project_complexity(self, description: str) -> str:
        """Analyze project complexity to adjust agent responses."""
        complexity_indicators = {
            'simple': ['simple', 'basic', 'minimal', 'small', 'prototype'],
            'medium': ['web app', 'api', 'database', 'authentication', 'responsive'],
            'complex': ['microservices', 'distributed', 'scalable', 'enterprise', 'machine learning', 'ai', 'real-time', 'multi-tenant']
        }
        
        description_lower = description.lower()
        scores = {level: sum(1 for keyword in keywords if keyword in description_lower) 
                 for level, keywords in complexity_indicators.items()}
        
        if scores['complex'] >= 2:
            return "complex"
        elif scores['medium'] >= 2:
            return "medium"
        else:
            return "simple"
    
    def _get_context_summary(self, previous_outputs: List[AgentMessage]) -> str:
        """Create a concise summary of previous agent outputs."""
        if not previous_outputs:
            return ""
        
        summary = "PREVIOUS TEAM INSIGHTS:\n"
        for msg in previous_outputs[-3:]:  # Only use last 3 to avoid token limits
            role_name = msg.role.value.replace('_', ' ').title()
            # Extract key points (first 200 chars of each output)
            content_preview = msg.content[:200] + "..." if len(msg.content) > 200 else msg.content
            summary += f"• {role_name}: {content_preview}\n"
        
        return summary
    
    def _retrieve_kb_context(self, project_description: str) -> str:
        """Retrieve knowledge base snippets for this agent's role if available."""
        if not self._kb:
            return ""
        try:
            # Use a focused query; could be improved with dynamic query generation
            base_query = project_description.split('\n')[0][:200]
            docs = self._kb.retrieve(self.role.value, base_query, k=3)
            if not docs:
                # Lazy ingest attempt (e.g., first time) already handled inside retrieve
                return ""
            snippets = []
            for d in docs:
                text = d.page_content.strip().replace('\n', ' ')[:500]
                snippets.append(f"- {text}")
            if snippets:
                return "ROLE KNOWLEDGE BASE EXTRACTS:\n" + "\n".join(snippets)
        except Exception as e:
            logger.debug(f"KB retrieval failed for {self.role.value}: {e}")
        return ""

    def process(self, project_description: str, previous_outputs: List[AgentMessage] = None) -> AgentResponse:
        """Process the project description and previous outputs to generate a response."""
        try:
            # Analyze project complexity
            complexity = self._extract_project_complexity(project_description)
            
            # Create context from previous outputs
            context = ""
            if previous_outputs:
                if len(previous_outputs) <= 2:
                    # For first few agents, include full context
                    context = "\n\n".join([f"=== {msg.role.value.upper()} OUTPUT ===\n{msg.content}" for msg in previous_outputs])
                else:
                    # For later agents, use summary to avoid token limits
                    context = self._get_context_summary(previous_outputs)
            
            # Retrieve knowledge base context
            kb_context = self._retrieve_kb_context(project_description)
            if kb_context:
                context = f"{kb_context}\n\n{context}" if context else kb_context
            
            # Add complexity guidance to prompt
            complexity_guidance = {
                'simple': "Focus on essential features and straightforward implementation.",
                'medium': "Balance functionality with maintainability. Consider standard patterns.",
                'complex': "Emphasize scalability, security, and enterprise patterns. Plan for growth."
            }
            
            prompt_template = (
                f"{self.system_prompts[self.role]}\n\n"
                f"PROJECT COMPLEXITY: {complexity.upper()} - {complexity_guidance[complexity]}\n\n"
                "PROJECT DESCRIPTION:\n{project_description}\n\n"
                "{context}\n\n"
                "Based on the project description, complexity level, knowledge base extracts (if any), and previous team outputs, "
                "provide your response according to your role's output format. "
                "Be comprehensive but focused on your specific role. "
                "Ensure your output builds upon and references previous team work and knowledge base material when relevant."
            )
            
            prompt = ChatPromptTemplate.from_template(prompt_template)
            
            formatted_prompt = prompt.format(
                project_description=project_description,
                context=context
            )
            
            result = self.llm.invoke(formatted_prompt)
            
            # Post-process the response to ensure quality
            processed_output = self._post_process_output(result.content, complexity)
            
            return AgentResponse(
                role=self.role,
                output=processed_output
            )
        except Exception as e:
            logger.error(f"Error in {self.role.value} agent: {str(e)}")
            return AgentResponse(
                role=self.role,
                output=f"Error processing request: {str(e)}"
            )
    
    def _post_process_output(self, content: str, complexity: str) -> str:
        """Post-process agent output to ensure quality and consistency."""
        # Add role identifier at the start if not present
        role_name = self.role.value.replace('_', ' ').title()
        if not content.startswith(f"# {role_name}") and not content.startswith(f"## {role_name}"):
            content = f"# {role_name} Analysis\n\n{content}"
        
        # Enhance Mermaid diagrams if present
        from .utils import enhance_mermaid_diagrams
        content = enhance_mermaid_diagrams(content)
        
        # Ensure minimum content length for complex projects
        if complexity == "complex" and len(content) < 500:
            content += f"\n\n*Note: This {role_name.lower()} analysis has been kept concise but should be expanded for production use.*"
        
        return content
