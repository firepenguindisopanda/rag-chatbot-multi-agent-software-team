from typing import List, Dict, Optional, Callable
from .schemas import TeamRole, AgentResponse, AgentMessage, ProjectRequest
from .agents import Agent
from .utils import validate_team_composition
import logging
import time

logger = logging.getLogger(__name__)

class Orchestrator:
    def __init__(self, llm):
        self.llm = llm
        self.execution_order = [
            TeamRole.PRODUCT_OWNER,
            TeamRole.ANALYST,
            TeamRole.ARCHITECT,
            TeamRole.DEVELOPER,
            TeamRole.TESTER,
            TeamRole.DESIGNER,
            TeamRole.REVIEWER,
            TeamRole.TECH_WRITER
        ]
    
    def run_pipeline(self, project_request: ProjectRequest, progress_callback: Optional[Callable] = None) -> List[AgentResponse]:
        """Run the multi-agent pipeline on the project request."""
        try:
            description = project_request.description
            if project_request.file_content:
                description += f"\n\nFILE CONTENT:\n{project_request.file_content}"
            
            responses = []
            messages = []
            total_agents = len(self.execution_order)
            
            for i, role in enumerate(self.execution_order, 1):
                start_time = time.time()
                logger.info(f"Running agent {i}/{total_agents}: {role.value}")
                
                if progress_callback:
                    progress_callback(f"🤖 Running {role.value.replace('_', ' ').title()} ({i}/{total_agents})")
                
                agent = Agent(role, self.llm)
                response = agent.process(description, messages)
                responses.append(response)
                messages.append(AgentMessage(role=role, content=response.output))
                
                elapsed = time.time() - start_time
                logger.info(f"Completed agent: {role.value} in {elapsed:.2f}s")
            
            if progress_callback:
                progress_callback(f"✅ All {total_agents} agents completed successfully!")
            
            return responses
        except Exception as e:
            logger.error(f"Error in orchestration: {str(e)}")
            if progress_callback:
                progress_callback(f"❌ Error: {str(e)}")
            return []

    def run_selected_roles(self, project_request: ProjectRequest, roles: List[TeamRole], progress_callback: Optional[Callable] = None) -> List[AgentResponse]:
        """Run only selected roles in the multi-agent pipeline."""
        try:
            # Validate team composition
            is_valid, message = validate_team_composition(roles)
            if not is_valid:
                logger.warning(f"Invalid team composition: {message}")
                if progress_callback:
                    progress_callback(f"⚠️ {message}")
                return []
            
            description = project_request.description
            if project_request.file_content:
                description += f"\n\nFILE CONTENT:\n{project_request.file_content}"
            
            responses = []
            messages = []
            
            # Sort roles by execution order
            sorted_roles = [role for role in self.execution_order if role in roles]
            total_agents = len(sorted_roles)
            
            logger.info(f"Starting pipeline with {total_agents} agents: {[r.value for r in sorted_roles]}")
            
            for i, role in enumerate(sorted_roles, 1):
                start_time = time.time()
                logger.info(f"Running agent {i}/{total_agents}: {role.value}")
                
                if progress_callback:
                    progress_callback(f"🤖 Running {role.value.replace('_', ' ').title()} ({i}/{total_agents})")
                
                agent = Agent(role, self.llm)
                response = agent.process(description, messages)
                
                if "Error processing request" in response.output:
                    logger.warning(f"Agent {role.value} encountered an error")
                    if progress_callback:
                        progress_callback(f"⚠️ {role.value.replace('_', ' ').title()} encountered issues but continued")
                
                responses.append(response)
                messages.append(AgentMessage(role=role, content=response.output))
                
                elapsed = time.time() - start_time
                logger.info(f"Completed agent: {role.value} in {elapsed:.2f}s")
            
            if progress_callback:
                progress_callback(f"✅ All {total_agents} agents completed successfully!")
            
            return responses
        except Exception as e:
            logger.error(f"Error in orchestration: {str(e)}")
            if progress_callback:
                progress_callback(f"❌ Error: {str(e)}")
            return []

    def get_execution_plan(self, roles: List[TeamRole]) -> str:
        """Get a readable execution plan for the selected roles."""
        if not roles:
            return "No roles selected"
        
        sorted_roles = [role for role in self.execution_order if role in roles]
        role_names = [role.value.replace('_', ' ').title() for role in sorted_roles]
        
        plan = f"**Execution Plan ({len(sorted_roles)} agents):**\n"
        for i, name in enumerate(role_names, 1):
            plan += f"{i}. {name}\n"
        
        return plan
