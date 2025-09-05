from enum import Enum
from pydantic import BaseModel
from typing import Dict, List, Optional

class TeamRole(Enum):
    PRODUCT_OWNER = "product_owner"
    ANALYST = "analyst"
    ARCHITECT = "architect"
    SOLUTION_ARCHITECT = "solution_architect"  # Alias for ARCHITECT
    DEVELOPER = "developer"
    SENIOR_DEVELOPER = "senior_developer"
    FULL_STACK_DEVELOPER = "full_stack_developer"
    REVIEWER = "reviewer"
    TESTER = "tester"
    DESIGNER = "designer"
    TECH_WRITER = "tech_writer"
    DEVOPS_ENGINEER = "devops_engineer"

class AgentMessage(BaseModel):
    role: TeamRole
    content: str

class AgentResponse(BaseModel):
    role: TeamRole
    output: str

class ProjectRequest(BaseModel):
    description: str
    file_content: Optional[str] = None
    selected_roles: Optional[List[TeamRole]] = None
