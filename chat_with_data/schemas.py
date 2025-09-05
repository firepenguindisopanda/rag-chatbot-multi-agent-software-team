from enum import Enum
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import pandas as pd

class DataAnalysisType(Enum):
    EXPLORATORY = "exploratory"
    PREDICTIVE = "predictive"
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"
    TIME_SERIES = "time_series"

class DataRequest(BaseModel):
    file_path: str
    user_context: Optional[str] = None
    problem_description: Optional[str] = None
    target_variable: Optional[str] = None

class DataSummary(BaseModel):
    shape: tuple
    columns: List[str]
    data_types: Dict[str, str]
    missing_values: Dict[str, int]
    numerical_columns: List[str]
    categorical_columns: List[str]
    datetime_columns: List[str]

class AnalysisResult(BaseModel):
    summary: str
    visualizations: List[str]  # Base64 encoded plots
    insights: List[str]
    recommended_ml_models: List[str]
    suggested_metrics: List[str]
    next_steps: List[str]

class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str
    timestamp: str
