from pydantic import BaseModel
from typing import List, Dict, Any, Optional


class VersionInfo(BaseModel):
    version_id: str
    description: Optional[str] = None


class CorrelationMatrix(BaseModel):
    columns: List[str]
    values: List[List[Optional[float]]]


class TargetCorrelationItem(BaseModel):
    colonne: str
    correlation: Optional[float]


class FunctionalDependencyItem(BaseModel):
    determinants: List[str]
    dependent: str


class MultiColumnProfilingResponse(BaseModel):
    selected_version: str
    selected_attraction: str
    correlation_matrix: Dict[str, Any]
    target_correlation: List[Dict[str, Any]]
    functional_dependencies: List[FunctionalDependencyItem]
    sample_data: List[Dict[str, Any]]
