from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class CorrelationMatrix(BaseModel):
    columns: List[str]
    values: List[List[Optional[float]]]  # Chấp nhận None nếu có giá trị NaN/null

class TargetCorrelationItem(BaseModel):
    colonne: str
    correlation: Optional[float]          # Chấp nhận None

class FunctionalDependencyItem(BaseModel):
    determinants: List[str]  # Đổi từ determinant: str -> determinants: List[str]
    dependent: str

class MultiColumnProfilingResponse(BaseModel):
    correlation_matrix: Dict[str, Any]
    target_correlation: List[Dict[str, Any]]
    functional_dependencies: List[FunctionalDependencyItem]  # Validate theo model mới
    sample_data: List[Dict[str, Any]]