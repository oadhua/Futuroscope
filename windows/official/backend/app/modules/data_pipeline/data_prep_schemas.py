from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional


class SharedVersionMetadataResponse(BaseModel):
    version_id: str
    parent_version_id: Optional[str] = None
    id_attraction: Optional[str] = "ALL"
    step_type: str  # "missing_value_imputation" hoặc "outlier_treatment"
    method: str
    method_label_fr: str
    action: Optional[str] = None  # None đối với missing value, "cap"/"drop"... đối với outlier
    target_columns: List[str] = []
    parameters: Dict[str, Any] = Field(default_factory=dict)
    stats: Dict[str, Any] = Field(default_factory=dict)  # Chấp nhận linh hoạt mọi cấu trúc stats
    file_path: str
    created_at: str


class DeleteVersionResponse(BaseModel):
    status: str
    message: str