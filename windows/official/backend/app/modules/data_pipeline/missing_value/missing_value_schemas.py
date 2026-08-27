from enum import Enum
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional


class ImputationMethodEnum(str, Enum):
    MEAN = "mean"
    MEDIAN = "median"
    MODE = "mode"
    CONSTANT = "constant"
    BFILL = "bfill"
    FFILL = "ffill"
    KNN = "knn"
    LINEAR = "linear"
    POLYNOMIAL = "polynomial"
    VISITOR_DOMAIN_RULES = "visitor_domain_rules"
    ENERGY_DOMAIN_RULES = "energy_domain_rules"


class ImputationRequest(BaseModel):
    parent_version_id: str = Field(
        ..., description="ID của phiên bản dữ liệu nguồn (VD: v0_raw)"
    )
    id_attraction: Optional[str] = Field(
        default="ALL",
        description="Mã trò chơi / điểm tham quan (VD: 'H03', 'H07' hoặc 'ALL')",
    )
    target_columns: Optional[List[str]] = Field(
        default=[], description="Danh sách cột áp dụng xử lý"
    )
    method: ImputationMethodEnum = Field(
        ..., description="Phương pháp xử lý missing values"
    )
    params: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Tham số phụ: constant_value, n_neighbors, order",
    )


class NullStats(BaseModel):
    null_count_before: Dict[str, int]
    null_count_after: Dict[str, int]


class VersionMetadataResponse(BaseModel):
    version_id: str
    parent_version_id: Optional[str]
    id_attraction: Optional[str]
    step_type: str
    method: str
    method_label_fr: str
    target_columns: List[str]
    parameters: Dict[str, Any]
    stats: NullStats
    file_path: str
    created_at: str


class DeleteVersionResponse(BaseModel):
    status: str
    message: str


class VersionStatsResponse(BaseModel):
    version_id: str
    null_counts: Dict[str, int]
    available_attractions: List[str] = Field(
        default_factory=list,
        description="Danh sách danh mục id_attraction có trong phiên bản dữ liệu",
    )
