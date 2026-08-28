from enum import Enum
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional


class OutlierDetectionMethodEnum(str, Enum):
    IQR = "iqr"
    Z_SCORE = "z_score"
    ISOLATION_FOREST = "isolation_forest"
    LOF = "lof"
    VISITOR_DOMAIN_RULES = "visitor_domain_rules"
    ENERGY_DOMAIN_RULES = "energy_domain_rules"


class OutlierTreatmentActionEnum(str, Enum):
    CAP = "cap"  # Winsorization (Cắt ngọn / Giới hạn biên)
    CLIP_IQR = "clip_iqr"  # Giới hạn theo Q1 - 1.5*IQR và Q3 + 1.5*IQR
    NULLIFY = "nullify"  # Biến outlier thành NaN (để xử lý missing value sau)
    DROP = "drop"  # Xóa các dòng chứa outlier
    NONE = "none"  # Chỉ phát hiện/đánh dấu, không thay đổi giá trị


class OutlierRequest(BaseModel):
    parent_version_id: str = Field(
        ..., description="ID của phiên bản dữ liệu nguồn (VD: v0_raw hoặc v1_knn)"
    )
    id_attraction: Optional[str] = Field(
        default="ALL",
        description="Mã trò chơi / điểm tham quan (VD: 'H03', 'H07' hoặc 'ALL')",
    )
    target_columns: Optional[List[str]] = Field(
        default=[], description="Danh sách cột áp dụng kiểm tra/xử lý outlier"
    )
    method: OutlierDetectionMethodEnum = Field(
        ...,
        description="Phương pháp phát hiện nhiễu (IQR, Z-Score, Isolation Forest...)",
    )
    action: OutlierTreatmentActionEnum = Field(
        default=OutlierTreatmentActionEnum.CAP,
        description="Hành động xử lý nhiễu (cap, nullify, drop, none)",
    )
    params: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Tham số phụ: iqr_factor, z_threshold, contamination, n_neighbors",
    )


class OutlierStats(BaseModel):
    outliers_detected: Dict[str, int]
    rows_affected: int


class OutlierVersionMetadataResponse(BaseModel):
    version_id: str
    parent_version_id: Optional[str]
    id_attraction: Optional[str]
    step_type: str
    method: str
    method_label_fr: str
    action: str
    target_columns: List[str]
    parameters: Dict[str, Any]
    stats: OutlierStats
    file_path: str
    created_at: str


class DeleteVersionResponse(BaseModel):
    status: str
    message: str


class OutlierVersionStatsResponse(BaseModel):
    version_id: str
    outlier_counts: Dict[str, int] = Field(
        ..., description="Thống kê số lượng Outliers tìm thấy trên từng cột dữ liệu"
    )
    total_records: int = Field(..., description="Tổng số dòng dữ liệu hiện tại")
    available_attractions: List[str] = Field(
        default_factory=list,
        description="Danh sách danh mục id_attraction có trong phiên bản dữ liệu",
    )
