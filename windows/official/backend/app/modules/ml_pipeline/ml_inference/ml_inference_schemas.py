from typing import List, Optional, Dict, Any
from enum import Enum
from pydantic import BaseModel, Field


class ImputationMode(str, Enum):
    FILL_MISSING = "fill_missing"  # Chỉ UPDATE ở những nơi có giá trị NULL / NaN
    OVERWRITE = "overwrite"  # UPDATE tất cả dòng được chỉ định trong dải dữ liệu


class ExportFileFormat(str, Enum):
    CSV = "csv"
    EXCEL = "excel"


class PredictRequest(BaseModel):
    model_id: str = Field(..., description="ID của mô hình đã lưu trong registry")
    version_id: str = Field(
        ..., description="Tên bảng dữ liệu nguồn trong schema data_prep"
    )
    id_attraction: str = Field("ALL", description="ID Attraction hoặc ALL")
    start_date: Optional[str] = Field(
        None, description="Ngày bắt đầu lọc dữ liệu (YYYY-MM-DD)"
    )
    end_date: Optional[str] = Field(
        None, description="Ngày kết thúc lọc dữ liệu (YYYY-MM-DD)"
    )


class CompareRequest(BaseModel):
    version_id: str = Field(
        ..., description="Tên bảng dữ liệu nguồn trong schema data_prep"
    )
    model_ids: List[str] = Field(
        ...,
        min_items=1,
        description="Danh sách các ID mô hình cần so sánh (ví dụ: mô hình dự báo visitor_count, elec_kwh,...)",
    )
    id_attraction: str = Field("ALL", description="ID Attraction hoặc ALL")
    start_date: Optional[str] = Field(
        None, description="Ngày bắt đầu lọc dữ liệu (YYYY-MM-DD)"
    )
    end_date: Optional[str] = Field(
        None, description="Ngày kết thúc lọc dữ liệu (YYYY-MM-DD)"
    )


class PredictionPoint(BaseModel):
    datetime: str = Field(..., description="Mốc thời gian ISO")
    actual: Optional[float] = Field(
        None, description="Giá trị thực tế trong bảng DB (nếu có)"
    )
    predicted: float = Field(..., description="Giá trị mô hình dự đoán (không âm)")


class PredictResponse(BaseModel):
    status: str
    model_id: str
    version_id: str
    id_attraction: str
    target_column: Optional[str] = None
    count: int
    predictions: List[PredictionPoint]


# ==================== SCHEMAS BÁO CÁO SO SÁNH ĐỘNG (DYNAMIC TARGET) ====================


class MetricDetail(BaseModel):
    model_id: str
    target_column: str
    target_type: str
    rmse: float
    mae: float
    r2: float
    total_actual: float
    total_predicted: float
    total_difference: float
    error_percentage: float


class ComparePredictionResponse(BaseModel):
    status: str
    version_id: str
    metrics: Dict[str, MetricDetail] = Field(
        ..., description="Kết quả đánh giá gom nhóm theo target_column hoặc model_id"
    )
    monthly_summary: List[Dict[str, Any]] = Field(
        ..., description="Dữ liệu tổng hợp theo tháng cho từng target_column"
    )
    html_dashboard: Optional[str] = Field(
        None,
        description="Mã HTML Plotly Dashboard tương tác đầy đủ giao diện Multi-tabs",
    )


class ApplyImputationRequest(BaseModel):
    version_id: str = Field(
        ..., description="Tên bảng trong schema data_prep cần ghi đè"
    )
    id_attraction: str = Field("ALL", description="Mã Attraction hoặc ALL")
    target_column: str = Field(
        ...,
        description="Cột target cần ghi đè/lấp dữ liệu (VD: visitor_count, elec_kwh, ec_kwh)",
    )
    mode: ImputationMode = Field(
        ImputationMode.FILL_MISSING,
        description="Chế độ 'fill_missing' hoặc 'overwrite'",
    )
    data: List[Dict[str, Any]] = Field(
        ..., description="Danh sách các điểm dữ liệu cần áp dụng"
    )


class ApplyImputationResponse(BaseModel):
    status: str
    message: str
    version_id: str
    target_column: str
    mode: ImputationMode
    updated_count: int


class ExportPredictionToSchemaRequest(BaseModel):
    model_id: str = Field(..., description="ID của mô hình đã huấn luyện")
    version_id: str = Field(..., description="Tên bảng nguồn trong data_prep")
    id_attraction: Optional[str] = Field("ALL", description="ID Attraction hoặc ALL")
    start_date: Optional[str] = Field(None, description="Ngày bắt đầu (YYYY-MM-DD)")
    end_date: Optional[str] = Field(None, description="Ngày kết thúc (YYYY-MM-DD)")
    target_schema: Optional[str] = Field("predictions", description="Schema đích")
    output_table_name: Optional[str] = Field(
        None, description="Tên bảng xuất ra (tùy chọn)"
    )
    if_exists: Optional[str] = Field(
        "replace", description="Hành vi nếu bảng đã tồn tại: 'replace' hoặc 'append'"
    )


class ExportPredictionResponse(BaseModel):
    status: str
    message: str
    target_schema: str
    target_table: str
    total_rows: int
    kept_target_column: Optional[str] = None
    dropped_columns: Optional[List[str]] = []


class ExportPredictionFileRequest(BaseModel):
    version_id: str = Field(..., description="Tên bảng nguồn trong schema data_prep")
    model_ids: List[str] = Field(..., min_items=1, description="Danh sách model_id cần xuất kết quả dự đoán")
    file_format: ExportFileFormat = Field(ExportFileFormat.EXCEL, description="Định dạng xuất file: csv hoặc excel")
    id_attraction: Optional[str] = Field("ALL", description="ID Attraction hoặc ALL")
    start_date: Optional[str] = Field(None, description="Ngày bắt đầu lọc (YYYY-MM-DD)")
    end_date: Optional[str] = Field(None, description="Ngày kết thúc lọc (YYYY-MM-DD)")