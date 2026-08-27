import re
from enum import Enum
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Union

from app.modules.data_pipeline.timeseries_analysis.timeseries_analysis_services import (
    get_available_attractions,
    get_available_features,
)

def sanitize_enum_key(name: str) -> str:
    """Chuyển chuỗi bất kỳ thành Key hợp lệ cho Enum Python (VD: 'Danse des Robots' -> 'DANSE_DES_ROBOTS')."""
    sanitized = re.sub(r'\W+', '_', str(name)).strip('_').upper()
    if not sanitized or sanitized[0].isdigit():
        sanitized = f"ATTR_{sanitized}"
    return sanitized

def create_dynamic_enums():
    raw_attractions = get_available_attractions() or ["DEFAULT"]
    raw_features = get_available_features() or ["value"]

    # Tạo dict {KEY_HỢP_LỆ: giá_trị_thực_tế_trong_db}
    attr_dict = {sanitize_enum_key(a): str(a) for a in raw_attractions}
    feat_dict = {sanitize_enum_key(f): str(f) for f in raw_features}

    AttractionEnum = Enum("AttractionEnum", attr_dict, type=str)
    FeatureEnum = Enum("FeatureEnum", feat_dict, type=str)
    
    return AttractionEnum, FeatureEnum

# Khởi tạo Enum
AttractionEnum, FeatureEnum = create_dynamic_enums()

class TimeSeriesAnalysisRequest(BaseModel):
    id_attraction: AttractionEnum = Field(
        ..., 
        description="Chọn Attraction từ danh sách có sẵn trong DB"
    )
    feature_column: FeatureEnum = Field(
        ..., 
        description="Chọn biến/cột dữ liệu đầu vào để phân tích"
    )
    version: str = Field(default="v1", description="Phiên bản mô hình/phân tích")

# --- Common Request Schema ---
class BaseTimeSeriesRequest(BaseModel):
    version: str = Field(
        default="v1", description="Data version ID (e.g., v1, v2, all)"
    )
    id_attraction: Optional[str] = Field(
        default=None, description="ID của attraction (nếu lọc theo attraction cụ thể)"
    )
    col_name: str = Field(
        ...,
        description="Tên cột chuỗi thời gian (e.g., visitor_count, conso_elec, temperature)",
    )


# --- 1. Stationarity Test Schemas ---
class StationarityRequest(BaseTimeSeriesRequest):
    method: str = Field(
        default="adf",
        description="Phương pháp kiểm định: 'adf' (Augmented Dickey-Fuller) hoặc 'kpss'",
    )


class StationarityResponse(BaseModel):
    col_name: str
    method: str
    is_stationary: bool
    test_statistic: float
    p_value: float
    critical_values: Dict[str, float]
    message: str


# --- 2. Time Series Decomposition Schemas ---
class DecompositionRequest(BaseTimeSeriesRequest):
    model_type: str = Field(
        default="additive",
        description="Mô hình phân rã: 'additive' hoặc 'multiplicative'",
    )
    period: Optional[int] = Field(
        default=24,
        description="Tần số chuỗi thời gian (mặc định 24 cho dữ liệu theo giờ)",
    )


class DecompositionComponent(BaseModel):
    timestamps: List[str]
    observed: List[Optional[float]]
    trend: List[Optional[float]]
    seasonal: List[Optional[float]]
    residual: List[Optional[float]]


# --- 3. ACF and PACF Schemas ---
class AcfPacfRequest(BaseTimeSeriesRequest):
    plot_type: str = Field(
        default="acf",
        description="Loại hàm: 'acf' (Autocorrelation) hoặc 'pacf' (Partial Autocorrelation)",
    )
    lags: int = Field(
        default=25, ge=1, le=100, description="Số lượng lags cần tính toán"
    )


class AcfPacfResponse(BaseModel):
    col_name: str
    plot_type: str
    lags: List[int]
    values: List[float]
    confidence_interval_upper: List[float]
    confidence_interval_lower: List[float]
