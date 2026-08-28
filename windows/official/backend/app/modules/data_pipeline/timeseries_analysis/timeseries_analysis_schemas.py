from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Union


# --- Common Request Schema ---
class BaseTimeSeriesRequest(BaseModel):
    version: str = Field(
        default="v0_raw", description="Phiên bản dữ liệu (e.g., v0_raw, v1, v2)"
    )
    id_attraction: str = Field(
        default="ALL",
        description="ID của attraction (mặc định 'ALL' nếu không lọc theo từng attraction)",
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
    period: int = Field(
        default=24,
        ge=1,
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
