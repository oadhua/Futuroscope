from typing import Dict, List, Optional, Any
from enum import Enum
from pydantic import BaseModel, Field


class TargetType(str, Enum):
    VISITOR = "visitor"
    ENERGY = "energy"


class TrainModelRequest(BaseModel):
    version_id: str = Field("v0_raw", description="ID version (tên bảng) trong schema data_prep")
    id_attraction: str = Field("ALL", description="ID Attraction ('H03', 'H07', 'ALL')")
    target_column: str = Field("elec_01", description="Cột target dự báo (vd: visitor_count, elec_01, ec_01)")
    target_type: TargetType = Field(
        TargetType.ENERGY, 
        description="Loại target: 'visitor' hoặc 'energy'"
    )
    model_type: str = Field(
        "xgboost", 
        description="Loại mô hình: 'xgboost', 'lightgbm', 'random_forest', 'ridge', 'lstm', 'gru'"
    )
    
    # --- CẤU HÌNH PHÂN CHIA TRAIN/TEST ---
    split_method: str = Field("ratio", description="Phương pháp chia tập dữ liệu: 'ratio' (tỷ lệ) hoặc 'date' (theo ngày)")
    test_size: float = Field(0.2, description="Tỷ lệ tập test (dùng khi split_method='ratio')")
    
    start_date: Optional[str] = Field(None, description="Ngày bắt đầu lấy dữ liệu (YYYY-MM-DD)")
    end_date: Optional[str] = Field(None, description="Ngày kết thúc lấy dữ liệu (YYYY-MM-DD)")
    split_date: Optional[str] = Field(None, description="Mốc ngày chia Train/Test (Dữ liệu trước date là Train, sau date là Test)")

    hyperparameters: Optional[Dict[str, Any]] = Field(
        default_factory=dict, 
        description="Tham số tùy chỉnh cho mô hình"
    )


class TrainModelResponse(BaseModel):
    status: str
    model_id: str
    model_type: str
    target_type: str
    version_id: str
    id_attraction: str
    target_column: str
    train_rows: int
    test_rows: int
    metrics: Dict[str, float] = Field(..., description="Metrics: R2, RMSE, MAE, MAPE")
    feature_names: List[str]
    created_at: str


class ModelRegistryItem(BaseModel):
    model_id: str
    model_type: str
    target_type: str
    version_id: str
    id_attraction: str
    target_column: str
    r2_score: float
    rmse: float
    mae: float
    created_at: Optional[str] = None
    
    
