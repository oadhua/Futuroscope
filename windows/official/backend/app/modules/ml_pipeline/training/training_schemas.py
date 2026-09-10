from typing import Dict, List, Optional, Any
from enum import Enum
from pydantic import BaseModel, Field


class TargetType(str, Enum):
    VISITOR = "visitor"
    ELEC = "elec"
    THERMAL = "thermal"


class TrainModelRequest(BaseModel):
    version_id: str = Field("v0_raw", description="ID version (tên bảng) trong schema data_prep")
    id_attraction: str = Field("ALL", description="ID Attraction ('H03', 'H07', 'ALL')")
    target_column: str = Field("elec_01", description="Cột target dự báo")
    target_type: TargetType = Field(TargetType.ELEC, description="Loại target: 'visitor' hoặc 'energy'")
    model_type: str = Field("xgboost", description="Loại mô hình: 'xgboost', 'lightgbm', 'random_forest', 'ridge', 'lstm', 'gru'")
    
    split_method: str = Field("ratio", description="Phương pháp chia: 'ratio' hoặc 'date'")
    test_size: float = Field(0.2, description="Tỷ lệ test")
    start_date: Optional[str] = Field(None)
    end_date: Optional[str] = Field(None)
    split_date: Optional[str] = Field(None)

    hyperparameters: Optional[Dict[str, Any]] = Field(default_factory=dict)


class FeatureImportanceItem(BaseModel):
    feature: str
    importance: float


class PerformanceComparison(BaseModel):
    previous_model_id: Optional[str] = None
    r2_diff: Optional[float] = None
    rmse_diff: Optional[float] = None
    mae_diff: Optional[float] = None
    improvement_summary: str


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
    metrics: Dict[str, float]
    feature_names: List[str]
    
    # --- CÁC MỤC MỚI BỔ SUNG ---
    shap_importance: List[FeatureImportanceItem] = Field(default_factory=list, description="Top biến quan trọng theo SHAP")
    pfi_importance: List[FeatureImportanceItem] = Field(default_factory=list, description="Top biến quan trọng theo PFI")
    performance_improvement: PerformanceComparison
    ai_explanation: str = Field("", description="AI phân tích và giải thích tự động")
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