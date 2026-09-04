from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ScalingMethodEnum(str, Enum):
    STANDARD = "standard"  # Normalisation standard (Z-score)
    MIN_MAX = "min_max"  # Normalisation Min-Max [0, 1]
    ROBUST = "robust"  # Normalisation robuste (basée sur l'IQR : Q1 & Q3)


class ScalingRequest(BaseModel):
    parent_version_id: str = Field(
        ..., description="ID de la version source (ex: 'v1_iqr_cap' ou 'v2_knn')"
    )
    id_attraction: Optional[str] = Field(
        default="ALL",
        description="Identifiant de l'attraction (ex: 'H03', 'H07' ou 'ALL')",
    )
    target_columns: Optional[List[str]] = Field(
        default=[],
        description="Liste des colonnes à normaliser (vide = toutes les colonnes numériques)",
    )
    method: ScalingMethodEnum = Field(
        ..., description="Méthode de mise à l'échelle : standard, min_max ou robust"
    )
    feature_range: Optional[List[float]] = Field(
        default=[0.0, 1.0],
        description="Plage de valeurs pour Min-Max (par défaut [0, 1])",
    )


class ScalingStats(BaseModel):
    columns_scaled: List[str]
    rows_affected: int


class ScalingVersionMetadataResponse(BaseModel):
    version_id: str
    parent_version_id: Optional[str]
    id_attraction: Optional[str]
    step_type: str
    method: str
    method_label_fr: str
    target_columns: List[str]
    parameters: Dict[str, Any]
    stats: ScalingStats
    file_path: str
    created_at: str
    
class ScalingVersionStatsResponse(BaseModel):
    version_id: str
    available_columns: List[str]
    total_records: int
    available_attractions: List[str]