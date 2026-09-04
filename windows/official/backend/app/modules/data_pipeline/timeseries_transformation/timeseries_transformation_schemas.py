from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class TimeSeriesTransformMethodEnum(str, Enum):
    FIRST_DIFF = "first_diff"       # Différenciation première
    SEASONAL_DIFF = "seasonal_diff" # Différenciation saisonnière
    LOG_TRANSFORM = "log_transform" # Différenciation logarithmique


class TimeSeriesTransformRequest(BaseModel):
    parent_version_id: str = Field(
        ..., description="ID de la version source (ex: 'v0_raw', 'v1_scale_standard')"
    )
    id_attraction: Optional[str] = Field(
        default="ALL",
        description="Identifiant de l'attraction (ex: 'H03', 'H07' ou 'ALL')",
    )
    target_columns: List[str] = Field(
        ..., description="Liste des colonnes temporelles/numériques à transformer"
    )
    method: TimeSeriesTransformMethodEnum = Field(
        ..., description="Méthode de transformation"
    )
    seasonal_period: Optional[int] = Field(
        default=24,
        description="Période saisonnière (ex: 24 pour 24h, 7 pour 7 jours) - Utile pour seasonal_diff",
    )
    drop_na: Optional[bool] = Field(
        default=True,
        description="Supprimer les lignes contenant des NaN créés par le lag/diff",
    )


class TimeSeriesTransformStats(BaseModel):
    rows_before: int
    rows_after: int
    nan_rows_dropped: int
    transformed_columns: List[str]


class TimeSeriesTransformVersionMetadataResponse(BaseModel):
    version_id: str
    parent_version_id: Optional[str]
    id_attraction: Optional[str]
    step_type: str
    method: str
    method_label_fr: str
    target_columns: List[str]
    parameters: Dict[str, Any]
    stats: TimeSeriesTransformStats
    file_path: str
    created_at: str


class TimeSeriesTransformVersionStatsResponse(BaseModel):
    version_id: str
    available_columns: List[str]
    numeric_columns: List[str]
    total_records: int
    available_attractions: List[str]