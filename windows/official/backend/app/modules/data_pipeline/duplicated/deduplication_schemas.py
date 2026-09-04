from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class DeduplicationKeepEnum(str, Enum):
    FIRST = "first"  # Giữ lại bản ghi đầu tiên
    LAST = "last"    # Giữ lại bản ghi cuối cùng
    NONE = "none"    # Xóa toàn bộ các bản ghi bị trùng (không giữ bản ghi nào)


class DeduplicationRequest(BaseModel):
    parent_version_id: str = Field(
        ..., description="ID de la version source (ex: 'v1_scale_standard' ou 'v0_raw')"
    )
    id_attraction: Optional[str] = Field(
        default="ALL",
        description="Identifiant de l'attraction (ex: 'H03', 'H07' ou 'ALL')",
    )
    subset_columns: Optional[List[str]] = Field(
        default=[],
        description="Liste des colonnes à vérifier pour les doublons (vide = vérifier toutes les colonnes)",
    )
    keep: DeduplicationKeepEnum = Field(
        default=DeduplicationKeepEnum.FIRST,
        description="Quelle occurrence conserver : 'first', 'last', ou 'none'",
    )


class DeduplicationStats(BaseModel):
    rows_before: int
    rows_after: int
    duplicates_removed: int
    subset_columns_used: List[str]


class DeduplicationVersionMetadataResponse(BaseModel):
    version_id: str
    parent_version_id: Optional[str]
    id_attraction: Optional[str]
    step_type: str
    method: str
    method_label_fr: str
    target_columns: List[str]
    parameters: Dict[str, Any]
    stats: DeduplicationStats
    file_path: str
    created_at: str


class DeduplicationVersionStatsResponse(BaseModel):
    version_id: str
    available_columns: List[str]
    total_records: int
    total_duplicates: int
    available_attractions: List[str]