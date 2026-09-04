from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class EncodingMethodEnum(str, Enum):
    ONE_HOT = "one_hot"  # One-Hot Encoding (Variables Indicatrices)
    LABEL = "label"      # Label / Ordinal Encoding
    TARGET = "target"    # Target Encoding (Mean Encoding)


class EncodingRequest(BaseModel):
    parent_version_id: str = Field(
        ..., description="ID de la version source (ex: 'v1_iqr_cap' ou 'v0_raw')"
    )
    id_attraction: Optional[str] = Field(
        default="ALL",
        description="Identifiant de l'attraction (ex: 'H03', 'H07' ou 'ALL')",
    )
    target_columns: Optional[List[str]] = Field(
        default=[],
        description="Liste des colonnes catégorielles à encoder (vide = toutes les colonnes d'objets/catégories)",
    )
    method: EncodingMethodEnum = Field(
        ..., description="Méthode d'encodage : one_hot, label ou target"
    )
    drop_first: Optional[bool] = Field(
        default=False,
        description="Pour One-Hot : supprimer la première catégorie pour éviter la colinéarité",
    )
    target_variable: Optional[str] = Field(
        default=None,
        description="Pour Target Encoding : nom de la colonne cible (y)",
    )


class EncodingStats(BaseModel):
    columns_encoded: List[str]
    generated_columns: List[str]
    rows_affected: int


class EncodingVersionMetadataResponse(BaseModel):
    version_id: str
    parent_version_id: Optional[str]
    id_attraction: Optional[str]
    step_type: str
    method: str
    method_label_fr: str
    target_columns: List[str]
    parameters: Dict[str, Any]
    stats: EncodingStats
    file_path: str
    created_at: str


class EncodingVersionStatsResponse(BaseModel):
    version_id: str
    available_columns: List[str]
    total_records: int
    available_attractions: List[str]