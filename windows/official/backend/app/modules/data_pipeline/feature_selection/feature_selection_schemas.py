from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class FeatureSelectionMethodEnum(str, Enum):
    CORRELATION = "correlation"      # Pearson/Spearman
    MUTUAL_INFO = "mutual_info"      # Mutual Information
    RANDOM_FOREST = "random_forest"  # Random Forest Feature Importance
    XGBOOST = "xgboost"              # XGBoost Feature Importance
    SHAP = "shap"                    # SHAP Values
    LASSO = "lasso"                  # L1 Regularization Lasso


class FeatureSelectionRequest(BaseModel):
    parent_version_id: str = Field(
        ..., description="ID de la version source (ex: 'v0_raw')"
    )
    id_attraction: Optional[str] = Field(
        default="ALL",
        description="Code de l'attraction (ex: 'H03', 'H07' ou 'ALL')",
    )
    target_column: str = Field(
        ..., description="Colonne cible (ex: 'visitor_count')"
    )
    selected_features: List[str] = Field(
        ..., description="Liste des caractéristiques sélectionnées à conserver"
    )
    method: str = Field(
        default="random_forest",
        description="Méthode utilisée pour la sélection des caractéristiques",
    )
    custom_version_id: Optional[str] = Field(
        default=None, description="Nom de version personnalisé (optionnel)"
    )


class FeatureAnalyzeRequest(BaseModel):
    version_id: str = Field(..., description="ID de la version de données à analyser")
    id_attraction: Optional[str] = Field(default="ALL", description="Code de l'attraction")
    target_column: str = Field(..., description="Colonne cible (Variable Cible)")
    method: str = Field(default="random_forest", description="Méthode de calcul d'importance")
    top_k: Optional[int] = Field(default=None, description="Nombre de top K caractéristiques à retourner")
    calculate_all: Optional[bool] = Field(default=True, description="Calculer pour toutes les colonnes")


class FeatureItem(BaseModel):
    name: str
    score: float
    cumulative_score: Optional[float] = None


class FeatureImportanceAnalysisResponse(BaseModel):
    version_id: str
    target_column: str
    method: str
    features: List[FeatureItem]


class FeatureSelectionStats(BaseModel):
    target_column: str
    original_feature_count: int
    selected_feature_count: int
    retained_features: List[str]
    dropped_features: List[str]
    rows_affected: int


class FeatureSelectionMetadataResponse(BaseModel):
    version_id: str
    parent_version_id: Optional[str]
    id_attraction: Optional[str]
    step_type: str
    method: str
    method_label_fr: str
    target_columns: List[str]
    parameters: Dict[str, Any]
    stats: FeatureSelectionStats
    file_path: str
    created_at: str


class FeatureSelectionVersionStatsResponse(BaseModel):
    version_id: str
    available_columns: List[str]
    numeric_columns: List[str]
    total_records: int
    available_attractions: List[str]


class DeleteVersionResponse(BaseModel):
    message: str
    deleted_version_id: str