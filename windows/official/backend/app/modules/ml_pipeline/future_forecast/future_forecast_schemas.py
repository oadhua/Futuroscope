from datetime import datetime
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field
from enum import Enum


# ==========================================
# 1. SCHÉMAS DE BASE & FEATURES HORAIRES
# ==========================================

class OuvertModeEnum(str, Enum):
    PROFILE = "profile"
    IDEAL = "ideal"
    
class HourlyFeatureRow(BaseModel):
    datetime: str
    heure: int
    is_open: float
    type_frequentation: int
    interrompu: float
    ouvert: Optional[float] = None
    temperature: Optional[float] = None
    humidite: Optional[float] = None
    rayonnement_solaire: Optional[float] = None
    operation: float
    day_degree_cold: Optional[float] = None
    day_degree_hot: Optional[float] = None

    class Config:
        extra = "allow"


# ==========================================
# 2. SCHÉMAS POUR LA SIMULATION DE SCÉNARIO EN CHAÎNE
# ==========================================


class ForecastRequest(BaseModel):
    target_type: str = Field(
        default="visitor",
        description="Type de cible : 'visitor', 'electricity', ou 'thermal'",
    )
    id_attraction: str = Field(..., json_schema_extra={"example": "ATT_01"})
    start_date: datetime = Field(
        ..., json_schema_extra={"example": "2026-09-10T00:00:00"}
    )
    end_date: datetime = Field(
        ..., json_schema_extra={"example": "2026-09-10T23:00:00"}
    )

    type_frequentation_default: int = Field(
        default=1,
        ge=1,
        le=4,
        description="1: BF - Basse, 2: MF - Moyenne, 3: HF - Haute, 4: THF - Très Haute",
    )
    temp_offset: float = Field(
        default=0.0, description="Écart de température hypothétique (°C)"
    )
    operation_factor: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Facteur d'exploitation"
    )
    model_id: Optional[str] = Field(
        default=None, description="ID spécifique du modèle cible"
    )
    visitor_model_id: Optional[str] = Field(
        default=None, description="ID spécifique du modèle visitor pour le chaînage"
    )
    custom_hourly_features: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Lignes de features horaires personnalisées sous forme de dictionnaire flexible",
    )
    ouvert_mode: Union[OuvertModeEnum, str] = Field(
        default=OuvertModeEnum.IDEAL, 
        description="Chế độ điền ouvert: 'ideal' (gán 1.0 nếu có mở cửa) hoặc 'profile' (giữ giá trị trung bình)"
    )
    open_threshold: float = Field(
        default=0.1, 
        description="Ngưỡng profil_moyen tối thiểu để tính là trò chơi có mở cửa trong khung giờ đó"
    )

class HourlyPredictionItem(BaseModel):
    datetime: str
    heure: int
    semaine: Optional[int] = None
    type_frequentation: Optional[int] = None
    predicted_value: float
    ouvert: float
    operation: float
    temperature: Optional[float] = None
    visitor_count: Optional[int] = None


class MetricSummary(BaseModel):
    total: float
    mean: float
    median: float
    max: float
    min: float


class ForecastSummaryStats(BaseModel):
    unit: str
    predicted_value: MetricSummary
    visitor_count: Optional[MetricSummary] = None


class ForecastResponse(BaseModel):
    status: str
    target_type: str
    id_attraction: str
    target_model_used: str
    visitor_model_used: Optional[str] = None
    total_records: int
    summary: ForecastSummaryStats
    predictions: List[HourlyPredictionItem]


# ==========================================
# 3. SCHÉMAS POUR L'INFÉRENCE DIRECTE DE MODÈLE
# ==========================================


class FeatureMetadata(BaseModel):
    name: str
    label: str
    type: str = "number"
    default_value: Union[float, int, str] = 0


class FeatureImportanceItem(BaseModel):
    name: str
    importance: float


class ModelMetadataResponse(BaseModel):
    model_id: str
    name: str
    algorithm: str
    target_variable: str
    features: List[FeatureMetadata]
    feature_importances: Optional[List[FeatureImportanceItem]] = None


class ModelListItem(BaseModel):
    model_id: str
    name: str
    target: str
    algorithm: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None


class PredictRequest(BaseModel):
    model_id: str = Field(..., description="Identifiant du modèle entraîné à utiliser")
    features: Dict[str, Any] = Field(
        ..., description="Dictionnaire clé-valeur des variables d'entrée"
    )


class PredictResponse(BaseModel):
    status: str = "success"
    model_id: str
    predicted_value: float
    unit: str
    confidence_interval: Optional[List[float]] = None
    timestamp: str