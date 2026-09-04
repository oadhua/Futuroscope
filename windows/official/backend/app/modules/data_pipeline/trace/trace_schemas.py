from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class DatasetOverviewStats(BaseModel):
    num_rows: int = Field(..., description="Nombre total de lignes")
    num_columns: int = Field(..., description="Nombre total de colonnes")
    missing_values: int = Field(..., description="Nombre total de valeurs manquantes")
    outliers_count: int = Field(..., description="Nombre total de valeurs aberrantes (Outliers)")
    zero_values: int = Field(..., description="Nombre total de valeurs égales à zéro")
    duplicate_rows: int = Field(..., description="Nombre de lignes dupliquées")


class ColumnDetailedStats(BaseModel):
    column_name: str = Field(..., description="Nom de la colonne analysée")
    min_val: Optional[float] = Field(None, description="Valeur minimale")
    max_val: Optional[float] = Field(None, description="Valeur maximale")
    mean_val: Optional[float] = Field(None, description="Valeur moyenne")
    unique_count: int = Field(..., description="Nombre de valeurs uniques")
    missing_values: int = Field(..., description="Nombre de valeurs manquantes dans cette colonne")
    outliers_count: int = Field(..., description="Nombre de valeurs aberrantes")
    negative_count: int = Field(..., description="Nombre de valeurs négatives")
    zero_count: int = Field(..., description="Nombre de valeurs égales à zéro")


class ChartDataPoint(BaseModel):
    timestamp: str = Field(..., description="Horodatage de la mesure")
    original_value: Optional[float] = Field(None, description="Valeur avant prétraitement")
    processed_value: Optional[float] = Field(None, description="Valeur après prétraitement")


class PreprocessingTraceabilityComparisonResponse(BaseModel):
    version_id: str = Field(..., description="Identifiant de la version sélectionnée")
    parent_version_id: str = Field(..., description="Identifiant de la version parente (source)")
    id_attraction: str = Field(..., description="Identifiant de l'attraction")

    # Bloc 1 : Métadonnées de l'opération
    operation: str = Field(..., description="Type d'opération effectuée (ex: Encoding, Scaling, Imputation)")
    column_name: str = Field(..., description="Colonne principale analysée")
    method: str = Field(..., description="Méthode appliquée (ex: one_hot, imputation_mean)")

    # Bloc 2 : Comparaison globale du Jeu de Données
    original_overview: DatasetOverviewStats = Field(..., description="Statistiques globales des données originales")
    processed_overview: DatasetOverviewStats = Field(..., description="Statistiques globales des données prétraitées")

    # Bloc 3 : Comparaison détaillée par colonne
    original_column_stats: Optional[ColumnDetailedStats] = Field(None, description="Statistiques détaillées de la colonne originale")
    processed_column_stats: Optional[ColumnDetailedStats] = Field(None, description="Statistiques détaillées de la colonne prétraitée")

    # Bloc 4 : Données de la série temporelle pour le graphique
    chart_data: List[ChartDataPoint] = Field(..., description="Points de données pour le graphique comparatif")