from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field


class UtilisationMemoire(BaseModel):
    nom_colonne: str = Field(..., example="visitor_count")
    octets: int = Field(..., example=1024)
    pourcentage: float = Field(..., example=12.5)


class StatistiqueColonneComparaison(BaseModel):
    nom_colonne: str = Field(..., example="visitor_count")
    type_donnees: str = Field(..., example="int64")
    valeurs_manquantes: int = Field(..., example=15)
    valeurs_aberrantes: int = Field(..., example=2)
    moyenne: Optional[Union[int, float]] = None
    ecart_type: Optional[float] = None
    val_min: Optional[Union[int, float]] = None
    mediane: Optional[Union[int, float]] = None
    val_max: Optional[Union[int, float]] = None

class RepartitionAttraction(BaseModel):
    id_attraction: str = Field(..., example="H07")
    total_lignes: int = Field(..., example=5000)
    pourcentage: float = Field(..., example=50.0)
    
class RepartitionVersion(BaseModel):
    name: str = Field(..., example="v0_raw")
    total_lignes: int = Field(..., example=1200)
    pourcentage: float = Field(..., example=30.0)

class ReponseAnalyseGlobale(BaseModel):
    statut: str = Field(..., example="succes")
    version_id: str = Field(..., example="v1_mean_H03")
    id_attraction_filtre: Optional[str] = Field("ALL", example="H03")
    liste_attractions: List[str] = Field(default=[], example=["H03", "H07"])
    repartition_par_attraction: List[RepartitionAttraction] = Field(default=[])
    repartition_par_version: Optional[List[RepartitionVersion]] = Field(
        default=None, 
        description="Phân bổ số lượng dòng theo từng phiên bản khi chọn một id_attraction cụ thể"
    )
    total_lignes: int = Field(..., example=1000)
    total_colonnes: int = Field(..., example=28)
    nb_valeurs_manquantes: int = Field(..., example=15)
    pourcentage_manquants: float = Field(..., example=1.5)
    nb_outliers: int = Field(..., example=2)
    pourcentage_outliers: float = Field(..., example=0.8)
    nb_valeurs_negatives: int = Field(..., example=0)
    pourcentage_negatifs: float = Field(..., example=0.0)
    nb_doublons: int = Field(..., example=0)
    pourcentage_dupliques: float = Field(..., example=0.0)
    date_debut: str = Field(..., example="2024-02-10 00:00:00")
    date_fin: str = Field(..., example="2024-02-10 15:00:00")
    utilisation_memoire: List[UtilisationMemoire]
    comparaison_colonnes: List[StatistiqueColonneComparaison]
    apercu_donnees: List[Dict[str, Any]]


class DeltaColonne(BaseModel):
    nom_colonne: str = Field(..., example="visitor_count")
    type_v1: str = Field(..., example="float64")
    type_v2: str = Field(..., example="float64")
    diff_valeurs_manquantes: int = Field(
        ..., description="Différence de valeurs manquantes (V2 - V1)."
    )
    diff_valeurs_aberrantes: int = Field(
        ..., description="Différence de valeurs aberrantes (V2 - V1)."
    )
    diff_moyenne: Optional[float] = None
    diff_mediane: Optional[float] = None


class ReponseComparaisonVersions(BaseModel):
    statut: str = Field("succes", example="succes")
    v1_version_id: str = Field(..., example="v0_raw")
    v2_version_id: str = Field(..., example="v1_mean_H03")
    id_attraction: str = Field(..., example="H03")
    delta_global: Dict[str, Any] = Field(
        ..., description="Ecart global (lignes, valeurs manquantes, outliers)"
    )
    delta_colonnes: List[DeltaColonne]