from typing import List, Optional, Union, Dict, Any
from pydantic import BaseModel, Field

# ==============================================================================
# 5. SCHÉMAS POUR L'ANALYSE DÉTAILLÉE D'UNE COLONNE (SINGLE COLUMN PROFILING)
# ==============================================================================

class StatistiquesColonne(BaseModel):
    type_donnees: str = Field(..., example="int64")
    total_lignes: int = Field(..., example=8760)
    valeurs_manquantes: int = Field(..., example=12)
    pourcentage_manquants: float = Field(..., example=0.14)
    valeurs_uniques: int = Field(..., example=320)
    valeurs_zero: int = Field(..., example=5)
    pourcentage_zeros: float = Field(..., example=0.06)
    valeurs_negatives: int = Field(..., example=0)
    pourcentage_negatifs: float = Field(..., example=0.0)
    valeurs_aberrantes: int = Field(..., example=3)
    pourcentage_outliers: float = Field(..., example=0.03)


class DonneesBoxPlot(BaseModel):
    min: Optional[Union[int, float]] = None
    q25: Optional[Union[int, float]] = None
    mediane: Optional[Union[int, float]] = None
    q75: Optional[Union[int, float]] = None
    max: Optional[Union[int, float]] = None
    borne_inf: Optional[Union[int, float]] = None
    borne_sup: Optional[Union[int, float]] = None
    outliers_samples: List[Union[int, float]] = Field(default=[])


class BinHistogramme(BaseModel):
    bin_range: str = Field(..., example="0 - 5")
    count: int = Field(..., example=120)


class PointHeure(BaseModel):
    heure: str = Field(..., example="2024-02-10 08:00")  # Chuyển int -> str cho khớp định dạng YYYY-MM-DD HH:MM
    valeur_moyenne: Optional[Union[int, float]] = None   # Cho phép nhận int lẫn float


class PointJour(BaseModel):
    date: str = Field(..., example="2024-02-10")
    valeur_moyenne: Optional[Union[int, float]] = None


class PointMois(BaseModel):
    mois: str = Field(..., example="2024-02")            # Chuyển int -> str cho khớp định dạng YYYY-MM
    valeur_moyenne: Optional[Union[int, float]] = None


class PointAnnee(BaseModel):
    annee: str = Field(..., example="2024")              # Chuyển int -> str cho khớp định dạng YYYY
    valeur_moyenne: Optional[Union[int, float]] = None


class CourbesTemporelles(BaseModel):
    par_heure: List[PointHeure] = Field(default=[])
    par_jour: List[PointJour] = Field(default=[])
    par_mois: List[PointMois] = Field(default=[])
    par_annee: List[PointAnnee] = Field(default=[])


class GraphiquesColonne(BaseModel):
    box_plot: Optional[DonneesBoxPlot] = None
    histogramme: Optional[List[BinHistogramme]] = None
    courbes_temporelles: CourbesTemporelles


class ReponseColonneDetail(BaseModel):
    nom_colonne: str = Field(..., example="visitor_count")
    id_attraction: str = Field(..., example="H03")
    selected_version: str = Field(default="v0_raw", example="v1_visitor_domain_rules_H03")  # Bổ sung
    statistiques: StatistiquesColonne
    graphiques: GraphiquesColonne
    
class ReponseMultiVersionDetail(BaseModel):
    nom_colonne: str = Field(..., example="visitor_count")
    id_attraction: str = Field(..., example="H03")
    # Dictionary chứa kết quả chi tiết ứng với từng version_id (VD: {"v0_raw": {...}, "v1_mean_H03": {...}})
    versions_data: Dict[str, Union[ReponseColonneDetail, Dict[str, Any]]] = Field(default={})