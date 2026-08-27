from datetime import date, datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field


# ==============================================================================
# 1. SCHÉMAS POUR L'UPLOAD ET LE PIPELINE DE DONNÉES
# ==============================================================================

class UploadResponse(BaseModel):
    message: str
    filename: str
    total_rows: int
    columns: List[str]


class UploadSixFilesDetails(BaseModel):
    dim_temps: int = Field(..., example=8760)
    fact_attraction: int = Field(..., example=17520)
    fact_elec: int = Field(..., example=35040)
    fact_ec: int = Field(..., example=17520)


class UploadSixFilesResponse(BaseModel):
    status: str = Field(..., example="success")
    message: str = Field(
        ...,
        example="Téléchargement, suppression des anciennes données et sauvegarde des 6 fichiers réussies!",
    )
    details: UploadSixFilesDetails


# ==============================================================================
# 2. SCHÉMAS POUR LA CONSULTATION VUE BDD & ML FEATURES
# ==============================================================================

class VisitorDataResponse(BaseModel):
    datetime: datetime
    id_attraction: str
    visitor_count: Optional[float] = None
    ouvert: Optional[int] = None
    interrompu: Optional[int] = None
    operation: Optional[int] = None
    temperature: Optional[float] = None
    humidite: Optional[float] = None
    rayonnement_solaire: Optional[float] = None
    is_weekend: Optional[int] = None
    is_open: Optional[int] = None

    model_config = ConfigDict(from_attributes=True, extra="allow")


class PaginatedResponse(BaseModel):
    total: int
    page: int
    size: int
    items: List[VisitorDataResponse]


class MLFeaturesResponse(BaseModel):
    status: str = Field(..., example="success")
    id_attraction: str = Field(..., example="H03")
    total_rows: int = Field(..., example=8760)
    columns: List[str]
    data: List[Dict[str, Any]]


# ==============================================================================
# 3. SCHÉMA POUR LA GÉNÉRATION DE TIMEFRAME KHUNG
# ==============================================================================

class GenerateTimeframeRequest(BaseModel):
    start_date: str = Field(..., example="2026-01-01")
    end_date: str = Field(..., example="2026-12-31")
    attractions: List[str] = Field(default=["H03", "H07"], example=["H03", "H07"])


# ==============================================================================
# 4. SCHÉMAS POUR L'ANALYSE GLOBALE (GLOBAL PROFILING DASHBOARD)
# ==============================================================================

class UtilisationMemoire(BaseModel):
    nom_colonne: str = Field(..., example="visitor_count")
    octets: int = Field(..., example=1024)
    pourcentage: float = Field(..., example=12.5)


class StatistiqueColonneComparaison(BaseModel):
    nom_colonne: str = Field(..., example="visitor_count")
    type_donnees: str = Field(..., example="float64")
    valeurs_manquantes: int = Field(..., example=15)
    valeurs_aberrantes: int = Field(..., example=2)
    moyenne: Optional[float] = None
    ecart_type: Optional[float] = None
    val_min: Optional[float] = None
    mediane: Optional[float] = None
    val_max: Optional[float] = None


class ReponseAnalyseGlobale(BaseModel):
    statut: str = Field(..., example="succes")
    nom_table: str = Field(..., example="fact_attraction_hourly")
    id_attraction_filtre: Optional[str] = Field(None, example="H03")
    liste_attractions: List[str] = Field(default=[], example=["H03", "H07"])
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


# ==============================================================================
# 5. SCHÉMAS POUR L'ANALYSE DÉTAILLÉE D'UNE COLONNE (SINGLE COLUMN PROFILING)
# ==============================================================================

class StatistiquesColonne(BaseModel):
    type_donnees: str = Field(..., example="float64")
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
    min: float
    q25: float
    mediane: float
    q75: float
    max: float
    borne_inf: float
    borne_sup: float
    outliers_samples: List[float] = Field(default=[])


class BinHistogramme(BaseModel):
    bin_range: str = Field(..., example="0.0 - 5.0")
    count: int = Field(..., example=120)


class PointHeure(BaseModel):
    heure: int
    valeur_moyenne: float


class PointJour(BaseModel):
    date: str
    valeur_moyenne: float


class PointMois(BaseModel):
    mois: int
    valeur_moyenne: float


class PointAnnee(BaseModel):
    annee: int
    valeur_moyenne: float


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
    nom_colonne: str = Field(..., example="temperature")
    id_attraction: str = Field(..., example="H03")
    statistiques: StatistiquesColonne
    graphiques: GraphiquesColonne