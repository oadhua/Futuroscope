from typing import List
from pydantic import BaseModel, Field

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
# 3. SCHÉMA POUR LA GÉNÉRATION DE TIMEFRAME KHUNG
# ==============================================================================

class GenerateTimeframeRequest(BaseModel):
    start_date: str = Field(..., example="2026-01-01")
    end_date: str = Field(..., example="2026-12-31")
    attractions: List[str] = Field(default=["H03", "H07"], example=["H03", "H07"])