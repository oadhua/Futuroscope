from fastapi import APIRouter, HTTPException, status
from typing import Optional

from app.modules.data_pipeline.encoding.encode_schemas import (
    EncodingRequest,
    EncodingVersionMetadataResponse,
    EncodingVersionStatsResponse,
)
from app.modules.data_pipeline.encoding.encode_services import (
    EncodingService,
)

router = APIRouter(prefix="/data-prep", tags=["Data Preparation - Encoding"])


@router.post(
    "/encoding",
    response_model=EncodingVersionMetadataResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Exécuter l'encodage des variables catégorielles (One-Hot, Label, Target)",
)
def process_encoding(req: EncodingRequest):
    try:
        metadata = EncodingService.execute_and_version(
            parent_version_id=req.parent_version_id,
            id_attraction=req.id_attraction,
            target_columns=req.target_columns,
            method=req.method.value,
            params={
                "drop_first": req.drop_first,
                "target_variable": req.target_variable,
            },
        )
        return metadata
    except FileNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur interne du système : {str(e)}",
        )


@router.get(
    "/encoding/versions/{version_id}/stats",
    response_model=EncodingVersionStatsResponse,
    summary="Obtenir les colonnes catégorielles disponibles et les statistiques de la version",
)
def get_encoding_version_stats(
    version_id: str,
    id_attraction: Optional[str] = "ALL",
):
    try:
        cols, total_records, attractions = EncodingService.get_version_stats(
            version_id=version_id,
            id_attraction=id_attraction,
        )
        return EncodingVersionStatsResponse(
            version_id=version_id,
            available_columns=cols,
            total_records=total_records,
            available_attractions=attractions,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur interne du système : {str(e)}",
        )