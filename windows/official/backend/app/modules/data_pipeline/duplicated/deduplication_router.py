from fastapi import APIRouter, HTTPException, status
from typing import Optional

from app.modules.data_pipeline.duplicated.deduplication_schemas import (
    DeduplicationRequest,
    DeduplicationVersionMetadataResponse,
    DeduplicationVersionStatsResponse,
)
from app.modules.data_pipeline.duplicated.deduplication_services import (
    DeduplicationService,
)

router = APIRouter(prefix="/data-prep", tags=["Data Preparation - Deduplication"])


@router.post(
    "/deduplication",
    response_model=DeduplicationVersionMetadataResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Exécuter la suppression des doublons",
)
def process_deduplication(req: DeduplicationRequest):
    try:
        metadata = DeduplicationService.execute_and_version(
            parent_version_id=req.parent_version_id,
            id_attraction=req.id_attraction,
            subset_columns=req.subset_columns,
            keep=req.keep.value,
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
    "/deduplication/versions/{version_id}/stats",
    response_model=DeduplicationVersionStatsResponse,
    summary="Obtenir les statistiques sur les doublons et les colonnes d'une version",
)
def get_deduplication_version_stats(
    version_id: str,
    id_attraction: Optional[str] = "ALL",
):
    try:
        cols, total_records, total_duplicates, attractions = (
            DeduplicationService.get_version_stats(
                version_id=version_id,
                id_attraction=id_attraction,
            )
        )
        return DeduplicationVersionStatsResponse(
            version_id=version_id,
            available_columns=cols,
            total_records=total_records,
            total_duplicates=total_duplicates,
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