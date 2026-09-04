from fastapi import APIRouter, HTTPException, status
from typing import Optional

from app.modules.data_pipeline.scaling.scaling_schemas import (
    ScalingRequest,
    ScalingVersionMetadataResponse,
    ScalingVersionStatsResponse,
)
from app.modules.data_pipeline.scaling.scaling_services import (
    ScalingService,
)

router = APIRouter(prefix="/data-prep", tags=["Data Preparation - Scaling"])


@router.post(
    "/scaling",
    response_model=ScalingVersionMetadataResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Exécuter la mise à l'échelle des données (Standard, Min-Max, Robuste)",
)
def process_scaling(req: ScalingRequest):
    try:
        metadata = ScalingService.execute_and_version(
            parent_version_id=req.parent_version_id,
            id_attraction=req.id_attraction,
            target_columns=req.target_columns,
            method=req.method.value,
            params={"feature_range": req.feature_range},
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
    "/scaling/versions/{version_id}/stats",
    response_model=ScalingVersionStatsResponse,
    summary="Obtenir les colonnes disponibles et les statistiques de la version",
)
def get_scaling_version_stats(
    version_id: str,
    id_attraction: Optional[str] = "ALL",
):
    try:
        cols, total_records, attractions = ScalingService.get_version_stats(
            version_id=version_id,
            id_attraction=id_attraction,
        )
        return ScalingVersionStatsResponse(
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