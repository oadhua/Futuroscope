from fastapi import APIRouter, HTTPException, status
from typing import Optional

from app.modules.data_pipeline.timeseries_transformation.timeseries_transformation_schemas import (
    TimeSeriesTransformRequest,
    TimeSeriesTransformVersionMetadataResponse,
    TimeSeriesTransformVersionStatsResponse,
)
from app.modules.data_pipeline.timeseries_transformation.timeseries_transformation_services import (
    TimeSeriesTransformService,
)

router = APIRouter(prefix="/data-prep", tags=["Data Preparation - Time Series Transformation"])


@router.post(
    "/time-series-transform",
    response_model=TimeSeriesTransformVersionMetadataResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Exécuter la transformation de séries temporelles",
)
def process_time_series_transform(req: TimeSeriesTransformRequest):
    try:
        metadata = TimeSeriesTransformService.execute_and_version(
            parent_version_id=req.parent_version_id,
            id_attraction=req.id_attraction,
            target_columns=req.target_columns,
            method=req.method.value,
            seasonal_period=req.seasonal_period or 24,
            drop_na=req.drop_na if req.drop_na is not None else True,
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
    "/time-series-transform/versions/{version_id}/stats",
    response_model=TimeSeriesTransformVersionStatsResponse,
    summary="Obtenir les statistiques des colonnes numériques pour les séries temporelles",
)
def get_time_series_version_stats(
    version_id: str,
    id_attraction: Optional[str] = "ALL",
):
    try:
        cols, num_cols, total_records, attractions = (
            TimeSeriesTransformService.get_version_stats(
                version_id=version_id,
                id_attraction=id_attraction,
            )
        )
        return TimeSeriesTransformVersionStatsResponse(
            version_id=version_id,
            available_columns=cols,
            numeric_columns=num_cols,
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