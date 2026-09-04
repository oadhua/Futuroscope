from typing import Optional
from fastapi import APIRouter, HTTPException, Query, status

from app.modules.data_pipeline.trace.trace_schemas import (
    PreprocessingTraceabilityComparisonResponse,
)
from app.modules.data_pipeline.trace.trace_services import (
    PreprocessingTraceabilityService,
)

router = APIRouter(
    prefix="/data-prep", tags=["Data Preparation - Traceability"]
)


@router.get(
    "/compare/{version_id}",
    response_model=PreprocessingTraceabilityComparisonResponse,
    summary="Obtenir l'analyse comparative détaillée entre les données originales et prétraitées",
)
def compare_version_data(
    version_id: str,
    id_attraction: Optional[str] = Query(
        "ALL", description="Identifiant de l'attraction (ex: 'H03', 'H07' ou 'ALL')"
    ),
    column_name: Optional[str] = Query(
        None, description="Nom de la colonne spécifique à comparer (ex: 'Gas_01')"
    ),
):
    try:
        comparison_data = (
            PreprocessingTraceabilityService.get_comparison_traceability(
                version_id=version_id,
                id_attraction=id_attraction,
                target_column=column_name,
            )
        )
        return comparison_data
    except FileNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur interne du système : {str(e)}",
        )