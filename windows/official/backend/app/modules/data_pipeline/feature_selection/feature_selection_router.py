from fastapi import APIRouter, HTTPException, Query, status
from typing import Optional

from app.modules.data_pipeline.feature_selection.feature_selection_schemas import (
    FeatureSelectionRequest,
    FeatureAnalyzeRequest,
    FeatureSelectionMetadataResponse,
    FeatureSelectionVersionStatsResponse,
    FeatureImportanceAnalysisResponse,
    FeatureItem,
    DeleteVersionResponse,
)
from app.modules.data_pipeline.feature_selection.feature_selection_services import (
    FeatureSelectionService,
)

router = APIRouter(prefix="/data-prep", tags=["Data Preparation - Feature Selection"])


@router.post(
    "/feature-selection/analyze",
    response_model=FeatureImportanceAnalysisResponse,
    summary="Analyser et classer l'importance des caractéristiques",
)
def analyze_features(req: FeatureAnalyzeRequest):
    """Calcule les scores d'importance pour toutes les colonnes éligibles."""
    try:
        df = FeatureSelectionService.load_version_dataframe(
            version_id=req.version_id, id_attraction=req.id_attraction or "ALL"
        )
        scores = FeatureSelectionService.analyze_feature_importance(
            df=df,
            target_column=req.target_column,
            method=req.method,
        )

        feature_items = [FeatureItem(**s) for s in scores]

        return FeatureImportanceAnalysisResponse(
            version_id=req.version_id,
            target_column=req.target_column,
            method=req.method,
            features=feature_items,
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


@router.get(
    "/feature-selection/versions/{version_id}/stats",
    response_model=FeatureSelectionVersionStatsResponse,
    summary="Obtenir les colonnes disponibles et les statistiques de la version",
)
def get_feature_selection_version_stats(
    version_id: str,
    id_attraction: Optional[str] = "ALL",
):
    """Retourne la liste complète des colonnes disponibles et des caractéristiques numériques."""
    try:
        cols, num_cols, total_records, attractions = (
            FeatureSelectionService.get_version_stats(
                version_id=version_id,
                id_attraction=id_attraction,
            )
        )
        return FeatureSelectionVersionStatsResponse(
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


@router.post(
    "/feature-selection/save",
    response_model=FeatureSelectionMetadataResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Sauvegarder une nouvelle version filtrée par les caractéristiques choisies",
)
def save_feature_selected_version(req: FeatureSelectionRequest):
    """Sauvegarde les colonnes sélectionnées dans une nouvelle table de données."""
    try:
        result = FeatureSelectionService.create_feature_selected_version(
            parent_version_id=req.parent_version_id,
            target_column=req.target_column,
            selected_features=req.selected_features,
            id_attraction=req.id_attraction or "ALL",
            method=req.method,
            custom_version_id=req.custom_version_id,
        )
        return FeatureSelectionMetadataResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur lors de la création de la version : {str(e)}",
        )


@router.delete(
    "/feature-selection/versions/{version_id}",
    response_model=DeleteVersionResponse,
    summary="Supprimer une version de données",
)
def delete_version(version_id: str):
    """Supprime une version spécifique de la base de données."""
    try:
        FeatureSelectionService.delete_version(version_id)
        return DeleteVersionResponse(
            message=f"La version '{version_id}' a été supprimée avec succès.",
            deleted_version_id=version_id,
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur lors de la suppression de la version : {str(e)}",
        )
