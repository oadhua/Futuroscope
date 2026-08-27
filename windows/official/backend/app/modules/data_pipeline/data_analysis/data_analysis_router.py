import traceback
from typing import Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.modules.data_pipeline.data_analysis.data_analysis_schemas import (
    ReponseAnalyseGlobale,
    ReponseComparaisonVersions,
)
from app.modules.data_pipeline.data_analysis.data_analysis_services import (
    compare_versions_service,
    generate_profilage_service,
    get_data_prep_versions_service,
)
from app.modules.data_pipeline.utils import clean_nan_and_inf

router = APIRouter(prefix="/analysis", tags=["Data Pipeline - Analysis & Profiling"])


@router.get(
    "/versions-schema",
    response_model=Dict[str, List[str]],
    summary="Récupérer toutes les versions disponibles dans le schéma data_prep",
)
def obtenir_versions_schema(db: Session = Depends(get_db)):
    try:
        versions = get_data_prep_versions_service(db)
        return {"versions": versions}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Erreur serveur: {str(e)}")


@router.get(
    "/analyse-globale",
    response_model=ReponseAnalyseGlobale,
    summary="Obtenir l'analyse globale filtrée par version et par attraction",
)
def obtenir_analyse_globale(
    version_id: str = Query(
        "v0_raw", description="Identifiant de version (ex: 'v0_raw', 'v1_mean_H03')"
    ),
    id_attraction: Optional[str] = Query(
        "ALL", description="Code de l'attraction (ex: 'H03', 'H07', 'ALL')"
    ),
    nb_lignes_apercu: Optional[int] = Query(
        10, description="Nombre de lignes d'aperçu à retourner"
    ),
    db: Session = Depends(get_db),
):
    try:
        resultats = generate_profilage_service(
            db=db,
            version_id=version_id,
            id_attraction=id_attraction,
            nb_lignes_apercu=nb_lignes_apercu,
        )
        return clean_nan_and_inf(resultats)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Erreur serveur: {str(e)}")


@router.get(
    "/compare-versions",
    response_model=ReponseComparaisonVersions,
    summary="Comparer l'évolution des métriques entre deux versions de données selon l'attraction",
)
def comparer_versions(
    v1_version_id: str = Query(
        ..., description="Version initiale / source (ex: v0_raw ou v1_mean_H03)"
    ),
    v2_version_id: str = Query(
        ..., description="Version cible à comparer (ex: v2_knn_H03)"
    ),
    id_attraction: Optional[str] = Query(
        "ALL", description="Filtre par identifiant d'attraction"
    ),
    db: Session = Depends(get_db),
):
    try:
        res = compare_versions_service(
            db=db,
            v1=v1_version_id,
            v2=v2_version_id,
            id_attraction=id_attraction,
        )
        return clean_nan_and_inf(res)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Erreur serveur: {str(e)}")