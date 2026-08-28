import traceback
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.modules.data_pipeline.single_column.single_column_schemas import (
    ReponseColonneDetail,
    ReponseMultiVersionDetail,
)
from app.modules.data_pipeline.single_column.single_column_services import (
    analyser_profil_colonne_seule,
    get_dataframe_from_version_local,
    analyser_multi_versions,
)
from app.modules.data_pipeline.utils import clean_nan_and_inf

router = APIRouter(prefix="/analysis", tags=["Data Pipeline - Analysis & Profiling"])


@router.get(
    "/colonne-detail",
    response_model=ReponseColonneDetail,
    summary="Analyse détaillée d'une seule colonne (Single Column Profiling)",
)
def get_column_detail_profiling(
    col_name: str = Query(..., description="Nom de la colonne à analyser"),
    id_attraction: Optional[str] = Query(
        "ALL", description="Code de l'attraction (ex: 'H03', 'H07', 'ALL')"
    ),
    version: Optional[str] = Query(
        "v0_raw", description="Identifiant de version (ex: 'v0_raw', 'v1_mean_H03')"
    ),
    db: Session = Depends(get_db),
):
    try:
        # Lấy DataFrame chuẩn xác theo Version và Attraction
        df = get_dataframe_from_version_local(
            db=db, version_id=version, id_attraction=id_attraction
        )

        if df.empty:
            raise HTTPException(
                status_code=404,
                detail="Aucune donnée trouvée pour la version et l'attraction spécifiées.",
            )

        resultat = analyser_profil_colonne_seule(
            df=df, col_name=col_name, id_attraction=id_attraction, version=version
        )
        if "erreur" in resultat:
            raise HTTPException(status_code=400, detail=resultat["erreur"])

        return clean_nan_and_inf(resultat)
    except HTTPException as he:
        raise he
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Erreur serveur: {str(e)}")
    
@router.get(
    "/multi-column-detail",
    response_model=ReponseMultiVersionDetail,
    summary="Analyse détaillée d'une colonne à travers plusieurs versions en un seul appel",
)
def get_multi_version_column_detail(
    col_name: str = Query(..., description="Nom de la colonne à analyser"),
    versions: str = Query(..., description="Liste des versions séparées par des virgules (ex: v0_raw,v1_test)"),
    id_attraction: Optional[str] = Query(
        "ALL", description="Code de l'attraction (ex: 'H03', 'H07', 'ALL')"
    ),
    db: Session = Depends(get_db),
):
    try:
        version_list = [v.strip() for v in versions.split(",") if v.strip()]
        if not version_list:
            raise HTTPException(status_code=400, detail="La liste des versions est vide.")

        # Gọi service phân tích hàng loạt phiên bản
        multi_data = analyser_multi_versions(
            db=db, col_name=col_name, versions=version_list, id_attraction=id_attraction
        )

        return {
            "nom_colonne": col_name,
            "id_attraction": id_attraction or "ALL",
            "versions_data": multi_data
        }
    except HTTPException as he:
        raise he
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Erreur serveur multi-version: {str(e)}")