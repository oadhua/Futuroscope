import traceback
import pandas as pd
from typing import Optional
from fastapi import APIRouter, HTTPException, Query
from sqlalchemy import text

from app.core.database import engine
from app.modules.data_pipeline.queries import get_dynamic_gathering_query
from app.modules.data_pipeline.single_column.single_column_services import (
    analyser_profil_colonne_seule,
)
from app.modules.data_pipeline.utils import clean_nan_and_inf

router = APIRouter(prefix="/analysis", tags=["Data Pipeline - Analysis & Profiling"])

@router.get(
    "/colonne-detail",
    summary="Analyse détaillée d'une seule colonne (Single Column Profiling)",
)
def get_column_detail_profiling(
    col_name: str = Query(...),
    id_attraction: Optional[str] = Query("ALL"),
):
    try:
        with engine.connect() as conn:
            query_base = get_dynamic_gathering_query(
                engine, id_attraction=id_attraction
            )
            df = pd.read_sql(text(query_base), con=conn)

        if df.empty:
            raise HTTPException(status_code=404, detail="Aucune donnée trouvée.")

        resultat = analyser_profil_colonne_seule(
            df=df, col_name=col_name, id_attraction=id_attraction
        )
        if "erreur" in resultat:
            raise HTTPException(status_code=400, detail=resultat["erreur"])

        return clean_nan_and_inf(resultat)
    except HTTPException as he:
        raise he
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Erreur serveur: {str(e)}")