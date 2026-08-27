import io
import math
import traceback
from typing import List, Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from sqlalchemy import text
from sqlalchemy.engine import Engine

from app.core.database import engine
from windows.web.schemas import (
    GenerateTimeframeRequest,
    ReponseAnalyseGlobale,
)
from windows.web.services import (
    analyser_profil_colonne_seule,
    effectuer_collecte_analyse_globale,
    generate_all_fact_shells_in_db,
    process_and_store_dynamic_data,
    process_and_store_static_data,
)

# Khai báo router
router = APIRouter(tags=["Data Pipeline"])


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================


def clean_nan_and_inf(obj):
    """Xử lý các giá trị NaN và Infinity để tránh lỗi JSON serialization."""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    elif isinstance(obj, dict):
        return {k: clean_nan_and_inf(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_nan_and_inf(v) for v in obj]
    elif isinstance(obj, np.generic):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return obj.item()
    return obj


def format_alias(attr_id: str, metric_name: str) -> str:
    """Tạo tên cột dạng H03_puissance_kw."""
    if not attr_id:
        return metric_name
    attr_str = str(attr_id).strip()
    metric_str = str(metric_name).strip()

    if metric_str.lower().startswith(attr_str.lower()):
        return metric_str
    return f"{attr_str}_{metric_str}"


def get_dynamic_gathering_query(
    engine: Engine, id_attraction: Optional[str] = None
) -> str:
    """Tự động tạo câu SQL Pivot ghép id_attraction vào đầu tên cột năng lượng.

    Bao gồm sẵn cả WHERE và ORDER BY.
    """
    has_filter = (
        id_attraction is not None
        and str(id_attraction).strip() != ""
        and str(id_attraction).upper() != "ALL"
    )

    clean_id = str(id_attraction).strip() if has_filter else None

    where_clause = (
        f"WHERE id_attraction = '{clean_id}' AND metric_name IS NOT NULL"
        if has_filter
        else "WHERE metric_name IS NOT NULL"
    )

    with engine.connect() as conn:
        # 1. Truy vấn metric Điện
        try:
            df_elec = pd.read_sql(
                text(
                    f"SELECT DISTINCT id_attraction, metric_name FROM"
                    f" fact_elec_hourly {where_clause}"
                ),
                conn,
            )
            elec_pairs = df_elec.dropna().to_dict(orient="records")
        except Exception:
            elec_pairs = []

        # 2. Truy vấn metric Nước
        try:
            df_ec = pd.read_sql(
                text(
                    f"SELECT DISTINCT id_attraction, metric_name FROM"
                    f" fact_ec_hourly {where_clause}"
                ),
                conn,
            )
            ec_pairs = df_ec.dropna().to_dict(orient="records")
        except Exception:
            ec_pairs = []

    # 3. Build Pivot Điện
    elec_pivot_list = []
    elec_select_cols = []
    for item in elec_pairs:
        attr = item["id_attraction"]
        m = item["metric_name"]
        alias = format_alias(attr, m)
        elec_pivot_list.append(
            f"MAX(CASE WHEN id_attraction = '{attr}' AND metric_name = '{m}'"
            f' THEN value END) AS "{alias}"'
        )
        elec_select_cols.append(f'e."{alias}"')

    # 4. Build Pivot Nước
    ec_pivot_list = []
    ec_select_cols = []
    for item in ec_pairs:
        attr = item["id_attraction"]
        m = item["metric_name"]
        alias = format_alias(attr, m)
        ec_pivot_list.append(
            f"MAX(CASE WHEN id_attraction = '{attr}' AND metric_name = '{m}'"
            f' THEN value END) AS "{alias}"'
        )
        ec_select_cols.append(f'c."{alias}"')

    elec_pivot_cols = (
        ",\n            ".join(elec_pivot_list)
        if elec_pivot_list
        else "NULL AS _no_elec"
    )
    ec_pivot_cols = (
        ",\n            ".join(ec_pivot_list) if ec_pivot_list else "NULL AS _no_ec"
    )

    # Ghép danh sách cột năng lượng
    energy_select_cols = elec_select_cols + ec_select_cols
    energy_select_str = ""
    if energy_select_cols:
        energy_select_str = ",\n        ".join(energy_select_cols) + ",\n        "

    cte_where = f"WHERE id_attraction = '{clean_id}'" if has_filter else ""
    main_where = f"WHERE f.id_attraction = '{clean_id}'" if has_filter else ""

    query = f"""
    WITH elec_pivoted AS (
        SELECT 
            temps_id,
            id_attraction,
            {elec_pivot_cols}
        FROM fact_elec_hourly
        {cte_where}
        GROUP BY temps_id, id_attraction
    ),
    ec_pivoted AS (
        SELECT 
            temps_id,
            id_attraction,
            {ec_pivot_cols}
        FROM fact_ec_hourly
        {cte_where}
        GROUP BY temps_id, id_attraction
    )
    SELECT 
        f.datetime,
        f.id_attraction,
        f.visitor_count,
        f.ouvert,
        f.interrompu,
        f.operation,
        
        {energy_select_str}
        w.temperature,
        w.humidite,
        w.rayonnement_solaire,
        w.day_degree_cold,
        w.day_degree_hot,
        w.temp_max,
        w.temp_min,
        w.temp_moy,
        w.humidite_max,
        w.humidite_min,
        w.humidite_moy,

        h.jf,
        t.is_weekend,
        h.is_open,
        h.h_ouv,
        h.h_ferm,
        h.frequentation,
        h.type_frequentation,
        t.heure,
        t.jour,
        t.mois,
        t.annee
    FROM fact_attraction_hourly f
    LEFT JOIN dim_temps t ON f.temps_id = t.temps_id
    LEFT JOIN dim_weather w ON f.temps_id = w.temps_id
    LEFT JOIN dim_horaire h ON CAST(f.datetime AS DATE) = h.date
    LEFT JOIN elec_pivoted e ON f.temps_id = e.temps_id AND f.id_attraction = e.id_attraction
    LEFT JOIN ec_pivoted c ON f.temps_id = c.temps_id AND f.id_attraction = c.id_attraction
    {main_where}
    ORDER BY f.datetime ASC
    """
    return query


# ==============================================================================
# API ENDPOINTS
# ==============================================================================


@router.post("/upload-static-files")
async def upload_static_files(
    surface_file: UploadFile = File(...),
    cadence_file: UploadFile = File(...),
):
    """API tiếp nhận 2 file dữ liệu tĩnh (surface.xlsx, cadence.xlsx)
    và nạp mới vào dim_attraction & dim_cadence.
    """
    try:
        surface_bytes = await surface_file.read()
        cadence_bytes = await cadence_file.read()

        summary_result = process_and_store_static_data(
            surface_bytes=surface_bytes,
            cadence_bytes=cadence_bytes,
            engine=engine,
        )

        return {
            "status": "success",
            "message": "Téléchargement et sauvegarde des données statiques (surface & cadence) réussis!",
            "details": summary_result,
        }
    except Exception as e:
        print("\n=== ERREUR TRAITEMENT FICHIERS STATIQUES ===")
        traceback.print_exc()
        print("============================================")
        raise HTTPException(
            status_code=500,
            detail=f"Erreur de traitement des fichiers statiques: {str(e)}",
        )


@router.post("/upload-6-files")
async def upload_six_files(
    visitor_file: UploadFile = File(...),
    etat_file: UploadFile = File(...),
    weather_file: UploadFile = File(...),
    horaire_file: UploadFile = File(...),
    elec_file: UploadFile = File(...),
    ec_file: UploadFile = File(...),
):
    """API tiếp nhận 6 file dữ liệu động gốc (CSV/Excel), tự động TRUNCATE dữ liệu động cũ
    và nạp mới vào Star Schema.
    """
    try:
        v_bytes = await visitor_file.read()
        e_bytes = await etat_file.read()
        w_bytes = await weather_file.read()
        h_bytes = await horaire_file.read()
        elec_bytes = await elec_file.read()
        ec_bytes = await ec_file.read()

        summary_result = process_and_store_dynamic_data(
            visitor_bytes=v_bytes,
            etat_bytes=e_bytes,
            weather_bytes=w_bytes,
            horaire_bytes=h_bytes,
            elec_bytes=elec_bytes,
            ec_bytes=ec_bytes,
            engine=engine,
        )

        return {
            "status": "success",
            "message": "Téléchargement, suppression des anciennes données dynamiques et sauvegarde des 6 fichiers réussis!",
            "details": summary_result,
        }
    except Exception as e:
        print("\n=== ERREUR TRAITEMENT 6 FICHIERS ===")
        traceback.print_exc()
        print("====================================")
        raise HTTPException(
            status_code=500,
            detail=f"Erreur de traitement des fichiers: {str(e)}",
        )


@router.post("/upload")
async def upload_dataset(file: UploadFile = File(...)):
    """API Upload file CSV hoặc Excel dữ liệu tổng hợp vào bảng raw."""
    if not (file.filename.endswith(".csv") or file.filename.endswith(".xlsx")):
        raise HTTPException(
            status_code=400,
            detail="Seuls les formats de fichiers .csv ou .xlsx sont pris en charge",
        )

    contents = await file.read()

    try:
        if file.filename.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(contents))
        else:
            df = pd.read_excel(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Erreur de lecture du fichier: {str(e)}"
        )

    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])

    try:
        with engine.begin() as conn:
            df.to_sql("fact_visitor_raw", con=conn, if_exists="replace", index=False)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur de sauvegarde dans la base de données: {str(e)}",
        )

    return {
        "message": "Téléchargement du fichier réussi!",
        "filename": file.filename,
        "total_rows": len(df),
        "columns": list(df.columns),
    }


@router.get("/data")
def get_table_data(
    page: int = Query(1, ge=1),
    size: int = Query(50, ge=1, le=500),
    id_attraction: Optional[str] = None,
):
    """Lấy dữ liệu phân trang kết hợp giữa bảng Fact và các bảng Dimension."""
    offset = (page - 1) * size

    query = text("""
        SELECT 
            f.datetime, 
            f.id_attraction, 
            f.visitor_count, 
            f.ouvert, 
            f.interrompu, 
            f.operation,
            w.temperature, 
            w.humidite, 
            w.rayonnement_solaire,
            t.is_weekend, 
            h.is_open
        FROM fact_attraction_hourly f
        LEFT JOIN dim_weather w ON f.temps_id = w.temps_id
        LEFT JOIN dim_temps t ON f.temps_id = t.temps_id
        LEFT JOIN dim_horaire h ON f.datetime = h.datetime
        WHERE (:id_attraction IS NULL OR f.id_attraction = :id_attraction)
        ORDER BY f.datetime ASC
        LIMIT :limit OFFSET :offset
    """)

    count_query = text("""
        SELECT COUNT(*) 
        FROM fact_attraction_hourly f
        WHERE (:id_attraction IS NULL OR f.id_attraction = :id_attraction)
    """)

    params = {
        "id_attraction": id_attraction if id_attraction else None,
        "limit": size,
        "offset": offset,
    }

    try:
        with engine.connect() as conn:
            total_rows = conn.execute(count_query, params).scalar()
            df_page = pd.read_sql(query, con=conn, params=params)

        df_page = df_page.astype(object).where(pd.notnull(df_page), None)

        return {
            "total": int(total_rows or 0),
            "page": page,
            "size": size,
            "items": df_page.to_dict(orient="records"),
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Erreur de requête des données: {str(e)}"
        )


@router.post("/generate-fact-shells")
def api_generate_fact_shells(req: GenerateTimeframeRequest):
    """API khởi tạo các dòng Fact khung trống cho khoảng thời gian tùy chọn."""
    try:
        inserted_count = generate_all_fact_shells_in_db(
            start_date=req.start_date,
            end_date=req.end_date,
            engine=engine,
            target_attractions=req.attractions,
        )
        return {
            "status": "success",
            "message": f"Initialisation réussie de la ligne de faits vide (NULL): {inserted_count} lignes dans PostgreSQL.",
            "inserted_rows": inserted_count,
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la génération des Time Frame: {str(e)}",
        )


@router.get(
    "/analyse-globale",
    response_model=ReponseAnalyseGlobale,
    summary="Obtenir le tableau de bord d'analyse globale filtré par attraction",
)
def obtenir_analyse_globale(
    nom_table: str = Query(
        "fact_attraction_hourly", description="Nom de la table principale"
    ),
    id_attraction: Optional[str] = Query(
        None, description="Filtrer par id_attraction (ex: H03, H07)"
    ),
    nb_lignes_apercu: Optional[int] = Query(
        10,
        description="Nombre de lignes à retourner pour l'aperçu (ex: 10, 50, 100)",
    ),
):
    try:
        with engine.connect() as conn:
            liste_attractions = []
            try:
                df_attractions = pd.read_sql(
                    text(
                        "SELECT DISTINCT id_attraction FROM fact_attraction_hourly WHERE id_attraction IS NOT NULL ORDER BY id_attraction"
                    ),
                    con=conn,
                )
                liste_attractions = df_attractions["id_attraction"].tolist()
            except Exception:
                pass

            if nom_table == "fact_attraction_hourly":
                query_base = get_dynamic_gathering_query(
                    engine, id_attraction=id_attraction
                )
                df = pd.read_sql(text(query_base), con=conn)
            else:
                query_simple = text(
                    'SELECT * FROM "' + nom_table.replace('"', '""') + '"'
                )
                df = pd.read_sql(query_simple, con=conn)
                if "datetime" in df.columns:
                    df = df.sort_values(by="datetime", ascending=True)

        if df.empty:
            raise HTTPException(
                status_code=404,
                detail=f"Aucune donnée trouvée pour l'attraction {id_attraction}"
                if id_attraction
                else "Aucune donnée trouvée.",
            )

        if "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"])

        resultats = effectuer_collecte_analyse_globale(df, nom_table, nb_lignes_apercu)

        response_payload = {
            "statut": "succes",
            "nom_table": nom_table,
            "id_attraction_filtre": id_attraction,
            "liste_attractions": liste_attractions,
            **resultats,
        }
        return clean_nan_and_inf(response_payload)

    except HTTPException as he:
        raise he
    except Exception as e:
        print("\n=== ERREUR DETECTEE DANS ANALYSE GLOBALE FILTRÉE ===")
        traceback.print_exc()
        print("====================================================\n")
        raise HTTPException(status_code=500, detail=f"Erreur serveur : {str(e)}")


@router.get(
    "/colonne-detail",
    summary="Analyse détaillée d'une seule colonne (Single Column Profiling)",
)
def get_column_detail_profiling(
    col_name: str = Query(..., description="Nom de la colonne à analyser"),
    id_attraction: Optional[str] = Query(
        "ALL", description="Mã attraction (H03, H07... ou ALL)"
    ),
):
    try:
        with engine.connect() as conn:
            query_base = get_dynamic_gathering_query(
                engine, id_attraction=id_attraction
            )
            df = pd.read_sql(text(query_base), con=conn)

        if df.empty:
            raise HTTPException(
                status_code=404,
                detail=f"Aucune donnée trouvée pour l'attraction {id_attraction}",
            )

        resultat = analyser_profil_colonne_seule(
            df=df, col_name=col_name, id_attraction=id_attraction
        )

        if "erreur" in resultat:
            raise HTTPException(status_code=400, detail=resultat["erreur"])

        return clean_nan_and_inf(resultat)

    except HTTPException as he:
        raise he
    except Exception as e:
        print("\n=== ERREUR DETECTEE DANS COLONNE DETAIL PROFILING ===")
        traceback.print_exc()
        print("====================================================\n")
        raise HTTPException(status_code=500, detail=f"Erreur serveur : {str(e)}")
