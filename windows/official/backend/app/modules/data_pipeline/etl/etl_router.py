import io
import pandas as pd
import traceback
from fastapi import APIRouter, File, HTTPException, UploadFile, Query
from typing import Optional
from sqlalchemy import text
from app.core.database import engine
from app.modules.data_pipeline.etl.etl_schemas import GenerateTimeframeRequest
from app.modules.data_pipeline.etl.etl_services import (
    process_and_store_static_data,
    process_and_store_dynamic_data,
    generate_all_fact_shells_in_db,
)

router = APIRouter(prefix="/etl", tags=["Data Pipeline - ETL"])


@router.post("/upload-static-files")
async def upload_static_files(
    surface_file: UploadFile = File(...),
    cadence_file: UploadFile = File(...),
):
    """Nạp file tĩnh (surface.xlsx, cadence.xlsx)."""
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
            "message": "Téléchargement et sauvegarde des données statiques réussis!",
            "details": summary_result,
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(
            status_code=500, detail=f"Erreur dữ liệu tĩnh: {str(e)}"
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
    """Nạp 6 file dữ liệu động gốc và ghi đè vào Star Schema."""
    try:
        summary_result = process_and_store_dynamic_data(
            visitor_bytes=await visitor_file.read(),
            etat_bytes=await etat_file.read(),
            weather_bytes=await weather_file.read(),
            horaire_bytes=await horaire_file.read(),
            elec_bytes=await elec_file.read(),
            ec_bytes=await ec_file.read(),
            engine=engine,
        )
        return {
            "status": "success",
            "message": "Téléchargement et sauvegarde des 6 fichiers réussis!",
            "details": summary_result,
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Erreur 6 files: {str(e)}")


@router.post("/generate-fact-shells")
def api_generate_fact_shells(req: GenerateTimeframeRequest):
    """Tạo khung Fact Shells trống theo khoảng thời gian chọn lựa."""
    try:
        inserted_count = generate_all_fact_shells_in_db(
            start_date=req.start_date,
            end_date=req.end_date,
            engine=engine,
            target_attractions=req.attractions,
        )
        return {
            "status": "success",
            "message": "Initialisation réussie de la ligne de faits vide.",
            "inserted_rows": inserted_count,
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Erreur générer Fact Shells: {str(e)}"
        )

@router.post("/upload-raw")
async def upload_dataset(file: UploadFile = File(...)):
    """Upload dữ liệu thô tổng hợp vào bảng raw."""
    if not (file.filename.endswith(".csv") or file.filename.endswith(".xlsx")):
        raise HTTPException(
            status_code=400, detail="Chỉ hỗ trợ file .csv hoặc .xlsx"
        )

    contents = await file.read()

    try:
        df = (
            pd.read_csv(io.BytesIO(contents))
            if file.filename.endswith(".csv")
            else pd.read_excel(io.BytesIO(contents))
        )
        if "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"])

        with engine.begin() as conn:
            df.to_sql("fact_visitor_raw", con=conn, if_exists="replace", index=False)

        return {
            "message": "Upload thành công!",
            "filename": file.filename,
            "total_rows": len(df),
            "columns": list(df.columns),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur upload raw: {str(e)}")


@router.get("/records")
def get_table_data(
    page: int = Query(1, ge=1),
    size: int = Query(50, ge=1, le=500),
    id_attraction: Optional[str] = None,
):
    """Trích xuất danh sách dữ liệu phân trang."""
    offset = (page - 1) * size

    query = text("""
        SELECT f.datetime, f.id_attraction, f.visitor_count, f.ouvert, f.interrompu, f.operation,
               w.temperature, w.humidite, w.rayonnement_solaire, t.is_weekend, h.is_open
        FROM fact_attraction_hourly f
        LEFT JOIN dim_weather w ON f.temps_id = w.temps_id
        LEFT JOIN dim_temps t ON f.temps_id = t.temps_id
        LEFT JOIN dim_horaire h ON f.datetime = h.datetime
        WHERE (:id_attraction IS NULL OR f.id_attraction = :id_attraction)
        ORDER BY f.datetime ASC
        LIMIT :limit OFFSET :offset
    """)

    count_query = text("""
        SELECT COUNT(*) FROM fact_attraction_hourly f
        WHERE (:id_attraction IS NULL OR f.id_attraction = :id_attraction)
    """)

    params = {"id_attraction": id_attraction or None, "limit": size, "offset": offset}

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
        raise HTTPException(status_code=500, detail=f"Erreur requête: {str(e)}")