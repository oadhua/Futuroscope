import logging
from typing import Optional, List
from fastapi import APIRouter, HTTPException, Query, status
from sqlalchemy import text

from app.core.database import engine
from app.modules.ml_pipeline.training.training_schemas import (
    TrainModelRequest,
    TrainModelResponse,
)
from app.modules.ml_pipeline.training.training_services import MLTrainingService, SCHEMA_NAME

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ml/training", tags=["Machine Learning - Model Training"])


@router.get("/attractions", summary="Lấy danh sách tất cả id_attraction khả dụng trong hệ thống")
def get_attractions_list():
    try:
        attractions = MLTrainingService.get_distinct_attractions()
        return {"status": "success", "attractions": attractions}
    except Exception as e:
        logger.error(f"Lỗi khi lấy danh sách attractions: {str(e)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Không thể lấy danh sách attractions: {str(e)}")


@router.get("/versions", summary="Lấy danh sách tất cả phiên bản dữ liệu khả dụng")
def get_data_prep_versions(id_attraction: Optional[str] = Query("ALL", description="Lọc phiên bản theo Attraction ID")):
    try:
        versions = MLTrainingService.list_available_versions(id_attraction=id_attraction)
        return {"versions": versions}
    except Exception as e:
        logger.error(f"Lỗi truy vấn danh sách versions: {str(e)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Không thể lấy danh sách phiên bản: {str(e)}")


@router.get("/versions/{version_id}/columns", summary="Lấy danh sách các cột khả dụng trong phiên bản")
def get_version_columns(version_id: str, id_attraction: Optional[str] = Query("ALL")):
    try:
        columns = MLTrainingService.get_version_columns(version_id, id_attraction)
        return {"version_id": version_id, "columns": columns}
    except Exception as e:
        logger.error(f"Lỗi truy vấn danh sách cột: {str(e)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Không thể lấy danh sách cột: {str(e)}")


@router.delete("/versions", status_code=status.HTTP_200_OK, summary="Xóa một phiên bản dữ liệu")
def delete_prep_version(version_id: str = Query(..., description="Mã phiên bản cần xóa"), id_attraction: Optional[str] = Query("ALL")):
    try:
        MLTrainingService.delete_prep_version(version_id)
        return {"status": "success", "message": f"Phiên bản '{version_id}' đã được xóa thành công.", "version_id": version_id}
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except Exception as e:
        logger.error(f"Lỗi khi xóa phiên bản {version_id}: {str(e)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Không thể xóa phiên bản dữ liệu: {str(e)}")


@router.post("/run", response_model=TrainModelResponse, status_code=status.HTTP_201_CREATED, summary="Huấn luyện mô hình ML/DL kèm SHAP, PFI và AI Giải thích")
def run_training(req: TrainModelRequest):
    try:
        result = MLTrainingService.train_and_evaluate(
            version_id=req.version_id,
            id_attraction=req.id_attraction,
            target_column=req.target_column,
            target_type=req.target_type.value,
            model_type=req.model_type,
            split_method=req.split_method,
            test_size=req.test_size,
            start_date=req.start_date,
            end_date=req.end_date,
            split_date=req.split_date,
            hyperparameters=req.hyperparameters,
        )
        return result
    except Exception as e:
        logger.error(f"Lỗi huấn luyện mô hình: {str(e)}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Échec de l'entraînement du modèle : {str(e)}")


@router.get("/models", summary="Obtenir le registre de tous les modèles entraînés")
def list_trained_models(target_type: Optional[str] = Query(None, description="Filtrer par type: 'visitor' ou 'energy'")):
    try:
        where_clause = ""
        params = {}
        if target_type:
            where_clause = "WHERE target_type = :t_type"
            params["t_type"] = target_type.lower()

        query = text(f"""
            SELECT model_id, model_type, target_type, version_id, id_attraction, target_column, r2_score, rmse, mae, created_at::text
            FROM {SCHEMA_NAME}.ml_model_registry
            {where_clause}
            ORDER BY created_at DESC;
        """)
        with engine.connect() as conn:
            rows = conn.execute(query, params).mappings().fetchall()
            models = [dict(r) for r in rows]
        return {"models": models}
    except Exception as e:
        logger.error(f"Lỗi danh sách mô hình: {str(e)}")
        return {"models": []}


@router.delete("/models/{model_id}", status_code=status.HTTP_200_OK, summary="Xóa một mô hình đã huấn luyện")
def delete_trained_model(model_id: str):
    try:
        MLTrainingService.delete_model(model_id)
        return {"status": "success", "message": f"Mô hình '{model_id}' đã được xóa thành công.", "model_id": model_id}
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(ve))
    except Exception as e:
        logger.error(f"Lỗi khi xóa mô hình {model_id}: {str(e)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Không thể xóa mô hình: {str(e)}")


@router.get("/versions/{version_id}/date-range", summary="Lấy khoảng thời gian (min_date, max_date)")
def get_version_date_range(version_id: str, id_attraction: Optional[str] = Query("ALL")):
    try:
        return MLTrainingService.get_dataset_date_range(version_id, id_attraction)
    except Exception as e:
        logger.error(f"Lỗi truy vấn date range: {str(e)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Không thể lấy khoảng thời gian: {str(e)}")