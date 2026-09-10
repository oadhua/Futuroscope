import logging
from typing import List, Optional
from fastapi import APIRouter, HTTPException, status, Query
from fastapi.responses import HTMLResponse, StreamingResponse

from app.modules.ml_pipeline.training.training_services import MLTrainingService

from app.modules.ml_pipeline.ml_inference.ml_inference_services import (
    MLInferenceService,
)
from app.modules.ml_pipeline.ml_inference.ml_inference_schemas import (
    PredictRequest,
    PredictResponse,
    CompareRequest,
    ComparePredictionResponse,
    ApplyImputationRequest,
    ApplyImputationResponse,
    ExportPredictionToSchemaRequest,
    ExportPredictionResponse,
    ExportPredictionFileRequest,
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/ml/inference", tags=["Machine Learning - Inference & Imputation"]
)


@router.post(
    "/predict", response_model=PredictResponse, summary="Dự đoán / Castback cho mô hình"
)
def predict_and_castback(req: PredictRequest):
    """
    Tải mô hình đã huấn luyện, thực hiện dự đoán trên bảng `data_prep."{version_id}"`
    cho cột target tương ứng và trả về chuỗi Actual vs Predicted.
    """
    try:
        res = MLInferenceService.run_prediction(
            model_id=req.model_id,
            version_id=req.version_id,
            id_attraction=req.id_attraction,
            start_date=req.start_date,
            end_date=req.end_date,
        )
        return res
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except Exception as e:
        logger.error(f"Lỗi khi thực hiện inference: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Lỗi xử lý dự đoán: {str(e)}",
        )


@router.post(
    "/compare",
    response_model=ComparePredictionResponse,
    summary="So sánh & Đánh giá mô hình theo target_column động (RMSE, MAE, R2, Dashboard HTML)",
)
def compare_and_evaluate_predictions(
    req: CompareRequest,
    include_html: bool = Query(True, description="Trả về mã HTML Plotly Dashboard"),
):
    """
    Thực hiện dự đoán và so sánh Thực Tế vs Dự Đoán động theo đúng `target_column`
    của danh sách mô hình truyền vào (`model_ids`).
    """
    try:
        res = MLInferenceService.compare_and_evaluate(
            version_id=req.version_id,
            model_ids=req.model_ids,
            id_attraction=req.id_attraction,
            start_date=req.start_date,
            end_date=req.end_date,
            include_html=include_html,
        )
        return res
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except Exception as e:
        logger.error(f"Lỗi khi thực hiện so sánh đánh giá: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Lỗi hệ thống khi so sánh dự đoán: {str(e)}",
        )


@router.post(
    "/apply-imputation",
    response_model=ApplyImputationResponse,
    summary="Lấp lỗ trống / Ghi đè kết quả dự đoán vào bảng PostgreSQL",
)
def apply_imputation(req: ApplyImputationRequest):
    if not req.data:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Danh sách dữ liệu cập nhật trống.",
        )

    try:
        updated_count = MLInferenceService.apply_imputation_to_db(
            version_id=req.version_id,
            id_attraction=req.id_attraction,
            target_column=req.target_column,
            mode=req.mode.value if hasattr(req.mode, "value") else str(req.mode),
            data=req.data,
        )

        return ApplyImputationResponse(
            status="success",
            message=f"Đã áp dụng thành công ({req.mode}). Đã cập nhật {updated_count} bản ghi trong bảng '{req.version_id}'.",
            version_id=req.version_id,
            target_column=req.target_column,
            mode=req.mode,
            updated_count=updated_count,
        )
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except Exception as e:
        logger.error(f"Lỗi khi cập nhật bảng Postgres: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Không thể cập nhật cơ sở dữ liệu: {str(e)}",
        )


@router.post(
    "/export-prediction-schema",
    response_model=ExportPredictionResponse,
    summary="Xuất toàn bộ bảng Version + Kết quả dự đoán ra Schema riêng",
)
def export_prediction_to_schema(req: ExportPredictionToSchemaRequest):
    try:
        result = MLInferenceService.export_predictions_to_schema(
            model_id=req.model_id,
            version_id=req.version_id,
            id_attraction=req.id_attraction,
            start_date=req.start_date,
            end_date=req.end_date,
            target_schema=req.target_schema,
            output_table_name=req.output_table_name,
            if_exists=req.if_exists,
        )
        return result
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except Exception as e:
        logger.error(f"Lỗi khi xuất bảng dự đoán ra schema: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Lỗi hệ thống khi tạo bảng dự đoán: {str(e)}",
        )


@router.post(
    "/export-file",
    summary="Xuất file CSV hoặc Excel chứa tất cả các cột nguồn + kết quả dự đoán",
)
def export_prediction_file(req: ExportPredictionFileRequest):
    """
    Truy vấn toàn bộ dữ liệu từ bảng nguồn trong `data_prep`, chạy dự đoán cho các `model_ids`,
    kết hợp tất cả cột nguồn cùng các cột dự đoán (`pred_<model_id>`) và trả về file tải về trực tiếp.
    """
    try:
        file_stream, filename, media_type = MLInferenceService.export_predictions_to_file(
            version_id=req.version_id,
            model_ids=req.model_ids,
            file_format=req.file_format,
            id_attraction=req.id_attraction or "ALL",
            start_date=req.start_date,
            end_date=req.end_date,
        )

        return StreamingResponse(
            file_stream,
            media_type=media_type,
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"',
                "Access-Control-Expose-Headers": "Content-Disposition",
            },
        )
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except Exception as e:
        logger.error(f"Lỗi khi xuất file dự đoán: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Lỗi hệ thống khi xuất file: {str(e)}",
        )


@router.post(
    "/dashboard-html",
    response_class=HTMLResponse,
    summary="Xuất trực tiếp giao diện HTML Dashboard (POST)",
)
def get_prediction_dashboard_html_post(req: CompareRequest):
    """Thực thi so sánh và trả về trực tiếp mã HTML Dashboard tương tác (Plotly)."""
    try:
        res = MLInferenceService.compare_and_evaluate(
            version_id=req.version_id,
            model_ids=req.model_ids,
            id_attraction=req.id_attraction,
            start_date=req.start_date,
            end_date=req.end_date,
            include_html=True,
        )
        html_content = res.get("html_dashboard")
        if not html_content:
            raise ValueError("Không thể tạo nội dung HTML Dashboard.")

        return HTMLResponse(content=html_content, status_code=200)

    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except Exception as e:
        logger.error(f"Lỗi khi khởi tạo HTML Dashboard: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Lỗi khi khởi tạo HTML Dashboard: {str(e)}",
        )


@router.get(
    "/dashboard-html",
    response_class=HTMLResponse,
    summary="Xem trực tiếp HTML Dashboard trên trình duyệt (GET)",
)
def get_prediction_dashboard_html_get(
    version_id: str,
    model_ids: List[str] = Query(..., description="Danh sách model_id cần so sánh"),
    id_attraction: str = "ALL",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
):
    """Mở trực tiếp giao diện Dashboard HTML trên trình duyệt thông qua URL parameters."""
    try:
        res = MLInferenceService.compare_and_evaluate(
            version_id=version_id,
            model_ids=model_ids,
            id_attraction=id_attraction,
            start_date=start_date,
            end_date=end_date,
            include_html=True,
        )
        html_content = res.get("html_dashboard")
        if not html_content:
            raise ValueError("Không thể tạo nội dung HTML Dashboard.")

        return HTMLResponse(content=html_content, status_code=200)

    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except Exception as e:
        logger.error(f"Lỗi khi hiển thị HTML Dashboard: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Lỗi khi hiển thị HTML Dashboard: {str(e)}",
        )


@router.get("/versions", summary="Lấy danh sách các bảng phiên bản dữ liệu (data_prep)")
def get_versions(
    id_attraction: Optional[str] = Query("ALL", description="Mã Attraction cần lọc (VD: ATTR_01, ALL)")
):
    """
    Trả về danh sách các phiên bản bảng trong schema data_prep,
    đã được lọc theo id_attraction nếu truyền vào.
    """
    try:
        all_versions = MLTrainingService.list_available_versions() 

        if not id_attraction or id_attraction.upper() == "ALL":
            return {"versions": all_versions}

        target_attr = id_attraction.strip().upper()
        filtered_versions = []

        for item in all_versions:
            if isinstance(item, dict):
                v_id = str(item.get("version_id") or item.get("table_name") or "").upper()
                v_attr = str(item.get("id_attraction") or item.get("attraction_id") or "").upper()
                
                if v_attr == target_attr or target_attr in v_id:
                    filtered_versions.append(item)
            elif isinstance(item, str):
                if target_attr in item.upper():
                    filtered_versions.append(item)

        return {"versions": filtered_versions}

    except Exception as e:
        logger.error(f"Lỗi khi lấy danh sách versions: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Lỗi khi lấy danh sách versions: {str(e)}"
        )