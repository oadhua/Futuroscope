from typing import List, Dict, Any
import pandas as pd
import logging
import numpy as np
from fastapi import APIRouter, HTTPException, status

from app.modules.ml_pipeline.future_forecast.future_forecast_schemas import (
    ForecastRequest,
    ForecastResponse,
    HourlyPredictionItem,
    ModelMetadataResponse,
    PredictRequest,
    PredictResponse,
    ModelListItem,
    MetricSummary,
    ForecastSummaryStats,
)
from app.modules.ml_pipeline.future_forecast.future_forecast_services import (
    ForecastService,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/predict-visitor", tags=["Prédiction & Inférence ML"])


# ==========================================
# 1. ENDPOINTS D'INFÉRENCE EN TEMPS RÉEL (MODÈLE SEUL)
# ==========================================


@router.get(
    "/attractions",
    summary="Lấy danh sách tất cả id_attraction khả dụng trong hệ thống cho module dự báo",
)
def get_inference_attractions():
    try:
        attractions = ForecastService.get_available_attractions()
        return {"status": "success", "attractions": attractions}
    except Exception as e:
        logger.error(f"Lỗi khi lấy danh sách attractions cho inference: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Không thể lấy danh sách attractions: {str(e)}",
        )


@router.get(
    "/ml/models",
    response_model=List[ModelListItem],
    summary="Lister tous les modèles entraînés",
)
def get_trained_models():
    try:
        return ForecastService.list_trained_models()
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la récupération des modèles : {str(e)}",
        )


@router.get(
    "/ml/models/{model_id}",
    response_model=ModelMetadataResponse,
    summary="Obtenir les métadonnées d'un modèle",
)
def get_model_details(model_id: str):
    try:
        return ForecastService.get_model_metadata(model_id)
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur d'extraction des métadonnées du modèle : {str(e)}",
        )


@router.post(
    "/ml/predict",
    response_model=PredictResponse,
    summary="Exécuter une prédiction instantanée",
)
def predict_single_instance(req: PredictRequest):
    try:
        return ForecastService.predict_single_instance(req)
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de l'exécution de la prédiction : {str(e)}",
        )


# ==========================================
# 2. ENDPOINT MOTEUR DE SIMULATION HOURLY CHAÎNÉ
# ==========================================


@router.post(
    "/forecast",
    response_model=ForecastResponse,
    summary="Calculer une simulation de prévision temporelle par heure",
)
def predict_future_hourly(req: ForecastRequest):
    try:
        df_result, target_type, target_model, visitor_model = ForecastService.predict(
            req
        )

        prediction_items = []
        preds_list = []
        visitors_list = []

        for idx, row in df_result.iterrows():
            # Ép giá trị năng lượng đảm bảo luôn lớn hơn 0 nếu là target năng lượng
            raw_pred = float(row["predicted_value"])
            if target_type in ["electricity", "thermal"]:
                pred_val = max(0.01, round(raw_pred, 2))
            else:
                pred_val = round(raw_pred, 2)

            preds_list.append(pred_val)

            # Ép visitor_count bắt buộc là số nguyên (int >= 0)
            vis_count = None
            if "visitor_count" in row and not pd.isna(row["visitor_count"]):
                vis_count = max(0, int(round(float(row["visitor_count"]))))
                visitors_list.append(vis_count)

            item = HourlyPredictionItem(
                datetime=row["datetime"].strftime("%Y-%m-%d %H:%M:%S"),
                heure=int(row["heure"]),
                predicted_value=pred_val,
                ouvert=float(row["ouvert"]) if not pd.isna(row.get("ouvert")) else 1.0,
                operation=float(row["operation"])
                if not pd.isna(row.get("operation"))
                else 1.0,
                temperature=float(row["temperature"])
                if "temperature" in row and not pd.isna(row["temperature"])
                else None,
                visitor_count=vis_count,
            )
            prediction_items.append(item)

        # Đơn vị tính
        unit = "kWh" if target_type in ["electricity", "thermal"] else "pers."

        # Tính toán bảng thống kê (Summary Table)
        pred_arr = np.array(preds_list)
        pred_summary = MetricSummary(
            total=round(float(np.sum(pred_arr)), 2),
            mean=round(float(np.mean(pred_arr)), 2),
            median=round(float(np.median(pred_arr)), 2),
            max=round(float(np.max(pred_arr)), 2),
            min=round(float(np.min(pred_arr)), 2),
        )

        vis_summary = None
        if len(visitors_list) > 0:
            vis_arr = np.array(visitors_list)
            vis_summary = MetricSummary(
                total=round(float(np.sum(vis_arr)), 2),
                mean=round(float(np.mean(vis_arr)), 2),
                median=round(float(np.median(vis_arr)), 2),
                max=round(float(np.max(vis_arr)), 2),
                min=round(float(np.min(vis_arr)), 2),
            )

        summary_stats = ForecastSummaryStats(
            unit=unit,
            predicted_value=pred_summary,
            visitor_count=vis_summary,
        )

        return ForecastResponse(
            status="success",
            target_type=target_type,
            id_attraction=req.id_attraction,
            target_model_used=target_model,
            visitor_model_used=visitor_model,
            total_records=len(prediction_items),
            summary=summary_stats,
            predictions=prediction_items,
        )

    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Erreur système interne : {str(e)}"
        )


@router.post(
    "/forecast/preview-features",
    summary="Générer et prévisualiser TOUTES les variables d'entrée horaires générées",
)
def preview_forecast_features(req: ForecastRequest):
    try:
        df_features = ForecastService.generate_future_dataframe(req)

        # 1. Chuyển cột datetime thành chuỗi định dạng ISO
        if "datetime" in df_features.columns:
            df_features["datetime"] = df_features["datetime"].dt.strftime(
                "%Y-%m-%d %H:%M:%S"
            )

        # 2. XỬ LÝ TRIỆT ĐỂ NaN, inf, -inf TRƯỚC KHIN JSON ENCODE
        df_clean = df_features.replace([np.inf, -np.inf], np.nan)

        records = (
            df_clean.astype(object)
            .where(pd.notnull(df_clean), None)
            .to_dict(orient="records")
        )

        # 3. Lấy thông tin model features (nếu có)
        used_features = []
        if req.model_id:
            try:
                artifacts = ForecastService.load_model_artifacts(req.model_id)
                used_features = artifacts.get("feature_names", [])
            except Exception as artifact_err:
                logger.warning(
                    f"Không thể load artifacts cho model_id {req.model_id}: {str(artifact_err)}"
                )

        return {
            "status": "success",
            "total_records": len(records),
            "model_required_features": used_features,
            "features": records,
        }
    except Exception as e:
        logger.exception(f"Erreur preview_forecast_features: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Erreur preview features : {str(e)}"
        )