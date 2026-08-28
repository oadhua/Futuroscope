import logging
import traceback
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.modules.data_pipeline.timeseries_analysis.timeseries_analysis_schemas import (
    AcfPacfResponse,
    DecompositionComponent,
    StationarityResponse,
)
from app.modules.data_pipeline.timeseries_analysis.timeseries_analysis_services import (
    TimeSeriesService,
    get_available_attractions,
    get_available_features,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/time-series", tags=["Data Pipeline - Time Series Analysis"])


@router.get("/meta/options", summary="Lấy danh sách điểm tham quan và biến khả dụng")
def get_meta_options(
    id_attraction: Optional[str] = Query("ALL"),
    version: str = Query("v0_raw"),
    db: Session = Depends(get_db),
):
    try:
        attractions = get_available_attractions(db)
        features = get_available_features(
            db, id_attraction=id_attraction, version=version
        )

        return {"attractions": attractions, "features": features}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Lỗi truy vấn metadata: {str(e)}")


@router.get(
    "/stationarity",
    response_model=StationarityResponse,
    summary="Kiểm định tính dừng (ADF/KPSS)",
)
def get_stationarity_test(
    version: str = Query("v0_raw"),
    col_name: str = Query(...),
    method: str = Query("adf"),
    id_attraction: Optional[str] = Query("ALL"),
    db: Session = Depends(get_db),
):
    try:
        return TimeSeriesService.analyze_stationarity(
            db=db,
            version=version,
            col_name=col_name,
            method=method,
            id_attraction=id_attraction,
        )
    except Exception as e:
        logger.error(f"Error /stationarity: {traceback.format_exc()}")
        raise HTTPException(status_code=400, detail=str(e))


@router.get(
    "/decomposition",
    response_model=DecompositionComponent,
    summary="Phân rã chuỗi thời gian",
)
def get_time_series_decomposition(
    version: str = Query("v0_raw"),
    col_name: str = Query(...),
    model_type: str = Query("additive"),
    period: int = Query(24),
    id_attraction: Optional[str] = Query("ALL"),
    db: Session = Depends(get_db),
):
    try:
        return TimeSeriesService.decompose_time_series(
            db=db,
            version=version,
            col_name=col_name,
            model_type=model_type,
            period=period,
            id_attraction=id_attraction,
        )
    except Exception as e:
        logger.error(f"Error /decomposition: {traceback.format_exc()}")
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/acf-pacf", response_model=AcfPacfResponse, summary="Tính ACF và PACF")
def get_acf_pacf(
    version: str = Query("v0_raw"),
    col_name: str = Query(...),
    plot_type: str = Query("acf"),
    lags: int = Query(25),
    id_attraction: Optional[str] = Query("ALL"),
    db: Session = Depends(get_db),
):
    try:
        return TimeSeriesService.calculate_acf_pacf(
            db=db,
            version=version,
            col_name=col_name,
            plot_type=plot_type,
            lags=lags,
            id_attraction=id_attraction,
        )
    except Exception as e:
        logger.error(f"Error /acf-pacf: {traceback.format_exc()}")
        raise HTTPException(status_code=400, detail=str(e))
