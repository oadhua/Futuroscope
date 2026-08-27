from fastapi import APIRouter, HTTPException, Query, Depends, status
from sqlalchemy.orm import Session
from typing import Optional
import traceback, logging

from app.core.database import get_db
from app.modules.data_pipeline.timeseries_analysis.timeseries_analysis_schemas import (
    StationarityResponse,
    DecompositionComponent,
    AcfPacfResponse,
    TimeSeriesAnalysisRequest,
)
from app.modules.data_pipeline.timeseries_analysis.timeseries_analysis_services import (
    TimeSeriesService,
    get_available_features,
    get_available_attractions,
)
   

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/time-series", tags=["Time Series Analysis"])

@router.post("/analyze")
def analyze(payload: TimeSeriesAnalysisRequest):
    # Lấy giá trị chuỗi gốc trong DB
    selected_attraction = payload.id_attraction.value
    selected_feature = payload.feature_column.value
    
    # Truyền vào service
    data = TimeSeriesService._fetch_series_data(
        version=payload.version,
        col_name=selected_feature,
        id_attraction=selected_attraction
    )
    return {"status": "success"}

# API hỗ trợ FE lấy danh sách động bất kỳ lúc nào mà không cần reload schema
@router.get("/meta/options")
def get_meta_options(id_attraction: Optional[str] = None):
    attractions = get_available_attractions()
    features = get_available_features(id_attraction=id_attraction)
    
    return {
        "attractions": attractions,
        "features": features
    }

@router.get("/stationarity", response_model=StationarityResponse)
async def get_stationarity_test(
    version: str = Query("v1"),
    col_name: str = Query(...),
    method: str = Query("adf"),
    id_attraction: Optional[str] = Query(None),
    db: Session = Depends(get_db)
):
    try:
        return TimeSeriesService.analyze_stationarity(version, col_name, method, id_attraction)
    except Exception as e:
        logger.error(f"Error /stationarity: {traceback.format_exc()}")
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/decomposition", response_model=DecompositionComponent)
async def get_time_series_decomposition(
    version: str = Query("v1"),
    col_name: str = Query(...),
    model_type: str = Query("additive"),
    period: int = Query(24),
    id_attraction: Optional[str] = Query(None),
    db: Session = Depends(get_db)
):
    try:
        return TimeSeriesService.decompose_time_series(version, col_name, model_type, period, id_attraction)
    except Exception as e:
        logger.error(f"Error /decomposition: {traceback.format_exc()}")
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/acf-pacf", response_model=AcfPacfResponse)
async def get_acf_pacf(
    version: str = Query("v1"),
    col_name: str = Query(...),
    plot_type: str = Query("acf"),
    lags: int = Query(25),
    id_attraction: Optional[str] = Query(None),
    db: Session = Depends(get_db)
):
    try:
        return TimeSeriesService.calculate_acf_pacf(version, col_name, plot_type, lags, id_attraction)
    except Exception as e:
        logger.error(f"Error /acf-pacf: {traceback.format_exc()}")
        raise HTTPException(status_code=400, detail=str(e))