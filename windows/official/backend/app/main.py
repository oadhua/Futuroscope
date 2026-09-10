from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Import router từ module data_pipeline mà chúng ta đã viết
from app.modules.data_pipeline.models import Base
from app.modules.data_pipeline.etl.etl_router import router as etl_router
from app.modules.data_pipeline.initial_profiling.initial_profiling_router import router as initial_profiling_router
from app.modules.data_pipeline.data_analysis.data_analysis_router import router as data_analysis_router
from app.modules.data_pipeline.single_column.single_column_router import router as single_column_router
from app.modules.data_pipeline.multi_column.multi_column_router import router as multi_column_router
from app.modules.data_pipeline.timeseries_analysis.timeseries_analysis_router import router as timeseries_analysis_router
from app.modules.data_pipeline.data_prep_router import router as shared_router
from app.modules.data_pipeline.missing_value.missing_value_router import router as missing_value_router
from app.modules.data_pipeline.outliers.outliers_router import router as outliers_router
from app.modules.data_pipeline.scaling.scaling_router import router as scaling_router
from app.modules.data_pipeline.encoding.encode_router import router as encoding_router
from app.modules.data_pipeline.duplicated.deduplication_router import router as duplicated_router
from app.modules.data_pipeline.timeseries_transformation.timeseries_transformation_router import router as timeseries_trans_router
from app.modules.data_pipeline.trace.trace_router import router as trace_router
from app.modules.data_pipeline.feature_selection.feature_selection_router import router as feature_selection_router
from app.modules.ml_pipeline.training.training_router import router as training_router
from app.modules.ml_pipeline.ml_inference.ml_inference_router import router as ml_inference_router
from app.modules.ml_pipeline.future_forecast.future_forecast_router import router as future_forecast_router
from app.core.database import engine

# Khởi tạo ứng dụng FastAPI
app = FastAPI(
    title="Futuroscope HVAC & Visitor Forecasting API",
    description="Système de traitement des données, prévisions de fréquentation, consommation d'énergie et optimisation du CVC.",
    version="1.0.0"
)

# -----------------------------------------------------------------------------
# Cấu hình CORS (Cho phép Frontend React.js gọi API)
# -----------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # React mặc định (Create React App)
        "http://localhost:5173",  # React Vite mặc định
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------------------------------
# Đăng ký các Router API vào hệ thống
# -----------------------------------------------------------------------------
app.include_router(etl_router)
app.include_router(initial_profiling_router)
app.include_router(data_analysis_router)
app.include_router(single_column_router)
app.include_router(multi_column_router)
app.include_router(timeseries_analysis_router)
app.include_router(shared_router)
app.include_router(missing_value_router)
app.include_router(outliers_router)
app.include_router(scaling_router)
app.include_router(encoding_router)
app.include_router(duplicated_router)
app.include_router(timeseries_trans_router)
app.include_router(trace_router)
app.include_router(feature_selection_router)
app.include_router(training_router)
app.include_router(ml_inference_router)
app.include_router(future_forecast_router)
Base.metadata.create_all(bind=engine)


# Endpoint trang chủ để test server
@app.get("/")
def read_root():
    return {
        "status": "Online",
        "message": "Bienvenue sur l'API d'optimisation énergétique de Futuroscope!",
        "docs_url": "http://127.0.0.1:8000/docs"
    }