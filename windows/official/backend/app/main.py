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
from app.modules.data_pipeline.missing_value.missing_value_router import router as missing_value_router
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
app.include_router(missing_value_router)
Base.metadata.create_all(bind=engine)


# Endpoint trang chủ để test server
@app.get("/")
def read_root():
    return {
        "status": "Online",
        "message": "Bienvenue sur l'API d'optimisation énergétique de Futuroscope!",
        "docs_url": "http://127.0.0.1:8000/docs"
    }