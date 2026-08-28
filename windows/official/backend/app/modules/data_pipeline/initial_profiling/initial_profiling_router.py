from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session
from typing import List, Optional

# 1. Import đúng hàm get_db từ database.py của bạn
from app.core.database import get_db

from app.modules.data_pipeline.initial_profiling.initial_profiling_schemas import (
    InitialProfilingResponse,
    DataVersionOption,
)
from app.modules.data_pipeline.initial_profiling.initial_profiling_services import (
    InitialProfilingService,
)

router = APIRouter(prefix="/initial-profiling", tags=["Initial Profiling"])


@router.get(
    "/versions",
    response_model=List[DataVersionOption],
    summary="Lấy danh sách các Data Version",
)
def get_versions(
    id_attraction: Optional[str] = Query(
        None, description="Lọc version theo id_attraction"
    ),
    db: Session = Depends(get_db),  # 1. Khai báo dependency db
):
    """Lấy danh sách các phiên bản dữ liệu hiện có trong schema data_prep."""
    # 2. Truyền db và id_attraction vào service
    return InitialProfilingService.get_available_versions(
        db=db, id_attraction=id_attraction
    )


@router.get(
    "/attractions",
    response_model=List[str],
    summary="Lấy danh sách tất cả id_attraction hiện có trong CSDL",
)
def get_attractions(db: Session = Depends(get_db)):
    """Trả về danh sách danh mục các trò chơi/công trình."""
    return InitialProfilingService.get_attractions(db)


@router.get(
    "",
    response_model=InitialProfilingResponse,
    summary="Lấy thông tin Initial Profiling của CSDL",
)
def get_initial_profiling(
    version: str = Query("v1", description="Phiên bản dữ liệu (v1, v2, all)"),
    id_attraction: Optional[str] = Query(
        None, description="Lọc theo mã trò chơi (ví dụ: H01, H03, ALL)"
    ),
    db: Session = Depends(get_db),  # get_db đã được import hợp lệ
):
    """
    Tự động truy vấn CSDL PostgreSQL Futuroscope, phân tích bảng Data Warehouse
    và trả về thông tin Profiling chi tiết cho từng cột dữ liệu.
    """
    try:
        return InitialProfilingService.get_initial_profiling(
            db=db, version_id=version, id_attraction=id_attraction
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Lỗi hệ thống khi phân tích dữ liệu profiling: {str(e)}",
        )
