import traceback
from typing import Optional, List
from fastapi import APIRouter, Depends, Query, HTTPException
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.modules.data_pipeline.multi_column.multi_column_schemas import (
    MultiColumnProfilingResponse,
    VersionInfo,
)
from app.modules.data_pipeline.multi_column.multi_column_services import (
    calculate_multi_column_profiling,
    load_multi_column_data_from_db,
)

router = APIRouter(prefix="/analysis", tags=["Data Pipeline - Multi-Column Profiling"])


@router.get(
    "/attractions", response_model=List[str], summary="Lấy danh sách ID Attractions"
)
def get_attractions(db: Session = Depends(get_db)):
    """Lấy danh sách ID điểm tham quan trực tiếp từ bảng fact_attraction_hourly."""
    try:
        query_str = text(
            "SELECT DISTINCT id_attraction FROM fact_attraction_hourly WHERE id_attraction IS NOT NULL ORDER BY id_attraction;"
        )
        result = db.execute(query_str).fetchall()
        return [row[0] for row in result if row[0]]
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(
            status_code=500, detail=f"Lỗi khi lấy danh sách attractions: {str(e)}"
        )


@router.get(
    "/versions",
    response_model=List[VersionInfo],
    summary="Lấy danh sách các Version dữ liệu động từ DB",
)
def get_versions(
    id_attraction: Optional[str] = Query(
        None,
        description="Lọc version theo attraction (nếu tên version chứa mã attraction)",
    ),
    db: Session = Depends(get_db),
):
    """Truy vấn các bảng/views thực tế từ schema data_prep trong PostgreSQL."""
    try:
        versions = [
            VersionInfo(
                version_id="v0_raw", description="Dữ liệu gốc chưa qua xử lý (Pivot)"
            )
        ]

        # Lấy tất cả bảng trong schema data_prep
        query_str = text("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'data_prep' 
            ORDER BY table_name;
        """)
        rows = db.execute(query_str).fetchall()

        target_attr = (
            id_attraction.upper()
            if id_attraction and id_attraction.upper() != "ALL"
            else None
        )

        for row in rows:
            tbl_name = row[0]
            # Nếu người dùng chọn id_attraction cụ thể, ưu tiên lọc các version tương ứng hoặc phiên bản chung
            if (
                target_attr
                and target_attr not in tbl_name.upper()
                and not tbl_name.endswith("_all")
            ):
                continue

            versions.append(
                VersionInfo(
                    version_id=tbl_name,
                    description=f"Phiên bản dữ liệu tiền xử lý: {tbl_name}",
                )
            )

        return versions
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(
            status_code=500, detail=f"Lỗi khi lấy danh sách versions: {str(e)}"
        )


@router.get(
    "/multi-column-profiling",
    response_model=MultiColumnProfilingResponse,
    summary="Phân tích tương quan và phụ thuộc đa cột theo Version & Attraction",
)
def get_multi_column_profiling(
    version: str = Query(
        "v0_raw", description="Version của dữ liệu (v0_raw, v1_..., ...)"
    ),
    id_attraction: Optional[str] = Query(
        "ALL", description="ID điểm tham quan (H03, H07, ... hoặc ALL)"
    ),
    target_var: str = Query("visitor_count", description="Biến mục tiêu"),
    corr_method: str = Query(
        "pearson", description="Phương pháp tính: pearson, kendall, spearman"
    ),
    db: Session = Depends(get_db),
):
    try:
        # Load data chuẩn hóa theo version & id_attraction
        df = load_multi_column_data_from_db(
            db=db, version_id=version, id_attraction=id_attraction
        )

        if df.empty:
            raise HTTPException(
                status_code=404, detail="Không tìm thấy dữ liệu cho lựa chọn này."
            )

        result = calculate_multi_column_profiling(
            df=df,
            target_var=target_var,
            corr_method=corr_method,
            selected_version=version,
            selected_attraction=id_attraction or "ALL",
        )
        return result

    except HTTPException as http_ex:
        raise http_ex
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(
            status_code=500, detail=f"Lỗi hệ thống multi-column profiling: {str(e)}"
        )
