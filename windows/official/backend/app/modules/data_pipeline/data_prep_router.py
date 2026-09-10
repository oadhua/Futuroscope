from fastapi import APIRouter, HTTPException, Query, status
from typing import Dict, Optional

from app.modules.data_pipeline.data_prep_schemas import (
    SharedVersionMetadataResponse,
    DeleteVersionResponse,
)
from app.modules.data_pipeline.missing_value.missing_value_services import (
    MissingValueService,
)

router = APIRouter(prefix="/data-prep", tags=["Data Preparation - Shared"])


@router.get(
    "/versions",
    response_model=Dict[str, SharedVersionMetadataResponse],
    summary="Lấy danh sách phiên bản dữ liệu (Có thể lọc theo step_type)",
)
def list_all_versions(
    step_type: Optional[str] = Query(
        None, 
        description="Filter theo step_type (vd: missing_value_imputation, outlier_treatment)"
    )
):
    try:
        # Truyền step_type vào service để lọc nếu có
        return MissingValueService._load_metadata_store(step_type=step_type)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )


@router.delete(
    "/versions/{version_id}",
    response_model=DeleteVersionResponse,
    summary="Xóa phiên bản dữ liệu khỏi PostgreSQL Schema data_prep",
)
def delete_version(version_id: str):
    try:
        return MissingValueService.delete_version(version_id)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )