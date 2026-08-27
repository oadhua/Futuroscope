from fastapi import APIRouter, HTTPException, status
from typing import Dict, Optional

from app.modules.data_pipeline.missing_value.missing_value_schemas import (
    ImputationRequest,
    VersionMetadataResponse,
    DeleteVersionResponse,
    VersionStatsResponse,
)
from app.modules.data_pipeline.missing_value.missing_value_services import (
    MissingValueService,
)

router = APIRouter(prefix="/data-prep", tags=["Data Preparation - Missing Values"])


@router.post(
    "/missing-values",
    response_model=VersionMetadataResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Thực hiện xử lý missing values theo cột/luật nghiệp vụ và tự động đánh số phiên bản",
)
def process_missing_values(req: ImputationRequest):
    try:
        metadata = MissingValueService.execute_and_version(
            parent_version_id=req.parent_version_id,
            id_attraction=req.id_attraction,
            target_columns=req.target_columns,
            method=req.method.value,
            params=req.params,
        )
        return metadata
    except FileNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Lỗi hệ thống: {str(e)}",
        )


@router.get(
    "/versions",
    response_model=Dict[str, VersionMetadataResponse],
    summary="Lấy danh sách tất cả phiên bản dữ liệu (Lineage)",
)
def list_versions():
    try:
        return MissingValueService._load_metadata_store()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )


@router.get(
    "/versions/{version_id}/stats",
    response_model=VersionStatsResponse,
    summary="Lấy danh sách cột, số lượng null và danh sách id_attraction khả dụng",
)
def get_version_stats(version_id: str, id_attraction: Optional[str] = "ALL"):
    try:
        null_counts, attractions = MissingValueService.get_version_null_stats(version_id, id_attraction)
        return VersionStatsResponse(
            version_id=version_id, 
            null_counts=null_counts,
            available_attractions=attractions
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
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