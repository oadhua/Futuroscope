from fastapi import APIRouter, HTTPException, status
from typing import Dict, Optional

from app.modules.data_pipeline.outliers.outliers_schemas import (
    OutlierRequest,
    OutlierVersionMetadataResponse,
    DeleteVersionResponse,
    OutlierVersionStatsResponse,
)
from app.modules.data_pipeline.outliers.outliers_services import (
    OutlierService,
)

router = APIRouter(prefix="/data-prep", tags=["Data Preparation - Outliers"])


@router.post(
    "/outliers",
    response_model=OutlierVersionMetadataResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Thực hiện kiểm tra và xử lý nhiễu (outliers) theo luật/thuật toán",
)
def process_outliers(req: OutlierRequest):
    try:
        metadata = OutlierService.execute_and_version(
            parent_version_id=req.parent_version_id,
            id_attraction=req.id_attraction,
            target_columns=req.target_columns,
            method=req.method.value,
            action=req.action.value,
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
    response_model=Dict[str, OutlierVersionMetadataResponse],
    summary="Lấy danh sách tất cả phiên bản dữ liệu xử lý Outliers",
)
def list_outlier_versions():
    try:
        return OutlierService._load_metadata_store()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )


@router.get(
    "/outliers/versions/{version_id}/stats",
    response_model=OutlierVersionStatsResponse,
    summary="Lấy thông tin thống kê số lượng nhiễu (outliers) và danh sách id_attraction",
)
def get_outlier_version_stats(
    version_id: str,
    id_attraction: Optional[str] = "ALL",
    method: Optional[str] = "iqr",
    iqr_factor: Optional[float] = 1.5,
    z_threshold: Optional[float] = 3.0,
):
    try:
        params = {"iqr_factor": iqr_factor, "z_threshold": z_threshold}
        outlier_counts, total_records, attractions = (
            OutlierService.get_version_outlier_stats(
                version_id=version_id,
                id_attraction=id_attraction,
                method=method,
                params=params,
            )
        )
        return OutlierVersionStatsResponse(
            version_id=version_id,
            outlier_counts=outlier_counts,
            total_records=total_records,
            available_attractions=attractions,
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
    summary="Xóa phiên bản dữ liệu Outliers khỏi PostgreSQL Schema data_prep",
)
def delete_outlier_version(version_id: str):
    try:
        return OutlierService.delete_version(version_id)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )
