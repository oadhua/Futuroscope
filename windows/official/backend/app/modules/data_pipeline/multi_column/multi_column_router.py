from typing import Optional, List
from fastapi import APIRouter, Query, HTTPException, Depends
import pandas as pd
import traceback
from sqlalchemy import text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from app.core.database import engine
from app.modules.data_pipeline.multi_column.multi_column_schemas import MultiColumnProfilingResponse
from app.modules.data_pipeline.multi_column.multi_column_services import calculate_multi_column_profiling
from app.modules.data_pipeline.queries import get_dynamic_gathering_query  # File queries.py[cite: 6]

router = APIRouter(prefix="/analysis", tags=["Multi-Column Profiling"])


def load_multi_column_data_from_db(engine: Engine, id_attraction: Optional[str] = None) -> pd.DataFrame:
    """
    Sử dụng query động để lấy toàn bộ dữ liệu đã được Pivot
    (bao gồm visitor_count, các cột năng lượng, thời tiết, lịch)[cite: 6].
    """
    query_str = get_dynamic_gathering_query(engine, id_attraction=id_attraction)
    
    with engine.connect() as conn:
        df = pd.read_sql(text(query_str), conn)
        
    return df

@router.get(
    "/multi-column-profiling", 
    response_model=MultiColumnProfilingResponse,
    summary="Phân tích tương quan và phụ thuộc đa cột"
)
def get_multi_column_profiling(
    id_attraction: Optional[str] = Query(None, description="ID điểm tham quan (H01, H02, ... hoặc ALL/bỏ trống)"),
    target_var: str = Query("visitor_count", description="Biến mục tiêu"),
    corr_method: str = Query("pearson", description="Phương pháp tính: pearson, kendall, spearman")
):
    try:
        # Nếu chọn "ALL" hoặc để trống -> gán filter_id = None để lấy tất cả điểm tham quan
        filter_id = None if not id_attraction or id_attraction.upper() == "ALL" else id_attraction
        
        # Load data đã Pivot & JOIN
        df = load_multi_column_data_from_db(engine=engine, id_attraction=filter_id)
        
        if df.empty:
            raise HTTPException(status_code=404, detail="Không tìm thấy dữ liệu cho lựa chọn này.")

        result = calculate_multi_column_profiling(df, target_var=target_var, corr_method=corr_method)
        return result

    except HTTPException as http_ex:
        raise http_ex
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi hệ thống: {str(e)}")