from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.modules.data_pipeline.queries import get_dynamic_gathering_query
from app.modules.data_pipeline.utils import clean_nan_and_inf


def load_multi_column_data_from_db(
    db: Session, version_id: str = "v0_raw", id_attraction: Optional[str] = None
) -> pd.DataFrame:
    """
    Load dữ liệu theo Version và Attraction.
    Đồng bộ hoàn toàn logic với module single_column.
    """
    conn = db.connection()
    target_attr = id_attraction.upper() if id_attraction else "ALL"

    # 1. Trường hợp phiên bản thô v0_raw: Gọi query Pivot động
    if not version_id or version_id == "v0_raw":
        query_sql = get_dynamic_gathering_query(
            db.bind, id_attraction=target_attr, version="v0_raw"
        )
        return pd.read_sql(text(query_sql), conn)

    # 2. Trường hợp các version trong schema data_prep
    schema_name = "data_prep"
    query_str = f'SELECT * FROM "{schema_name}"."{version_id}"'

    # Kiểm tra cột id_attraction trong schema
    cols_query = text("""
        SELECT column_name FROM information_schema.columns 
        WHERE table_schema = :s AND table_name = :t AND column_name = 'id_attraction'
    """)
    has_id_attr_col = conn.execute(
        cols_query, {"s": schema_name, "t": version_id}
    ).fetchone()

    if target_attr != "ALL" and has_id_attr_col:
        query_str += " WHERE id_attraction = :attr_id"
        df = pd.read_sql(text(query_str), conn, params={"attr_id": target_attr})
    else:
        df = pd.read_sql(text(query_str), conn)

    # Lọc dự phòng ở Python nếu cần
    if target_attr != "ALL" and "id_attraction" in df.columns:
        df = df[df["id_attraction"] == target_attr].copy()

    return df


def calculate_multi_column_profiling(
    df: pd.DataFrame,
    target_var: str = "visitor_count",
    corr_method: str = "pearson",
    selected_version: str = "v0_raw",
    selected_attraction: str = "ALL",
) -> Dict[str, Any]:
    if df.empty:
        return clean_nan_and_inf(
            {
                "selected_version": selected_version,
                "selected_attraction": selected_attraction,
                "correlation_matrix": {"columns": [], "values": []},
                "target_correlation": [],
                "functional_dependencies": [],
                "sample_data": [],
            }
        )

    # 1. Tách dữ liệu số để tính tương quan
    numeric_df = df.select_dtypes(include=[np.number])

    if numeric_df.empty:
        return clean_nan_and_inf(
            {
                "selected_version": selected_version,
                "selected_attraction": selected_attraction,
                "correlation_matrix": {"columns": [], "values": []},
                "target_correlation": [],
                "functional_dependencies": [],
                "sample_data": [],
            }
        )

    # 2. Tính Ma trận tương quan
    method = (
        corr_method.lower()
        if corr_method.lower() in ["pearson", "kendall", "spearman"]
        else "pearson"
    )
    corr_matrix = numeric_df.corr(method=method).round(3).fillna(0.0)

    corr_data = {
        "columns": [str(c) for c in corr_matrix.columns],
        "values": corr_matrix.values.tolist(),
    }

    # 3. Tính bảng tương quan với Target Var
    target_corr = []
    if target_var in numeric_df.columns:
        series_corr = (
            corr_matrix[target_var]
            .drop(index=target_var, errors="ignore")
            .sort_values(ascending=False)
        )
        target_corr = [
            {
                "colonne": str(col),
                "correlation": float(val) if not np.isnan(val) else 0.0,
            }
            for col, val in series_corr.items()
        ]

    # 4. Tính Phụ thuộc hàm
    dep_results = []
    if target_var in df.columns:
        candidate_cols = [
            c for c in df.columns if c != target_var and df[c].nunique() > 1
        ]

        potential_sets: List[List[str]] = []
        time_or_cat_cols = [
            c
            for c in candidate_cols
            if any(
                k in c.lower()
                for k in [
                    "date",
                    "time",
                    "hour",
                    "heure",
                    "day",
                    "jour",
                    "month",
                    "mois",
                    "year",
                    "annee",
                    "week",
                    "semaine",
                    "id",
                    "attraction",
                ]
            )
        ]

        if len(time_or_cat_cols) >= 2:
            potential_sets.append(time_or_cat_cols[:2])
        if len(time_or_cat_cols) >= 3:
            potential_sets.append(time_or_cat_cols[:3])
        if len(time_or_cat_cols) >= 4:
            potential_sets.append(time_or_cat_cols[:4])

        other_cols = [c for c in candidate_cols if c not in time_or_cat_cols]
        for col in other_cols[:2]:
            if len(time_or_cat_cols) >= 2:
                potential_sets.append(time_or_cat_cols[:2] + [col])

        for col_set in potential_sets:
            if not col_set:
                continue

            max_unique_per_group = df.groupby(col_set)[target_var].nunique().max()
            if max_unique_per_group == 1:
                dep_results.append(
                    {
                        "determinants": [str(c) for c in col_set],
                        "dependent": str(target_var),
                    }
                )

        unique_deps = []
        seen = set()
        for d in dep_results:
            key = tuple(sorted(d["determinants"]))
            if key not in seen:
                seen.add(key)
                unique_deps.append(d)
        dep_results = unique_deps

    # 5. Lấy Sample Data
    time_cols = [
        c
        for c in df.columns
        if "date" in c.lower() or "time" in c.lower() or "heure" in c.lower()
    ]
    keep_cols = list(numeric_df.columns) + [
        c for c in time_cols if c not in numeric_df.columns
    ]

    sample_df = df[keep_cols].head(200).copy()

    for tc in time_cols:
        sample_df[tc] = sample_df[tc].astype(str)
        if "datetime" not in sample_df.columns:
            sample_df["datetime"] = sample_df[tc]

    sample_df = sample_df.replace({np.nan: None, np.inf: None, -np.inf: None})
    sample_data = sample_df.to_dict(orient="records")

    raw_result = {
        "selected_version": selected_version,
        "selected_attraction": selected_attraction,
        "correlation_matrix": corr_data,
        "target_correlation": target_corr,
        "functional_dependencies": dep_results,
        "sample_data": sample_data,
    }

    return clean_nan_and_inf(raw_result)
