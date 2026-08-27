import pandas as pd
import numpy as np
from typing import Dict, Any, List
from app.modules.data_pipeline.utils import clean_nan_and_inf  # File utils.py

def calculate_multi_column_profiling(
    df: pd.DataFrame, 
    target_var: str = "visitor_count",
    corr_method: str = "pearson"
) -> Dict[str, Any]:
    
    # 1. Tách dữ liệu số để tính tương quan
    numeric_df = df.select_dtypes(include=[np.number])
    
    if numeric_df.empty:
        return {
            "correlation_matrix": {"columns": [], "values": []},
            "target_correlation": [],
            "functional_dependencies": [],
            "sample_data": []
        }

    # 2. Tính Ma trận tương quan theo method (pearson, kendall, spearman)
    method = corr_method.lower() if corr_method.lower() in ["pearson", "kendall", "spearman"] else "pearson"
    corr_matrix = numeric_df.corr(method=method).round(3).fillna(0.0)
    
    corr_data = {
        "columns": [str(c) for c in corr_matrix.columns],
        "values": corr_matrix.values.tolist()
    }

    # 3. Tính bảng tương quan với Target Var
    target_corr = []
    if target_var in numeric_df.columns:
        series_corr = corr_matrix[target_var].drop(index=target_var, errors="ignore").sort_values(ascending=False)
        target_corr = [
            {"colonne": str(col), "correlation": float(val) if not np.isnan(val) else 0.0}
            for col, val in series_corr.items()
        ]

    # 4. Tính Phụ thuộc hàm dạng {Determinant_1, Determinant_2, ...} -> Target
    dep_results = []
    
    if target_var in df.columns:
        # Lấy danh sách các cột ứng viên (loại bỏ target_var)
        candidate_cols = [c for c in df.columns if c != target_var and df[c].nunique() > 1]
        
        # Các tập hợp phổ biến xác định biến target trong dữ liệu thời gian/vận hành
        potential_sets: List[List[str]] = []
        
        # Tìm các cột thời gian / phân loại
        time_or_cat_cols = [
            c for c in candidate_cols 
            if any(k in c.lower() for k in ['date', 'time', 'hour', 'heure', 'day', 'jour', 'month', 'mois', 'year', 'annee', 'week', 'semaine', 'id', 'attraction'])
        ]

        # Tạo các tập hợp ứng viên 2, 3 và 4 thuộc tính
        if len(time_or_cat_cols) >= 2:
            potential_sets.append(time_or_cat_cols[:2])
        if len(time_or_cat_cols) >= 3:
            potential_sets.append(time_or_cat_cols[:3])
        if len(time_or_cat_cols) >= 4:
            potential_sets.append(time_or_cat_cols[:4])
            
        # Bổ sung các biến số / đặc trưng môi trường vào tập hợp
        other_cols = [c for c in candidate_cols if c not in time_or_cat_cols]
        for col in other_cols[:2]:
            if len(time_or_cat_cols) >= 2:
                potential_sets.append(time_or_cat_cols[:2] + [col])

        # Kiểm tra tính phụ thuộc hàm tuyệt đối cho từng tập hợp
        for col_set in potential_sets:
            if not col_set:
                continue
            
            # Nếu nhóm theo col_set mà mỗi nhóm chỉ tương ứng với 1 giá trị duy nhất của target_var
            max_unique_per_group = df.groupby(col_set)[target_var].nunique().max()
            
            if max_unique_per_group == 1:
                dep_results.append({
                    "determinants": [str(c) for c in col_set],
                    "dependent": str(target_var)
                })

        # Loại bỏ các tập hợp trùng lặp nếu có
        unique_deps = []
        seen = set()
        for d in dep_results:
            key = tuple(sorted(d["determinants"]))
            if key not in seen:
                seen.add(key)
                unique_deps.append(d)
        dep_results = unique_deps

    # 5. Lấy Sample Data (Giữ lại các cột thời gian nếu có để vẽ Line Chart)
    time_cols = [c for c in df.columns if 'date' in c.lower() or 'time' in c.lower() or 'heure' in c.lower()]
    keep_cols = list(numeric_df.columns) + [c for c in time_cols if c not in numeric_df.columns]
    
    sample_df = df[keep_cols].head(200).copy()
    
    # Ép kiểu cột thời gian về chuỗi để không bị lỗi JSON
    for tc in time_cols:
        sample_df[tc] = sample_df[tc].astype(str)
        if 'datetime' not in sample_df.columns:
            sample_df['datetime'] = sample_df[tc]

    sample_df = sample_df.replace({np.nan: None, np.inf: None, -np.inf: None})
    sample_data = sample_df.to_dict(orient="records")

    raw_result = {
        "correlation_matrix": corr_data,
        "target_correlation": target_corr,
        "functional_dependencies": dep_results,
        "sample_data": sample_data
    }

    return clean_nan_and_inf(raw_result)