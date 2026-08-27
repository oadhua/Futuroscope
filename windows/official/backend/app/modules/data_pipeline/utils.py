import io
import re
import pandas as pd
import math
from typing import Optional, Tuple, List, Any
import numpy as np


def parse_energy_column(col_name: str) -> Tuple[Optional[str], str]:
    """Bóc tách tên cột năng lượng dài (điện/nhiệt)."""
    col_str = str(col_name).strip()
    match_id = re.match(r"^(H\d{2})", col_str)
    id_attraction = match_id.group(1) if match_id else None

    clean_name = re.sub(r"\[.*?\]", "", col_str).strip()
    clean_name = re.sub(r"^H\d{2}[_ ]*", "", clean_name).strip()
    clean_name = re.sub(r"[^a-zA-Z0-9_]", "_", clean_name)
    clean_name = re.sub(r"_+", "_", clean_name).strip("_").lower()

    return id_attraction, clean_name


def load_file_safely(file_bytes: bytes, file_name: str = "") -> pd.DataFrame:
    """Tự động nhận diện định dạng (CSV hoặc Excel) và đọc dữ liệu an toàn."""
    is_excel = (
        file_name.endswith(".xlsx")
        or file_name.endswith(".xls")
        or file_bytes.startswith(b"PK")
    )

    if is_excel:
        df = pd.read_excel(io.BytesIO(file_bytes))
    else:
        try:
            df = pd.read_csv(
                io.BytesIO(file_bytes), encoding="utf-8-sig", sep=None, engine="python"
            )
        except Exception:
            try:
                df = pd.read_csv(io.BytesIO(file_bytes), encoding="utf-8-sig", sep=";")
            except Exception:
                df = pd.read_csv(
                    io.BytesIO(file_bytes), encoding="latin1", sep=None, engine="python"
                )

    df.columns = (
        df.columns.astype(str).str.replace("\ufeff", "", regex=False).str.strip()
    )

    date_col = None
    for col in df.columns:
        if col.lower() in [
            "datetime",
            "date",
            "timestamp",
            "date_heure",
            "horodate",
            "Date",
        ]:
            date_col = col
            break

    if date_col and date_col != "datetime":
        df = df.rename(columns={date_col: "datetime"})

    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])
    elif "Date" in df.columns:
        df["datetime"] = pd.to_datetime(df["Date"])

    return df


def create_temps_id(df: pd.DataFrame) -> pd.DataFrame:
    """Tạo khóa chính thời gian cấp giờ YYYYMMDDHH (ví dụ: 2026081210)."""
    if "datetime" in df.columns:
        df["temps_id"] = df["datetime"].dt.strftime("%Y%m%d%H").astype("int64")
    return df


def prepare_energy_fact(
    df_raw: pd.DataFrame, target_attractions: List[str]
) -> pd.DataFrame:
    """Chuẩn hóa dữ liệu năng lượng từ định dạng Wide sang Long."""
    df = df_raw.copy()
    if "temps_id" not in df.columns:
        df = create_temps_id(df)

    val_cols = [
        c
        for c in df.columns
        if c.lower() not in ["date", "hour", "datetime", "temps_id"]
    ]

    df_melted = df.melt(
        id_vars=["temps_id", "datetime"],
        value_vars=val_cols,
        var_name="raw_col",
        value_name="value",
    )

    parsed_meta = df_melted["raw_col"].apply(parse_energy_column)
    df_melted["id_attraction"] = [p[0] for p in parsed_meta]
    df_melted["metric_name"] = [p[1] for p in parsed_meta]

    mask = df_melted["id_attraction"].isin(target_attractions)
    final_df = df_melted[mask].copy()
    final_df["value"] = pd.to_numeric(final_df["value"], errors="coerce").fillna(0.0)

    cols = ["temps_id", "datetime", "id_attraction", "metric_name", "value"]
    return final_df[cols].drop_duplicates(
        subset=["temps_id", "id_attraction", "metric_name"]
    )
    
def clean_nan_and_inf(obj: Any) -> Any:
    """Xử lý các giá trị NaN và Infinity để tránh lỗi JSON serialization."""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    elif isinstance(obj, dict):
        return {k: clean_nan_and_inf(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_nan_and_inf(v) for v in obj]
    elif isinstance(obj, np.generic):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return obj.item()
    return obj


def format_alias(attr_id: str, metric_name: str) -> str:
    """Tạo tên cột dạng H03_puissance_kw."""
    if not attr_id:
        return metric_name
    attr_str = str(attr_id).strip()
    metric_str = str(metric_name).strip()

    if metric_str.lower().startswith(attr_str.lower()):
        return metric_str
    return f"{attr_str}_{metric_str}"