import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from sqlalchemy import text
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose

# Import từ các file helper của bạn
from app.core.database import engine
from app.modules.data_pipeline.queries import get_dynamic_gathering_query
from app.modules.data_pipeline.utils import clean_nan_and_inf

logger = logging.getLogger(__name__)


def get_available_attractions() -> list[str]:
    try:
        base_sql = get_dynamic_gathering_query(engine)
        # Thêm ORDER BY 1 ASC ở cuối query
        query = text(
            f"SELECT DISTINCT id_attraction FROM ({base_sql}) AS subquery "
            f"WHERE id_attraction IS NOT NULL ORDER BY 1 ASC;"
        )

        with engine.connect() as conn:
            result = conn.execute(query)
            attractions = [str(row[0]) for row in result.fetchall() if row[0]]
            return attractions if attractions else ["DEFAULT"]
    except Exception as e:
        return ["DEFAULT"]


def get_available_features(id_attraction: Optional[str] = None) -> list[str]:
    """Lấy danh sách các cột tính năng thực sự thuộc về id_attraction."""
    try:
        is_all = id_attraction in [None, "", "all", "ALL", "Toutes", "toutes"]
        query_attraction = None if is_all else id_attraction
        
        base_sql = get_dynamic_gathering_query(engine, id_attraction=query_attraction)
        
        # 1. Lấy 1 dòng để trích xuất danh sách tất cả các tên cột có trong Query
        query = text(f"SELECT * FROM ({base_sql}) AS subquery LIMIT 1;")

        with engine.connect() as conn:
            df_schema = pd.read_sql_query(query, conn)

        excluded = {"datetime", "id_attraction", "created_at", "updated_at"}
        all_cols = [col for col in df_schema.columns if col not in excluded]

        # Nếu là 'Toutes les attractions', giữ nguyên toàn bộ danh sách cột
        if is_all:
            return all_cols if all_cols else ["value"]

        # 2. Kiểm tra dữ liệu thực tế cho attraction cụ thể
        # Kiểm tra 500 dòng MỚI NHẤT (ORDER BY datetime DESC) thay vì 100 dòng ĐẦU TIÊN
        sample_query = text(
            f"SELECT * FROM ({base_sql}) AS subquery "
            f"ORDER BY datetime DESC LIMIT 500;"
        )
        
        with engine.connect() as conn:
            df_sample = pd.read_sql_query(sample_query, conn)

        valid_features = []
        for col in all_cols:
            # Lọc các biến cốt lõi luôn muốn giữ lại nếu tồn tại trong schema
            is_core_metric = any(k in col.lower() for k in ["visitor", "count", "energy", "ec", "elec", "conso"])
            
            # Giữ cột nếu là biến cốt lõi HOẶC có chứa giá trị không NULL trong mẫu dữ liệu mới
            if is_core_metric or (col in df_sample.columns and df_sample[col].notna().any()):
                valid_features.append(col)

        return valid_features if valid_features else all_cols

    except Exception as e:
        logger.warning(f"Không thể lấy danh sách features từ DB: {e}")
        return ["value"]


class TimeSeriesService:
    @classmethod
    def _fetch_series_data(
        cls, version: str, col_name: str, id_attraction: Optional[str] = None
    ) -> pd.Series:
        try:
            if hasattr(col_name, "value"):
                col_name = col_name.value
            if hasattr(id_attraction, "value"):
                id_attraction = id_attraction.value

            is_all_attractions = id_attraction in [None, "", "all", "ALL", "Toutes", "toutes"]
            query_attraction = None if is_all_attractions else id_attraction
            raw_sql = get_dynamic_gathering_query(engine, id_attraction=query_attraction)

            with engine.connect() as conn:
                df = pd.read_sql_query(text(raw_sql), conn)

            if df.empty:
                raise ValueError("Không có dữ liệu trả về từ DB.")

            if col_name not in df.columns:
                raise ValueError(f"Không tìm thấy cột '{col_name}'.")

            df["datetime"] = pd.to_datetime(df["datetime"])
            df[col_name] = pd.to_numeric(df[col_name], errors="coerce")

            # 1. Gom nhóm theo thời gian
            if is_all_attractions and "id_attraction" in df.columns:
                if any(k in col_name.lower() for k in ["visitor", "count", "energy", "ec", "elec", "conso"]):
                    df_grouped = df.groupby("datetime", as_index=False)[col_name].sum(min_count=1)
                else:
                    df_grouped = df.groupby("datetime", as_index=False)[col_name].mean()
            else:
                df_grouped = df.sort_values("datetime").drop_duplicates(subset=["datetime"])

            # 2. Tạo khung thời gian ĐỘNG theo min & max thực tế trong DB
            min_date = df_grouped["datetime"].min()
            max_date = df_grouped["datetime"].max()
            
            # Tạo index theo giờ phủ trọn dải thời gian từ min -> max
            full_index = pd.date_range(start=min_date, end=max_date, freq="h")

            # Tạo Series và Reindex
            series = pd.Series(
                data=df_grouped[col_name].values, 
                index=df_grouped["datetime"]
            )
            series = series.reindex(full_index)

            # 3. Lấy mốc thời gian đầu tiên CÓ DỮ LIỆU THỰC TẾ
            first_valid_idx = series.first_valid_index()

            if first_valid_idx is not None:
                # Chỉ xử lý nội suy từ mốc đầu tiên có dữ liệu đến hết
                # Loại bỏ đoạn toàn NaN trước đó để Décomposition không bị méo lệch
                series = series.loc[first_valid_idx:].interpolate(method="linear").ffill().fillna(0)
            else:
                series = series.fillna(0)

            return series

        except Exception as e:
            logger.error(f"Lỗi trong _fetch_series_data: {str(e)}", exc_info=True)
            raise ValueError(f"Lỗi xử lý dữ liệu: {str(e)}")

    @classmethod
    def analyze_stationarity(
        cls,
        version: str,
        col_name: str,
        method: str,
        id_attraction: Optional[str] = None,
    ) -> Dict[str, Any]:
        series = cls._fetch_series_data(version, col_name, id_attraction)

        if method.lower() == "adf":
            res = adfuller(series, autolag="AIC")
            test_stat = float(res[0])
            p_val = float(res[1])
            crit_vals = {str(k): float(v) for k, v in res[4].items()}
            is_stat = p_val < 0.05
            msg = (
                f"The {col_name} time series is Stationary!"
                if is_stat
                else f"The {col_name} time series is Non-Stationary."
            )

        elif method.lower() == "kpss":
            res = kpss(series, regression="c", nlags="auto")
            test_stat = float(res[0])
            p_val = float(res[1])
            crit_vals = {str(k): float(v) for k, v in res[3].items()}
            is_stat = p_val >= 0.05
            msg = (
                f"The {col_name} time series is Stationary!"
                if is_stat
                else f"The {col_name} time series is Non-Stationary."
            )
        else:
            raise ValueError(
                "Phương pháp kiểm định không hợp lệ. Chọn 'adf' hoặc 'kpss'."
            )

        response_data = {
            "col_name": col_name,
            "method": method.upper(),
            "is_stationary": is_stat,
            "test_statistic": round(test_stat, 4),
            "p_value": round(p_val, 4),
            "critical_values": crit_vals,
            "message": msg,
        }
        # Làm sạch NaN / Inf trước khi trả về
        return clean_nan_and_inf(response_data)

    @classmethod
    def decompose_time_series(
        cls,
        version: str,
        col_name: str,
        model_type: str,
        period: int,
        id_attraction: Optional[str] = None,
    ) -> Dict[str, Any]:
        series = cls._fetch_series_data(version, col_name, id_attraction)

        if model_type == "multiplicative" and (series <= 0).any():
            series = series + abs(series.min()) + 1e-5

        decomposition = seasonal_decompose(
            series, model=model_type, period=period, extrapolate_trend="freq"
        )

        response_data = {
            "timestamps": [d.strftime("%Y-%m-%d %H:%M:%S") for d in series.index],
            "observed": decomposition.observed.tolist(),
            "trend": decomposition.trend.tolist(),
            "seasonal": decomposition.seasonal.tolist(),
            "residual": decomposition.resid.tolist(),
        }
        # Áp dụng clean_nan_and_inf để biến NaN trong Trend/Residual thành None (null trong JSON)
        return clean_nan_and_inf(response_data)

    @classmethod
    def calculate_acf_pacf(
        cls,
        version: str,
        col_name: str,
        plot_type: str,
        lags: int,
        id_attraction: Optional[str] = None,
    ) -> Dict[str, Any]:
        series = cls._fetch_series_data(version, col_name, id_attraction)

        if plot_type.lower() == "acf":
            vals, confint = acf(series, nlags=lags, alpha=0.05)
        elif plot_type.lower() == "pacf":
            vals, confint = pacf(series, nlags=lags, method="ywm", alpha=0.05)
        else:
            raise ValueError("Loại hàm không hợp lệ. Chọn 'acf' hoặc 'pacf'.")

        lag_indices = list(range(len(vals)))
        lower_bounds = [float(c[0] - v) for c, v in zip(confint, vals)]
        upper_bounds = [float(c[1] - v) for c, v in zip(confint, vals)]

        response_data = {
            "col_name": col_name,
            "plot_type": plot_type.upper(),
            "lags": lag_indices,
            "values": [float(v) for v in vals],
            "confidence_interval_upper": upper_bounds,
            "confidence_interval_lower": lower_bounds,
        }
        return clean_nan_and_inf(response_data)
