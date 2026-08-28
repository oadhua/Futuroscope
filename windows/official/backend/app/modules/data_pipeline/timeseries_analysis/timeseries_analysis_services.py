import logging
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np
import warnings
from sqlalchemy import text
from sqlalchemy.orm import Session
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tools.sm_exceptions import InterpolationWarning

from app.modules.data_pipeline.queries import get_dynamic_gathering_query
from app.modules.data_pipeline.utils import clean_nan_and_inf

logger = logging.getLogger(__name__)


def get_available_attractions(db: Session) -> List[str]:
    """Lấy danh sách ID điểm tham quan từ bảng fact_attraction_hourly."""
    try:
        query = text(
            "SELECT DISTINCT id_attraction FROM fact_attraction_hourly "
            "WHERE id_attraction IS NOT NULL ORDER BY 1 ASC;"
        )
        result = db.execute(query).fetchall()
        attractions = [str(row[0]) for row in result if row[0]]
        return attractions if attractions else ["DEFAULT"]
    except Exception as e:
        logger.warning(f"Không thể lấy danh sách attractions: {e}")
        return ["DEFAULT"]


def get_available_features(
    db: Session, id_attraction: Optional[str] = None, version: str = "v0_raw"
) -> List[str]:
    """Lấy danh sách cột khả dụng dựa trên version và id_attraction."""
    try:
        df_sample = TimeSeriesService.fetch_dataframe_version(
            db=db, version=version, id_attraction=id_attraction, limit=500
        )

        excluded = {"datetime", "id_attraction", "created_at", "updated_at"}
        all_cols = [col for col in df_sample.columns if col not in excluded]

        valid_features = []
        for col in all_cols:
            is_core_metric = any(
                k in col.lower()
                for k in ["visitor", "count", "energy", "ec", "elec", "conso"]
            )
            if is_core_metric or (
                col in df_sample.columns and df_sample[col].notna().any()
            ):
                valid_features.append(col)

        return valid_features if valid_features else all_cols

    except Exception as e:
        logger.warning(f"Không thể lấy danh sách features từ DB: {e}")
        return ["visitor_count"]


class TimeSeriesService:
    @classmethod
    def fetch_dataframe_version(
        cls,
        db: Session,
        version: str = "v0_raw",
        id_attraction: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        """Đọc DataFrame chuẩn xác theo Version và Attraction (Schema data_prep)."""
        conn = db.connection()
        target_attr = id_attraction.upper() if id_attraction else "ALL"

        # 1. Version thô v0_raw: SQL Pivot động
        if not version or version == "v0_raw":
            query_sql = get_dynamic_gathering_query(
                db.bind, id_attraction=target_attr, version="v0_raw"
            )
            if limit:
                query_sql = f"SELECT * FROM ({query_sql}) AS subquery LIMIT {limit}"
            return pd.read_sql(text(query_sql), conn)

        # 2. Các version khác trong schema data_prep
        schema_name = "data_prep"
        query_str = f'SELECT * FROM "{schema_name}"."{version}"'

        cols_query = text("""
            SELECT column_name FROM information_schema.columns 
            WHERE table_schema = :s AND table_name = :t AND column_name = 'id_attraction'
        """)
        has_id_attr_col = conn.execute(
            cols_query, {"s": schema_name, "t": version}
        ).fetchone()

        if target_attr != "ALL" and has_id_attr_col:
            query_str += " WHERE id_attraction = :attr_id"
            if limit:
                query_str += f" LIMIT {limit}"
            df = pd.read_sql(text(query_str), conn, params={"attr_id": target_attr})
        else:
            if limit:
                query_str += f" LIMIT {limit}"
            df = pd.read_sql(text(query_str), conn)

        if target_attr != "ALL" and "id_attraction" in df.columns:
            df = df[df["id_attraction"] == target_attr].copy()

        return df

    @classmethod
    def _fetch_series_data(
        cls,
        db: Session,
        version: str,
        col_name: str,
        id_attraction: Optional[str] = None,
    ) -> pd.Series:
        try:
            if hasattr(col_name, "value"):
                col_name = col_name.value
            if hasattr(id_attraction, "value"):
                id_attraction = id_attraction.value

            df = cls.fetch_dataframe_version(
                db=db, version=version, id_attraction=id_attraction
            )

            if df.empty:
                raise ValueError(
                    f"Không có dữ liệu cho version '{version}' và attraction '{id_attraction}'."
                )

            if col_name not in df.columns:
                raise ValueError(
                    f"Không tìm thấy cột '{col_name}' trong phiên bản '{version}'."
                )

            df["datetime"] = pd.to_datetime(df["datetime"])
            df[col_name] = pd.to_numeric(df[col_name], errors="coerce")

            is_all_attractions = not id_attraction or id_attraction.upper() == "ALL"

            # Gom nhóm chuỗi thời gian
            if is_all_attractions and "id_attraction" in df.columns:
                if any(
                    k in col_name.lower()
                    for k in ["visitor", "count", "energy", "ec", "elec", "conso"]
                ):
                    df_grouped = df.groupby("datetime", as_index=False)[col_name].sum(
                        min_count=1
                    )
                else:
                    df_grouped = df.groupby("datetime", as_index=False)[col_name].mean()
            else:
                df_grouped = df.sort_values("datetime").drop_duplicates(
                    subset=["datetime"]
                )

            min_date = df_grouped["datetime"].min()
            max_date = df_grouped["datetime"].max()
            full_index = pd.date_range(start=min_date, end=max_date, freq="h")

            series = pd.Series(
                data=df_grouped[col_name].values, index=df_grouped["datetime"]
            )
            series = series.reindex(full_index)

            first_valid_idx = series.first_valid_index()
            if first_valid_idx is not None:
                series = (
                    series.loc[first_valid_idx:]
                    .interpolate(method="linear")
                    .ffill()
                    .fillna(0)
                )
            else:
                series = series.fillna(0)

            return series

        except Exception as e:
            logger.error(f"Lỗi trong _fetch_series_data: {str(e)}", exc_info=True)
            raise ValueError(f"Lỗi xử lý dữ liệu chuỗi thời gian: {str(e)}")

    @classmethod
    def analyze_stationarity(
        cls,
        db: Session,
        version: str,
        col_name: str,
        method: str,
        id_attraction: Optional[str] = None,
    ) -> Dict[str, Any]:
        series = cls._fetch_series_data(db, version, col_name, id_attraction)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", InterpolationWarning)

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
        return clean_nan_and_inf(response_data)

    @classmethod
    def decompose_time_series(
        cls,
        db: Session,
        version: str,
        col_name: str,
        model_type: str,
        period: int,
        id_attraction: Optional[str] = None,
    ) -> Dict[str, Any]:
        series = cls._fetch_series_data(db, version, col_name, id_attraction)

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
        return clean_nan_and_inf(response_data)

    @classmethod
    def calculate_acf_pacf(
        cls,
        db: Session,
        version: str,
        col_name: str,
        plot_type: str,
        lags: int,
        id_attraction: Optional[str] = None,
    ) -> Dict[str, Any]:
        series = cls._fetch_series_data(db, version, col_name, id_attraction)

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
