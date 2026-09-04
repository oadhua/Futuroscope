import json
import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sqlalchemy import text
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

from app.core.database import engine
from app.modules.data_pipeline.queries import get_dynamic_gathering_query
from app.modules.data_pipeline.utils import clean_nan_and_inf

logger = logging.getLogger(__name__)

SCHEMA_NAME = "data_prep"


class OutlierService:
    METHOD_LABELS_FR = {
        "iqr": "Écart Interquartile (IQR)",
        "z_score": "Score Z (Z-Score)",
        "isolation_forest": "Forêt d'isolement (Isolation Forest)",
        "lof": "Facteur Local d'Anomalie (LOF)",
        "visitor_domain_rules": "Règles Métier Outliers - Fréquentation",
        "energy_domain_rules": "Règles Métier Outliers - Énergie",
    }

    ATTRACTION_CAPACITIES = {
        "H03": {"visitor": 750, "electricite": 250.0, "thermal": 325.0},
        "H07": {"visitor": 800, "electricite": 410.0, "thermal": 350.0},
    }
    DEFAULT_VISITOR_CAPACITY = 1200
    DEFAULT_ELECTRICITE_CAPACITY = 500.0
    DEFAULT_THERMAL_CAPACITY = 500.0
    MIN_CONSECUTIVE_FLAT_HOURS = 3

    # ==================== KHO LƯU TRỮ POSTGRESQL ====================

    @classmethod
    def _init_db_schema(cls):
        queries = [
            f"CREATE SCHEMA IF NOT EXISTS {SCHEMA_NAME};",
            f"""
            CREATE TABLE IF NOT EXISTS {SCHEMA_NAME}.data_version_registry (
                version_id VARCHAR(100) PRIMARY KEY,
                parent_version_id VARCHAR(100),
                id_attraction VARCHAR(50) DEFAULT 'ALL',
                step_type VARCHAR(100),
                method VARCHAR(100),
                method_label_fr VARCHAR(255),
                target_columns TEXT,
                parameters JSONB,
                stats JSONB,
                file_path VARCHAR(255),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """,
        ]
        with engine.connect() as conn:
            for q in queries:
                conn.execute(text(q))
            conn.commit()

    @classmethod
    def _load_metadata_store(cls) -> Dict[str, Any]:
        cls._init_db_schema()
        query = text(
            f"SELECT * FROM {SCHEMA_NAME}.data_version_registry ORDER BY created_at DESC"
        )

        with engine.connect() as conn:
            rows = conn.execute(query).mappings().all()

        metadata_store = {}
        for row in rows:
            v_id = row["version_id"]
            cols = row["target_columns"].split(", ") if row["target_columns"] else []
            metadata_store[v_id] = {
                "version_id": v_id,
                "parent_version_id": row["parent_version_id"],
                "id_attraction": row.get("id_attraction", "ALL"),
                "step_type": row["step_type"],
                "method": row["method"],
                "method_label_fr": row["method_label_fr"],
                "action": row["parameters"].get("action", "cap")
                if row["parameters"]
                else "cap",
                "target_columns": cols,
                "parameters": row["parameters"] if row["parameters"] else {},
                "stats": row["stats"]
                if row["stats"]
                else {"outliers_detected": {}, "rows_affected": 0},
                "file_path": row["file_path"],
                "created_at": row["created_at"].isoformat()
                if row["created_at"]
                else "",
            }
        return metadata_store

    @classmethod
    def load_version_dataframe(
        cls, version_id: str, id_attraction: str = "ALL"
    ) -> pd.DataFrame:
        cls._init_db_schema()

        if version_id == "v0_raw":
            try:
                query_sql = get_dynamic_gathering_query(
                    engine, id_attraction=id_attraction
                )
                with engine.connect() as conn:
                    return pd.read_sql(text(query_sql), conn)
            except Exception as e:
                raise ValueError(f"Lỗi khi truy vấn v0_raw từ Database: {str(e)}")

        if not re.match(r"^[a-zA-Z0-9_]+$", version_id):
            raise ValueError("Tên version_id chứa ký tự không hợp lệ.")

        with engine.connect() as conn:
            check_sql = text("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_schema = :s AND table_name = :t
                );
            """)
            exists = conn.execute(
                check_sql, {"s": SCHEMA_NAME, "t": version_id}
            ).scalar()

            if not exists:
                raise FileNotFoundError(
                    f"Không tìm thấy phiên bản '{version_id}' trong Database."
                )

            order_clause = ""
            cols_query = text("""
                SELECT column_name FROM information_schema.columns 
                WHERE table_schema = :s AND table_name = :t AND column_name = 'datetime'
            """)
            if conn.execute(cols_query, {"s": SCHEMA_NAME, "t": version_id}).fetchone():
                order_clause = " ORDER BY datetime ASC"

            df = pd.read_sql(
                text(f'SELECT * FROM {SCHEMA_NAME}."{version_id}"{order_clause}'), conn
            )

            if (
                id_attraction
                and id_attraction != "ALL"
                and "id_attraction" in df.columns
            ):
                df = df[
                    df["id_attraction"].astype(str) == str(id_attraction)
                ].reset_index(drop=True)

            return df

    @classmethod
    def get_version_outlier_stats(
        cls,
        version_id: str,
        id_attraction: str = "ALL",
        method: str = "iqr",
        params: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, int], int, List[str]]:
        params = params or {}
        df_full = cls.load_version_dataframe(version_id, id_attraction="ALL")

        attractions = []
        if "id_attraction" in df_full.columns:
            attractions = sorted(
                [str(x) for x in df_full["id_attraction"].unique() if pd.notnull(x)]
            )

        df_target = df_full
        if (
            id_attraction
            and id_attraction != "ALL"
            and "id_attraction" in df_full.columns
        ):
            df_target = df_full[
                df_full["id_attraction"].astype(str) == str(id_attraction)
            ].reset_index(drop=True)

        numeric_cols = df_target.select_dtypes(include=[np.number]).columns
        outlier_counts = {}

        if method == "iqr":
            factor = float(params.get("iqr_factor", 1.5))
            for col in numeric_cols:
                q1 = df_target[col].quantile(0.25)
                q3 = df_target[col].quantile(0.75)
                iqr = q3 - q1
                lower = q1 - factor * iqr
                upper = q3 + factor * iqr
                mask = (df_target[col] < lower) | (df_target[col] > upper)
                outlier_counts[col] = int(mask.sum())
        elif method == "z_score":
            threshold = float(params.get("z_threshold", 3.0))
            for col in numeric_cols:
                std_val = df_target[col].std()
                if std_val > 0:
                    mean_val = df_target[col].mean()
                    z_scores = (df_target[col] - mean_val).abs() / std_val
                    outlier_counts[col] = int((z_scores > threshold).sum())
                else:
                    outlier_counts[col] = 0
        else:
            for col in numeric_cols:
                outlier_counts[col] = 0

        return (
            clean_nan_and_inf({k: int(v) for k, v in outlier_counts.items()}),
            len(df_target),
            attractions,
        )

    @classmethod
    def save_version_dataframe(cls, version_id: str, df: pd.DataFrame) -> str:
        cls._init_db_schema()

        if not re.match(r"^[a-zA-Z0-9_]+$", version_id):
            raise ValueError("Tên version_id chứa ký tự không hợp lệ.")

        df.to_sql(
            name=version_id,
            con=engine,
            schema=SCHEMA_NAME,
            if_exists="replace",
            index=False,
            chunksize=5000,
            method="multi",
        )
        return f"postgresql://{SCHEMA_NAME}/{version_id}"

    # ==================== LUẬT NGHIỆP VỤ OUTLIER FUTUROSCOPE ====================

    @classmethod
    def apply_visitor_outlier_rules(
        cls, df: pd.DataFrame, action: str = "cap"
    ) -> Tuple[pd.DataFrame, Dict[str, int], int]:
        """Xử lý Physical Outliers, Capping sức chứa + IQR, và Flatlines cho visitor_count."""
        df_out = df.copy()
        outliers_count = {}
        rows_affected_set = set()

        if "visitor_count" not in df_out.columns:
            return df_out, outliers_count, 0

        if "id_attraction" in df_out.columns and "datetime" in df_out.columns:
            df_out["datetime"] = pd.to_datetime(df_out["datetime"])
            df_out = df_out.sort_values(by=["id_attraction", "datetime"]).reset_index(
                drop=True
            )
        else:
            df_out = df_out.reset_index(drop=True)

        df_out["_visitor_orig"] = df_out["visitor_count"].copy()

        # 1. Ép trần sức chứa thực tế trước (Physical Capping)
        if "id_attraction" in df_out.columns:
            for att_id in df_out["id_attraction"].unique():
                str_id = str(att_id)
                cap_capacity = cls.ATTRACTION_CAPACITIES.get(str_id, {}).get(
                    "visitor", cls.DEFAULT_VISITOR_CAPACITY
                )
                att_mask = df_out["id_attraction"] == att_id
                df_out.loc[att_mask, "visitor_count"] = df_out.loc[
                    att_mask, "visitor_count"
                ].clip(upper=cap_capacity)

        cond_negative = df_out["visitor_count"] < 0
        cond_closed = pd.Series(False, index=df_out.index)
        if "ouvert" in df_out.columns:
            cond_closed |= df_out["ouvert"] == 0
        if "is_open" in df_out.columns:
            cond_closed |= df_out["is_open"] == 0

        upper_bounds = {}

        # 2. Tính IQR Upper Bound & kết hợp max(iqr_upper, cap_capacity)
        if "id_attraction" in df_out.columns:
            open_mask = df_out["visitor_count"] > 0
            open_df = df_out[open_mask].copy()

            if not open_df.empty:
                q1_series = open_df.groupby("id_attraction")["visitor_count"].transform(
                    lambda s: s.quantile(0.25)
                )
                q3_series = open_df.groupby("id_attraction")["visitor_count"].transform(
                    lambda s: s.quantile(0.75)
                )
                iqr_series = q3_series - q1_series
                open_df["_iqr_upper"] = q3_series + 1.5 * iqr_series

                upper_map = (
                    open_df.groupby("id_attraction")["_iqr_upper"].first().to_dict()
                )
            else:
                upper_map = {}

            for att_id in df_out["id_attraction"].unique():
                str_id = str(att_id)
                cap_capacity = cls.ATTRACTION_CAPACITIES.get(str_id, {}).get(
                    "visitor", cls.DEFAULT_VISITOR_CAPACITY
                )
                iqr_upper = upper_map.get(att_id, cap_capacity)

                upper_bounds[str_id] = (
                    max(iqr_upper, cap_capacity) if cap_capacity else iqr_upper
                )

            target_upper = (
                df_out["id_attraction"]
                .astype(str)
                .map(upper_bounds)
                .fillna(cls.DEFAULT_VISITOR_CAPACITY)
            )
        else:
            open_data = df_out[df_out["visitor_count"] > 0]["visitor_count"]
            if len(open_data) > 0:
                Q1 = open_data.quantile(0.25)
                Q3 = open_data.quantile(0.75)
                IQR = Q3 - Q1
                final_upper_bound = max(Q3 + 1.5 * IQR, cls.DEFAULT_VISITOR_CAPACITY)
            else:
                final_upper_bound = cls.DEFAULT_VISITOR_CAPACITY
            upper_bounds["DEFAULT"] = final_upper_bound
            target_upper = final_upper_bound

        cond_exceed = df_out["visitor_count"] > target_upper
        outlier_mask = cond_negative | cond_exceed

        rows_affected_set.update(df_out[outlier_mask].index.tolist())

        if action in ["cap", "clip_iqr"]:
            if isinstance(target_upper, pd.Series):
                df_out["visitor_count"] = np.clip(
                    df_out["visitor_count"], 0, target_upper
                )
            else:
                df_out["visitor_count"] = df_out["visitor_count"].clip(
                    lower=0, upper=target_upper
                )
        elif action == "nullify":
            df_out.loc[outlier_mask, "visitor_count"] = np.nan

        df_out.loc[cond_closed, "visitor_count"] = 0

        # 3. Xử lý Flatlines
        plateau_indices_all = []
        att_list = (
            df_out["id_attraction"].unique()
            if "id_attraction" in df_out.columns
            else ["ALL"]
        )

        for att_id in att_list:
            sub_df = (
                df_out[df_out["id_attraction"] == att_id] if att_id != "ALL" else df_out
            )

            is_diff = (sub_df["visitor_count"] != sub_df["visitor_count"].shift(1)) | (
                sub_df["visitor_count"] == 0
            )
            block_ids = is_diff.cumsum()

            block_sizes = sub_df.groupby(block_ids)["visitor_count"].transform("count")
            plateau_mask = (block_sizes >= cls.MIN_CONSECUTIVE_FLAT_HOURS) & (
                sub_df["visitor_count"] > 0
            )

            plateau_indices_all.extend(sub_df[plateau_mask].index.tolist())

        if plateau_indices_all:
            rows_affected_set.update(plateau_indices_all)
            df_out.loc[plateau_indices_all, "visitor_count"] = np.nan

            if "id_attraction" in df_out.columns:
                df_out["prev_day_visitor"] = df_out.groupby("id_attraction")[
                    "_visitor_orig"
                ].shift(24)
            else:
                df_out["prev_day_visitor"] = df_out["_visitor_orig"].shift(24)

            nan_mask = df_out["visitor_count"].isna()
            df_out.loc[nan_mask, "visitor_count"] = df_out.loc[
                nan_mask, "prev_day_visitor"
            ]

            if df_out["visitor_count"].isna().sum() > 0:
                if "id_attraction" in df_out.columns:
                    df_out["visitor_count"] = df_out.groupby("id_attraction")[
                        "visitor_count"
                    ].transform(
                        lambda grp: grp.interpolate(method="linear").ffill().bfill()
                    )
                else:
                    df_out["visitor_count"] = (
                        df_out["visitor_count"]
                        .interpolate(method="linear")
                        .ffill()
                        .bfill()
                    )

            df_out.drop(columns=["prev_day_visitor"], inplace=True, errors="ignore")

        df_out.loc[cond_closed, "visitor_count"] = 0

        if action in ["cap", "clip_iqr"]:
            if "id_attraction" in df_out.columns:
                for att_id, bound in upper_bounds.items():
                    mask = df_out["id_attraction"].astype(str) == str(att_id)
                    df_out.loc[mask, "visitor_count"] = df_out.loc[
                        mask, "visitor_count"
                    ].clip(lower=0, upper=bound)
            else:
                bound = upper_bounds.get("DEFAULT", cls.DEFAULT_VISITOR_CAPACITY)
                df_out["visitor_count"] = df_out["visitor_count"].clip(
                    lower=0, upper=bound
                )

        df_out["visitor_count"] = df_out["visitor_count"].fillna(0).round().astype(int)
        df_out.drop(columns=["_visitor_orig"], inplace=True, errors="ignore")

        outliers_count["visitor_count"] = len(rows_affected_set)

        return df_out, outliers_count, len(rows_affected_set)

    @classmethod
    def apply_energy_outlier_rules(
        cls, df: pd.DataFrame, regle_config: dict, action: str = "impute"
    ) -> Tuple[pd.DataFrame, Dict[str, int], int]:
        """Tự động xử lý Outliers đồng thời cho Điện năng (logic 2 Tầng H03_elec_5.py),

        Nhiệt năng (H03_thermal_2.py) và Lượt khách. Trực tiếp sử dụng các cột
        thời gian tiếng Pháp (heure, jour, mois, semaine, annee).
        """
        df_out = df.copy()
        outliers_count: Dict[str, int] = {}
        rows_affected_set = set()

        if df_out.empty:
            return df_out, outliers_count, 0

        # Trích xuất thời gian tạm thời nếu thiếu
        hour_col = (
            "heure"
            if "heure" in df_out.columns
            else ("hour" if "hour" in df_out.columns else None)
        )
        day_col = (
            "jour"
            if "jour" in df_out.columns
            else ("day" if "day" in df_out.columns else None)
        )
        month_col = (
            "mois"
            if "mois" in df_out.columns
            else ("month" if "month" in df_out.columns else None)
        )
        week_col = (
            "semaine"
            if "semaine" in df_out.columns
            else ("week" if "week" in df_out.columns else None)
        )
        year_col = (
            "annee"
            if "annee" in df_out.columns
            else ("year" if "year" in df_out.columns else None)
        )

        if "datetime" in df_out.columns:
            df_out["_dt"] = pd.to_datetime(df_out["datetime"])
        elif "date" in df_out.columns and hour_col and hour_col in df_out.columns:
            df_out["_dt"] = pd.to_datetime(
                df_out["date"].astype(str)
                + " "
                + df_out[hour_col].astype(str)
                + ":00:00"
            )

        if "_dt" in df_out.columns:
            dt_s = df_out["_dt"].dt
            if not hour_col:
                df_out["heure"] = dt_s.hour
                hour_col = "heure"
            if not month_col:
                df_out["mois"] = dt_s.month
                month_col = "mois"
            if not day_col:
                df_out["jour"] = dt_s.day
                day_col = "jour"
            if not week_col:
                df_out["semaine"] = dt_s.isocalendar().week
                week_col = "semaine"
            if not year_col:
                df_out["annee"] = dt_s.year
                year_col = "annee"

        attractions = (
            df_out["id_attraction"].dropna().unique()
            if "id_attraction" in df_out.columns
            else ["DEFAULT"]
        )

        # 1. Bắt danh sách cột Điện năng chuẩn
        elec_cols = regle_config.get(
            "elec_cols",
            [
                c
                for c in df_out.columns
                if any(k in c.lower() for k in ["elec", "kwh", "kvarh", "puissance"])
            ],
        )
        elec_cols = [c for c in elec_cols if c in df_out.columns]

        # 2. Bắt danh sách cột Nhiệt năng (Loại trừ hoàn toàn các cột Điện năng ở trên)
        thermal_cols = regle_config.get(
            "thermal_cols",
            [
                c
                for c in df_out.columns
                if any(
                    k in c.lower()
                    for k in ["ec_", "nhiet", "thermal", "chaud", "froid"]
                )
                and c not in elec_cols  # <--- CHỐT CHẶN TRÁNH BỊ QUÉT TRÙNG
                and not any(
                    k in c.lower() for k in ["elec", "kwh", "kvarh", "puissance"]
                )
            ],
        )
        thermal_cols = [c for c in thermal_cols if c in df_out.columns]

        for col in elec_cols + thermal_cols:
            outliers_count[col] = 0
        if "visitor_count" in df_out.columns:
            outliers_count["visitor_count"] = 0

        processed_dfs = []

        for att_id in attractions:
            if "id_attraction" in df_out.columns:
                sub_df = df_out[df_out["id_attraction"] == att_id].copy()
            else:
                sub_df = df_out.copy()

            if "_dt" in sub_df.columns:
                sub_df = sub_df.sort_values("_dt")

            str_att_id = str(att_id)
            att_limits = cls.ATTRACTION_CAPACITIES.get(str_att_id, {})
            default_v_cap = att_limits.get("visitor", cls.DEFAULT_VISITOR_CAPACITY)
            default_t_cap = att_limits.get("thermal", cls.DEFAULT_THERMAL_CAPACITY)

            # =========================================================================
            # XỬ LÝ ĐIỆN NĂNG (Elec) - CHUẨN 100% LOCAL (H03_ELEC_5.PY / H03_ELEC_3.PY)
            # =========================================================================
            if elec_cols:
                iqr_coef_elec = regle_config.get("iqr_multiplier_elec", 2.5)

                for col in elec_cols:
                    if col not in sub_df.columns:
                        continue

                    s = sub_df[col].copy()

                    # ---------------- TẦNG 1: LỌC CỤC BỘ (LOCAL MAD & RULES) ----------------
                    window_size = 7
                    roll_med = s.rolling(
                        window=window_size, center=True, min_periods=1
                    ).median()
                    roll_mad = (
                        (s - roll_med)
                        .abs()
                        .rolling(window=window_size, center=True, min_periods=1)
                        .median()
                    )

                    dynamic_threshold = np.maximum(4.0 * roll_mad, 0.15 * roll_med)
                    mask_spike_local = (s - roll_med).abs() > dynamic_threshold

                    mask_flatline = (
                        s.groupby(s.diff().ne(0).cumsum()).transform("count") >= 2
                    )
                    mask_zero_op = (
                        (sub_df["is_open"] == 1) & (s <= 0.5)
                        if "is_open" in sub_df.columns
                        else pd.Series(False, index=sub_df.index)
                    )
                    mask_negative = s < 0.0

                    mask_tier1 = (
                        mask_spike_local | mask_flatline | mask_zero_op | mask_negative
                    )

                    # ---------------- TẦNG 2: LỌC IQR [year x week] ----------------
                    s_temp = s.copy()
                    s_temp[mask_tier1] = np.nan

                    year_week_group = [
                        c for c in [year_col, week_col] if c and c in sub_df.columns
                    ]

                    if len(year_week_group) == 2:
                        grp = s_temp.groupby([sub_df[c] for c in year_week_group])
                        q1_yw = grp.transform(lambda x: x.quantile(0.25))
                        q3_yw = grp.transform(lambda x: x.quantile(0.75))
                        iqr_yw = q3_yw - q1_yw

                        upper_bound_iqr = q3_yw + iqr_coef_elec * iqr_yw
                        lower_bound_iqr = np.maximum(
                            0.0, q1_yw - iqr_coef_elec * iqr_yw
                        )

                        mask_tier2 = (s_temp > upper_bound_iqr) | (
                            s_temp < lower_bound_iqr
                        )
                    else:
                        mask_tier2 = pd.Series(False, index=sub_df.index)

                    # Tổng hợp mask Outlier
                    outlier_mask = mask_tier1 | mask_tier2
                    cnt = int(outlier_mask.sum())
                    outliers_count[col] += cnt

                    # ---------------- ĐẮP DỮ LIỆU CHUẨN LOCAL ----------------
                    if cnt > 0:
                        rows_affected_set.update(sub_df[outlier_mask].index.tolist())

                        # 1. Tạo bản sao sạch (Gán NaN các điểm Outlier)
                        df_clean_col = s.copy()
                        df_clean_col[outlier_mask] = np.nan

                        # 2. Điền Trung vị theo [annee x week x is_open x hour]
                        group_keys = [
                            c
                            for c in [year_col, week_col, "is_open", hour_col]
                            if c and c in sub_df.columns
                        ]

                        if group_keys:
                            context_median = df_clean_col.groupby(
                                [sub_df[c] for c in group_keys]
                            ).transform("median")
                            imputed_s = df_clean_col.fillna(context_median)
                        else:
                            imputed_s = df_clean_col.fillna(roll_med)

                        # 3. Fallback Trung vị theo [week x hour]
                        fallback_keys = [
                            c for c in [week_col, hour_col] if c and c in sub_df.columns
                        ]
                        if len(fallback_keys) == 2:
                            fallback_median = df_clean_col.groupby(
                                [sub_df[c] for c in fallback_keys]
                            ).transform("median")
                            imputed_s = imputed_s.fillna(fallback_median)

                        # 4. Nội suy mượt giáp ranh (limit=2) & Khóa sàn >= 0
                        sub_df[col] = imputed_s.interpolate(
                            method="linear", limit=2, limit_direction="both"
                        ).clip(lower=0.0)

            # =========================================================================
            # 2. XỬ LÝ NHIỆT NĂNG (Thermal) - TỐI GIẢN (ĐỒNG BỘ VỚI ĐIỆN NĂNG)
            # =========================================================================
            if thermal_cols:
                max_thermal_cap = (
                    regle_config.get("max_thermal_capacity") or default_t_cap
                )
                iqr_coef_thermal = regle_config.get("iqr_multiplier_thermal", 8.0)

                if month_col and month_col in sub_df.columns:
                    m = sub_df[month_col]
                elif "_dt" in sub_df.columns:
                    m = sub_df["_dt"].dt.month
                else:
                    m = pd.Series(1, index=sub_df.index)

                if day_col and day_col in sub_df.columns:
                    d = sub_df[day_col]
                elif "_dt" in sub_df.columns:
                    d = sub_df["_dt"].dt.day
                else:
                    d = pd.Series(1, index=sub_df.index)

                # Mùa hè tại Futuroscope: 15/05 đến 15/10
                mask_summer = (
                    ((m > 5) & (m < 10))
                    | ((m == 5) & (d >= 15))
                    | ((m == 10) & (d <= 15))
                )
                mask_winter = ~mask_summer

                for target_col in thermal_cols:
                    if target_col not in sub_df.columns:
                        continue

                    s = sub_df[target_col].copy()

                    # 2.1. Nhận diện Flatline (>= 4 giờ liên tiếp giữ nguyên giá trị)
                    diff = s.diff().ne(0)
                    run_id = diff.cumsum()
                    run_lengths = s.groupby(run_id).transform("count")

                    flat_positive = (run_lengths >= 4) & (s > 0)
                    flat_zero_winter = (run_lengths >= 4) & (s == 0) & mask_winter
                    mask_flatline = flat_positive | flat_zero_winter

                    # 2.2. Nhận diện giá trị âm & Outlier động
                    mask_negative = s < 0.0

                    valid_grp_cols = [
                        c
                        for c in ["type_frequentation", "is_open", hour_col]
                        if c and c in sub_df.columns
                    ]

                    if valid_grp_cols:
                        grp_q1 = sub_df.groupby(valid_grp_cols)[target_col].transform(
                            lambda x: x.quantile(0.25)
                        )
                        grp_q3 = sub_df.groupby(valid_grp_cols)[target_col].transform(
                            lambda x: x.quantile(0.75)
                        )
                        grp_iqr = grp_q3 - grp_q1
                        dynamic_upper = np.maximum(
                            grp_q3 + iqr_coef_thermal * grp_iqr, max_thermal_cap
                        )
                        mask_profile_outlier = (s > dynamic_upper) & mask_winter
                    else:
                        mask_profile_outlier = (s > max_thermal_cap) & mask_winter

                    mask_summer_noise = mask_summer & (s > 0.5)

                    outlier_mask = (
                        mask_negative
                        | mask_flatline
                        | mask_profile_outlier
                        | mask_summer_noise
                    )

                    cnt = int(outlier_mask.sum())
                    outliers_count[target_col] += cnt

                    # 2.3. Xử lý trực tiếp: Chuyển Outlier thành NaN & Nội suy đa tầng
                    if cnt > 0:
                        rows_affected_set.update(sub_df[outlier_mask].index.tolist())

                        sub_df.loc[outlier_mask, target_col] = np.nan

                        # Tầng 1: Mùa hè ép về 0.0
                        sub_df.loc[
                            mask_summer & sub_df[target_col].isna(),
                            target_col,
                        ] = 0.0

                        # Tầng 2: Impute Mùa Lạnh theo Profile Local
                        p_keys = [
                            k
                            for k in [
                                month_col,
                                "type_frequentation",
                                "is_open",
                                hour_col,
                            ]
                            if k and k in sub_df.columns
                        ]
                        if p_keys and mask_winter.any():
                            mean_l1 = (
                                sub_df[mask_winter]
                                .groupby(p_keys)[target_col]
                                .transform("median")
                            )
                            sub_df[target_col] = sub_df[target_col].fillna(mean_l1)

                        # Tầng 3: Dự phòng [is_open x heure]
                        backup_keys_l2 = [
                            k
                            for k in ["is_open", hour_col]
                            if k and k in sub_df.columns
                        ]
                        if (
                            backup_keys_l2
                            and sub_df[target_col].isna().sum() > 0
                            and mask_winter.any()
                        ):
                            mean_l2 = (
                                sub_df[mask_winter]
                                .groupby(backup_keys_l2)[target_col]
                                .transform("median")
                            )
                            sub_df[target_col] = sub_df[target_col].fillna(mean_l2)

                        # Tầng 4: Dự phòng [heure]
                        if (
                            hour_col
                            and hour_col in sub_df.columns
                            and sub_df[target_col].isna().sum() > 0
                            and mask_winter.any()
                        ):
                            mean_l3 = (
                                sub_df[mask_winter]
                                .groupby(hour_col)[target_col]
                                .transform("median")
                            )
                            sub_df[target_col] = sub_df[target_col].fillna(mean_l3)

                        # Tầng 5: Nội suy thời gian 2 chiều
                        if (
                            "_dt" in sub_df.columns
                            and sub_df[target_col].isna().sum() > 0
                        ):
                            s_time = (
                                sub_df.set_index("_dt")[target_col]
                                .interpolate(
                                    method="time",
                                    limit_direction="both",
                                )
                                .reset_index(drop=True)
                            )
                            s_time.index = sub_df.index
                            sub_df[target_col] = sub_df[target_col].fillna(s_time)

                        # Tầng 6: Global Mean
                        if sub_df[target_col].isna().sum() > 0:
                            global_mean = sub_df[target_col].mean()
                            sub_df[target_col] = sub_df[target_col].fillna(
                                global_mean if pd.notna(global_mean) else 0.0
                            )

                        sub_df[target_col] = sub_df[target_col].clip(lower=0.0)

            # =========================================================================
            # 3. XỬ LÝ LƯỢT KHÁCH (Visitor Count - Cố định nghiệp vụ theo Attraction Capacity)
            # =========================================================================
            if "visitor_count" in sub_df.columns:
                cutoff_str = regle_config.get("visitor_cutoff_date", "2024-02-10")

                # Lấy capacity riêng của từng attraction (H03: 750, H07: 800, ...)
                max_v_cap = regle_config.get("max_visitor_capacity") or default_v_cap

                # 1. Mask thời gian <= 10/02/2024
                dt_mask = (
                    sub_df["_dt"].dt.floor("D") <= pd.to_datetime(cutoff_str)
                    if "_dt" in sub_df.columns
                    else pd.Series(True, index=sub_df.index)
                )

                # 2. Tín hiệu vận hành & Đóng cửa thực sự
                has_op = (sub_df.get("ouvert", 0) > 0) | (
                    sub_df.get("operation", 0) > 0
                )
                pure_closed = (sub_df.get("is_open", 0) == 0) & (~has_op)

                # 3. Phát hiện Outliers lượt khách (chỉ trong khoảng cutoff)
                mask_v_out = dt_mask & (
                    (pure_closed & (sub_df["visitor_count"] > 0))
                    | (sub_df["visitor_count"] > max_v_cap)
                    | (sub_df["visitor_count"] < 0)
                )

                cnt_v = int(mask_v_out.sum())
                outliers_count["visitor_count"] += cnt_v

                # 4. Trực tiếp thực thi xử lý Outlier
                if cnt_v > 0:
                    rows_affected_set.update(sub_df[mask_v_out].index.tolist())

                    # a. Ép lượt khách = 0 nếu đóng cửa thực sự (chỉ áp dụng trong khoảng cutoff)
                    sub_df.loc[dt_mask & pure_closed, "visitor_count"] = 0

                    # b. Capping max capacity riêng của attraction và chặn sàn >= 0
                    v_sub = sub_df.loc[dt_mask, "visitor_count"]
                    sub_df.loc[dt_mask, "visitor_count"] = v_sub.clip(
                        lower=0, upper=max_v_cap
                    )

            # --- [QUAN TRỌNG]: LƯU SUB_DF ĐÃ XỬ LÝ VÀO LIST PROCESSED_DFS ---
            processed_dfs.append(sub_df)

        # Gộp dữ liệu và Restore nguyên vẹn Index ban đầu
        df_res = pd.concat(processed_dfs, axis=0)

        if action == "drop" and rows_affected_set:
            df_res = df_res.drop(index=list(rows_affected_set), errors="ignore")
        else:
            df_res = df_res.reindex(df_out.index)

        if "_dt" in df_res.columns:
            df_res.drop(columns=["_dt"], inplace=True)

        return df_res, outliers_count, len(rows_affected_set)

    # ==================== ĐIỀU PHỐI VÀ BIẾN ĐỔI ====================

    @classmethod
    def process_outliers(
        cls,
        df: pd.DataFrame,
        target_columns: List[str],
        method: str,
        action: str,
        params: Dict[str, Any],
    ) -> Tuple[pd.DataFrame, Dict[str, int], int]:
        method = method.lower()
        action = action.lower()
        params = params or {}

        if method == "visitor_domain_rules":
            return cls.apply_visitor_outlier_rules(df, action)
        elif method == "energy_domain_rules":
            return cls.apply_energy_outlier_rules(
                df, regle_config=params, action=action
            )

        df_out = df.copy()

        valid_cols = (
            [c for c in target_columns if c in df_out.columns]
            if target_columns
            else [
                c for c in df_out.columns if np.issubdtype(df_out[c].dtype, np.number)
            ]
        )

        numeric_cols = [
            c for c in valid_cols if np.issubdtype(df_out[c].dtype, np.number)
        ]
        if not numeric_cols:
            return df_out, {}, 0

        outliers_count = {col: 0 for col in numeric_cols}
        rows_affected_set = set()

        if method == "iqr":
            factor = float(params.get("iqr_factor", 1.5))
            for col in numeric_cols:
                q1 = df_out[col].quantile(0.25)
                q3 = df_out[col].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - factor * iqr
                upper_bound = q3 + factor * iqr

                mask = (df_out[col] < lower_bound) | (df_out[col] > upper_bound)
                outliers_count[col] = int(mask.sum())
                rows_affected_set.update(df_out[mask].index.tolist())

                if action in ["cap", "clip_iqr"]:
                    df_out[col] = df_out[col].clip(lower=lower_bound, upper=upper_bound)
                elif action == "nullify":
                    df_out.loc[mask, col] = np.nan

        elif method == "z_score":
            threshold = float(params.get("z_threshold", 3.0))
            for col in numeric_cols:
                mean_val = df_out[col].mean()
                std_val = df_out[col].std()
                if std_val > 0:
                    z_scores = (df_out[col] - mean_val) / std_val
                    mask = z_scores.abs() > threshold
                    outliers_count[col] = int(mask.sum())
                    rows_affected_set.update(df_out[mask].index.tolist())

                    if action in ["cap", "clip_iqr"]:
                        lower_bound = mean_val - threshold * std_val
                        upper_bound = mean_val + threshold * std_val
                        df_out[col] = df_out[col].clip(
                            lower=lower_bound, upper=upper_bound
                        )
                    elif action == "nullify":
                        df_out.loc[mask, col] = np.nan

        elif method in ["isolation_forest", "lof"]:
            clean_data = df_out[numeric_cols].fillna(df_out[numeric_cols].median())

            if method == "isolation_forest":
                contamination = float(params.get("contamination", 0.05))
                model = IsolationForest(contamination=contamination, random_state=42)
            else:
                n_neighbors = int(params.get("n_neighbors", 20))
                model = LocalOutlierFactor(n_neighbors=n_neighbors)

            preds = model.fit_predict(clean_data)
            outlier_mask = preds == -1
            rows_affected_set.update(df_out[outlier_mask].index.tolist())

            q_lows = df_out[numeric_cols].quantile(0.01)
            q_highs = df_out[numeric_cols].quantile(0.99)
            medians = df_out[numeric_cols].median()

            for col in numeric_cols:
                outliers_count[col] = int(outlier_mask.sum())
                if action in ["cap", "clip_iqr"]:
                    col_vals = df_out.loc[outlier_mask, col]
                    cond_high = col_vals > q_highs[col]
                    cond_low = col_vals < q_lows[col]

                    df_out.loc[outlier_mask, col] = np.select(
                        [cond_high, cond_low],
                        [q_highs[col], q_lows[col]],
                        default=medians[col],
                    )
                elif action == "nullify":
                    df_out.loc[outlier_mask, col] = np.nan

        if action == "drop" and rows_affected_set:
            df_out = df_out.drop(index=list(rows_affected_set)).reset_index(drop=True)

        return df_out, outliers_count, len(rows_affected_set)

    # ==================== QUẢN LÝ VERSION ====================

    @classmethod
    def execute_and_version(
        cls,
        parent_version_id: str,
        id_attraction: Optional[str],
        target_columns: Optional[List[str]],
        method: str,
        action: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        cls._init_db_schema()
        id_attr = id_attraction or "ALL"
        parent_df = cls.load_version_dataframe(parent_version_id, id_attraction=id_attr)
        target_columns = target_columns or []

        processed_df, outliers_count, rows_affected = cls.process_outliers(
            parent_df, target_columns, method, action, params
        )

        with engine.connect() as conn:
            rows = conn.execute(
                text(
                    f"SELECT version_id FROM {SCHEMA_NAME}.data_version_registry WHERE id_attraction = :id_attr"
                ),
                {"id_attr": id_attr},
            ).fetchall()

        existing_nums = []
        for r in rows:
            v_id = r[0]
            match = re.match(r"^v(\d+)_", v_id)
            if match:
                existing_nums.append(int(match.group(1)))

        next_ver = max(existing_nums) + 1 if existing_nums else 1
        attr_suffix = f"_{id_attr}" if id_attr != "ALL" else ""

        new_version_id = f"v{next_ver}_{method}_{action}{attr_suffix}"

        saved_file_path = cls.save_version_dataframe(new_version_id, processed_df)

        # =========================================================================
        # CẬP NHẬT: TỰ ĐỘNG BẮT DÚNG CỘT BỊ TÁC ĐỘNG THAY VÌ ĐỂ "ALL_NUMERIC"
        # =========================================================================
        if method in ["visitor_domain_rules", "energy_domain_rules"]:
            # Chỉ lấy các cột thực sự xuất hiện trong kết quả thống kê outliers_count
            effective_cols = list(outliers_count.keys())
        else:
            effective_cols = target_columns

        cols_str = ", ".join(effective_cols) if effective_cols else "NONE"
        params_dict = params or {}
        params_dict["action"] = action

        stats_dict = {
            "outliers_detected": clean_nan_and_inf(outliers_count),
            "rows_affected": rows_affected,
        }

        insert_sql = text(f"""
            INSERT INTO {SCHEMA_NAME}.data_version_registry 
            (version_id, parent_version_id, id_attraction, step_type, method, method_label_fr, target_columns, parameters, stats, file_path)
            VALUES (:v_id, :p_id, :id_attr, :step, :method, :label_fr, :cols, :params, :stats, :f_path)
        """)

        with engine.connect() as conn:
            conn.execute(
                insert_sql,
                {
                    "v_id": new_version_id,
                    "p_id": parent_version_id,
                    "id_attr": id_attr,
                    "step": "outlier_treatment",
                    "method": method,
                    "label_fr": cls.METHOD_LABELS_FR.get(method, method),
                    "cols": cols_str,
                    "params": json.dumps(params_dict),
                    "stats": json.dumps(stats_dict),
                    "f_path": saved_file_path,
                },
            )
            conn.commit()

        return {
            "version_id": new_version_id,
            "parent_version_id": parent_version_id,
            "id_attraction": id_attr,
            "step_type": "outlier_treatment",
            "method": method,
            "method_label_fr": cls.METHOD_LABELS_FR.get(method, method),
            "action": action,
            "target_columns": effective_cols,
            "parameters": params_dict,
            "stats": stats_dict,
            "file_path": saved_file_path,
            "created_at": datetime.now().isoformat(),
        }

    @classmethod
    def delete_version(cls, version_id: str) -> Dict[str, str]:
        if version_id == "v0_raw":
            raise ValueError("Không thể xóa phiên bản dữ liệu gốc (v0_raw).")

        metadata_store = cls._load_metadata_store()

        if version_id not in metadata_store:
            raise ValueError(f"Không tìm thấy phiên bản '{version_id}'.")

        children = [
            v_id
            for v_id, v_info in metadata_store.items()
            if v_info.get("parent_version_id") == version_id
        ]
        if children:
            raise ValueError(
                f"Không thể xóa '{version_id}' vì đang có các phiên bản phụ thuộc: {children}"
            )

        if not re.match(r"^[a-zA-Z0-9_]+$", version_id):
            raise ValueError("Tên version_id không hợp lệ.")

        with engine.connect() as conn:
            conn.execute(
                text(f'DROP TABLE IF EXISTS {SCHEMA_NAME}."{version_id}" CASCADE;')
            )
            conn.execute(
                text(
                    f"DELETE FROM {SCHEMA_NAME}.data_version_registry WHERE version_id = :v_id"
                ),
                {"v_id": version_id},
            )
            conn.commit()

        return {
            "status": "success",
            "message": f"Đã xóa thành công phiên bản '{version_id}' khỏi Database.",
        }
