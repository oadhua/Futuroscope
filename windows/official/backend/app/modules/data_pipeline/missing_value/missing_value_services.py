import json
import logging
import re
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from sqlalchemy import text
from sklearn.impute import KNNImputer

from app.core.database import engine
from app.modules.data_pipeline.queries import get_dynamic_gathering_query
from app.modules.data_pipeline.utils import clean_nan_and_inf

logger = logging.getLogger(__name__)

SCHEMA_NAME = "data_prep"


class MissingValueService:
    METHOD_LABELS_FR = {
        "mean": "Moyenne",
        "median": "Médiane",
        "mode": "Mode",
        "constant": "Constant",
        "bfill": "Remplissage en arrière",
        "ffill": "Remplissage en avant",
        "knn": "KNN",
        "linear": "Interpolation linéaire",
        "polynomial": "Interpolation polynomiale",
        "visitor_domain_rules": "Règles Métier - Fréquentation & Météo",
        "energy_domain_rules": "Règles Métier - Énergie (Elec/EC)",
    }

    # ==================== KHO LƯU TRỮ POSTGRESQL (SCHEMA DATA_PREP) ====================

    @classmethod
    def _init_db_schema(cls):
        """Khởi tạo Schema data_prep và bảng registry."""
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
            f"""
            ALTER TABLE {SCHEMA_NAME}.data_version_registry 
            ADD COLUMN IF NOT EXISTS id_attraction VARCHAR(50) DEFAULT 'ALL';
            """,
        ]
        with engine.connect() as conn:
            for q in queries:
                conn.execute(text(q))
            conn.commit()

    @classmethod
    def _load_metadata_store(cls) -> Dict[str, Any]:
        """Tải toàn bộ Metadata từ PostgreSQL Registry."""
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
                "target_columns": cols,
                "parameters": row["parameters"] if row["parameters"] else {},
                "stats": row["stats"]
                if row["stats"]
                else {"null_count_before": {}, "null_count_after": {}},
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
        """Đọc dữ liệu phiên bản từ Database PostgreSQL."""
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
            cols_query = text(f"""
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
                df = df[df["id_attraction"].astype(str) == str(id_attraction)]

            return df

    @classmethod
    def get_version_null_stats(
        cls, version_id: str, id_attraction: str = "ALL"
    ) -> Tuple[Dict[str, int], List[str]]:
        """Lấy danh sách các cột, số lượng null và danh sách id_attraction trong bảng."""
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
            ]

        null_counts = df_target.isnull().sum().to_dict()
        return clean_nan_and_inf(
            {k: int(v) for k, v in null_counts.items()}
        ), attractions

    @classmethod
    def save_version_dataframe(cls, version_id: str, df: pd.DataFrame) -> str:
        cls._init_db_schema()
        df.to_sql(
            name=version_id,
            con=engine,
            schema=SCHEMA_NAME,
            if_exists="replace",
            index=False,
        )
        return f"postgresql://{SCHEMA_NAME}/{version_id}"

    # ==================== LUẬT NGHIỆP VỤ FUTUROSCOPE ====================

    MAX_CAPACITY_MAP = {
        "H03": 750,
        "H07": 800,
    }
    DEFAULT_MAX_CAPACITY = 1200

    @classmethod
    def apply_visitor_domain_rules(cls, df: pd.DataFrame) -> pd.DataFrame:
        df_out = df.copy()

        # 1. Sắp xếp chuỗi thời gian chính xác
        if "datetime" in df_out.columns:
            df_out["datetime"] = pd.to_datetime(df_out["datetime"])
            sort_cols = [
                c for c in ["id_attraction", "datetime"] if c in df_out.columns
            ]
            if sort_cols:
                df_out = df_out.sort_values(by=sort_cols).reset_index(drop=True)

        # 2. Xử lý Trạng thái Hoạt động
        if "h_ouv" in df_out.columns:
            df_out["h_ouv"] = df_out["h_ouv"].fillna("00:00")
        if "h_ferm" in df_out.columns:
            df_out["h_ferm"] = df_out["h_ferm"].fillna("00:00")

        if "is_open" not in df_out.columns:
            df_out["is_open"] = 1
        else:
            df_out["is_open"] = df_out["is_open"].fillna(1)

        if "ouvert" in df_out.columns:
            if "type_frequentation" in df_out.columns and "heure" in df_out.columns:
                mask_open_valid = df_out["is_open"] == 1
                profil_moyen = (
                    df_out[mask_open_valid]
                    .groupby(["type_frequentation", "heure"])["ouvert"]
                    .transform("mean")
                )
                df_out["ouvert"] = df_out["ouvert"].fillna(profil_moyen)

            df_out["ouvert"] = (
                df_out["ouvert"].fillna(0.0).ffill().bfill().clip(0.0, 1.0).round(1)
            )

        df_out.loc[df_out["is_open"] == 0, "ouvert"] = 0.0
        cond_closed = (df_out["is_open"] == 0) | (df_out.get("ouvert", 1) == 0)

        # 3. Xử lý Trạng thái phụ
        if "interrompu" in df_out.columns:
            df_out["interrompu"] = (
                df_out["interrompu"].fillna(0.0).astype(float).round(1)
            )
            df_out.loc[cond_closed, "interrompu"] = 0.0

        if "ouvert" in df_out.columns and "interrompu" in df_out.columns:
            df_out["operation"] = (
                (df_out["ouvert"] - df_out["interrompu"]).clip(0.0, 1.0).round(1)
            )
            df_out.loc[cond_closed, "operation"] = 0.0

        # 4. Xử lý Lượt khách (visitor_count)
        if "visitor_count" in df_out.columns:
            df_out.loc[cond_closed, "visitor_count"] = 0.0

            if "id_attraction" in df_out.columns:
                df_out["prev_day_visitor"] = df_out.groupby("id_attraction")[
                    "visitor_count"
                ].shift(24)
            else:
                df_out["prev_day_visitor"] = df_out["visitor_count"].shift(24)

            nan_mask = df_out["visitor_count"].isna()
            df_out.loc[nan_mask, "visitor_count"] = df_out.loc[
                nan_mask, "prev_day_visitor"
            ]
            df_out.drop(columns=["prev_day_visitor"], inplace=True, errors="ignore")

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

            df_out.loc[cond_closed, "visitor_count"] = 0.0

            if "id_attraction" in df_out.columns:

                def cap_visitor(group):
                    att_id = str(group["id_attraction"].iloc[0])
                    cap = cls.MAX_CAPACITY_MAP.get(att_id, cls.DEFAULT_MAX_CAPACITY)
                    group["visitor_count"] = group["visitor_count"].clip(
                        lower=0, upper=cap
                    )
                    return group

                df_out = df_out.groupby("id_attraction", group_keys=False).apply(
                    cap_visitor
                )
            else:
                df_out["visitor_count"] = df_out["visitor_count"].clip(
                    lower=0, upper=cls.DEFAULT_MAX_CAPACITY
                )

            df_out["visitor_count"] = (
                df_out["visitor_count"].fillna(0).round().astype(int)
            )

        # 5. Xử lý dữ liệu thời tiết
        weather_cols = [
            "temperature",
            "humidite",
            "rayonnement_solaire",
            "day_degree_cold",
            "day_degree_hot",
            "temp_max",
            "temp_min",
            "temp_moy",
            "humidite_max",
            "humidite_min",
            "humidite_moy",
        ]
        weather_cols_in_df = [c for c in weather_cols if c in df_out.columns]

        if weather_cols_in_df:
            if "id_attraction" in df_out.columns:
                df_out[weather_cols_in_df] = df_out.groupby("id_attraction")[
                    weather_cols_in_df
                ].transform(
                    lambda grp: grp.interpolate(method="linear").ffill().bfill()
                )
            else:
                df_out[weather_cols_in_df] = (
                    df_out[weather_cols_in_df]
                    .interpolate(method="linear")
                    .ffill()
                    .bfill()
                )
            df_out[weather_cols_in_df] = df_out[weather_cols_in_df].round(1)

        return df_out

    @classmethod
    def apply_energy_domain_rules(cls, df: pd.DataFrame) -> pd.DataFrame:
        df_out = df.copy()

        if "datetime" in df_out.columns:
            df_out["datetime"] = pd.to_datetime(df_out["datetime"])
            sort_cols = [
                c for c in ["id_attraction", "datetime"] if c in df_out.columns
            ]
            if sort_cols:
                df_out = df_out.sort_values(by=sort_cols).reset_index(drop=True)

            if "heure" not in df_out.columns:
                df_out["heure"] = df_out["datetime"].dt.hour

        if "is_open" not in df_out.columns:
            df_out["is_open"] = 1
        else:
            df_out["is_open"] = df_out["is_open"].fillna(1)

        weather_cols = [
            c
            for c in ["temperature", "temp_moy", "rayonnement_solaire"]
            if c in df_out.columns
        ]
        if weather_cols:
            if "id_attraction" in df_out.columns:
                df_out[weather_cols] = df_out.groupby("id_attraction")[
                    weather_cols
                ].transform(
                    lambda grp: grp.interpolate(method="linear").ffill().bfill()
                )
            else:
                df_out[weather_cols] = (
                    df_out[weather_cols].interpolate(method="linear").ffill().bfill()
                )

        # Tự động quét các cột năng lượng động (bao gồm cột ghép alias dạng H03_elec_kwh...)
        energy_cols = [
            c
            for c in df_out.columns
            if any(k in c.lower() for k in ["elec", "ec", "kwh", "kvarh"])
        ]

        cond_closed = df_out["is_open"] == 0
        cond_open = df_out["is_open"] == 1

        for col in energy_cols:
            if cond_closed.any():
                closed_vals = df_out.loc[cond_closed, col].dropna()
                baseload = closed_vals.median() if not closed_vals.empty else 0.0
                df_out.loc[cond_closed & df_out[col].isna(), col] = baseload

            if cond_open.any() and df_out.loc[cond_open, col].isna().any():
                if "id_attraction" in df_out.columns:
                    hourly_profile = (
                        df_out[cond_open]
                        .groupby(["id_attraction", "heure"])[col]
                        .transform("mean")
                    )
                else:
                    hourly_profile = (
                        df_out[cond_open].groupby("heure")[col].transform("mean")
                    )
                df_out.loc[cond_open & df_out[col].isna(), col] = hourly_profile

            if df_out[col].isna().any():
                if "id_attraction" in df_out.columns:
                    df_out[col] = df_out.groupby("id_attraction")[col].transform(
                        lambda grp: grp.interpolate(method="linear").ffill().bfill()
                    )
                else:
                    df_out[col] = (
                        df_out[col].interpolate(method="linear").ffill().bfill()
                    )

            df_out[col] = df_out[col].clip(lower=0.0).round(2)

        return df_out

    # ==================== ĐIỀU PHỐI VÀ BIẾN ĐỔI ====================

    @classmethod
    def _apply_method_to_series(
        cls, series: pd.Series, method: str, params: Dict[str, Any]
    ) -> pd.Series:
        s = series.copy()

        if method == "mean":
            return s.fillna(s.mean())
        elif method == "median":
            return s.fillna(s.median())
        elif method == "mode":
            mode_val = s.mode()
            return s.fillna(mode_val[0]) if not mode_val.empty else s
        elif method == "constant":
            return s.fillna(params.get("constant_value", 0))
        elif method == "bfill":
            return s.bfill().ffill()
        elif method == "ffill":
            return s.ffill().bfill()
        elif method == "linear":
            return s.interpolate(method="linear").ffill().bfill()
        elif method == "polynomial":
            order = int(params.get("order", 2))
            if s.dropna().count() > order:
                s = s.interpolate(method="polynomial", order=order)
            else:
                s = s.interpolate(method="linear")
            return s.ffill().bfill()

        return s

    @classmethod
    @classmethod
    def apply_imputation(
        cls,
        df: pd.DataFrame,
        target_columns: List[str],
        method: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> pd.DataFrame:
        method = method.lower()
        params = params or {}

        # 1. Các phương pháp luật nghiệp vụ áp dụng trên toàn bộ dataframe
        if method == "visitor_domain_rules":
            return cls.apply_visitor_domain_rules(df)
        elif method == "energy_domain_rules":
            return cls.apply_energy_domain_rules(df)

        df_out = df.copy()

        # 2. Lấy danh sách cột cần xử lý
        if not target_columns:
            # Nếu người dùng không chọn cột nào, lấy tất cả các cột có chứa missing value
            valid_cols = [c for c in df_out.columns if df_out[c].isnull().sum() > 0]
        else:
            # Chọn danh sách các cột người dùng đã gửi lên từ UI
            valid_cols = [c for c in target_columns if c in df_out.columns]

        if not valid_cols:
            return df_out

        # 3. Lọc danh sách cột số (Numeric) để tránh lỗi khi dùng các thuật toán toán học/KNN
        numeric_cols = [
            c for c in valid_cols if np.issubdtype(df_out[c].dtype, np.number)
        ]
        non_numeric_cols = [c for c in valid_cols if c not in numeric_cols]

        # 4. Xử lý theo phương pháp KNN (Cần chạy đồng thời trên tập danh sách nhiều cột số)
        if method == "knn":
            n_neighbors = int(params.get("n_neighbors", 5))
            if numeric_cols:
                imputer = KNNImputer(n_neighbors=n_neighbors)
                # Chạy KNNImputer trên tất cả các cột số được chọn cùng lúc
                df_out[numeric_cols] = imputer.fit_transform(df_out[numeric_cols])
                # Quét lấp kín mép biên nếu có
                df_out[numeric_cols] = df_out[numeric_cols].ffill().bfill()

            # Đối với cột non-numeric khi chọn KNN, fallback sang ffill/bfill
            for col in non_numeric_cols:
                df_out[col] = df_out[col].ffill().bfill()

            return df_out

        # 5. Xử lý các phương pháp đơn biến (mean, median, linear, polynomial...) cho DANH SÁCH NHIỀU CỘT
        if "id_attraction" in df_out.columns and df_out["id_attraction"].nunique() > 1:
            # Nếu có nhiều id_attraction, nhóm theo id_attraction và apply từng cột trong danh sách
            for col in valid_cols:
                df_out[col] = df_out.groupby("id_attraction")[col].transform(
                    lambda grp: cls._apply_method_to_series(grp, method, params)
                )
        else:
            # Chạy vòng lặp qua từng cột trong danh sách nhiều cột được chọn
            for col in valid_cols:
                df_out[col] = cls._apply_method_to_series(df_out[col], method, params)

        return df_out

    # ==================== QUẢN LÝ VERSION ====================

    @classmethod
    def execute_and_version(
        cls,
        parent_version_id: str,
        id_attraction: Optional[str],
        target_columns: Optional[List[str]],
        method: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        cls._init_db_schema()
        id_attr = id_attraction or "ALL"
        parent_df = cls.load_version_dataframe(parent_version_id, id_attraction=id_attr)
        target_columns = target_columns or []

        check_cols = (
            target_columns
            if target_columns
            else [c for c in parent_df.columns if parent_df[c].isnull().sum() > 0]
        )

        before_nulls = clean_nan_and_inf(
            {
                col: int(parent_df[col].isnull().sum())
                for col in check_cols
                if col in parent_df.columns
            }
        )

        processed_df = cls.apply_imputation(parent_df, target_columns, method, params)

        after_nulls = clean_nan_and_inf(
            {col: int(processed_df[col].isnull().sum()) for col in processed_df.columns}
        )

        # ĐÁNH SỐ TĂNG DẦN v1, v2, v3 TOÀN CỤC (Không phụ thuộc id_attr nữa để tránh trùng lặp)
        with engine.connect() as conn:
            rows = conn.execute(
                text(f"SELECT version_id FROM {SCHEMA_NAME}.data_version_registry")
            ).fetchall()

        existing_nums = []
        for r in rows:
            v_id = r[0]
            match = re.match(r"^v(\d+)_", v_id)
            if match:
                existing_nums.append(int(match.group(1)))

        next_ver = max(existing_nums) + 1 if existing_nums else 1
        attr_suffix = f"_{id_attr}" if id_attr != "ALL" else ""

        # Tên version lúc này sẽ dạng: v1_mean_H03, v2_iqr_cap_H03, v3_iqr_cap_ALL,...
        new_version_id = f"v{next_ver}_{method}_{attr_suffix}"  # (hoặc _{method} bên missing value)

        saved_file_path = cls.save_version_dataframe(new_version_id, processed_df)

        cols_str = ", ".join(check_cols) if check_cols else ""
        params_dict = params or {}
        stats_dict = {
            "null_count_before": before_nulls,
            "null_count_after": after_nulls,
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
                    "step": "missing_value_imputation",
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
            "step_type": "missing_value_imputation",
            "method": method,
            "method_label_fr": cls.METHOD_LABELS_FR.get(method, method),
            "target_columns": check_cols,
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
