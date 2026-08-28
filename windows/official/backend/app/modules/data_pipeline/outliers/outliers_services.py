import json
import logging
import re
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
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

    MAX_CAPACITY_MAP = {
        "H03": 750,
        "H07": 800,
    }
    DEFAULT_MAX_CAPACITY = 1200

    # ==================== KHO LƯU TRỮ POSTGRESQL ====================

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
        """Lấy danh sách các cột số, thống kê số lượng outlier và danh sách id_attraction khả dụng."""
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
        )
        return f"postgresql://{SCHEMA_NAME}/{version_id}"

    # ==================== LUẬT NGHIỆP VỤ OUTLIER FUTUROSCOPE ====================

    @classmethod
    def apply_visitor_outlier_rules(
        cls, df: pd.DataFrame, action: str
    ) -> Tuple[pd.DataFrame, Dict[str, int], int]:
        df_out = df.copy()
        outliers_count = {}
        rows_affected_set = set()

        if "visitor_count" in df_out.columns:
            cond_negative = df_out["visitor_count"] < 0

            if "id_attraction" in df_out.columns:
                max_caps = (
                    df_out["id_attraction"]
                    .astype(str)
                    .map(cls.MAX_CAPACITY_MAP)
                    .fillna(cls.DEFAULT_MAX_CAPACITY)
                )
            else:
                max_caps = cls.DEFAULT_MAX_CAPACITY

            cond_exceed = df_out["visitor_count"] > max_caps
            outlier_mask = cond_negative | cond_exceed

            outliers_count["visitor_count"] = int(outlier_mask.sum())
            rows_affected_set.update(df_out[outlier_mask].index.tolist())

            if action in ["cap", "clip_iqr"]:
                df_out["visitor_count"] = df_out["visitor_count"].clip(
                    lower=0, upper=max_caps
                )
            elif action == "nullify":
                df_out.loc[outlier_mask, "visitor_count"] = np.nan
            elif action == "drop":
                df_out = df_out[~outlier_mask]

        return df_out, outliers_count, len(rows_affected_set)

    @classmethod
    def apply_energy_outlier_rules(
        cls, df: pd.DataFrame, action: str
    ) -> Tuple[pd.DataFrame, Dict[str, int], int]:
        df_out = df.copy()
        outliers_count = {}
        rows_affected_set = set()

        energy_cols = [
            c
            for c in df_out.columns
            if any(k in c.lower() for k in ["elec", "ec", "kwh", "kvarh"])
        ]

        for col in energy_cols:
            cond_neg = df_out[col] < 0
            cond_high = df_out[col] > 500.0

            outlier_mask = cond_neg | cond_high
            cnt = int(outlier_mask.sum())
            outliers_count[col] = cnt
            if cnt > 0:
                rows_affected_set.update(df_out[outlier_mask].index.tolist())

            if action in ["cap", "clip_iqr"]:
                df_out[col] = df_out[col].clip(lower=0.0, upper=500.0)
            elif action == "nullify":
                df_out.loc[outlier_mask, col] = np.nan
            elif action == "drop":
                df_out = df_out[~outlier_mask]

        return df_out, outliers_count, len(rows_affected_set)

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
            return cls.apply_energy_outlier_rules(df, action)

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

        # 1. IQR Method
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

        # 2. Z-Score Method
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

        # 3. Isolation Forest
        elif method == "isolation_forest":
            contamination = float(params.get("contamination", 0.05))
            clean_data = df_out[numeric_cols].fillna(df_out[numeric_cols].median())

            model = IsolationForest(contamination=contamination, random_state=42)
            preds = model.fit_predict(clean_data)
            outlier_mask = preds == -1

            rows_affected_set.update(df_out[outlier_mask].index.tolist())
            for col in numeric_cols:
                outliers_count[col] = int(outlier_mask.sum())
                if action in ["cap", "clip_iqr"]:
                    q_low = df_out[col].quantile(0.01)
                    q_high = df_out[col].quantile(0.99)
                    df_out.loc[outlier_mask, col] = df_out.loc[outlier_mask, col].apply(
                        lambda x: q_high
                        if x > q_high
                        else (q_low if x < q_low else df_out[col].median())
                    )
                elif action == "nullify":
                    df_out.loc[outlier_mask, col] = np.nan

        # 4. Local Outlier Factor (LOF)
        elif method == "lof":
            n_neighbors = int(params.get("n_neighbors", 20))
            clean_data = df_out[numeric_cols].fillna(df_out[numeric_cols].median())

            lof = LocalOutlierFactor(n_neighbors=n_neighbors)
            preds = lof.fit_predict(clean_data)
            outlier_mask = preds == -1

            rows_affected_set.update(df_out[outlier_mask].index.tolist())
            for col in numeric_cols:
                outliers_count[col] = int(outlier_mask.sum())
                if action in ["cap", "clip_iqr"]:
                    q_low = df_out[col].quantile(0.01)
                    q_high = df_out[col].quantile(0.99)
                    df_out.loc[outlier_mask, col] = df_out.loc[outlier_mask, col].apply(
                        lambda x: q_high
                        if x > q_high
                        else (q_low if x < q_low else df_out[col].median())
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
        new_version_id = f"v{next_ver}_{method}_{action}{attr_suffix}"  # (hoặc _{method} bên missing value)

        saved_file_path = cls.save_version_dataframe(new_version_id, processed_df)

        cols_str = ", ".join(target_columns) if target_columns else "ALL_NUMERIC"
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
            "target_columns": target_columns,
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
