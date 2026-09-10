import json
import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sqlalchemy import text
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

from app.core.database import engine
from app.modules.data_pipeline.queries import get_dynamic_gathering_query
from app.modules.data_pipeline.utils import clean_nan_and_inf

logger = logging.getLogger(__name__)

SCHEMA_NAME = "data_prep"


class ScalingService:
    METHOD_LABELS_FR = {
        "standard": "Normalisation Standard (StandardScaler)",
        "min_max": "Normalisation Min-Max (MinMaxScaler)",
        "robust": "Normalisation Robuste (RobustScaler)",
    }

    # Tập hợp các cột cố định cần loại trừ khỏi Scaling
    EXCLUDE_COLS = {
        "id_attraction",
        "datetime",
        "date",
    }

    # ==================== GESTION DU STOCKAGE POSTGRESQL ====================

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
                raise ValueError(
                    f"Erreur lors de la requête v0_raw depuis la base de données : {str(e)}"
                )

        if not re.match(r"^[a-zA-Z0-9_]+$", version_id):
            raise ValueError("Le nom version_id contient des caractères invalides.")

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
                    f"La version '{version_id}' est introuvable dans la base de données."
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
    def save_version_dataframe(cls, version_id: str, df: pd.DataFrame) -> str:
        cls._init_db_schema()

        if not re.match(r"^[a-zA-Z0-9_]+$", version_id):
            raise ValueError("Le nom version_id contient des caractères invalides.")

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

    # ==================== LOGIQUE DE SCALING ====================

    @classmethod
    def process_scaling(
        cls,
        df: pd.DataFrame,
        target_columns: List[str],
        method: str,
        params: Dict[str, Any],
    ) -> Tuple[pd.DataFrame, List[str], int]:
        df_out = df.copy()

        # Lọc nghiêm ngặt: Chỉ chọn cột là kiểu số (np.number) và không thuộc EXCLUDE_COLS
        if target_columns:
            valid_cols = [
                c
                for c in target_columns
                if c in df_out.columns
                and c not in cls.EXCLUDE_COLS
                and np.issubdtype(df_out[c].dtype, np.number)
            ]
        else:
            valid_cols = [
                c
                for c in df_out.columns
                if np.issubdtype(df_out[c].dtype, np.number)
                and c not in cls.EXCLUDE_COLS
            ]

        if not valid_cols:
            return df_out, [], 0

        # Initialisation du Scaler
        if method == "standard":
            scaler = StandardScaler()
        elif method == "min_max":
            f_range = params.get("feature_range", [0.0, 1.0])
            scaler = MinMaxScaler(feature_range=(f_range[0], f_range[1]))
        elif method == "robust":
            scaler = RobustScaler()
        else:
            raise ValueError(
                f"La méthode de normalisation '{method}' n'est pas prise en charge."
            )

        # Application de la transformation
        df_out[valid_cols] = scaler.fit_transform(df_out[valid_cols])

        return df_out, valid_cols, len(df_out)

    # ==================== GESTION DES VERSIONS ====================

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
        params = params or {}

        processed_df, columns_scaled, rows_affected = cls.process_scaling(
            parent_df, target_columns, method, params
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

        new_version_id = f"v{next_ver}_scale_{method}{attr_suffix}"

        saved_file_path = cls.save_version_dataframe(new_version_id, processed_df)

        cols_str = ", ".join(columns_scaled) if columns_scaled else "ALL_NUMERIC"

        stats_dict = {
            "columns_scaled": columns_scaled,
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
                    "step": "feature_scaling",
                    "method": method,
                    "label_fr": cls.METHOD_LABELS_FR.get(method, method),
                    "cols": cols_str,
                    "params": json.dumps(params),
                    "stats": json.dumps(stats_dict),
                    "f_path": saved_file_path,
                },
            )
            conn.commit()

        return {
            "version_id": new_version_id,
            "parent_version_id": parent_version_id,
            "id_attraction": id_attr,
            "step_type": "feature_scaling",
            "method": method,
            "method_label_fr": cls.METHOD_LABELS_FR.get(method, method),
            "target_columns": columns_scaled,
            "parameters": params,
            "stats": stats_dict,
            "file_path": saved_file_path,
            "created_at": datetime.now().isoformat(),
        }

    @classmethod
    def get_version_stats(
        cls, version_id: str, id_attraction: str = "ALL"
    ) -> Tuple[List[str], int, List[str]]:
        """
        Lấy danh sách các cột số khả dụng, tổng số bản ghi và danh sách id_attraction từ version.
        """
        # 1. Load TOÀN BỘ dữ liệu (id_attraction="ALL") để lấy danh sách đầy đủ tất cả ID
        full_df = cls.load_version_dataframe(version_id, id_attraction="ALL")
        available_attractions = []
        if "id_attraction" in full_df.columns:
            available_attractions = sorted(
                full_df["id_attraction"].astype(str).unique().tolist()
            )

        # 2. Load dữ liệu theo id_attraction truyền vào để tính số dòng và các cột tương ứng
        df = cls.load_version_dataframe(version_id, id_attraction=id_attraction)

        available_columns = [
            c
            for c in df.columns
            if np.issubdtype(df[c].dtype, np.number) and c not in cls.EXCLUDE_COLS
        ]

        total_records = len(df)

        return available_columns, total_records, available_attractions
