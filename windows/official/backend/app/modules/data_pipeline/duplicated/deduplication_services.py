import json
import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from sqlalchemy import text

from app.core.database import engine
from app.modules.data_pipeline.queries import get_dynamic_gathering_query

logger = logging.getLogger(__name__)

SCHEMA_NAME = "data_prep"


class DeduplicationService:
    METHOD_LABELS_FR = {
        "keep_first": "Suppression des doublons (Conserver le premier)",
        "keep_last": "Suppression des doublons (Conserver le dernier)",
        "drop_all": "Suppression de toutes les occurrences dupliquées",
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
            if conn.execute(
                cols_query, {"s": SCHEMA_NAME, "t": version_id}
            ).fetchone():
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

    # ==================== LOGIQUE DE DEDUPLICATION ====================

    @classmethod
    def process_deduplication(
        cls,
        df: pd.DataFrame,
        subset_columns: Optional[List[str]],
        keep: str,
    ) -> Tuple[pd.DataFrame, List[str], int, int, int]:
        """
        Thực thi kiểm tra và xóa các dòng trùng lặp trong DataFrame
        """
        rows_before = len(df)

        # Kiểm tra danh sách cột hợp lệ
        if subset_columns:
            valid_subset = [c for c in subset_columns if c in df.columns]
        else:
            valid_subset = list(df.columns)

        if not valid_subset:
            valid_subset = list(df.columns)

        # Chuyển tham số keep cho pandas drop_duplicates
        # keep='none' trong Schema tương ứng keep=False trong pandas
        keep_param: Any = False if keep == "none" else keep

        # Thực thi xóa trùng lặp
        df_out = df.drop_duplicates(subset=valid_subset, keep=keep_param).reset_index(drop=True)
        rows_after = len(df_out)
        duplicates_removed = rows_before - rows_after

        return df_out, valid_subset, rows_before, rows_after, duplicates_removed

    # ==================== GESTION DES VERSIONS ====================

    @classmethod
    def execute_and_version(
        cls,
        parent_version_id: str,
        id_attraction: Optional[str],
        subset_columns: Optional[List[str]],
        keep: str,
    ) -> Dict[str, Any]:
        cls._init_db_schema()
        id_attr = id_attraction or "ALL"
        parent_df = cls.load_version_dataframe(
            parent_version_id, id_attraction=id_attr
        )
        subset_columns = subset_columns or []

        processed_df, valid_subset, rows_before, rows_after, duplicates_removed = (
            cls.process_deduplication(parent_df, subset_columns, keep)
        )

        # Xác định key của phương thức
        method_key = f"keep_{keep}" if keep in ["first", "last"] else "drop_all"

        # Lấy số version tiếp theo
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

        new_version_id = f"v{next_ver}_dedup_{keep}{attr_suffix}"

        saved_file_path = cls.save_version_dataframe(new_version_id, processed_df)

        cols_str = ", ".join(valid_subset) if len(valid_subset) < len(parent_df.columns) else "ALL_COLUMNS"

        stats_dict = {
            "rows_before": rows_before,
            "rows_after": rows_after,
            "duplicates_removed": duplicates_removed,
            "subset_columns_used": valid_subset,
        }

        params_dict = {
            "subset_columns": valid_subset,
            "keep": keep,
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
                    "step": "deduplication",
                    "method": method_key,
                    "label_fr": cls.METHOD_LABELS_FR.get(method_key, method_key),
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
            "step_type": "deduplication",
            "method": method_key,
            "method_label_fr": cls.METHOD_LABELS_FR.get(method_key, method_key),
            "target_columns": valid_subset,
            "parameters": params_dict,
            "stats": stats_dict,
            "file_path": saved_file_path,
            "created_at": datetime.now().isoformat(),
        }

    @classmethod
    def get_version_stats(
        cls, version_id: str, id_attraction: str = "ALL"
    ) -> Tuple[List[str], int, int, List[str]]:
        """
        Lấy thông tin các cột, tổng bản ghi, số bản ghi bị trùng lặp hiện tại và danh sách attraction
        """
        df = cls.load_version_dataframe(version_id, id_attraction=id_attraction)

        available_columns = list(df.columns)
        total_records = len(df)
        total_duplicates = int(df.duplicated().sum())

        available_attractions = []
        if "id_attraction" in df.columns:
            available_attractions = sorted(
                df["id_attraction"].astype(str).unique().tolist()
            )

        return available_columns, total_records, total_duplicates, available_attractions