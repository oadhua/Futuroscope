import json
import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sqlalchemy import text

from app.core.database import engine
from app.modules.data_pipeline.queries import get_dynamic_gathering_query

logger = logging.getLogger(__name__)

SCHEMA_NAME = "data_prep"


class TimeSeriesTransformService:
    METHOD_LABELS_FR = {
        "first_diff": "Différenciation première (t - t-1)",
        "seasonal_diff": "Différenciation saisonnière (t - t-s)",
        "log_transform": "Transform de Logarithme (ln(x+1))",
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

    # ==================== LOGIQUE DE TRANSFORMATION ====================

    @classmethod
    def process_transformation(
        cls,
        df: pd.DataFrame,
        target_columns: List[str],
        method: str,
        seasonal_period: int = 24,
        drop_na: bool = True,
    ) -> Tuple[pd.DataFrame, List[str], int, int, int]:
        """
        Thực thi biến đổi chuỗi thời gian cho các cột được chọn
        """
        df_out = df.copy()
        rows_before = len(df_out)

        # Lọc các cột số hợp lệ
        valid_cols = [
            c for c in target_columns 
            if c in df_out.columns and pd.api.types.is_numeric_dtype(df_out[c])
        ]

        if not valid_cols:
            raise ValueError("Aucune colonne numérique valide n'a été sélectionnée pour la transformation.")

        # Xử lý theo phương pháp trong hình
        if method == "first_diff":
            # Soustraire chaque observation par son observation précédente
            for col in valid_cols:
                df_out[col] = df_out[col].diff(periods=1)

        elif method == "seasonal_diff":
            # Supprimer les tendances et les variations saisonnières (t - t_s)
            period = max(1, int(seasonal_period))
            for col in valid_cols:
                df_out[col] = df_out[col].diff(periods=period)

        elif method == "log_transform":
            # Appliquer le logarithme aux valeurs (dùng np.log1p để tránh log(0) bị -inf)
            for col in valid_cols:
                # Nếu có giá trị âm thì shift lên dương hoặc áp dụng log1p cho phần dương
                min_val = df_out[col].min()
                if min_val < 0:
                    df_out[col] = np.log1p(df_out[col] - min_val)
                else:
                    df_out[col] = np.log1p(df_out[col])

        else:
            raise ValueError(f"Méthode de transformation inconnue : {method}")

        # Xử lý NaN tạo ra do diff/lag
        nan_rows_dropped = 0
        if drop_na:
            df_out = df_out.dropna(subset=valid_cols).reset_index(drop=True)
            rows_after = len(df_out)
            nan_rows_dropped = rows_before - rows_after
        else:
            rows_after = len(df_out)

        return df_out, valid_cols, rows_before, rows_after, nan_rows_dropped

    # ==================== GESTION DES VERSIONS ====================

    @classmethod
    def execute_and_version(
        cls,
        parent_version_id: str,
        id_attraction: Optional[str],
        target_columns: List[str],
        method: str,
        seasonal_period: int = 24,
        drop_na: bool = True,
    ) -> Dict[str, Any]:
        cls._init_db_schema()
        id_attr = id_attraction or "ALL"
        parent_df = cls.load_version_dataframe(
            parent_version_id, id_attraction=id_attr
        )

        processed_df, valid_cols, rows_before, rows_after, nan_rows_dropped = (
            cls.process_transformation(
                parent_df, target_columns, method, seasonal_period, drop_na
            )
        )

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

        new_version_id = f"v{next_ver}_ts_{method}{attr_suffix}"

        saved_file_path = cls.save_version_dataframe(new_version_id, processed_df)

        stats_dict = {
            "rows_before": rows_before,
            "rows_after": rows_after,
            "nan_rows_dropped": nan_rows_dropped,
            "transformed_columns": valid_cols,
        }

        params_dict = {
            "target_columns": valid_cols,
            "method": method,
            "seasonal_period": seasonal_period,
            "drop_na": drop_na,
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
                    "step": "time_series_transform",
                    "method": method,
                    "label_fr": cls.METHOD_LABELS_FR.get(method, method),
                    "cols": ", ".join(valid_cols),
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
            "step_type": "time_series_transform",
            "method": method,
            "method_label_fr": cls.METHOD_LABELS_FR.get(method, method),
            "target_columns": valid_cols,
            "parameters": params_dict,
            "stats": stats_dict,
            "file_path": saved_file_path,
            "created_at": datetime.now().isoformat(),
        }

    @classmethod
    def get_version_stats(
        cls, version_id: str, id_attraction: str = "ALL"
    ) -> Tuple[List[str], List[str], int, List[str]]:
        df = cls.load_version_dataframe(version_id, id_attraction=id_attraction)

        available_columns = list(df.columns)
        numeric_columns = [
            c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])
        ]
        total_records = len(df)

        available_attractions = []
        if "id_attraction" in df.columns:
            available_attractions = sorted(
                df["id_attraction"].astype(str).unique().tolist()
            )

        return available_columns, numeric_columns, total_records, available_attractions