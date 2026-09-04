import json
import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sqlalchemy import text
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

from app.core.database import engine
from app.modules.data_pipeline.queries import get_dynamic_gathering_query

logger = logging.getLogger(__name__)

SCHEMA_NAME = "data_prep"


class EncodingService:
    METHOD_LABELS_FR = {
        "one_hot": "Encodage One-Hot (OneHotEncoder)",
        "label": "Encodage Ordinal (LabelEncoder)",
        "target": "Encodage basés sur la Cible (Target Encoding)",
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

    # ==================== LOGIQUE D'ENCODAGE ====================

    @classmethod
    def process_encoding(
        cls,
        df: pd.DataFrame,
        target_columns: List[str],
        method: str,
        params: Dict[str, Any],
    ) -> Tuple[pd.DataFrame, List[str], List[str], int]:
        """
        Thực thi chuyển đổi Categorical Encoding:
        - One-Hot Encoding
        - Label Encoding
        - Target (Mean) Encoding
        """
        df_out = df.copy()

        exclude_cols = {
            "id_attraction",
            "datetime",
            "date",
            "heure",
            "jour",
            "mois",
            "annee",
        }

        # Xác định danh sách cột chuỗi/catégorie hợp lệ
        if target_columns:
            valid_cols = [
                c for c in target_columns if c in df_out.columns and c not in exclude_cols
            ]
        else:
            valid_cols = [
                c
                for c in df_out.columns
                if (
                    df_out[c].dtype == "object"
                    or isinstance(df_out[c].dtype, pd.CategoricalDtype)
                )
                and c not in exclude_cols
            ]

        if not valid_cols:
            return df_out, [], [], 0

        generated_cols = []

        # 1. ONE-HOT ENCODING
        if method == "one_hot":
            drop_first = params.get("drop_first", False)
            drop_param = "first" if drop_first else None

            # Khởi tạo encoder từ sklearn
            encoder = OneHotEncoder(sparse_output=False, drop=drop_param, handle_unknown="ignore")
            
            # Fit và transform các cột categorical
            encoded_array = encoder.fit_transform(df_out[valid_cols])
            
            # Lấy tên các cột mới tạo ra
            generated_cols = list(encoder.get_feature_names_out(valid_cols))
            
            # Tạo DataFrame mới từ mảng đã transform
            df_encoded = pd.DataFrame(
                encoded_array, 
                columns=generated_cols, 
                index=df_out.index
            )
            
            # Loại bỏ cột cũ và ghép các cột one-hot mới vào
            df_out = df_out.drop(columns=valid_cols)
            df_out = pd.concat([df_out, df_encoded], axis=1)

        # 2. LABEL / ORDINAL ENCODING
        elif method == "label":
            for col in valid_cols:
                le = LabelEncoder()
                df_out[col] = le.fit_transform(df_out[col].astype(str))
                generated_cols.append(col)

        # 3. TARGET ENCODING
        elif method == "target":
            target_var = params.get("target_variable")
            if not target_var or target_var not in df_out.columns:
                raise ValueError(
                    f"La variable cible '{target_var}' est requise et doit exister dans le DataFrame pour le Target Encoding."
                )

            global_mean = df_out[target_var].mean()
            for col in valid_cols:
                # Tính giá trị trung bình theo nhóm (Target Encoding đơn giản)
                means = df_out.groupby(col)[target_var].mean()
                df_out[f"{col}_encoded"] = df_out[col].map(means).fillna(global_mean)
                generated_cols.append(f"{col}_encoded")
                df_out = df_out.drop(columns=[col])

        else:
            raise ValueError(
                f"La méthode d'encodage '{method}' n'est pas prise en charge."
            )

        return df_out, valid_cols, generated_cols, len(df_out)

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
        parent_df = cls.load_version_dataframe(
            parent_version_id, id_attraction=id_attr
        )
        target_columns = target_columns or []
        params = params or {}

        processed_df, columns_encoded, generated_columns, rows_affected = (
            cls.process_encoding(parent_df, target_columns, method, params)
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

        new_version_id = f"v{next_ver}_encode_{method}{attr_suffix}"

        saved_file_path = cls.save_version_dataframe(new_version_id, processed_df)

        cols_str = ", ".join(columns_encoded) if columns_encoded else "ALL_CATEGORICAL"

        stats_dict = {
            "columns_encoded": columns_encoded,
            "generated_columns": generated_columns,
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
                    "step": "categorical_encoding",
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
            "step_type": "categorical_encoding",
            "method": method,
            "method_label_fr": cls.METHOD_LABELS_FR.get(method, method),
            "target_columns": columns_encoded,
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
        Lấy danh sách các cột catégorielle (chuỗi) khả dụng, tổng số bản ghi và danh sách id_attraction.
        """
        df = cls.load_version_dataframe(version_id, id_attraction=id_attraction)

        exclude_cols = {
            "id_attraction",
            "datetime",
            "date",
        }

        # Trả về các cột kiểu object/categorical để phục vụ lựa chọn bên Frontend
        available_columns = [
            c
            for c in df.columns
            if (
                df[c].dtype == "object"
                or isinstance(df[c].dtype, pd.CategoricalDtype)
            )
            and c not in exclude_cols
        ]

        total_records = len(df)

        available_attractions = []
        if "id_attraction" in df.columns:
            available_attractions = sorted(
                df["id_attraction"].astype(str).unique().tolist()
            )

        return available_columns, total_records, available_attractions