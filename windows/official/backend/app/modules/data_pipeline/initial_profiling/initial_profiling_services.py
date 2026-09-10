import traceback
from typing import List, Optional, Dict
import pandas as pd
from sqlalchemy import inspect, text
from sqlalchemy.orm import Session
from fastapi import HTTPException, status

from app.core.database import engine
from app.modules.data_pipeline.queries import get_dynamic_gathering_query
from app.modules.data_pipeline.utils import clean_nan_and_inf
from app.modules.data_pipeline.initial_profiling.initial_profiling_schemas import (
    InitialProfilingResponse,
    ColumnProfileSchema,
    DataVersionOption,
)


class InitialProfilingService:
    @staticmethod
    def get_data_prep_versions(
        db: Session, id_attraction: Optional[str] = None
    ) -> List[Dict]:
        """
        Lấy danh sách các bảng trong schema 'data_prep' kèm theo thời gian chỉnh sửa an toàn.
        Nếu gặp lỗi bảng hỏng, tự động rollback transaction và chuyển sang fallback schema.
        """
        # 1. Truy vấn bằng information_schema kết hợp pg_stat_file an toàn
        query = text("""
            SELECT 
                t.table_name,
                (pg_stat_file(pg_relation_filepath(quote_ident(t.table_schema) || '.' || quote_ident(t.table_name)))).modification AS created_at
            FROM information_schema.tables t
            WHERE t.table_schema = 'data_prep' 
              AND t.table_type = 'BASE TABLE'
              AND t.table_name NOT IN ('data_version_registry', 'ml_model_registry')
            ORDER BY created_at ASC;
        """)

        try:
            result = db.execute(query).fetchall()
            versions_data = [
                {
                    "version_id": row[0],
                    "created_at": row[1].isoformat() if row[1] else None,
                }
                for row in result
            ]
        except Exception as e:
            # 🛑 QUAN TRỌNG: Rollback transaction bị hỏng để giải phóng Session trước khi query fallback
            db.rollback()
            print(f"[WARNING] Không thể lấy created_at bằng pg_stat_file: {str(e)}")

            # Truy vấn an toàn từ information_schema không chạm vào ổ đĩa
            fallback_query = text("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'data_prep' 
                  AND table_type = 'BASE TABLE'
                  AND table_name NOT IN ('data_version_registry', 'ml_model_registry')
                ORDER BY table_name ASC;
            """)
            result = db.execute(fallback_query).fetchall()
            versions_data = [
                {"version_id": row[0], "created_at": None} for row in result
            ]

        # 2. Đảm bảo luôn có v0_raw ở đầu danh sách
        has_v0 = any(v["version_id"] == "v0_raw" for v in versions_data)
        if not has_v0:
            versions_data.insert(
                0, {"version_id": "v0_raw", "created_at": "1970-01-01T00:00:00"}
            )

        # 3. Lọc theo id_attraction (nếu có chọn attraction)
        if id_attraction and id_attraction.upper() != "ALL":
            attr_upper = id_attraction.upper()
            versions_data = [
                v
                for v in versions_data
                if v["version_id"].lower().startswith("v")
                or attr_upper in v["version_id"].upper()
            ]

        return versions_data

    @classmethod
    def get_available_versions(
        cls, db: Session, id_attraction: Optional[str] = None
    ) -> List[DataVersionOption]:
        """Trả về danh sách DataVersionOption chứa thông tin version_id và created_at."""
        versions_data = cls.get_data_prep_versions(db, id_attraction)
        options = []
        for item in versions_data:
            v_id = item["version_id"]
            created_at = item["created_at"]
            label = "Version 0 (Brute)" if v_id == "v0_raw" else f"Version ({v_id})"
            is_up = v_id != "v0_raw"

            # Đảm bảo DataVersionOption schema của bạn có nhận tham số created_at
            options.append(
                DataVersionOption(
                    version_id=v_id,
                    version_label=label,
                    is_updated=is_up,
                    created_at=created_at,
                )
            )
        return options

    @staticmethod
    def _map_dtype_to_ui_type(dtype_str: str) -> str:
        dtype_lower = str(dtype_str).lower()
        if any(t in dtype_lower for t in ["int", "bigint", "smallint", "integer"]):
            return "Integer"
        elif any(
            t in dtype_lower for t in ["float", "double", "numeric", "decimal", "real"]
        ):
            return "Float"
        elif "datetime" in dtype_lower or "date" in dtype_lower:
            return "Date"
        elif "bool" in dtype_lower:
            return "Boolean"
        return "String"

    @staticmethod
    def get_attractions(db: Session) -> List[str]:
        """Récupère la liste des id_attraction uniques depuis la base PostgreSQL"""
        try:
            query = text("""
                SELECT DISTINCT id_attraction 
                FROM fact_elec_hourly 
                WHERE id_attraction IS NOT NULL AND id_attraction != ''
                ORDER BY id_attraction;
            """)
            result = db.execute(query).fetchall()
            return [row[0] for row in result]
        except Exception as e:
            print(f"Erreur lors de la récupération des attractions : {str(e)}")
            return []

    @classmethod
    def get_initial_profiling(
        cls,
        db: Session,
        version_id: str = "v0_raw",
        id_attraction: Optional[str] = None,
    ) -> InitialProfilingResponse:
        try:
            conn = db.connection()
            target_attr = id_attraction.upper() if id_attraction else "ALL"

            # 1. Xác định câu truy vấn SQL dựa vào phiên bản
            if version_id in ["v0_raw", "v1"]:
                sql_query = get_dynamic_gathering_query(
                    engine, id_attraction=target_attr
                )
                sample_sql = text(
                    f"WITH full_data AS ({sql_query}) SELECT * FROM full_data LIMIT 50;"
                )
                count_sql = text(
                    f"WITH full_data AS ({sql_query}) SELECT COUNT(*) AS total FROM full_data;"
                )

                df_sample = pd.read_sql(sample_sql, conn)
                total_rows = db.execute(count_sql).scalar() or 0
            else:
                schema_name = "data_prep"
                base_query = f'SELECT * FROM "{schema_name}"."{version_id}"'

                # Kiểm tra cột id_attraction trong bảng data_prep
                cols_query = text("""
                    SELECT column_name FROM information_schema.columns 
                    WHERE table_schema = :s AND table_name = :t AND column_name = 'id_attraction'
                """)
                has_id_attr = conn.execute(
                    cols_query, {"s": schema_name, "t": version_id}
                ).fetchone()

                if target_attr != "ALL" and has_id_attr:
                    sample_sql = text(
                        f"{base_query} WHERE id_attraction = :attr LIMIT 50"
                    )
                    count_sql = text(
                        f'SELECT COUNT(*) FROM "{schema_name}"."{version_id}" WHERE id_attraction = :attr'
                    )
                    params = {"attr": target_attr}
                    df_sample = pd.read_sql(sample_sql, conn, params=params)
                    total_rows = db.execute(count_sql, params).scalar() or 0
                else:
                    sample_sql = text(f"{base_query} LIMIT 50")
                    count_sql = text(
                        f'SELECT COUNT(*) FROM "{schema_name}"."{version_id}"'
                    )
                    df_sample = pd.read_sql(sample_sql, conn)
                    total_rows = db.execute(count_sql).scalar() or 0

            # 2. Phân tích cấu trúc các cột dữ liệu
            columns_profile = []
            for col_name in df_sample.columns:
                raw_dtype = str(df_sample[col_name].dtype)
                col_lower = col_name.lower()

                if col_lower in [
                    "visitor_count",
                    "frequentation",
                    "temps_id",
                    "annee",
                    "mois",
                    "jour",
                    "heure",
                    "jour_semaine",
                    "is_weekend",
                    "jf",
                    "is_open",
                ]:
                    ui_type = "Integer"
                elif col_lower in [
                    "ouvert",
                    "interrompu",
                    "operation",
                    "conso_elec",
                    "conso_ec",
                    "surface",
                    "capacite_salle",
                    "capacite_file_attente",
                    "capacite_pre_salle",
                    "duree_longue",
                    "duree_courte",
                    "cycle_max_vl",
                    "duty_cycle_max_vl",
                    "cycle_max_vc",
                    "duty_cycle_max_vc",
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
                    "value",
                ]:
                    ui_type = "Float"
                else:
                    ui_type = cls._map_dtype_to_ui_type(raw_dtype)

                columns_profile.append(
                    ColumnProfileSchema(
                        column_name=col_name,
                        column_type=ui_type,
                        raw_data_type=raw_dtype,
                        column_description="",
                        column_owner="",
                    )
                )

            dataset_desc = (
                f"Jeu de données consolidé du Futuroscope (Version : {version_id})."
            )
            if target_attr != "ALL":
                dataset_desc += f" Filtré par attraction : {target_attr}."

            result = {
                "selected_version": version_id,
                "dataset_description": dataset_desc,
                "total_columns": len(columns_profile),
                "total_rows": total_rows,
                "columns": columns_profile,
            }

            return InitialProfilingResponse(**clean_nan_and_inf(result))

        except Exception as e:
            print("=== JOURNAL D'ERREUR BASE DE DONNÉES ===")
            traceback.print_exc()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Erreur SQL / Base de données : {str(e)}",
            )