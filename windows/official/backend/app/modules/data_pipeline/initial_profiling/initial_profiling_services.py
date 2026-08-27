import traceback
from typing import List, Optional, Dict
import pandas as pd
from sqlalchemy import text
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
    def get_available_versions() -> List[DataVersionOption]:
        return [
            DataVersionOption(
                version_id="v1", version_label="Version 1 (Brute)", is_updated=False
            ),
            DataVersionOption(
                version_id="v2", version_label="Version 2 (Nettoyée)", is_updated=True
            ),
            DataVersionOption(
                version_id="all", version_label="Toutes les versions", is_updated=False
            ),
        ]

    @staticmethod
    def _map_dtype_to_ui_type(dtype_str: str) -> str:
        dtype_lower = str(dtype_str).lower()
        # Mở rộng điều kiện nhận diện kiểu số nguyên và số thực/decimal
        if any(t in dtype_lower for t in ["int", "bigint", "smallint", "integer"]):
            return "Integer"
        elif any(t in dtype_lower for t in ["float", "double", "numeric", "decimal", "real"]):
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
        cls, db: Session, version_id: str = "v1", id_attraction: Optional[str] = None
    ) -> InitialProfilingResponse:
        try:
            # 1. Génération de la requête SQL dynamique
            sql_query = get_dynamic_gathering_query(engine, id_attraction=id_attraction)

            # 2. Requête d'un échantillon de données avec connexion sécurisée
            conn = db.connection()

            limited_sql = text(
                f"WITH full_data AS ({sql_query}) SELECT * FROM full_data LIMIT 50;"
            )
            df_sample = pd.read_sql(limited_sql, conn)

            # 3. Comptage du nombre total de lignes
            count_sql = text(
                f"WITH full_data AS ({sql_query}) SELECT COUNT(*) AS total FROM full_data;"
            )
            total_rows = db.execute(count_sql).scalar() or 0

            # 4. Analyse du profil des colonnes et association des métadonnées
            # Backend chỉ cần phân tích schema từ DataFrame và trả về
            columns_profile = []
            for col_name in df_sample.columns:
                raw_dtype = str(df_sample[col_name].dtype)
                
                # Ép kiểu thủ công dựa trên tên cột nếu gặp trường hợp bị NULL toàn bộ (pandas nhận nhầm là object/string)
                col_lower = col_name.lower()
                if col_lower in ["visitor_count", "frequentation", "temps_id", "annee", "mois", "jour", "heure", "jour_semaine", "is_weekend", "jf", "is_open"]:
                    ui_type = "Integer"
                elif col_lower in ["ouvert", "interrompu", "operation", "conso_elec", "conso_ec", "surface", "capacite_salle", "capacite_file_attente", "capacite_pre_salle", "duree_longue", "duree_courte", "cycle_max_vl", "duty_cycle_max_vl", "cycle_max_vc", "duty_cycle_max_vc", "temperature", "humidite", "rayonnement_solaire", "day_degree_cold", "day_degree_hot", "temp_max", "temp_min", "temp_moy", "humidite_max", "humidite_min", "humidite_moy", "value"]:
                    ui_type = "Float"
                else:
                    ui_type = cls._map_dtype_to_ui_type(raw_dtype)

                columns_profile.append(
                    ColumnProfileSchema(
                        column_name=col_name,
                        column_type=ui_type,
                        raw_data_type=raw_dtype,
                        column_description="", # Để trống để Frontend tự map
                        column_owner="",       # Để trống để Frontend tự map
                    )
                )

            dataset_desc = (
                f"Jeu de données consolidé du Futuroscope (Version : {version_id})."
            )
            if id_attraction and id_attraction.upper() != "ALL":
                dataset_desc += f" Filtré par attraction : {id_attraction}."

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